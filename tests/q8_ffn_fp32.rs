#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/env_restore.rs"]
mod env_restore;
#[path = "common/q8_projection.rs"]
#[allow(dead_code)]
mod fixture;

use anyhow::Result;
use env_restore::EnvRestore;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer, Q8FfnFp32Variant},
    infer::execution_weights::{
        DeviceQ8ExecutionView, DeviceQ8ExecutionWeight, HostQ8ExecutionWeight, Q8MatrixShape,
        Q8SourceIdentity, Q8SourceView,
    },
    profile,
};

const GUARD: usize = 8;
const GUARD_BITS: u32 = 0x4b12_3456;

#[derive(Clone, Copy, PartialEq, Eq)]
enum SplitKernel {
    Dynamic,
    TwoColumns,
    FixedK2048,
}

fn weights(k: usize, hidden: usize, seed: u8) -> Vec<u8> {
    let mut bytes = fixture::make_column_blocks(k, hidden);
    for (index, block) in bytes.chunks_exact_mut(34).enumerate() {
        if index % 7 == 0 {
            block[..2].copy_from_slice(&0u16.to_le_bytes());
        }
        for (lane, value) in block[2..].iter_mut().enumerate() {
            *value = [0x80, 0x7f, 0xff, 0x00, 0x01, 0x40, 0xc0, 0x08]
                [(lane + index + seed as usize) % 8];
        }
    }
    bytes
}

fn upload(ctx: &CudaContext, bytes: &[u8], tag: &'static str) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new_tagged(ctx, bytes.len(), tag)?;
    unsafe {
        ctx.memcpy_h2d(buffer.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())?;
    }
    Ok(buffer)
}

fn read(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<f32>> {
    let mut values = vec![0.0f32; buffer.bytes() / 4];
    unsafe {
        ctx.memcpy_d2h(values.as_mut_ptr().cast(), buffer.as_ptr(), buffer.bytes())?;
    }
    assert!(values[..GUARD]
        .iter()
        .chain(&values[values.len() - GUARD..])
        .all(|v| v.to_bits() == GUARD_BITS));
    Ok(values[GUARD..values.len() - GUARD].to_vec())
}

fn compare_case(
    ctx: &CudaContext,
    k: usize,
    hidden: usize,
    rows: usize,
    offset: usize,
) -> Result<()> {
    let input = fixture::make_input(rows, k);
    let mut guarded = vec![f32::from_bits(GUARD_BITS); rows * k + 2 * GUARD];
    guarded[GUARD..GUARD + rows * k].copy_from_slice(&input);
    let d_input = DeviceBuffer::new_tagged(ctx, guarded.len() * 4, "ffn-fp32:input")?;
    unsafe {
        ctx.memcpy_h2d(
            d_input.as_mut_ptr(),
            guarded.as_ptr().cast(),
            d_input.bytes(),
        )?;
    }
    let gate = weights(k, hidden, 0);
    let up = weights(k, hidden, 3);
    let guard_weights = |payload: &[u8]| {
        let mut values = vec![0x5au8; payload.len() + offset + 16];
        values[offset..offset + payload.len()].copy_from_slice(payload);
        values
    };
    let gate_all = guard_weights(&gate);
    let up_all = guard_weights(&up);
    let d_gate = upload(ctx, &gate_all, "ffn-fp32:gate")?;
    let d_up = upload(ctx, &up_all, "ffn-fp32:up")?;
    let output_values = vec![f32::from_bits(GUARD_BITS); rows * hidden + 2 * GUARD];
    let output = DeviceBuffer::new_tagged(ctx, output_values.len() * 4, "ffn-fp32:output")?;
    let reference = DeviceBuffer::new_tagged(ctx, output_values.len() * 4, "ffn-fp32:reference")?;
    let x = unsafe { d_input.as_ptr().cast::<f32>().add(GUARD) };
    let gate_pointer = unsafe { d_gate.as_ptr().cast::<u8>().add(offset).cast() };
    let up_pointer = unsafe { d_up.as_ptr().cast::<u8>().add(offset).cast() };
    let out = unsafe { output.as_mut_ptr().cast::<f32>().add(GUARD) };
    unsafe {
        ctx.memcpy_h2d(
            reference.as_mut_ptr(),
            output_values.as_ptr().cast(),
            reference.bytes(),
        )?;
        for row in 0..rows {
            ctx.mlp_gate_up_swiglu_f32xq8_0_gguf_decode_async(
                x.add(row * k).cast(),
                gate_pointer,
                up_pointer,
                reference
                    .as_mut_ptr()
                    .cast::<f32>()
                    .add(GUARD + row * hidden)
                    .cast(),
                hidden as i32,
                k as i32,
            )?;
        }
    }
    ctx.synchronize_stream(CudaStream::Decode)?;
    let native = read(ctx, &reference)?;
    let expected: Vec<_> = (0..rows)
        .flat_map(|row| {
            let input = &input;
            let gate = &gate;
            let up = &up;
            (0..hidden).map(move |column| {
                let g = fixture::project_cpu(input, gate, row, column, k);
                let u = fixture::project_cpu(input, up, row, column, k);
                g / (1.0 + (-g).exp()) * u
            })
        })
        .collect();
    let pack = |name: &str, bytes: &[u8]| -> Result<_> {
        let identity = Q8SourceIdentity::from_column_blocks(
            [0; 32],
            name.into(),
            0,
            Q8MatrixShape::new(k, hidden)?,
        )?;
        HostQ8ExecutionWeight::separate_scales_values(Q8SourceView::from_tensor_bytes(
            &identity, bytes,
        )?)
    };
    let host_planes = [pack("gate", &gate)?, pack("up", &up)?];
    let split_weights = [
        DeviceQ8ExecutionWeight::upload(ctx, &host_planes[0])?,
        DeviceQ8ExecutionWeight::upload(ctx, &host_planes[1])?,
    ];
    let mut scales = [std::ptr::null(); 2];
    let mut values = [std::ptr::null(); 2];
    for (index, weight) in split_weights.iter().enumerate() {
        let DeviceQ8ExecutionView::SeparateScalesValues {
            scales: s,
            values: v,
            ..
        } = weight.view()
        else {
            unreachable!()
        };
        scales[index] = s.as_ptr();
        values[index] = v.as_ptr();
    }
    let mut split_output: Option<Vec<f32>> = None;
    let mut split_projections: Option<[Vec<f32>; 2]> = None;
    for (variant, split_kernel) in Q8FfnFp32Variant::ALL
        .into_iter()
        .chain([
            Q8FfnFp32Variant::PackedWarp128FactoredScales,
            Q8FfnFp32Variant::PackedWarp128CoalescedPairs,
        ])
        .map(|variant| (Some(variant), SplitKernel::Dynamic))
        .chain([
            (None, SplitKernel::Dynamic),
            (None, SplitKernel::TwoColumns),
            (None, SplitKernel::FixedK2048),
        ])
        .filter(|(_, kernel)| *kernel != SplitKernel::FixedK2048 || k == 2048)
    {
        let enqueue_split = match split_kernel {
            SplitKernel::Dynamic => CudaContext::ffn_q8_split_f32_async,
            SplitKernel::TwoColumns => CudaContext::ffn_q8_split_f32_two_columns_async,
            SplitKernel::FixedK2048 => CudaContext::ffn_q8_split_f32_k2048_async,
        };
        unsafe {
            ctx.memcpy_h2d(
                output.as_mut_ptr(),
                output_values.as_ptr().cast(),
                output.bytes(),
            )?;
        }
        let before = profile::snapshot();
        unsafe {
            if let Some(variant) = variant {
                ctx.ffn_q8_f32_ablation_async(
                    x.cast(),
                    gate_pointer,
                    up_pointer,
                    out.cast(),
                    rows as i32,
                    hidden as i32,
                    k as i32,
                    variant,
                )?;
            } else {
                enqueue_split(
                    ctx,
                    x.cast(),
                    scales,
                    values,
                    out.cast(),
                    rows as i32,
                    hidden as i32,
                    k as i32,
                    None,
                )?;
            }
        }
        let delta = profile::counter_delta(&before, &profile::snapshot());
        assert_eq!(delta.launches, 1);
        assert_eq!(delta.device_allocs, 0);
        assert_eq!(delta.device_frees, 0);
        assert_eq!(delta.h2d_copies, 0);
        assert_eq!(delta.d2h_copies, 0);
        assert_eq!(delta.stream_syncs, 0);
        ctx.synchronize_stream(CudaStream::Decode)?;
        let actual = read(ctx, &output)?;
        if variant.is_none() {
            if split_kernel != SplitKernel::Dynamic {
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    split_output
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "split follow-up changed qualified one-column SwiGLU bits"
                );
            } else {
                split_output = Some(actual.clone());
            }
        }
        if variant.is_none_or(|v| v.has_intermediate_capture()) {
            let captured = DeviceBuffer::new_tagged(ctx, output.bytes(), "ffn-fp32:captured")?;
            let gate_out = DeviceBuffer::new_tagged(ctx, output.bytes(), "ffn-fp32:gate-out")?;
            let up_out = DeviceBuffer::new_tagged(ctx, output.bytes(), "ffn-fp32:up-out")?;
            for buffer in [&captured, &gate_out, &up_out] {
                unsafe {
                    ctx.memcpy_h2d(
                        buffer.as_mut_ptr(),
                        output_values.as_ptr().cast(),
                        buffer.bytes(),
                    )?;
                }
            }
            unsafe {
                if let Some(variant) = variant {
                    ctx.ffn_q8_f32_capture_async(
                        x.cast(),
                        gate_pointer,
                        up_pointer,
                        captured.as_mut_ptr().cast::<f32>().add(GUARD).cast(),
                        gate_out.as_mut_ptr().cast::<f32>().add(GUARD).cast(),
                        up_out.as_mut_ptr().cast::<f32>().add(GUARD).cast(),
                        rows as i32,
                        hidden as i32,
                        k as i32,
                        variant,
                    )?;
                } else {
                    enqueue_split(
                        ctx,
                        x.cast(),
                        scales,
                        values,
                        captured.as_mut_ptr().cast::<f32>().add(GUARD).cast(),
                        rows as i32,
                        hidden as i32,
                        k as i32,
                        Some((
                            gate_out.as_mut_ptr().cast::<f32>().add(GUARD).cast(),
                            up_out.as_mut_ptr().cast::<f32>().add(GUARD).cast(),
                        )),
                    )?;
                }
            }
            ctx.synchronize_stream(CudaStream::Decode)?;
            assert_eq!(
                read(ctx, &captured)?,
                actual,
                "capture changed FFN arithmetic"
            );
            let projections = [read(ctx, &gate_out)?, read(ctx, &up_out)?];
            if variant.is_none() {
                if split_kernel != SplitKernel::Dynamic {
                    for (a, b) in projections.iter().zip(split_projections.as_ref().unwrap()) {
                        assert_eq!(
                            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        );
                    }
                } else {
                    split_projections = Some(projections.clone());
                }
            }
            for (values, weight) in projections.into_iter().zip([&gate, &up]) {
                for row in 0..rows {
                    for column in 0..hidden {
                        let expected = fixture::project_cpu(&input, weight, row, column, k);
                        let actual = f64::from(values[row * hidden + column]);
                        assert!(
                            (actual - expected).abs() < 1e-3 + 2e-5 * expected.abs(),
                            "FFN projection differs: {actual} vs {expected}"
                        );
                    }
                }
            }
        }
        for ((&a, &b), &native) in actual.iter().zip(&expected).zip(&native) {
            assert!(a.is_finite() && b.is_finite());
            assert!(
                (f64::from(a) - b).abs() < 1e-3 + 2e-5 * b.abs(),
                "{variant:?} k={k} hidden={hidden} rows={rows}: {a} vs {b}"
            );
            if variant.is_some_and(|v| v.threads() == 256) {
                assert_eq!(
                    a.to_bits(),
                    native.to_bits(),
                    "{variant:?} k={k} hidden={hidden} rows={rows}"
                );
            }
        }
    }
    for (weight, host) in split_weights.iter().zip(&host_planes) {
        let DeviceQ8ExecutionView::SeparateScalesValues { scales, values, .. } = weight.view()
        else {
            unreachable!()
        };
        for (buffer, expected) in [scales, values].into_iter().zip(host.planes()) {
            let mut actual = vec![0u8; buffer.bytes()];
            unsafe {
                ctx.memcpy_d2h(actual.as_mut_ptr().cast(), buffer.as_ptr(), actual.len())?;
            }
            assert_eq!(&actual, expected);
        }
    }
    assert_eq!(read(ctx, &d_input)?, input);
    for (buffer, original) in [(&d_gate, &gate_all), (&d_up, &up_all)] {
        let mut after = vec![0u8; original.len()];
        unsafe {
            ctx.memcpy_d2h(after.as_mut_ptr().cast(), buffer.as_ptr(), after.len())?;
        }
        assert_eq!(&after, original);
    }
    Ok(())
}

/// All four 256-thread load/reduction variants preserve native split4 bits.
/// All 128-thread strategies obey the same numerical tolerance; follow-up
/// gate/up capture must match the ordinary kernel's SwiGLU bits.
/// The fixed split-plane/four-value strategy uses lossless typed packing and
/// must preserve source planes as well as output guards. Its two-column
/// experiment must keep one-column gate/up/SwiGLU bits, including odd tails.
/// The K=2048 address specialization must preserve those bits on its supported
/// width; other widths remain covered by the unchanged dynamic kernels.
/// Distinct rows,
/// signed extremes, zero scales, hidden tails and two-byte weight bases protect
/// the packed representation and one-warp reduction without changing defaults.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_fp32_ffn_variants_preserve_native_math_and_guards() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _split = EnvRestore::set("M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK", "4");
    let _stream = EnvRestore::set("M40LLM_Q8_DECODE_STREAM", "decode");
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for (k, hidden, offset) in [(32, 1, 0), (96, 37, 2), (2048, 257, 0), (4096, 37, 2)] {
        for rows in [1, 4] {
            compare_case(&ctx, k, hidden, rows, offset)?;
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Unsupported dimensions and alignment/alias errors fail before launch,
/// allocation, transfer or synchronization instead of choosing another kernel.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_fp32_ffn_rejects_unsupported_arguments_without_gpu_work() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let input = DeviceBuffer::new_tagged(&ctx, 512, "ffn-fp32:invalid-input")?;
        let weight = DeviceBuffer::new_tagged(&ctx, 512, "ffn-fp32:invalid-weight")?;
        let output = DeviceBuffer::new_tagged(&ctx, 512, "ffn-fp32:invalid-output")?;
        let before = profile::snapshot();
        unsafe {
            for enqueue in [
                CudaContext::ffn_q8_split_f32_async,
                CudaContext::ffn_q8_split_f32_two_columns_async,
                CudaContext::ffn_q8_split_f32_k2048_async,
            ] {
                for (rows, hidden, k, offset) in [
                    (0, 1, 2048, 0),
                    (1, 0, 2048, 0),
                    (1, 1, 2049, 0),
                    (1, 1, 2048, 8),
                ] {
                    assert!(enqueue(
                        &ctx,
                        input.as_ptr().cast::<u8>().add(offset).cast(),
                        [weight.as_ptr(); 2],
                        [weight.as_ptr(); 2],
                        output.as_mut_ptr(),
                        rows,
                        hidden,
                        k,
                        None
                    )
                    .is_err());
                }
                assert!(enqueue(
                    &ctx,
                    input.as_ptr(),
                    [std::ptr::null(), weight.as_ptr()],
                    [weight.as_ptr(); 2],
                    output.as_mut_ptr(),
                    1,
                    1,
                    2048,
                    None
                )
                .is_err());
                assert!(enqueue(
                    &ctx,
                    input.as_ptr(),
                    [weight.as_ptr(); 2],
                    [weight.as_ptr(); 2],
                    output.as_mut_ptr(),
                    1,
                    1,
                    2048,
                    Some((output.as_mut_ptr(), output.as_mut_ptr()))
                )
                .is_err());
            }
            for k in [32, 96, 4096] {
                let error = ctx
                    .ffn_q8_split_f32_k2048_async(
                        input.as_ptr(),
                        [weight.as_ptr(); 2],
                        [weight.as_ptr(); 2],
                        output.as_mut_ptr(),
                        1,
                        1,
                        k,
                        None,
                    )
                    .unwrap_err();
                assert!(error.to_string().contains("requires K=2048"));
            }
            assert!(ctx
                .ffn_q8_f32_ablation_async(
                    input.as_ptr().cast::<u8>().add(4).cast(),
                    weight.as_ptr(),
                    weight.as_ptr(),
                    output.as_mut_ptr(),
                    1,
                    1,
                    32,
                    Q8FfnFp32Variant::PackedWarp128CoalescedPairs
                )
                .is_err());
            for (g, u) in [
                (output.as_mut_ptr(), output.as_mut_ptr()),
                (std::ptr::null_mut(), output.as_mut_ptr()),
            ] {
                assert!(ctx
                    .ffn_q8_f32_capture_async(
                        input.as_ptr(),
                        weight.as_ptr(),
                        weight.as_ptr(),
                        output.as_mut_ptr(),
                        g,
                        u,
                        1,
                        1,
                        32,
                        Q8FfnFp32Variant::PackedWarp128FactoredScales
                    )
                    .is_err());
            }
        }
        for (rows, hidden, k) in [(0, 1, 32), (2, 1, 32), (4, 0, 32), (1, 1, 33), (1, 1, 0)] {
            assert!(unsafe {
                ctx.ffn_q8_f32_ablation_async(
                    input.as_ptr(),
                    weight.as_ptr(),
                    weight.as_ptr(),
                    output.as_mut_ptr(),
                    rows,
                    hidden,
                    k,
                    Q8FfnFp32Variant::PackedWarp256,
                )
            }
            .is_err());
        }
        unsafe {
            assert!(ctx
                .ffn_q8_f32_ablation_async(
                    std::ptr::null(),
                    weight.as_ptr(),
                    weight.as_ptr(),
                    output.as_mut_ptr(),
                    1,
                    1,
                    32,
                    Q8FfnFp32Variant::PackedWarp256
                )
                .is_err());
            assert!(ctx
                .ffn_q8_f32_ablation_async(
                    input.as_ptr(),
                    weight.as_ptr(),
                    weight.as_ptr(),
                    input.as_mut_ptr(),
                    1,
                    1,
                    32,
                    Q8FfnFp32Variant::PackedWarp256
                )
                .is_err());
            assert!(ctx
                .ffn_q8_f32_ablation_async(
                    input.as_ptr(),
                    weight.as_ptr().cast::<u8>().add(1).cast(),
                    weight.as_ptr(),
                    output.as_mut_ptr(),
                    1,
                    1,
                    32,
                    Q8FfnFp32Variant::PackedWarp256
                )
                .is_err());
        }
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()),
            Default::default()
        );
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

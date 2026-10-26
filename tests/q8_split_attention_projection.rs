#![cfg(all(feature = "cuda", nvcc))]
#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/q8_projection.rs"]
#[allow(dead_code)]
mod fixture;

use anyhow::Result;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    infer::execution_weights::{
        DeviceQ8ExecutionView, DeviceQ8ExecutionWeight, HostQ8ExecutionWeight, Q8MatrixShape,
        Q8SourceIdentity, Q8SourceView,
    },
    profile,
};
use std::ffi::c_void;

struct Guarded {
    device: DeviceBuffer,
    initial: Vec<f32>,
}

impl Guarded {
    fn new(ctx: &CudaContext, data: &[f32]) -> Result<Self> {
        let mut initial = vec![12345.0; data.len() + 16];
        initial[8..8 + data.len()].copy_from_slice(data);
        let device = DeviceBuffer::new_tagged(ctx, initial.len() * 4, "split_projection:test")?;
        unsafe {
            ctx.memcpy_h2d(device.as_mut_ptr(), initial.as_ptr().cast(), device.bytes())?;
        }
        Ok(Self { device, initial })
    }

    fn pointer(&self) -> *mut c_void {
        unsafe { self.device.as_mut_ptr().cast::<f32>().add(8).cast() }
    }

    fn read(&self, ctx: &CudaContext) -> Result<Vec<f32>> {
        let mut actual = vec![0.0; self.initial.len()];
        unsafe {
            ctx.memcpy_d2h(
                actual.as_mut_ptr().cast(),
                self.device.as_ptr(),
                self.device.bytes(),
            )?;
        }
        assert_eq!(&actual[..8], &self.initial[..8]);
        assert_eq!(
            &actual[actual.len() - 8..],
            &self.initial[self.initial.len() - 8..]
        );
        Ok(actual[8..actual.len() - 8].to_vec())
    }
}

struct SplitWeight {
    host: HostQ8ExecutionWeight,
    device: DeviceQ8ExecutionWeight,
    raw: DeviceBuffer,
    guarded_source: Vec<u8>,
}

impl SplitWeight {
    fn new(ctx: &CudaContext, k: usize, n: usize) -> Result<Self> {
        let mut source = fixture::make_column_blocks(k, n);
        for (index, block) in source.chunks_exact_mut(34).enumerate() {
            if index % 7 == 0 {
                block[..2].fill(0);
            }
            for (lane, value) in block[2..].iter_mut().enumerate() {
                *value = [0x80, 0x7f, 0xff, 0, 1, 0x40, 0xc0, 8][(lane + index) % 8];
            }
        }
        let identity = Q8SourceIdentity::from_column_blocks(
            [42; 32],
            "test.weight".into(),
            0,
            Q8MatrixShape::new(k, n)?,
        )?;
        let host = HostQ8ExecutionWeight::separate_scales_values(Q8SourceView::from_tensor_bytes(
            &identity, &source,
        )?)?;
        let device = DeviceQ8ExecutionWeight::upload(ctx, &host)?;
        let mut guarded_source = vec![0xa5; source.len() + 18];
        guarded_source[2..2 + source.len()].copy_from_slice(&source);
        let raw = DeviceBuffer::new_tagged(ctx, guarded_source.len(), "split_projection:raw")?;
        unsafe {
            ctx.memcpy_h2d(
                raw.as_mut_ptr(),
                guarded_source.as_ptr().cast(),
                raw.bytes(),
            )?;
        }
        Ok(Self {
            host,
            device,
            raw,
            guarded_source,
        })
    }

    fn planes(&self) -> (*const c_void, *const c_void) {
        let DeviceQ8ExecutionView::SeparateScalesValues { scales, values, .. } = self.device.view()
        else {
            panic!("expected typed split planes");
        };
        (scales.as_ptr(), values.as_ptr())
    }

    fn raw_pointer(&self) -> *const c_void {
        unsafe { self.raw.as_ptr().cast::<u8>().add(2).cast() }
    }

    fn source(&self) -> &[u8] {
        &self.guarded_source[2..self.guarded_source.len() - 16]
    }

    fn verify(&self, ctx: &CudaContext) -> Result<()> {
        let DeviceQ8ExecutionView::SeparateScalesValues { scales, values, .. } = self.device.view()
        else {
            panic!("expected typed split planes");
        };
        for (buffer, expected) in [scales, values].into_iter().zip(self.host.planes()) {
            let mut actual = vec![0; buffer.bytes()];
            unsafe {
                ctx.memcpy_d2h(actual.as_mut_ptr().cast(), buffer.as_ptr(), actual.len())?;
            }
            assert_eq!(actual, *expected);
        }
        let mut actual = vec![0; self.raw.bytes()];
        unsafe {
            ctx.memcpy_d2h(actual.as_mut_ptr().cast(), self.raw.as_ptr(), actual.len())?;
        }
        assert_eq!(actual, self.guarded_source);
        Ok(())
    }
}

fn compare_case(ctx: &CudaContext, k: usize, widths: [usize; 3]) -> Result<()> {
    let rows = 4;
    let weights = widths
        .into_iter()
        .chain([widths[0]])
        .map(|n| SplitWeight::new(ctx, k, n))
        .collect::<Result<Vec<_>>>()?;
    let input = fixture::make_input(rows, k);
    let x = Guarded::new(ctx, &input)?;
    let outputs = widths
        .into_iter()
        .chain([widths[0]])
        .map(|n| Guarded::new(ctx, &vec![0.0; rows * n]))
        .collect::<Result<Vec<_>>>()?;
    let biases = widths
        .iter()
        .enumerate()
        .map(|(index, &n)| {
            (0..n)
                .map(|column| ((column + index * 7) % 17) as f32 * 0.015625 - 0.125)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let bias_buffers = biases
        .iter()
        .map(|b| Guarded::new(ctx, b))
        .collect::<Result<Vec<_>>>()?;
    let original_bytes = CudaContext::total_device_bytes();
    for stream in [CudaStream::Decode, CudaStream::Prefill] {
        for has_bias in [[false; 3], [true, false, true], [true; 3]] {
            unsafe {
                ctx.memcpy_h2d(x.pointer(), input.as_ptr().cast(), input.len() * 4)?;
                ctx.residual_add_f32_async(x.pointer(), x.pointer(), x.pointer(), input.len())?;
            }
            if stream == CudaStream::Prefill {
                ctx.stream_wait_for_stream(
                    stream,
                    CudaStream::Decode,
                    "split_projection:producer",
                )?;
            }
            let b = std::array::from_fn(|i| {
                has_bias[i].then(|| bias_buffers[i].pointer().cast_const())
            });
            let mut control = Vec::new();
            for is_split in [false, true] {
                let before = profile::snapshot();
                for row in 0..rows {
                    unsafe {
                        let input = x.pointer().cast::<f32>().add(row * k).cast();
                        let out = std::array::from_fn(|i| {
                            outputs[i]
                                .pointer()
                                .cast::<f32>()
                                .add(row * widths[i])
                                .cast()
                        });
                        let output = outputs[3]
                            .pointer()
                            .cast::<f32>()
                            .add(row * widths[0])
                            .cast();
                        if is_split {
                            ctx.qkv_f32xq8_separate_packed_warp_async(
                                input,
                                std::array::from_fn(|i| weights[i].planes().0),
                                std::array::from_fn(|i| weights[i].planes().1),
                                b,
                                out,
                                widths.map(|n| n as i32),
                                k as i32,
                                stream,
                            )?;
                            ctx.gemm_f32xq8_separate_coltile4_packed_warp_async(
                                input,
                                weights[3].planes().0,
                                weights[3].planes().1,
                                output,
                                widths[0] as i32,
                                k as i32,
                                stream,
                            )?;
                        } else {
                            ctx.qkv_f32xq8_0_gguf_packed_warp_async(
                                input,
                                std::array::from_fn(|i| weights[i].raw_pointer()),
                                b,
                                out,
                                widths.map(|n| n as i32),
                                k as i32,
                                stream,
                            )?;
                            ctx.gemm_f32xq8_0_gguf_coltile4_packed_warp_async(
                                input,
                                weights[3].raw_pointer(),
                                output,
                                widths[0] as i32,
                                k as i32,
                                stream,
                            )?;
                        }
                    }
                }
                let after = profile::snapshot();
                let delta = profile::counter_delta(&before, &after);
                assert_eq!(delta.launches, 8);
                assert_eq!(
                    (
                        delta.device_allocs,
                        delta.device_frees,
                        delta.h2d_copies,
                        delta.d2h_copies,
                        delta.stream_syncs,
                        delta.stream_waits,
                        delta.cublas_calls
                    ),
                    (0, 0, 0, 0, 0, 0, 0)
                );
                if is_split {
                    for op in [
                        "qkv_f32xq8_separate_packed_warp",
                        "gemm_f32xq8_separate_coltile4_packed_warp",
                    ] {
                        assert_eq!(
                            after.by_op.get(op).map_or(0, |v| v.launches)
                                - before.by_op.get(op).map_or(0, |v| v.launches),
                            4
                        );
                    }
                }
                ctx.synchronize_stream(stream)?;
                let actual = outputs
                    .iter()
                    .map(|o| o.read(ctx))
                    .collect::<Result<Vec<_>>>()?;
                if is_split {
                    assert_eq!(
                        actual
                            .iter()
                            .flatten()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        control
                    );
                    let doubled = input.iter().map(|v| v + v).collect::<Vec<_>>();
                    for index in 0..4 {
                        let n = if index == 3 { widths[0] } else { widths[index] };
                        for row in 0..rows {
                            for column in 0..n {
                                let bias = if index < 3 && has_bias[index] {
                                    f64::from(biases[index][column])
                                } else {
                                    0.0
                                };
                                let cpu = fixture::project_cpu(
                                    &doubled,
                                    weights[index].source(),
                                    row,
                                    column,
                                    k,
                                ) + bias;
                                let observed = f64::from(actual[index][row * n + column]);
                                assert!(
                                    observed.is_finite()
                                        && (observed - cpu).abs() < 1e-3 + 2e-5 * cpu.abs()
                                );
                            }
                        }
                    }
                    assert_eq!(x.read(ctx)?, doubled);
                } else {
                    control = actual.iter().flatten().map(|v| v.to_bits()).collect();
                }
                assert_eq!(CudaContext::total_device_bytes(), original_bytes);
            }
        }
    }
    for weight in &weights {
        weight.verify(ctx)?;
    }
    for bias in &bias_buffers {
        bias.read(ctx)?;
    }
    Ok(())
}

/// Split storage retains the qualified QKV/output bits across unequal widths,
/// column tails, signed extremes, zero scales and biases. Four distinct rows,
/// both producer streams, immutable planes, guards and cleanup are exercised.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_split_projections_preserve_packed_controls_and_ownership() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for (k, widths) in [
        (32, [1, 2, 3]),
        (96, [5, 7, 11]),
        (2048, [257, 32, 32]),
        (4096, [37, 5, 9]),
    ] {
        compare_case(&ctx, k, widths)?;
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Invalid shapes, planes, aliases and alignments must fail before dispatch,
/// without allocation, transfer, synchronization or a different implementation.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_split_projections_reject_invalid_spans_without_work() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let x = DeviceBuffer::new_tagged(&ctx, 512, "split:invalid")?;
        let s = DeviceBuffer::new_tagged(&ctx, 512, "split:invalid")?;
        let v = DeviceBuffer::new_tagged(&ctx, 512, "split:invalid")?;
        let out = (0..3)
            .map(|_| DeviceBuffer::new_tagged(&ctx, 512, "split:invalid"))
            .collect::<Result<Vec<_>>>()?;
        let outputs = std::array::from_fn(|i| out[i].as_mut_ptr());
        let before = profile::snapshot();
        unsafe {
            for (k, widths) in [
                (0, [1; 3]),
                (33, [1; 3]),
                (32, [0, 1, 1]),
                (32, [i32::MAX; 3]),
            ] {
                assert!(ctx
                    .qkv_f32xq8_separate_packed_warp_async(
                        x.as_ptr(),
                        [s.as_ptr(); 3],
                        [v.as_ptr(); 3],
                        [None; 3],
                        outputs,
                        widths,
                        k,
                        CudaStream::Decode
                    )
                    .is_err());
            }
            for (input, scale, values, bias, output) in [
                (std::ptr::null(), s.as_ptr(), v.as_ptr(), None, outputs[0]),
                (x.as_ptr(), std::ptr::null(), v.as_ptr(), None, outputs[0]),
                (x.as_ptr(), s.as_ptr(), std::ptr::null(), None, outputs[0]),
                (
                    x.as_ptr().cast::<u8>().add(4).cast(),
                    s.as_ptr(),
                    v.as_ptr(),
                    None,
                    outputs[0],
                ),
                (
                    x.as_ptr(),
                    s.as_ptr().cast::<u8>().add(1).cast(),
                    v.as_ptr(),
                    None,
                    outputs[0],
                ),
                (
                    x.as_ptr(),
                    s.as_ptr(),
                    v.as_ptr().cast::<u8>().add(4).cast(),
                    None,
                    outputs[0],
                ),
                (
                    x.as_ptr(),
                    s.as_ptr(),
                    v.as_ptr(),
                    Some(std::ptr::null()),
                    outputs[0],
                ),
                (
                    x.as_ptr(),
                    s.as_ptr(),
                    v.as_ptr(),
                    None,
                    std::ptr::null_mut(),
                ),
                (x.as_ptr(), s.as_ptr(), v.as_ptr(), None, x.as_mut_ptr()),
                (x.as_ptr(), s.as_ptr(), v.as_ptr(), None, s.as_mut_ptr()),
                (x.as_ptr(), s.as_ptr(), v.as_ptr(), None, v.as_mut_ptr()),
                (x.as_ptr(), s.as_ptr(), v.as_ptr(), None, outputs[1]),
            ] {
                assert!(ctx
                    .qkv_f32xq8_separate_packed_warp_async(
                        input,
                        [scale; 3],
                        [values; 3],
                        [bias; 3],
                        [output, outputs[1], outputs[2]],
                        [1; 3],
                        32,
                        CudaStream::Decode
                    )
                    .is_err());
                if bias.is_none() && output != outputs[1] {
                    assert!(ctx
                        .gemm_f32xq8_separate_coltile4_packed_warp_async(
                            input,
                            scale,
                            values,
                            output,
                            1,
                            32,
                            CudaStream::Decode
                        )
                        .is_err());
                }
            }
            for (n, k) in [(0, 32), (1, 0), (1, 33), (-1, 32), (1, -32)] {
                assert!(ctx
                    .gemm_f32xq8_separate_coltile4_packed_warp_async(
                        x.as_ptr(),
                        s.as_ptr(),
                        v.as_ptr(),
                        outputs[0],
                        n,
                        k,
                        CudaStream::Decode
                    )
                    .is_err());
            }
        }
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()),
            profile::counter_delta(&before, &before)
        );
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

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
        DeviceQ8ExecutionView, DeviceQ8ExecutionWeight, DeviceQ8FfnPair, HostQ8ExecutionWeight,
        HostQ8FfnPair, Q8MatrixShape, Q8SourceIdentity, Q8SourceView,
    },
    profile,
};

fn upload(ctx: &CudaContext, bytes: &[u8]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new_tagged(ctx, bytes.len(), "joint_ffn:test")?;
    unsafe {
        ctx.memcpy_h2d(buffer.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())?;
    }
    Ok(buffer)
}

fn read(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<u8>> {
    let mut bytes = vec![0; buffer.bytes()];
    unsafe {
        ctx.memcpy_d2h(bytes.as_mut_ptr().cast(), buffer.as_ptr(), bytes.len())?;
    }
    Ok(bytes)
}

fn compare_case(ctx: &CudaContext, k: usize, n: usize, rows: usize) -> Result<()> {
    let identity = |name: &str| {
        Q8SourceIdentity::from_column_blocks([1; 32], name.into(), 0, Q8MatrixShape::new(k, n)?)
    };
    let ids = [identity("gate")?, identity("up")?];
    let source = |seed: usize| {
        let mut weights = fixture::make_column_blocks(k, n);
        for (b, block) in weights.chunks_exact_mut(34).enumerate() {
            if b % 7 == 0 {
                block[..2].fill(0);
            }
            for (i, v) in block[2..].iter_mut().enumerate() {
                *v = [0x80, 0x7f, 0xff, 0, 1, 0x40, 0xc0, 8][(i + b + seed) % 8];
            }
        }
        weights
    };
    let bytes = [source(0), source(3)];
    let views = [
        Q8SourceView::from_tensor_bytes(&ids[0], &bytes[0])?,
        Q8SourceView::from_tensor_bytes(&ids[1], &bytes[1])?,
    ];
    let host = HostQ8FfnPair::pack(views[0], views[1])?;
    let joint = DeviceQ8FfnPair::upload(ctx, &host)?;
    let separate = [
        HostQ8ExecutionWeight::separate_scales_values(views[0])?,
        HostQ8ExecutionWeight::separate_scales_values(views[1])?,
    ];
    let control = [
        DeviceQ8ExecutionWeight::upload(ctx, &separate[0])?,
        DeviceQ8ExecutionWeight::upload(ctx, &separate[1])?,
    ];
    let mut scales = [std::ptr::null(); 2];
    let mut values = [std::ptr::null(); 2];
    for (i, w) in control.iter().enumerate() {
        let DeviceQ8ExecutionView::SeparateScalesValues {
            scales: s,
            values: v,
            ..
        } = w.view()
        else {
            unreachable!()
        };
        scales[i] = s.as_ptr();
        values[i] = v.as_ptr();
    }
    let input = fixture::make_input(rows, k);
    let input_bytes: Vec<u8> = input.iter().flat_map(|x| x.to_le_bytes()).collect();
    let x = upload(ctx, &input_bytes)?;
    let guard = 0x4b123456u32.to_le_bytes();
    let initial: Vec<u8> = (0..rows * n + 16).flat_map(|_| guard).collect();
    let out = upload(ctx, &initial)?;
    let g = upload(ctx, &initial)?;
    let u = upload(ctx, &initial)?;
    let ptr = |buffer: &DeviceBuffer| unsafe { buffer.as_mut_ptr().cast::<u8>().add(32).cast() };
    let mut expected = Vec::new();
    for is_joint in [false, true] {
        let before = profile::snapshot();
        unsafe {
            if is_joint {
                joint.enqueue_async(
                    x.as_ptr(),
                    ptr(&out),
                    rows as i32,
                    Some((ptr(&g), ptr(&u))),
                )?;
            } else {
                ctx.ffn_q8_split_f32_async(
                    x.as_ptr(),
                    scales,
                    values,
                    ptr(&out),
                    rows as i32,
                    n as i32,
                    k as i32,
                    Some((ptr(&g), ptr(&u))),
                )?;
            }
        }
        let delta = profile::counter_delta(&before, &profile::snapshot());
        assert_eq!(delta.launches, 1);
        assert_eq!(
            (
                delta.device_allocs,
                delta.device_frees,
                delta.h2d_copies,
                delta.d2h_copies,
                delta.stream_syncs
            ),
            (0, 0, 0, 0, 0)
        );
        ctx.synchronize_stream(CudaStream::Decode)?;
        let actual = [read(ctx, &out)?, read(ctx, &g)?, read(ctx, &u)?];
        for buffer in &actual {
            assert_eq!(&buffer[..32], &initial[..32]);
            assert_eq!(&buffer[buffer.len() - 32..], &initial[initial.len() - 32..]);
        }
        if is_joint {
            assert_eq!(
                actual.as_slice(),
                expected.as_slice(),
                "joint packing changed gate/up/SwiGLU bits"
            );
        } else {
            expected = actual.to_vec();
        }
        for row in 0..rows {
            for column in 0..n {
                let gcpu = fixture::project_cpu(&input, &bytes[0], row, column, k);
                let ucpu = fixture::project_cpu(&input, &bytes[1], row, column, k);
                for (buffer, value) in
                    actual
                        .iter()
                        .zip([gcpu / (1.0 + (-gcpu).exp()) * ucpu, gcpu, ucpu])
                {
                    let start = 32 + (row * n + column) * 4;
                    let observed = f32::from_le_bytes(buffer[start..start + 4].try_into()?) as f64;
                    assert!((observed - value).abs() < 1e-3 + 2e-5 * value.abs());
                }
            }
        }
    }
    unsafe {
        joint.enqueue_async(x.as_ptr(), ptr(&out), rows as i32, None)?;
    }
    ctx.synchronize_stream(CudaStream::Decode)?;
    assert_eq!(
        read(ctx, &out)?,
        expected[0],
        "capture changed ordinary joint kernel"
    );
    for (device, original) in joint.planes().into_iter().zip(host.planes()) {
        assert_eq!(read(ctx, device)?, original);
    }
    assert_eq!(read(ctx, &x)?, input_bytes);
    Ok(())
}

/// Joint records retain one-column gate/up/SwiGLU bits and CPU agreement
/// across distinct rows, signed extremes, zero scales and odd hidden tails.
/// Every temporary allocation is released after decode completion.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_joint_ffn_preserves_math_guards_and_ownership() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for (k, n) in [(32, 1), (96, 37), (2048, 257), (4096, 37)] {
        for rows in [1, 4] {
            compare_case(&ctx, k, n, rows)?;
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Invalid dimensions, misaligned joint records and aliased outputs fail
/// before issuing GPU work, not by dispatching a different layout.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_joint_ffn_rejects_invalid_spans_without_launch() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let x = DeviceBuffer::new_tagged(&ctx, 512, "joint_ffn:invalid")?;
        let w = DeviceBuffer::new_tagged(&ctx, 512, "joint_ffn:invalid")?;
        let y = DeviceBuffer::new_tagged(&ctx, 512, "joint_ffn:invalid")?;
        let before = profile::snapshot();
        for (rows, k, xoff, soff, voff) in [
            (0, 32, 0, 0, 0),
            (2, 32, 0, 0, 0),
            (1, 33, 0, 0, 0),
            (1, 32, 4, 0, 0),
            (1, 32, 0, 2, 0),
            (1, 32, 0, 0, 4),
        ] {
            unsafe {
                assert!(ctx
                    .ffn_q8_joint_f32_async(
                        x.as_ptr().cast::<u8>().add(xoff).cast(),
                        w.as_ptr().cast::<u8>().add(soff).cast(),
                        w.as_ptr().cast::<u8>().add(voff).cast(),
                        y.as_mut_ptr(),
                        rows,
                        1,
                        k,
                        None
                    )
                    .is_err());
            }
        }
        unsafe {
            assert!(ctx
                .ffn_q8_joint_f32_async(
                    x.as_ptr(),
                    std::ptr::null(),
                    w.as_ptr(),
                    y.as_mut_ptr(),
                    1,
                    1,
                    32,
                    None
                )
                .is_err());
            assert!(ctx
                .ffn_q8_joint_f32_async(
                    x.as_ptr(),
                    w.as_ptr(),
                    w.as_ptr(),
                    x.as_mut_ptr(),
                    1,
                    1,
                    32,
                    None
                )
                .is_err());
            assert!(ctx
                .ffn_q8_joint_f32_async(
                    x.as_ptr(),
                    w.as_ptr(),
                    w.as_ptr(),
                    y.as_mut_ptr(),
                    1,
                    1,
                    32,
                    Some((y.as_mut_ptr(), y.as_mut_ptr()))
                )
                .is_err());
        }
        let delta = profile::counter_delta(&before, &profile::snapshot());
        assert_eq!(
            (
                delta.launches,
                delta.device_allocs,
                delta.device_frees,
                delta.h2d_copies,
                delta.d2h_copies,
                delta.stream_syncs
            ),
            (0, 0, 0, 0, 0, 0)
        );
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

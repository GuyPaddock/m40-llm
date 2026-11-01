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
        DeviceQ8FfnPair, HostQ8FfnPair, Q8MatrixShape, Q8SourceIdentity, Q8SourceView,
    },
    profile,
};

fn upload(ctx: &CudaContext, bytes: &[u8]) -> Result<DeviceBuffer> {
    let out = DeviceBuffer::new_tagged(ctx, bytes.len(), "r4:test")?;
    unsafe {
        ctx.memcpy_h2d(out.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())?;
    }
    Ok(out)
}

fn read(ctx: &CudaContext, b: &DeviceBuffer) -> Result<Vec<u8>> {
    let mut out = vec![0; b.bytes()];
    unsafe {
        ctx.memcpy_d2h(out.as_mut_ptr().cast(), b.as_ptr(), out.len())?;
    }
    Ok(out)
}

/// The single fixed shape preserves all control bits for four distinct rows,
/// signed extremes, zero/negative-zero scales, optional captures and a real
/// other-stream producer. Outputs and input/weight owners remain intact.
#[test]
#[ignore = "requires M40; run with --include-ignored --test-threads=1"]
fn test_four_row_reader_preserves_exact_math_and_owners() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let (k, n) = (2048, 11008);
        let identity = |name: &str| {
            Q8SourceIdentity::from_column_blocks([1; 32], name.into(), 0, Q8MatrixShape::new(k, n)?)
        };
        let ids = [identity("gate")?, identity("up")?];
        let source = |seed: usize| {
            let mut b = fixture::make_column_blocks(k, n);
            for (index, block) in b.chunks_exact_mut(34).enumerate() {
                if index % 7 == 0 {
                    block[..2].copy_from_slice(
                        &(if index % 14 == 0 { 0u16 } else { 0x8000u16 }).to_le_bytes(),
                    );
                }
                for (i, v) in block[2..].iter_mut().enumerate() {
                    *v = [0x80, 0x7f, 0xff, 0, 1, 0x40, 0xc0, 8][(i + index + seed) % 8];
                }
            }
            b
        };
        let bytes = [source(0), source(3)];
        let host = HostQ8FfnPair::pack(
            Q8SourceView::from_tensor_bytes(&ids[0], &bytes[0])?,
            Q8SourceView::from_tensor_bytes(&ids[1], &bytes[1])?,
        )?;
        let pair = DeviceQ8FfnPair::upload(&ctx, &host)?;
        let input = fixture::make_input(4, k);
        let input_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x = upload(&ctx, &input_bytes)?;
        let producer = upload(&ctx, &input_bytes)?;
        unsafe {
            ctx.memcpy_d2d_async(
                x.as_mut_ptr(),
                producer.as_ptr(),
                input_bytes.len(),
                CudaStream::Prefill,
            )?;
        }
        ctx.stream_wait_for_stream(CudaStream::Decode, CudaStream::Prefill, "r4:test_producer")?;
        let initial_output: Vec<u8> = (0..4 * n + 16)
            .flat_map(|_| 0x4b123456u32.to_le_bytes())
            .collect();
        let out = upload(&ctx, &initial_output)?;
        let g = upload(&ctx, &initial_output)?;
        let u = upload(&ctx, &initial_output)?;
        let ptr = |b: &DeviceBuffer, row: usize| unsafe {
            b.as_mut_ptr().cast::<u8>().add(32 + row * n * 4).cast()
        };
        let mut expected = None;
        for arm in 0..3 {
            let before = profile::snapshot();
            unsafe {
                match arm {
                    0 => {
                        for row in 0..4 {
                            pair.enqueue_shared_scales_async(
                                x.as_ptr().cast::<f32>().add(row * k).cast(),
                                ptr(&out, row),
                                1,
                                Some((ptr(&g, row), ptr(&u, row))),
                            )?;
                        }
                    }
                    1 => pair.enqueue_shared_scales_async(
                        x.as_ptr(),
                        ptr(&out, 0),
                        4,
                        Some((ptr(&g, 0), ptr(&u, 0))),
                    )?,
                    _ => ctx.ffn_q8_joint_four_rows_f32_async(
                        x.as_ptr(),
                        pair.planes()[0].as_ptr(),
                        pair.planes()[1].as_ptr(),
                        ptr(&out, 0),
                        4,
                        n as i32,
                        k as i32,
                        Some((ptr(&g, 0), ptr(&u, 0))),
                    )?,
                }
            }
            assert_eq!(
                profile::counter_delta(&before, &profile::snapshot()),
                profile::OpCounters {
                    launches: if arm == 0 { 4 } else { 1 },
                    ..Default::default()
                }
            );
            ctx.synchronize_stream(CudaStream::Decode)?;
            let actual = [read(&ctx, &out)?, read(&ctx, &g)?, read(&ctx, &u)?];
            for b in &actual {
                assert_eq!(&b[..32], &initial_output[..32]);
                assert_eq!(
                    &b[b.len() - 32..],
                    &initial_output[initial_output.len() - 32..]
                );
            }
            if let Some(reference) = &expected {
                assert_eq!(&actual, reference, "fixed reader changed control bits");
            }
            for row in 0..4 {
                for col in [0, 1, 257, n - 1] {
                    let gcpu = fixture::project_cpu(&input, &bytes[0], row, col, k);
                    let ucpu = fixture::project_cpu(&input, &bytes[1], row, col, k);
                    for (b, want) in
                        actual
                            .iter()
                            .zip([gcpu / (1. + (-gcpu).exp()) * ucpu, gcpu, ucpu])
                    {
                        let index = 32 + (row * n + col) * 4;
                        let value = f32::from_le_bytes(b[index..index + 4].try_into()?);
                        assert!(value.is_finite());
                        assert!((value as f64 - want).abs() < 1e-3 + 2e-5 * want.abs());
                    }
                }
            }
            expected = Some(actual);
        }
        unsafe {
            ctx.ffn_q8_joint_four_rows_f32_async(
                x.as_ptr(),
                pair.planes()[0].as_ptr(),
                pair.planes()[1].as_ptr(),
                ptr(&out, 0),
                4,
                n as i32,
                k as i32,
                None,
            )?;
        }
        ctx.synchronize_stream(CudaStream::Decode)?;
        assert_eq!(read(&ctx, &out)?, expected.unwrap()[0]);
        assert_eq!(read(&ctx, &x)?, input_bytes);
        assert_eq!(read(&ctx, &producer)?, input_bytes);
        for (device, original) in pair.planes().into_iter().zip(host.planes()) {
            assert_eq!(read(&ctx, device)?, original);
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Unsupported shapes and invalid spans fail before launch, without choosing a
/// fallback or touching undersized buffers. The probe has no tunable geometry.
#[test]
#[ignore = "requires M40; run with --include-ignored --test-threads=1"]
fn test_four_row_reader_rejects_invalid_requests() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let x = DeviceBuffer::new_tagged(&ctx, 64, "r4:invalid")?;
        let w = DeviceBuffer::new_tagged(&ctx, 64, "r4:invalid")?;
        let out = DeviceBuffer::new_tagged(&ctx, 64, "r4:invalid")?;
        let before = profile::snapshot();
        for (rows, n, k, off) in [
            (1, 11008, 2048, 0),
            (4, 11007, 2048, 0),
            (4, 11008, 4096, 0),
            (4, 11008, 2048, 4),
        ] {
            unsafe {
                assert!(ctx
                    .ffn_q8_joint_four_rows_f32_async(
                        x.as_ptr().cast::<u8>().add(off).cast(),
                        w.as_ptr(),
                        w.as_ptr(),
                        out.as_mut_ptr(),
                        rows,
                        n,
                        k,
                        None
                    )
                    .is_err());
            }
        }
        unsafe {
            assert!(ctx
                .ffn_q8_joint_four_rows_f32_async(
                    x.as_ptr(),
                    std::ptr::null(),
                    w.as_ptr(),
                    out.as_mut_ptr(),
                    4,
                    11008,
                    2048,
                    None
                )
                .is_err());
            assert!(ctx
                .ffn_q8_joint_four_rows_f32_async(
                    x.as_ptr(),
                    w.as_ptr(),
                    w.as_ptr(),
                    x.as_mut_ptr(),
                    4,
                    11008,
                    2048,
                    None
                )
                .is_err());
            assert!(ctx
                .ffn_q8_joint_four_rows_f32_async(
                    x.as_ptr(),
                    w.as_ptr(),
                    w.as_ptr(),
                    out.as_mut_ptr(),
                    4,
                    11008,
                    2048,
                    Some((out.as_mut_ptr(), out.as_mut_ptr()))
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

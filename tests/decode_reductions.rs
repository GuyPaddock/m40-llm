#![cfg(all(feature = "cuda", nvcc))]
#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/env_restore.rs"]
mod env_restore;
#[path = "common/guarded_bytes.rs"]
mod guarded_bytes;

use anyhow::Result;
use guarded_bytes::GuardedBytes;
use half::f16;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    profile,
};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn compare_norm(ctx: &CudaContext, rows: usize, dim: usize) -> Result<()> {
    let input: Vec<f32> = (0..rows * dim)
        .map(|i| {
            if dim == 32 {
                0.0
            } else {
                ((i * 17 % 101) as f32 - 50.0) * 0.00317
            }
        })
        .collect();
    let doubled: Vec<f32> = input.iter().map(|x| x + x).collect();
    let x = GuardedBytes::new(ctx, &f32_bytes(&input), 32)?;
    let y = GuardedBytes::new(ctx, &f32_bytes(&vec![0.0; input.len()]), 32)?;
    for dtype in [0, 1] {
        let weights: Vec<f32> = (0..dim)
            .map(|i| 0.37 + (i * 7 % 59) as f32 * 0.02171)
            .collect();
        let weight_values = if dtype == 0 {
            weights
                .iter()
                .map(|w| f16::from_f32(*w).to_f32())
                .collect::<Vec<_>>()
        } else {
            weights
        };
        let weight_bytes = if dtype == 0 {
            weight_values
                .iter()
                .flat_map(|w| f16::from_f32(*w).to_bits().to_le_bytes())
                .collect()
        } else {
            f32_bytes(&weight_values)
        };
        let w = GuardedBytes::new(ctx, &weight_bytes, if dtype == 0 { 34 } else { 32 })?;
        let eps = if dim == 31 { 0.0 } else { 1e-5 };
        unsafe {
            ctx.memcpy_h2d(
                x.pointer(),
                f32_bytes(&input).as_ptr().cast(),
                input.len() * 4,
            )?;
            ctx.residual_add_f32_async(x.pointer(), x.pointer(), x.pointer(), input.len())?;
            ctx.rms_norm_f32_weighted_async(
                x.pointer(),
                w.pointer(),
                y.pointer(),
                rows as u32,
                dim as u32,
                eps,
                dtype,
            )?;
        }
        ctx.synchronize_stream(CudaStream::Decode)?;
        let control = y.read(ctx)?;
        let resident = CudaContext::total_device_bytes();
        for stream in [CudaStream::Decode, CudaStream::Prefill] {
            for _ in 0..2 {
                unsafe {
                    ctx.memcpy_h2d(
                        x.pointer(),
                        f32_bytes(&input).as_ptr().cast(),
                        input.len() * 4,
                    )?;
                    ctx.residual_add_f32_async(x.pointer(), x.pointer(), x.pointer(), input.len())?;
                }
                if stream == CudaStream::Prefill {
                    ctx.stream_wait_for_stream(
                        stream,
                        CudaStream::Decode,
                        "weighted_norm:producer",
                    )?;
                }
                let before = profile::snapshot();
                unsafe {
                    ctx.rms_norm_f32_weighted_warp_async(
                        x.pointer(),
                        w.pointer(),
                        y.pointer(),
                        rows as u32,
                        dim as u32,
                        eps,
                        dtype,
                        stream,
                    )?;
                }
                let after = profile::snapshot();
                let delta = profile::counter_delta(&before, &after);
                assert_eq!(delta.launches, 1);
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
                let op = "rms_norm_f32_weighted_warp";
                assert_eq!(
                    after.by_op.get(op).map_or(0, |v| v.launches)
                        - before.by_op.get(op).map_or(0, |v| v.launches),
                    1
                );
                ctx.synchronize_stream(stream)?;
                let actual = y.read(ctx)?;
                assert_eq!(
                    actual, control,
                    "weighted norm changed original GPU bits, rows={rows},dim={dim},dtype={dtype}"
                );
                for (row, observed) in read_f32(&actual).chunks_exact(dim).enumerate() {
                    let source = &doubled[row * dim..(row + 1) * dim];
                    let ss: f64 = source.iter().map(|x| f64::from(*x).powi(2)).sum();
                    let scale = 1.0 / (ss / dim as f64 + f64::from(eps)).sqrt();
                    for ((value, x), weight) in observed.iter().zip(source).zip(&weight_values) {
                        let reference = f64::from(*x) * scale * f64::from(*weight);
                        assert!(
                            value.is_finite()
                                && (f64::from(*value) - reference).abs()
                                    < 1e-4 + 2e-5 * reference.abs()
                        );
                    }
                }
                assert_eq!(x.read(ctx)?, f32_bytes(&doubled));
                assert_eq!(w.read(ctx)?, weight_bytes);
                assert_eq!(CudaContext::total_device_bytes(), resident);
            }
        }
    }
    Ok(())
}

/// The warp finish must preserve original weighted RMSNorm bits for FP16 and
/// FP32 weights, zeros, ragged dimensions and distinct rows, including ordered
/// producers on both streams. Enqueue reuses spans and leaves guards intact.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_weighted_norm_warp_preserves_tree_and_ownership() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _cache = env_restore::EnvRestore::set("M40LLM_CACHE_EXPERIMENT", "off");
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for rows in [1, 4] {
        for dim in [1, 31, 32, 255, 256, 257, 2048, 11008] {
            compare_norm(&ctx, rows, dim)?;
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Invalid norm spans, shapes, dtype and epsilon fail without launch or any
/// transfer/allocation/synchronization, rather than using the old kernel.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_weighted_norm_warp_rejects_invalid_requests() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let input = DeviceBuffer::new_tagged(&ctx, 512, "norm:invalid")?;
        let weight = DeviceBuffer::new_tagged(&ctx, 512, "norm:invalid")?;
        let output = DeviceBuffer::new_tagged(&ctx, 512, "norm:invalid")?;
        let before = profile::snapshot();
        unsafe {
            for (rows, dim, eps, dtype) in [
                (0, 32, 1e-5, 1),
                (1, 0, 1e-5, 1),
                (u32::MAX, 1, 1e-5, 1),
                (1, 32, -1.0, 1),
                (1, 32, f32::NAN, 1),
                (1, 32, f32::INFINITY, 1),
                (1, 32, 1e-5, 2),
            ] {
                assert!(ctx
                    .rms_norm_f32_weighted_warp_async(
                        input.as_ptr(),
                        weight.as_ptr(),
                        output.as_mut_ptr(),
                        rows,
                        dim,
                        eps,
                        dtype,
                        CudaStream::Decode
                    )
                    .is_err());
            }
            for (x, w, y) in [
                (std::ptr::null(), weight.as_ptr(), output.as_mut_ptr()),
                (input.as_ptr(), std::ptr::null(), output.as_mut_ptr()),
                (input.as_ptr(), weight.as_ptr(), std::ptr::null_mut()),
                (
                    input.as_ptr().cast::<u8>().add(1).cast(),
                    weight.as_ptr(),
                    output.as_mut_ptr(),
                ),
                (
                    input.as_ptr(),
                    weight.as_ptr().cast::<u8>().add(2).cast(),
                    output.as_mut_ptr(),
                ),
                (
                    input.as_ptr(),
                    weight.as_ptr(),
                    output.as_mut_ptr().cast::<u8>().add(2).cast(),
                ),
                (input.as_ptr(), weight.as_ptr(), input.as_mut_ptr()),
                (input.as_ptr(), weight.as_ptr(), weight.as_mut_ptr()),
            ] {
                assert!(ctx
                    .rms_norm_f32_weighted_warp_async(x, w, y, 1, 32, 1e-5, 1, CudaStream::Decode)
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

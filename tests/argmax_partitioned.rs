#![cfg(all(feature = "cuda", nvcc))]
#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/guarded_bytes.rs"]
mod guarded_bytes;

use anyhow::Result;
use guarded_bytes::GuardedBytes;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    profile,
};

fn compare_case(ctx: &CudaContext, values: &[f32], expected: u32) -> Result<()> {
    let len = u32::try_from(values.len())?;
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let source = GuardedBytes::new(ctx, &bytes, 32)?;
    let input = GuardedBytes::new(ctx, &vec![0; bytes.len()], 32)?;
    let output = GuardedBytes::new(ctx, &0u32.to_le_bytes(), 32)?;
    let capacity = CudaContext::argmax_f32_partitioned_scratch_bytes(len)?;
    assert!(capacity <= 384);
    let scratch = GuardedBytes::new(ctx, &vec![0xa5; capacity], 32)?;
    unsafe {
        ctx.argmax_f32_async(source.pointer(), len, output.pointer(), CudaStream::Decode)?;
    }
    ctx.synchronize_stream(CudaStream::Decode)?;
    let control = u32::from_le_bytes(output.read(ctx)?.try_into().unwrap());
    assert_eq!(
        control, expected,
        "original argmax contract differs for len={len}"
    );
    let resident = CudaContext::total_device_bytes();
    let mut previous_partials = None;
    for stream in [CudaStream::Decode, CudaStream::Prefill] {
        for _ in 0..2 {
            unsafe {
                ctx.memcpy_h2d(
                    input.pointer(),
                    vec![0u8; bytes.len()].as_ptr().cast(),
                    bytes.len(),
                )?;
                ctx.memcpy_d2d_async(
                    input.pointer(),
                    source.pointer(),
                    bytes.len(),
                    CudaStream::Decode,
                )?;
            }
            if stream == CudaStream::Prefill {
                ctx.stream_wait_for_stream(
                    stream,
                    CudaStream::Decode,
                    "partitioned_argmax:producer",
                )?;
            }
            let before = profile::snapshot();
            unsafe {
                ctx.argmax_f32_partitioned_async(
                    input.pointer(),
                    len,
                    scratch.pointer(),
                    capacity,
                    output.pointer(),
                    stream,
                )?;
            }
            let after = profile::snapshot();
            let delta = profile::counter_delta(&before, &after);
            assert_eq!(delta.launches, 2);
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
            for op in [
                "argmax_f32_partition_partials",
                "argmax_f32_partition_reduce",
            ] {
                assert_eq!(
                    after.by_op.get(op).map_or(0, |v| v.launches)
                        - before.by_op.get(op).map_or(0, |v| v.launches),
                    1
                );
            }
            ctx.synchronize_stream(stream)?;
            assert_eq!(
                u32::from_le_bytes(output.read(ctx)?.try_into().unwrap()),
                control
            );
            assert_eq!(input.read(ctx)?, bytes);
            assert_eq!(source.read(ctx)?, bytes);
            let partials = scratch.read(ctx)?;
            if let Some(previous) = &previous_partials {
                assert_eq!(&partials, previous);
            }
            previous_partials = Some(partials);
            assert_eq!(CudaContext::total_device_bytes(), resident);
        }
    }
    Ok(())
}

/// Partitioned argmax preserves the old lowest-index winner with repeated
/// maxima, scan/partition tails and full Qwen vocabulary size. Ordered async
/// producers on either stream, immutable values, guards and scratch reuse hold.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_partitioned_argmax_preserves_indices_and_ownership() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for len in [
        1, 2, 31, 32, 33, 255, 256, 257, 511, 512, 513, 12287, 12288, 12289, 151935, 151936, 151937,
    ] {
        let mut values: Vec<f32> = (0..len)
            .map(|i| ((i * 17 % 199) as f32 - 99.0) * 0.0317)
            .collect();
        values[len / 2] = 1000.0;
        values[len - 1] = 1000.0;
        let cpu = values
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |best, (index, value)| {
                if *value > best.1 {
                    (index, *value)
                } else {
                    best
                }
            })
            .0;
        assert_eq!(cpu, len / 2);
        compare_case(&ctx, &values, cpu as u32)?;
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Nonfinite and sentinel-valued inputs preserve the old kernel's observable
/// behavior, including its index-zero result when no value beats -FLT_MAX.
/// This is compatibility testing, not a change to sampler numerical policy.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_partitioned_argmax_preserves_nonfinite_and_sentinel_behavior() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for (values, expected) in [
        (vec![f32::NAN; 33], 0),
        (vec![f32::NEG_INFINITY; 257], 0),
        (vec![-f32::MAX; 12289], 0),
        (vec![f32::NEG_INFINITY, -f32::MAX], 0),
        (vec![f32::INFINITY, f32::NAN, f32::INFINITY], 0),
        (vec![f32::NAN, f32::INFINITY, f32::INFINITY], 1),
        (vec![f32::NAN, f32::NEG_INFINITY, 17.0, 17.0], 2),
        (vec![0.0, -0.0, 0.0], 0),
        (vec![f32::from_bits(1), 0.0], 0),
        (vec![-f32::MAX, f32::MAX, f32::MAX], 1),
    ] {
        compare_case(&ctx, &values, expected)?;
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Invalid capacity, shape, alignment and aliases fail before either kernel
/// is enqueued, without allocating, copying, synchronizing or falling back.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_partitioned_argmax_rejects_invalid_scratch_and_spans() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    assert!(CudaContext::argmax_f32_partitioned_scratch_bytes(0).is_err());
    for (len, bytes) in [
        (1, 8),
        (256, 8),
        (257, 16),
        (12288, 384),
        (151936, 384),
        (u32::MAX, 384),
    ] {
        assert_eq!(
            CudaContext::argmax_f32_partitioned_scratch_bytes(len)?,
            bytes
        );
    }
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let input = DeviceBuffer::new_tagged(&ctx, 512, "argmax:invalid")?;
        let scratch = DeviceBuffer::new_tagged(&ctx, 512, "argmax:invalid")?;
        let output = DeviceBuffer::new_tagged(&ctx, 512, "argmax:invalid")?;
        let before = profile::snapshot();
        unsafe {
            for (len, capacity) in [(0, 384), (151936, 0), (151936, 383), (257, 8)] {
                assert!(ctx
                    .argmax_f32_partitioned_async(
                        input.as_ptr(),
                        len,
                        scratch.as_mut_ptr(),
                        capacity,
                        output.as_mut_ptr(),
                        CudaStream::Decode
                    )
                    .is_err());
            }
            for (x, s, y) in [
                (std::ptr::null(), scratch.as_mut_ptr(), output.as_mut_ptr()),
                (input.as_ptr(), std::ptr::null_mut(), output.as_mut_ptr()),
                (input.as_ptr(), scratch.as_mut_ptr(), std::ptr::null_mut()),
                (
                    input.as_ptr().cast::<u8>().add(2).cast(),
                    scratch.as_mut_ptr(),
                    output.as_mut_ptr(),
                ),
                (
                    input.as_ptr(),
                    scratch.as_mut_ptr().cast::<u8>().add(4).cast(),
                    output.as_mut_ptr(),
                ),
                (
                    input.as_ptr(),
                    scratch.as_mut_ptr(),
                    output.as_mut_ptr().cast::<u8>().add(2).cast(),
                ),
                (input.as_ptr(), input.as_mut_ptr(), output.as_mut_ptr()),
                (input.as_ptr(), scratch.as_mut_ptr(), input.as_mut_ptr()),
                (input.as_ptr(), scratch.as_mut_ptr(), scratch.as_mut_ptr()),
            ] {
                assert!(ctx
                    .argmax_f32_partitioned_async(x, 32, s, 384, y, CudaStream::Decode)
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

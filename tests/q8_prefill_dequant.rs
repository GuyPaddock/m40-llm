use anyhow::Result;
use m40_llm::infer::q8_prefill::{Q8PrefillWeightShape, Q8_PREFILL_MAX_WEIGHT_BYTES};

/// Temporary storage covers either real FFN projection orientation, while the
/// much larger vocabulary projection cannot accidentally become a FP32 cache.
#[test]
fn test_q8_prefill_shape_bounds_temporary_storage() -> Result<()> {
    for (input, output) in [(2048, 11008), (11008, 2048)] {
        let shape = Q8PrefillWeightShape::new(input, output)?;
        assert_eq!((shape.input_dim(), shape.output_dim()), (input, output));
        assert_eq!(shape.f32_bytes(), 90_177_536);
        assert_eq!(shape.q8_bytes(), 23_953_408);
    }
    assert_eq!(
        Q8PrefillWeightShape::new(32, Q8_PREFILL_MAX_WEIGHT_BYTES / 128)?.f32_bytes(),
        Q8_PREFILL_MAX_WEIGHT_BYTES
    );
    for (input, output) in [
        (0, 1),
        (32, 0),
        (33, 1),
        (2048, 151936),
        (32, Q8_PREFILL_MAX_WEIGHT_BYTES / 128 + 1),
        (usize::MAX - 31, 32),
    ] {
        assert!(Q8PrefillWeightShape::new(input, output).is_err());
    }
    Ok(())
}

#[cfg(all(feature = "cuda", nvcc))]
#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[cfg(all(feature = "cuda", nvcc))]
#[path = "common/q8_projection.rs"]
#[allow(dead_code)]
mod fixture;

#[cfg(all(feature = "cuda", nvcc))]
mod hardware {
    use super::*;
    use half::f16;
    use m40_llm::{
        cuda::{CudaContext, CudaStream, DeviceBuffer},
        profile,
    };

    const GUARD: usize = 4;
    const SENTINEL: u32 = 0x4b12_3456;

    fn upload(ctx: &CudaContext, bytes: &[u8], tag: &'static str) -> Result<DeviceBuffer> {
        let buffer = DeviceBuffer::new_tagged(ctx, bytes.len(), tag)?;
        unsafe { ctx.memcpy_h2d(buffer.as_mut_ptr(), bytes.as_ptr().cast(), buffer.bytes())? };
        Ok(buffer)
    }

    fn read(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<f32>> {
        let mut values = vec![0.0; buffer.bytes() / 4];
        unsafe {
            ctx.memcpy_d2h(values.as_mut_ptr().cast(), buffer.as_ptr(), buffer.bytes())?;
        }
        Ok(values)
    }

    fn compare_conversion(ctx: &CudaContext, k: usize, n: usize, offset: usize) -> Result<()> {
        let shape = Q8PrefillWeightShape::new(k, n)?;
        let mut payload = crate::fixture::make_column_blocks(k, n);
        payload[..2].copy_from_slice(&0u16.to_le_bytes());
        let mut source = vec![0x5a; offset + payload.len() + 16];
        source[offset..offset + payload.len()].copy_from_slice(&payload);
        let weight = upload(ctx, &source, "q8-prefill:weights")?;
        let guarded = vec![f32::from_bits(SENTINEL); k * n + 2 * GUARD];
        let output = DeviceBuffer::new_tagged(ctx, guarded.len() * 4, "q8-prefill:temporary")?;
        unsafe {
            ctx.memcpy_h2d(output.as_mut_ptr(), guarded.as_ptr().cast(), output.bytes())?;
        }
        for _ in 0..2 {
            let before = profile::snapshot();
            unsafe {
                ctx.dequantize_q8_0_to_f32_colmajor_async(
                    weight.as_ptr().cast::<u8>().add(offset).cast(),
                    output.as_mut_ptr().cast::<f32>().add(GUARD).cast(),
                    shape.f32_bytes(),
                    shape,
                )?;
            }
            let delta = profile::counter_delta(&before, &profile::snapshot());
            assert_eq!(delta.launches, 1);
            assert_eq!(
                delta.device_allocs
                    + delta.device_frees
                    + delta.h2d_copies
                    + delta.d2h_copies
                    + delta.stream_syncs
                    + delta.cublas_calls,
                0
            );
            ctx.synchronize_stream(CudaStream::Prefill)?;
            let actual = read(ctx, &output)?;
            assert!(actual[..GUARD]
                .iter()
                .chain(&actual[GUARD + k * n..])
                .all(|v| v.to_bits() == SENTINEL));
            for (index, &value) in actual[GUARD..GUARD + k * n].iter().enumerate() {
                let input = index / n;
                let column = index % n;
                let base = (column * (k / 32) + input / 32) * 34;
                let scale =
                    f16::from_bits(u16::from_le_bytes([payload[base], payload[base + 1]])).to_f32();
                let expected = scale * f32::from(payload[base + 2 + input % 32] as i8);
                assert_eq!(
                    value.to_bits(),
                    expected.to_bits(),
                    "k={k} n={n} index={index}"
                );
            }
        }
        let mut after = vec![0; source.len()];
        unsafe {
            ctx.memcpy_d2h(after.as_mut_ptr().cast(), weight.as_ptr(), after.len())?;
        }
        assert_eq!(after, source);
        Ok(())
    }

    /// Dequantization preserves every FP32 weight bit in SGEMM layout, including
    /// signed extremes, zero scales, output tails, guards and two-byte bases.
    /// Real Qwen gate/up and down shapes exercise the bounded allocation.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_q8_prefill_conversion_preserves_values_layout_and_reuse() -> Result<()> {
        let _lock = crate::cuda_test_scope::lock_cuda();
        let ctx = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        for (k, n, offset) in [(32, 1, 0), (96, 37, 2), (2048, 11008, 0), (11008, 2048, 2)] {
            compare_conversion(&ctx, k, n, offset)?;
        }
        assert_eq!(CudaContext::total_device_bytes(), initial);
        Ok(())
    }

    /// A queued source producer, temporary conversion and SGEMM obey explicit
    /// stream dependencies. Distinct batched rows match independent CPU dots
    /// without an enqueue allocation, host transfer or stream synchronization.
    #[cfg(have_cublas)]
    #[test]
    #[ignore = "requires M40/cuBLAS; run with --include-ignored --test-threads=1"]
    fn test_q8_prefill_temporary_layout_feeds_async_sgemm() -> Result<()> {
        let _lock = crate::cuda_test_scope::lock_cuda();
        let ctx = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        for (k, n, rows) in [(96, 37, 4), (96, 37, 64), (2048, 257, 4)] {
            let shape = Q8PrefillWeightShape::new(k, n)?;
            let payload = crate::fixture::make_column_blocks(k, n);
            let input = crate::fixture::make_input(rows, k);
            let source = upload(&ctx, &payload, "q8-prefill:source")?;
            let weights = DeviceBuffer::new_tagged(&ctx, payload.len(), "q8-prefill:queued")?;
            let input_bytes: Vec<_> = input.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let x = upload(&ctx, &input_bytes, "q8-prefill:input")?;
            let temporary =
                DeviceBuffer::new_tagged(&ctx, shape.f32_bytes(), "q8-prefill:sgemm-weight")?;
            let output = DeviceBuffer::new_tagged(&ctx, rows * n * 4, "q8-prefill:output")?;
            unsafe {
                ctx.memcpy_d2d_async(
                    weights.as_mut_ptr(),
                    source.as_ptr(),
                    payload.len(),
                    CudaStream::Decode,
                )?;
            }
            ctx.stream_wait_for_stream(
                CudaStream::Prefill,
                CudaStream::Decode,
                "test_q8_prefill_source",
            )?;
            let before = profile::snapshot();
            unsafe {
                ctx.dequantize_q8_0_to_f32_colmajor_async(
                    weights.as_ptr(),
                    temporary.as_mut_ptr(),
                    temporary.bytes(),
                    shape,
                )?;
                ctx.gemm_f32xf32_f32_stream_async(
                    x.as_ptr(),
                    temporary.as_ptr(),
                    output.as_mut_ptr(),
                    rows as i32,
                    n as i32,
                    k as i32,
                    CudaStream::Prefill,
                )?;
            }
            let delta = profile::counter_delta(&before, &profile::snapshot());
            assert_eq!((delta.launches, delta.cublas_calls), (1, 1));
            assert_eq!(
                delta.device_allocs
                    + delta.device_frees
                    + delta.h2d_copies
                    + delta.d2h_copies
                    + delta.stream_syncs,
                0
            );
            ctx.synchronize_stream(CudaStream::Prefill)?;
            let actual = read(&ctx, &output)?;
            for (index, &value) in actual.iter().enumerate() {
                let expected =
                    crate::fixture::project_cpu(&input, &payload, index / n, index % n, k);
                assert!(
                    value.is_finite()
                        && (f64::from(value) - expected).abs() < 1e-3 + 2e-5 * expected.abs(),
                    "k={k} n={n} rows={rows} index={index}: {value} vs {expected}"
                );
            }
        }
        assert_eq!(CudaContext::total_device_bytes(), initial);
        Ok(())
    }

    /// Invalid buffer spans are rejected before enqueue, not after an asynchronous
    /// out-of-bounds write or an implicit fallback to another projection path.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_q8_prefill_conversion_rejects_invalid_buffers_before_launch() -> Result<()> {
        let _lock = crate::cuda_test_scope::lock_cuda();
        let ctx = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        {
            let shape = Q8PrefillWeightShape::new(32, 1)?;
            let source = DeviceBuffer::new_tagged(&ctx, 64, "q8-prefill:invalid-source")?;
            let output = DeviceBuffer::new_tagged(&ctx, 256, "q8-prefill:invalid-output")?;
            let before = profile::snapshot();
            let bad = unsafe {
                [
                    (std::ptr::null(), output.as_mut_ptr(), 128),
                    (source.as_ptr(), std::ptr::null_mut(), 128),
                    (source.as_ptr(), source.as_mut_ptr(), 128),
                    (
                        source.as_ptr().cast::<u8>().add(1).cast(),
                        output.as_mut_ptr(),
                        128,
                    ),
                    (
                        source.as_ptr(),
                        output.as_mut_ptr().cast::<u8>().add(2).cast(),
                        128,
                    ),
                    (source.as_ptr(), output.as_mut_ptr(), 127),
                ]
            };
            for (source, output, capacity) in bad {
                assert!(unsafe {
                    ctx.dequantize_q8_0_to_f32_colmajor_async(source, output, capacity, shape)
                }
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
}

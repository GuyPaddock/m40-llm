use anyhow::Result;
use half::f16;
use m40_llm::infer::q8_activation::{quantize_reference, Q8ActivationBlock, Q8ActivationShape};

/// Zero, ties away from zero and FP32-before-FP16 scaling are part of the
/// pinned CUDA contract, not interchangeable quantizer implementation details.
#[test]
fn test_q8_activation_rounding_and_zero_contract() -> Result<()> {
    let shape = Q8ActivationShape::new(1, 32, 32)?;
    let zero = quantize_reference(&[0.0; 32], shape)?;
    assert_eq!(zero[0].to_bytes(), [0u8; 36]);
    let negative_zero = quantize_reference(&[-0.0; 32], shape)?;
    assert_eq!(negative_zero[0].sum_bits, 0x8000);
    assert_eq!(negative_zero[0].quants, [0; 32]);
    let mut input = [0.0; 32];
    input[..6].copy_from_slice(&[127.0, -127.0, 0.5, -0.5, 1.5, -1.5]);
    let block = &quantize_reference(&input, shape)?[0];
    assert_eq!(block.scale_bits, f16::from_f32(1.0).to_bits());
    assert_eq!(&block.quants[..6], &[127, -127, 1, -1, 2, -2]);
    input.fill(0.0);
    input[0] = 1.0;
    let scale = 1.0f32 / 127.0;
    input[1] = f32::from_bits((scale * 0.5).to_bits() - 1);
    let block = &quantize_reference(&input, shape)?[0];
    assert_eq!(block.quants[1], 0);
    assert_eq!(
        (input[1] / f16::from_bits(block.scale_bits).to_f32()).round() as i8,
        1
    );
    Ok(())
}

/// The sum field preserves original FP32 inputs through a warp XOR tree;
/// quantized integers cannot reconstruct it. Every row and padded block has
/// an explicit 36-byte layout, independent of Rust struct memory layout.
#[test]
fn test_q8_activation_sum_padding_and_serialization() -> Result<()> {
    let shape = Q8ActivationShape::new(1, 32, 32)?;
    let mut input = [0.001; 32];
    input[0] = 1.0;
    let block = &quantize_reference(&input, shape)?[0];
    assert_eq!(block.sum_bits, f16::from_f32(1.031).to_bits());
    assert_eq!(block.quants.iter().map(|&v| i32::from(v)).sum::<i32>(), 127);
    assert_ne!(
        block.sum_bits,
        f16::from_f32(block.reconstruct().iter().sum()).to_bits()
    );
    let shape = Q8ActivationShape::new(4, 33, 96)?;
    let input: Vec<_> = (0..4).flat_map(|row| vec![(row + 1) as f32; 33]).collect();
    let encoded = quantize_reference(&input, shape)?;
    assert_eq!(shape.bytes(), 4 * 3 * 36);
    assert_eq!(encoded.len(), shape.block_count());
    for row in 0..4 {
        assert_ne!(encoded[row * 3].scale_bits, 0);
        assert_eq!(encoded[row * 3 + 1].quants[0], 127);
        assert_eq!(&encoded[row * 3 + 1].quants[1..], &[0i8; 31]);
        assert_eq!(encoded[row * 3 + 2].to_bytes(), [0u8; 36]);
    }
    for block in &encoded {
        assert_eq!(*block, Q8ActivationBlock::from_bytes(&block.to_bytes())?);
    }
    assert!(Q8ActivationBlock::from_bytes(&[0; 35]).is_err());
    Ok(())
}

/// Invalid shapes, nonfinite values and unrepresentable FP32 scale underflow
/// must not manufacture a seemingly valid reference encoding.
#[test]
fn test_q8_activation_rejects_invalid_reference_inputs() -> Result<()> {
    for (rows, columns, padded) in [
        (0, 32, 32),
        (2, 32, 32),
        (1, 0, 32),
        (1, 33, 32),
        (4, 32, 33),
        (1, 32, usize::MAX),
    ] {
        assert!(Q8ActivationShape::new(rows, columns, padded).is_err());
    }
    let shape = Q8ActivationShape::new(1, 32, 32)?;
    assert!(quantize_reference(&[0.0; 31], shape).is_err());
    for value in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
    ] {
        let mut input = [0.0; 32];
        input[0] = value;
        assert!(quantize_reference(&input, shape).is_err());
    }
    Ok(())
}

#[cfg(all(feature = "cuda", nvcc))]
#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;

#[cfg(all(feature = "cuda", nvcc))]
mod hardware {
    use super::*;
    use m40_llm::{
        cuda::{CudaContext, CudaStream, DeviceBuffer},
        profile,
    };

    /// The actual M40 quantizer matches every CPU contract byte, including
    /// signed/tie cases, original-input sums, row padding and FP16 scale bits.
    /// Caller-owned guarded buffers are reused without per-enqueue churn.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_q8_activation_cuda_matches_reference_and_preserves_guards() -> Result<()> {
        let _lock = crate::cuda_test_scope::lock_cuda();
        let ctx = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        for (rows, columns, padded) in [
            (1, 32, 32),
            (4, 33, 96),
            (1, 2048, 2048),
            (4, 11008, 11008),
            (1, 257, 512),
        ] {
            let shape = Q8ActivationShape::new(rows, columns, padded)?;
            let input = DeviceBuffer::new_tagged(
                &ctx,
                (rows * columns + 16) * 4,
                "q8-activation:test-input",
            )?;
            let output =
                DeviceBuffer::new_tagged(&ctx, shape.bytes() + 32, "q8-activation:test-output")?;
            let guarded_output = vec![0xa5u8; shape.bytes() + 32];
            for iteration in 0..4 {
                let scale = 1.0f32 / 127.0;
                let values: Vec<_> = (0..rows * columns)
                    .map(|i| match iteration {
                        0 => 0.0,
                        1 => -0.0,
                        2 => [
                            1.0,
                            f32::from_bits((scale * 0.5).to_bits() - 1),
                            0.001,
                            -0.5,
                            0.5,
                            -1.0,
                            0.0,
                            -0.001,
                        ][i % 8],
                        _ => ((i * 13 % 127) as f32 - 63.0) * 0.0078125,
                    })
                    .collect();
                let expected: Vec<u8> = quantize_reference(&values, shape)?
                    .iter()
                    .flat_map(Q8ActivationBlock::to_bytes)
                    .collect();
                let mut guarded_input = vec![f32::from_bits(0x4b12_3456); rows * columns + 16];
                guarded_input[8..8 + values.len()].copy_from_slice(&values);
                unsafe {
                    ctx.memcpy_h2d(
                        input.as_mut_ptr(),
                        guarded_input.as_ptr().cast(),
                        input.bytes(),
                    )?;
                    ctx.memcpy_h2d(
                        output.as_mut_ptr(),
                        guarded_output.as_ptr().cast(),
                        output.bytes(),
                    )?;
                }
                let before = profile::snapshot();
                unsafe {
                    ctx.quantize_q8_1_async(
                        input.as_ptr().cast::<f32>().add(8).cast(),
                        output.as_mut_ptr().cast::<u8>().add(16).cast(),
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
                        + delta.stream_syncs,
                    0
                );
                ctx.synchronize_stream(CudaStream::Decode)?;
                let mut actual = vec![0u8; output.bytes()];
                let mut input_after = vec![0u8; input.bytes()];
                unsafe {
                    ctx.memcpy_d2h(actual.as_mut_ptr().cast(), output.as_ptr(), actual.len())?;
                    ctx.memcpy_d2h(
                        input_after.as_mut_ptr().cast(),
                        input.as_ptr(),
                        input_after.len(),
                    )?;
                }
                assert_eq!(&actual[..16], &[0xa5; 16]);
                assert_eq!(&actual[16 + shape.bytes()..], &[0xa5; 16]);
                assert_eq!(
                    &actual[16..16 + shape.bytes()],
                    &expected,
                    "rows={rows} columns={columns} iteration={iteration}"
                );
                let original: Vec<_> = guarded_input
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect();
                assert_eq!(input_after, original);
            }
        }
        assert_eq!(CudaContext::total_device_bytes(), initial);
        Ok(())
    }

    /// Invalid pointers/alignment fail before launch or synchronization rather
    /// than silently selecting another activation representation.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_q8_activation_cuda_rejects_bad_buffers_without_gpu_work() -> Result<()> {
        let _lock = crate::cuda_test_scope::lock_cuda();
        let ctx = CudaContext::new_m40()?;
        let input = DeviceBuffer::new_tagged(&ctx, 128, "q8-activation:invalid-input")?;
        let output = DeviceBuffer::new_tagged(&ctx, 128, "q8-activation:invalid-output")?;
        let shape = Q8ActivationShape::new(1, 32, 32)?;
        let before = profile::snapshot();
        unsafe {
            for pointer in [std::ptr::null(), input.as_ptr().cast::<u8>().add(1).cast()] {
                assert!(ctx
                    .quantize_q8_1_async(pointer, output.as_mut_ptr(), shape)
                    .is_err());
            }
            assert!(ctx
                .quantize_q8_1_async(input.as_ptr(), input.as_mut_ptr(), shape)
                .is_err());
            assert!(ctx
                .quantize_q8_1_async(
                    input.as_ptr(),
                    output.as_mut_ptr().cast::<u8>().add(2).cast(),
                    shape
                )
                .is_err());
        }
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()),
            Default::default()
        );
        Ok(())
    }
}

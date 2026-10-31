#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/guarded_bytes.rs"]
mod guarded_bytes;

use anyhow::Result;
use guarded_bytes::GuardedBytes;
use half::f16;
use m40_llm::{
    cuda::{CudaContext, CudaStream},
    profile,
    shiftadd::ShiftAddMatrix,
};
use std::ffi::c_void;

fn encode_floats(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn decode_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
        .collect()
}

fn fit_matrix(input_width: usize, output_width: usize) -> ShiftAddMatrix {
    let source: Vec<f32> = (0..input_width * output_width)
        .map(|index| ((index * 23 + index / input_width * 11) % 251) as f32 * 0.0003 - 0.0375)
        .collect();
    ShiftAddMatrix::fit(input_width, output_width, &source).unwrap()
}

/// Bound FP32 sign-add reductions against the FP64 decoded-weight dot. Each
/// path has four local adds, five shuffle adds, then at most two roundings per
/// plane/group contribution. The absolute sum uses plane amplitudes before
/// cancellation, so signed-weight cancellation cannot make the bound too small.
fn assert_projection(matrix: &ShiftAddMatrix, input: &[f32], actual: &[f32]) -> f64 {
    let expected = matrix.project_reference(input).unwrap();
    assert_eq!(actual.len(), expected.len());
    let k = matrix.get_input_width();
    let n = matrix.get_output_width();
    let operations = (12 + k / 16) as f64;
    let epsilon = 2.0_f64.powi(-24);
    let gamma = operations * epsilon / (1.0 - operations * epsilon);
    let mut maximum_fraction = 0.0_f64;
    for (index, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
        let row = index / n;
        let channel = index % n;
        let mut absolute_sum = 0.0;
        for block_index in 0..k / 256 {
            let block = &matrix.get_bytes()[(channel * (k / 256) + block_index) * 138..][..138];
            let scale = f16::from_le_bytes([block[136], block[137]]).to_f64();
            for group in 0..2 {
                let amplitude: f64 = block[128 + group * 4..132 + group * 4]
                    .iter()
                    .map(|&value| f64::from(value) * scale)
                    .sum();
                let start = row * k + block_index * 256 + group * 128;
                absolute_sum += amplitude
                    * input[start..start + 128]
                        .iter()
                        .map(|&value| f64::from(value).abs())
                        .sum::<f64>();
            }
        }
        let bound = gamma * absolute_sum + operations * f64::from(f32::MIN_POSITIVE);
        let error = (f64::from(a) - e).abs();
        assert!(
            a.is_finite() && error <= bound,
            "index {index}: actual={a} expected={e} error={error} bound={bound}"
        );
        maximum_fraction = maximum_fraction.max(error / bound);
    }
    maximum_fraction
}

/// All supported batch sizes, short and FFN input widths, output tails and
/// cancellation-heavy inputs preserve guards and encoded weights while meeting
/// an operation-count-derived FP64 reference bound on the actual M40.
#[test]
#[ignore = "explicit serial M40 ShiftAdd numerical gate"]
fn test_shiftadd_gemv_reference_and_guards() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    let mut maximum_fraction = 0.0_f64;
    let mut comparisons = 0;
    for k in [256, 512, 2048, 11008] {
        for n in [1, 5, 17] {
            let matrix = fit_matrix(k, n);
            {
                // 34-byte prefix forces legal two-byte-only scale alignment.
                let weights = GuardedBytes::new(&ctx, matrix.get_bytes(), 34)?;
                for rows in [1, 2, 4, 8] {
                    for pattern in 0..3 {
                        let input: Vec<f32> = (0..rows * k)
                            .map(|index| match pattern {
                                0 => ((index * 13 + index / k * 37) % 127) as f32 * 0.013 - 0.8,
                                1 => {
                                    if index % 2 == 0 {
                                        127.3
                                    } else {
                                        -127.25 + (index % 7) as f32 * 0.001
                                    }
                                }
                                _ => 0.0,
                            })
                            .collect();
                        let input_bytes = encode_floats(&input);
                        let device_input = GuardedBytes::new(&ctx, &input_bytes, 32)?;
                        let output = GuardedBytes::new(&ctx, &vec![0xcd; rows * n * 4], 32)?;
                        let before = profile::snapshot();
                        unsafe {
                            ctx.gemv_f32xsa4_128_async(
                                device_input.pointer(),
                                weights.pointer(),
                                output.pointer(),
                                rows as i32,
                                n as i32,
                                k as i32,
                                CudaStream::Decode,
                            )?;
                        }
                        ctx.synchronize_stream(CudaStream::Decode)?;
                        let delta = profile::counter_delta(&before, &profile::snapshot());
                        assert_eq!(delta.device_allocs, 0);
                        assert_eq!(delta.device_frees, 0);
                        let actual = decode_floats(&output.read(&ctx)?);
                        maximum_fraction =
                            maximum_fraction.max(assert_projection(&matrix, &input, &actual));
                        comparisons += actual.len();
                        assert_eq!(device_input.read(&ctx)?, input_bytes);
                        assert_eq!(weights.read(&ctx)?, matrix.get_bytes());
                    }
                }
            }
            assert_eq!(CudaContext::total_device_bytes(), initial);
        }
    }
    eprintln!("ShiftAdd reference comparisons={comparisons} maximum_bound_fraction={maximum_fraction} guards=pass immutable=pass cleanup=pass");
    Ok(())
}

/// Independent live workspaces and explicit producer/consumer dependencies
/// preserve two chained projections in both stream directions. Reusing the
/// input address with new contents must recompute the result on every call.
#[test]
#[ignore = "explicit serial M40 ShiftAdd stream and ownership gate"]
fn test_shiftadd_streams_reuse_and_cleanup() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let first = fit_matrix(256, 256);
        let second = fit_matrix(256, 5);
        let first_weights = GuardedBytes::new(&ctx, first.get_bytes(), 34)?;
        let second_weights = GuardedBytes::new(&ctx, second.get_bytes(), 34)?;
        let inputs = [
            GuardedBytes::new(&ctx, &vec![0; 4 * 256 * 4], 32)?,
            GuardedBytes::new(&ctx, &vec![0; 4 * 256 * 4], 32)?,
        ];
        let intermediates = [
            GuardedBytes::new(&ctx, &vec![0xcd; 4 * 256 * 4], 32)?,
            GuardedBytes::new(&ctx, &vec![0xcd; 4 * 256 * 4], 32)?,
        ];
        let outputs = [
            GuardedBytes::new(&ctx, &[0xcd; 4 * 5 * 4], 32)?,
            GuardedBytes::new(&ctx, &[0xcd; 4 * 5 * 4], 32)?,
        ];
        for (producer, consumer) in [
            (CudaStream::Prefill, CudaStream::Decode),
            (CudaStream::Decode, CudaStream::Prefill),
        ] {
            for generation in 0..3 {
                for workspace in 0..2 {
                    let other_output = outputs[1 - workspace].read(&ctx)?;
                    let other_intermediate = intermediates[1 - workspace].read(&ctx)?;
                    let input: Vec<f32> = (0..4 * 256)
                        .map(|i| {
                            ((i * 13 + generation * 43 + workspace * 17) % 97) as f32 * 0.007 - 0.3
                        })
                        .collect();
                    let bytes = encode_floats(&input);
                    unsafe {
                        ctx.memcpy_h2d(
                            inputs[workspace].pointer(),
                            bytes.as_ptr().cast(),
                            bytes.len(),
                        )?;
                        ctx.gemv_f32xsa4_128_async(
                            inputs[workspace].pointer(),
                            first_weights.pointer(),
                            intermediates[workspace].pointer(),
                            4,
                            256,
                            256,
                            producer,
                        )?;
                        ctx.stream_wait_for_stream(
                            consumer,
                            producer,
                            "shiftadd:test_producer_to_consumer",
                        )?;
                        ctx.gemv_f32xsa4_128_async(
                            intermediates[workspace].pointer(),
                            second_weights.pointer(),
                            outputs[workspace].pointer(),
                            4,
                            5,
                            256,
                            consumer,
                        )?;
                    }
                    ctx.synchronize_stream(consumer)?;
                    let intermediate = decode_floats(&intermediates[workspace].read(&ctx)?);
                    assert_projection(&first, &input, &intermediate);
                    assert_projection(
                        &second,
                        &intermediate,
                        &decode_floats(&outputs[workspace].read(&ctx)?),
                    );
                    assert_eq!(inputs[workspace].read(&ctx)?, bytes);
                    assert_eq!(outputs[1 - workspace].read(&ctx)?, other_output);
                    assert_eq!(intermediates[1 - workspace].read(&ctx)?, other_intermediate);
                }
            }
        }
        assert_eq!(first_weights.read(&ctx)?, first.get_bytes());
        assert_eq!(second_weights.read(&ctx)?, second.get_bytes());
    }
    ctx.synchronize_stream(CudaStream::Decode)?;
    ctx.synchronize_stream(CudaStream::Prefill)?;
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Invalid spans and dimensions must reject before enqueue, without mutating
/// inputs, weights or output. Partial overlaps are errors as well as equality.
#[test]
#[ignore = "explicit serial M40 ShiftAdd argument gate"]
fn test_shiftadd_rejects_invalid_spans() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let matrix = fit_matrix(256, 5);
        let input = GuardedBytes::new(&ctx, &vec![0; 256 * 4], 32)?;
        let weights = GuardedBytes::new(&ctx, matrix.get_bytes(), 32)?;
        let output = GuardedBytes::new(&ctx, &[0xcd; 20], 32)?;
        let before = profile::snapshot();
        let cases = [
            (
                input.pointer().cast_const(),
                weights.pointer().cast_const(),
                output.pointer(),
                3,
                5,
                256,
            ),
            (
                input.pointer().cast_const(),
                weights.pointer().cast_const(),
                output.pointer(),
                1,
                5,
                255,
            ),
            (
                input.pointer().cast_const(),
                weights.pointer().cast_const(),
                output.pointer(),
                1,
                0,
                256,
            ),
            (
                std::ptr::null(),
                weights.pointer().cast_const(),
                output.pointer(),
                1,
                5,
                256,
            ),
            (
                input.pointer().cast_const(),
                std::ptr::null(),
                output.pointer(),
                1,
                5,
                256,
            ),
            (
                (input.pointer() as usize + 1) as *const c_void,
                weights.pointer().cast_const(),
                output.pointer(),
                1,
                5,
                256,
            ),
            (
                input.pointer().cast_const(),
                (weights.pointer() as usize + 1) as *const c_void,
                output.pointer(),
                1,
                5,
                256,
            ),
            (
                input.pointer().cast_const(),
                weights.pointer().cast_const(),
                (output.pointer() as usize + 1) as *mut c_void,
                1,
                5,
                256,
            ),
            (
                input.pointer().cast_const(),
                weights.pointer().cast_const(),
                (input.pointer() as usize + 4) as *mut c_void,
                1,
                5,
                256,
            ),
            (
                input.pointer().cast_const(),
                weights.pointer().cast_const(),
                (weights.pointer() as usize + 4) as *mut c_void,
                1,
                5,
                256,
            ),
            (
                input.pointer().cast_const(),
                weights.pointer().cast_const(),
                (usize::MAX - 3) as *mut c_void,
                1,
                5,
                256,
            ),
        ];
        for (source, packed, destination, rows, n, k) in cases {
            assert!(unsafe {
                ctx.gemv_f32xsa4_128_async(
                    source,
                    packed,
                    destination,
                    rows,
                    n,
                    k,
                    CudaStream::Decode,
                )
            }
            .is_err());
        }
        let delta = profile::counter_delta(&before, &profile::snapshot());
        assert_eq!(delta.launches, 0);
        assert_eq!(delta.device_allocs, 0);
        assert_eq!(input.read(&ctx)?, vec![0; 256 * 4]);
        assert_eq!(weights.read(&ctx)?, matrix.get_bytes());
        assert_eq!(output.read(&ctx)?, [0xcd; 20]);
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

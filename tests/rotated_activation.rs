#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/q8_projection.rs"]
#[allow(dead_code)]
mod fixture;

use anyhow::Result;
use half::f16;
use m40_llm::{
    cuda::{
        CudaContext, CudaStream, DeviceBuffer, DeviceRotatedActivation, Q8FfnQ81Launch,
        RotatedPreparation,
    },
    infer::q8_activation::{quantize_reference, Q8ActivationBlock, Q8ActivationShape},
    profile,
};
use std::ffi::c_void;

const GUARD: u32 = 0x4b12_3456;

fn upload_bytes(ctx: &CudaContext, bytes: &[u8]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new_tagged(ctx, bytes.len(), "rotated_activation:test")?;
    unsafe {
        ctx.memcpy_h2d(buffer.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())?;
    }
    Ok(buffer)
}

fn upload_guarded(ctx: &CudaContext, values: &[f32]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new_tagged(ctx, (values.len() + 16) * 4, "rotated_activation:test")?;
    overwrite_guarded(ctx, &buffer, values)?;
    Ok(buffer)
}

fn overwrite_guarded(ctx: &CudaContext, buffer: &DeviceBuffer, values: &[f32]) -> Result<()> {
    let mut guarded = vec![f32::from_bits(GUARD); values.len() + 16];
    guarded[8..8 + values.len()].copy_from_slice(values);
    assert_eq!(buffer.bytes(), guarded.len() * 4);
    unsafe {
        ctx.memcpy_h2d(buffer.as_mut_ptr(), guarded.as_ptr().cast(), buffer.bytes())?;
    }
    Ok(())
}

fn get_payload(buffer: &DeviceBuffer) -> *mut c_void {
    unsafe { buffer.as_mut_ptr().cast::<f32>().add(8).cast() }
}

fn read_guarded(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<f32>> {
    let mut values = vec![0.0f32; buffer.bytes() / 4];
    unsafe {
        ctx.memcpy_d2h(values.as_mut_ptr().cast(), buffer.as_ptr(), buffer.bytes())?;
    }
    assert!(values[..8]
        .iter()
        .chain(&values[values.len() - 8..])
        .all(|v| v.to_bits() == GUARD));
    Ok(values[8..values.len() - 8].to_vec())
}

fn read_bytes(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<u8>> {
    let mut values = vec![0; buffer.bytes()];
    unsafe {
        ctx.memcpy_d2h(values.as_mut_ptr().cast(), buffer.as_ptr(), buffer.bytes())?;
    }
    Ok(values)
}

// Dense Walsh coefficients are independent of the CUDA butterfly schedule.
fn rotate_reference(input: &[f32], signs: &[i8]) -> Vec<f64> {
    input
        .chunks_exact(signs.len())
        .flat_map(|row| {
            row.chunks_exact(128)
                .enumerate()
                .flat_map(|(block, values)| {
                    (0..128usize).map(move |output| {
                        values
                            .iter()
                            .enumerate()
                            .map(|(input, &value)| {
                                let walsh = if (input & output).count_ones().is_multiple_of(2) {
                                    1.0
                                } else {
                                    -1.0
                                };
                                f64::from(value) * f64::from(signs[block * 128 + input]) * walsh
                            })
                            .sum::<f64>()
                            / 128.0f64.sqrt()
                    })
                })
        })
        .collect()
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

// The consumer multiplies integer dot products by block scales. Reconstructing
// each activation into FP32 first adds rounding that this kernel never performs.
// Bound its two scale/product roundings, per-thread sums and seven tree levels
// by gamma_n times the absolute sum, so cancellation does not hide input scale.
fn project_quantized_reference(
    input: &[Q8ActivationBlock],
    weights: &[u8],
    row: usize,
    channel: usize,
    columns: usize,
) -> (f64, f64) {
    let blocks = columns / 32;
    let mut value = 0.0;
    let mut absolute_sum = 0.0;
    for block in 0..blocks {
        let activation = &input[row * blocks + block];
        let base = (channel * blocks + block) * 34;
        let scale = f16::from_bits(activation.scale_bits).to_f64()
            * f16::from_le_bytes([weights[base], weights[base + 1]]).to_f64();
        for lane in 0..32 {
            let term = scale
                * f64::from(activation.quants[lane])
                * f64::from(weights[base + 2 + lane] as i8);
            value += term;
            absolute_sum += term.abs();
        }
    }
    let roundings = (blocks * 4).div_ceil(128) + 7 + 2;
    let epsilon = roundings as f64 * 2.0f64.powi(-24);
    (value, epsilon / (1.0 - epsilon) * absolute_sum)
}

fn check_case(ctx: &CudaContext, rows: usize, columns: usize) -> Result<()> {
    let signs: Vec<i8> = (0..columns)
        .map(|i| if i % 3 == 1 { -1 } else { 1 })
        .collect();
    let owner = DeviceRotatedActivation::upload(ctx, rows, &signs)?;
    let input = upload_guarded(ctx, &vec![0.0; rows * columns])?;
    let source = upload_guarded(ctx, &vec![0.0; rows * columns])?;
    let hidden = 33;
    let gate = fixture::make_column_blocks(columns, hidden);
    let mut up = gate.clone();
    for block in up.chunks_exact_mut(34) {
        block[2..].reverse();
    }
    let gate_device = upload_bytes(ctx, &gate)?;
    let up_device = upload_bytes(ctx, &up)?;
    let output = upload_guarded(ctx, &vec![0.0; rows * hidden])?;
    let shape = Q8ActivationShape::new(1, columns, columns)?;
    let flattened = Q8ActivationShape::new(1, rows * columns, rows * columns)?;
    let resident = CudaContext::total_device_bytes();
    for iteration in 0..4 {
        let values: Vec<_> = (0..rows * columns)
            .map(|i| match iteration {
                0 => 0.0,
                1 => {
                    if i % 2 == 0 {
                        0.5
                    } else {
                        -0.5
                    }
                }
                2 => ((i * 13 % 127) as f32 - 63.0) * 0.0078125 + (i / columns) as f32 * 0.03125,
                _ => [127.0, -127.0, 0.5, -0.5, 1.5, -1.5, 0.0, -0.0][i % 8],
            })
            .collect();
        overwrite_guarded(ctx, &source, &values)?;
        let reference = rotate_reference(&values, &signs);
        let mut previous: Option<(Vec<f32>, Vec<u8>, Vec<f32>)> = None;
        for producer in [CudaStream::Prefill, CudaStream::Decode] {
            let upstream = if producer == CudaStream::Decode {
                CudaStream::Prefill
            } else {
                CudaStream::Decode
            };
            unsafe {
                ctx.memcpy_d2d_async(
                    get_payload(&input),
                    get_payload(&source),
                    rows * columns * 4,
                    upstream,
                )?;
            }
            ctx.stream_wait_for_stream(producer, upstream, "rotated_test_input")?;
            for mode in [RotatedPreparation::Staged, RotatedPreparation::Fused] {
                let before = profile::snapshot();
                unsafe {
                    owner.enqueue_async(get_payload(&input), mode, producer)?;
                }
                let delta = profile::counter_delta(&before, &profile::snapshot());
                assert_eq!(
                    delta.launches,
                    if mode == RotatedPreparation::Staged {
                        3
                    } else {
                        1
                    }
                );
                assert_eq!(
                    delta.device_allocs
                        + delta.device_frees
                        + delta.h2d_copies
                        + delta.d2h_copies
                        + delta.stream_syncs
                        + delta.stream_waits
                        + delta.cublas_calls,
                    0
                );
                if producer != CudaStream::Decode {
                    ctx.stream_wait_for_stream(
                        CudaStream::Decode,
                        producer,
                        "rotated_test_consumer",
                    )?;
                }
                for row in 0..rows {
                    unsafe {
                        ctx.ffn_q8_1_async(
                            owner
                                .get_quantized_pointer()
                                .cast::<u8>()
                                .add(row * shape.bytes())
                                .cast(),
                            gate_device.as_ptr(),
                            up_device.as_ptr(),
                            get_payload(&output).cast::<f32>().add(row * hidden).cast(),
                            None,
                            hidden as i32,
                            shape,
                            Q8FfnQ81Launch::Threads128,
                        )?;
                    }
                }
                let (rotated, encoded) = owner.read_preparation()?;
                for (&actual, &expected) in rotated.iter().zip(&reference) {
                    assert!(
                        (f64::from(actual) - expected).abs() <= 1e-5 + 2e-6 * expected.abs(),
                        "rotation {actual} vs {expected}"
                    );
                }
                let blocks = quantize_reference(&rotated, flattened)?;
                assert_eq!(
                    encoded,
                    blocks.iter().flat_map(|b| b.to_bytes()).collect::<Vec<_>>()
                );
                let result = read_guarded(ctx, &output)?;
                for row in 0..rows {
                    for channel in 0..hidden {
                        let (g, g_error) =
                            project_quantized_reference(&blocks, &gate, row, channel, columns);
                        let (u, u_error) =
                            project_quantized_reference(&blocks, &up, row, channel, columns);
                        let expected = g / (1.0 + (-g).exp()) * u;
                        // |SiLU'| < 1.1 and |SiLU(g)| <= |g|. The final term
                        // covers the existing activation/product arithmetic.
                        let bound = 1.1 * g_error * (u.abs() + u_error)
                            + g.abs() * u_error
                            + 1e-3
                            + 2e-5 * expected.abs();
                        let actual = f64::from(result[row * hidden + channel]);
                        assert!(actual.is_finite() && (actual - expected).abs() <= bound,
                            "rows={rows} K={columns} iteration={iteration} row={row} channel={channel}: {actual} vs {expected}, bound={bound}");
                    }
                }
                if let Some((previous_rotated, previous_encoded, previous_output)) = &previous {
                    assert_bits(&rotated, previous_rotated);
                    assert_eq!(&encoded, previous_encoded);
                    assert_bits(&result, previous_output);
                }
                previous = Some((rotated, encoded, result));
            }
        }
        assert_bits(&read_guarded(ctx, &input)?, &values);
        assert_bits(&read_guarded(ctx, &source)?, &values);
        assert_eq!(
            read_bytes(ctx, owner.get_sign_buffer())?,
            signs.iter().map(|&s| s as u8).collect::<Vec<_>>()
        );
        assert_eq!(read_bytes(ctx, &gate_device)?, gate);
        assert_eq!(read_bytes(ctx, &up_device)?, up);
        assert_eq!(CudaContext::total_device_bytes(), resident);
    }
    Ok(())
}

/// Staged and fused producers match dense Walsh/CPU Q8_1, preserve every byte
/// of immutable inputs, and feed identical results to the unchanged consumer.
/// Distinct rows, address reuse and both dependency directions protect ownership.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_rotated_preparation_matches_reference_and_consumer() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for columns in [128, 2048, 11008] {
        for rows in [1, 2, 4, 8] {
            check_case(&ctx, rows, columns)?;
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Separate shapes can be enqueued on independent streams without sharing
/// scratch or changing the first owner's output when the second is reused.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_rotated_preparation_independent_owners() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let first = DeviceRotatedActivation::upload(&ctx, 2, &[1; 128])?;
        let second = DeviceRotatedActivation::upload(&ctx, 4, &[-1; 256])?;
        let input = upload_guarded(&ctx, &fixture::make_input(4, 256))?;
        unsafe {
            first.enqueue_async(
                get_payload(&input),
                RotatedPreparation::Fused,
                CudaStream::Prefill,
            )?;
            second.enqueue_async(
                get_payload(&input),
                RotatedPreparation::Staged,
                CudaStream::Decode,
            )?;
        }
        let (original, bytes) = first.read_preparation()?;
        let (second_rotated, second_bytes) = second.read_preparation()?;
        assert_eq!(original.len(), 256);
        assert_eq!(second_rotated.len(), 1024);
        assert_ne!(bytes, second_bytes[..bytes.len()]);
        overwrite_guarded(&ctx, &input, &vec![0.0; 1024])?;
        unsafe {
            second.enqueue_async(
                get_payload(&input),
                RotatedPreparation::Fused,
                CudaStream::Decode,
            )?;
        }
        assert!(second.read_preparation()?.0.iter().all(|&x| x == 0.0));
        let (retained, retained_bytes) = first.read_preparation()?;
        assert_bits(&retained, &original);
        assert_eq!(retained_bytes, bytes);
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Unsupported shapes/signs and aliased, null, misaligned or overflowing input
/// spans fail before GPU work; invalid construction cannot leak allocations.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_rotated_preparation_rejects_invalid_inputs() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    let before = profile::snapshot();
    for rows in [0, 3, 16, usize::MAX] {
        assert!(DeviceRotatedActivation::upload(&ctx, rows, &[1; 128]).is_err());
    }
    for signs in [&[][..], &[1; 32][..], &[0; 128][..], &[2; 128][..]] {
        assert!(DeviceRotatedActivation::upload(&ctx, 1, signs).is_err());
    }
    assert_eq!(
        profile::counter_delta(&before, &profile::snapshot()),
        Default::default()
    );
    {
        let owner = DeviceRotatedActivation::upload(&ctx, 1, &[1; 128])?;
        let before = profile::snapshot();
        for pointer in [
            std::ptr::null(),
            std::ptr::dangling::<u8>().cast::<c_void>(),
            (usize::MAX - 3) as *const c_void,
            owner.get_rotated_pointer(),
            owner.get_quantized_pointer(),
            owner.get_sign_buffer().as_ptr(),
        ] {
            for mode in [RotatedPreparation::Staged, RotatedPreparation::Fused] {
                assert!(unsafe { owner.enqueue_async(pointer, mode, CudaStream::Decode) }.is_err());
            }
        }
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()),
            Default::default()
        );
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

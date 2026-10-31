#![cfg(all(feature = "cuda", nvcc))]
#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;

use anyhow::Result;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer, DeviceTernaryScreen},
    evaluation::ternary_screen::{
        generate_signs, rotate_reference, select_coordinates, TernarySketch,
    },
    profile,
};

const GUARD: f32 = 12345.25;

fn encode_weights(k: usize, h: usize, seed: usize, signs: &[i8]) -> Result<TernarySketch> {
    let mut raw = Vec::with_capacity(k * h / 32 * 18);
    for block in 0..k * h / 32 {
        raw.extend_from_slice(
            &half::f16::from_f32(((block + seed) % 7) as f32 / 256.0).to_le_bytes(),
        );
        for i in 0..16 {
            raw.push(((i * 37 + block * 13 + seed * 19) % 256) as u8);
        }
    }
    TernarySketch::encode_q4(&raw, k, h, signs)
}

fn upload(ctx: &CudaContext, values: &[f32]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new_tagged(ctx, values.len() * 4, "ternary_test:buffer")?;
    unsafe {
        ctx.memcpy_h2d(buffer.as_mut_ptr(), values.as_ptr().cast(), buffer.bytes())?;
    }
    Ok(buffer)
}

fn read(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<f32>> {
    let mut values = vec![0.0; buffer.bytes() / 4];
    unsafe {
        ctx.memcpy_d2h(values.as_mut_ptr().cast(), buffer.as_ptr(), buffer.bytes())?;
    }
    Ok(values)
}

fn check_close(actual: f32, expected: f64, index: usize) {
    assert!(
        actual.is_finite() && (f64::from(actual) - expected).abs() < 1e-3 + 2e-5 * expected.abs(),
        "index {index}: {actual} vs {expected}"
    );
}

fn check_case(ctx: &CudaContext, k: usize, h: usize) -> Result<()> {
    let signs = if k == 128 {
        vec![1; k]
    } else {
        generate_signs(k, 0)?
    };
    let gate = encode_weights(k, h, 1, &signs)?;
    let up = encode_weights(k, h, 9, &signs)?;
    let norms: Vec<_> = (0..h).map(|i| (i % 17) as f32 / 16.0).collect();
    let owner = DeviceTernaryScreen::upload(ctx, &gate, &up, &norms)?;
    let input_device = upload(ctx, &vec![GUARD; k + 16])?;
    let input_pointer = unsafe { input_device.as_mut_ptr().cast::<f32>().add(8).cast() };
    let output = upload(ctx, &vec![GUARD; h * 4 + 16])?;
    let output_pointer = unsafe { output.as_mut_ptr().cast::<f32>().add(8).cast() };
    for case in 0..2 {
        let input: Vec<_> = (0..k)
            .map(|i| {
                if case == 0 {
                    0.25
                } else {
                    ((i * 31) % 59) as f32 / 113.0 - 0.25
                }
            })
            .collect();
        let expected_rotation = rotate_reference(&input, &signs)?;
        let producer = upload(ctx, &input)?;
        let mut previous = None;
        for stream in [CudaStream::Decode, CudaStream::Prefill] {
            let producer_stream = if stream == CudaStream::Decode {
                CudaStream::Prefill
            } else {
                CudaStream::Decode
            };
            unsafe {
                ctx.memcpy_d2d_async(
                    input_pointer,
                    producer.as_ptr(),
                    producer.bytes(),
                    producer_stream,
                )?;
            }
            ctx.stream_wait_for_stream(stream, producer_stream, "ternary_test:producer")?;
            for _ in 0..2 {
                let before = profile::snapshot();
                let resident = CudaContext::total_device_bytes();
                unsafe {
                    owner.enqueue_async(input_pointer, output_pointer, stream)?;
                }
                let delta = profile::counter_delta(&before, &profile::snapshot());
                assert_eq!(delta.launches, 2);
                assert_eq!(
                    delta.device_allocs
                        + delta.device_frees
                        + delta.h2d_copies
                        + delta.d2h_copies
                        + delta.stream_waits
                        + delta.stream_syncs
                        + delta.cublas_calls,
                    0
                );
                assert_eq!(resident, CudaContext::total_device_bytes());
                ctx.synchronize_stream(stream)?;
                let (rotated, coordinates) = owner.read_preparation()?;
                for (&actual, &expected) in rotated.iter().zip(&expected_rotation) {
                    assert!((f64::from(actual) - expected).abs() < 2e-6 + 2e-6 * expected.abs());
                }
                assert_eq!(coordinates, select_coordinates(&rotated, 80)?);
                let expected_gate = gate.project_reference(&rotated, &coordinates, 80)?;
                let expected_up = up.project_reference(&rotated, &coordinates, 80)?;
                let actual = read(ctx, &output)?;
                assert!(actual[..8]
                    .iter()
                    .chain(&actual[h * 4 + 8..])
                    .all(|&v| v == GUARD));
                let values = &actual[8..h * 4 + 8];
                let bits: Vec<_> = values.iter().map(|v| v.to_bits()).collect();
                if let Some(ref expected) = previous {
                    assert_eq!(&bits, expected);
                }
                previous = Some(bits);
                for channel in 0..h {
                    let g = expected_gate[channel];
                    let u = expected_up[channel];
                    let activation = g / (1.0 + (-g).exp()) * u;
                    for (plane, expected) in [
                        g,
                        u,
                        activation,
                        activation.abs() * f64::from(norms[channel]),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        check_close(values[plane * h + channel], expected, plane * h + channel);
                    }
                }
            }
        }
        let actual = read(ctx, &input_device)?;
        assert_eq!(&actual[8..k + 8], &input);
        assert!(actual[..8]
            .iter()
            .chain(&actual[k + 8..])
            .all(|&v| v == GUARD));
    }
    let sign_bytes: Vec<_> = signs.iter().map(|&s| s as u8).collect();
    let norm_bytes: Vec<_> = norms.iter().flat_map(|n| n.to_le_bytes()).collect();
    let [gs, gc] = gate.planes();
    let [us, uc] = up.planes();
    for (buffer, expected) in owner.immutable_buffers().into_iter().zip([
        sign_bytes.as_slice(),
        gs,
        gc,
        us,
        uc,
        norm_bytes.as_slice(),
    ]) {
        let mut bytes = vec![0; buffer.bytes()];
        unsafe {
            ctx.memcpy_d2h(bytes.as_mut_ptr().cast(), buffer.as_ptr(), bytes.len())?;
        }
        assert_eq!(bytes, expected);
    }
    Ok(())
}

/// Every score plane matches an independent reference with tied and changing
/// inputs, partial channel tiles, real FFN dimensions and cross-stream producers.
/// Scratch guards, immutable owners, zero-allocation reuse and cleanup must hold.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_screen_matches_encoded_reference() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let before = CudaContext::total_device_bytes();
    for (k, h) in [(128, 5), (256, 35), (2048, 11008)] {
        check_case(&ctx, k, h)?;
    }
    assert_eq!(before, CudaContext::total_device_bytes());
    Ok(())
}

/// Invalid shape, signs and norms fail before allocation; invalid spans fail
/// before any kernel launch and leave existing outputs unchanged.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_screen_rejects_invalid_inputs() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let resident = CudaContext::total_device_bytes();
    {
        let gate = encode_weights(128, 5, 1, &[1; 128])?;
        let different_signs = encode_weights(128, 5, 1, &[-1; 128])?;
        let different_shape = encode_weights(128, 6, 1, &[1; 128])?;
        let before = profile::snapshot();
        for up in [&different_signs, &different_shape] {
            assert!(DeviceTernaryScreen::upload(&ctx, &gate, up, &[1.0; 5]).is_err());
        }
        for norms in [
            vec![],
            vec![1.0; 4],
            vec![-1.0; 5],
            vec![f32::NAN; 5],
            vec![f32::INFINITY; 5],
        ] {
            assert!(DeviceTernaryScreen::upload(&ctx, &gate, &gate, &norms).is_err());
        }
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()).device_allocs,
            0
        );
        let owner = DeviceTernaryScreen::upload(&ctx, &gate, &gate, &[1.0; 5])?;
        let input = upload(&ctx, &[1.0; 128])?;
        let output = upload(&ctx, &[GUARD; 20])?;
        let before = profile::snapshot();
        for pointer in [std::ptr::null(), (usize::MAX - 3) as *const _, unsafe {
            input.as_ptr().cast::<u8>().add(1).cast()
        }] {
            assert!(unsafe {
                owner.enqueue_async(pointer, output.as_mut_ptr(), CudaStream::Decode)
            }
            .is_err());
        }
        for pointer in [
            input.as_mut_ptr(),
            owner.immutable_buffers()[1].as_mut_ptr(),
            std::ptr::null_mut(),
        ] {
            assert!(
                unsafe { owner.enqueue_async(input.as_ptr(), pointer, CudaStream::Decode) }
                    .is_err()
            );
        }
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()).launches,
            0
        );
        assert_eq!(read(&ctx, &output)?, vec![GUARD; 20]);
    }
    assert_eq!(resident, CudaContext::total_device_bytes());
    Ok(())
}

#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/q8_projection.rs"]
#[allow(dead_code)]
mod fixture;

use anyhow::Result;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer, Q8FfnQ81Launch},
    infer::q8_activation::{quantize_reference, Q8ActivationShape},
    profile,
};

const GUARD_BITS: u32 = 0x4b12_3456;

fn upload(ctx: &CudaContext, bytes: &[u8], tag: &str) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new_tagged(ctx, bytes.len(), tag)?;
    unsafe {
        ctx.memcpy_h2d(buffer.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())?;
    }
    Ok(buffer)
}

fn guarded_f32(ctx: &CudaContext, values: &[f32], tag: &str) -> Result<DeviceBuffer> {
    let mut all = vec![f32::from_bits(GUARD_BITS); values.len() + 16];
    all[8..8 + values.len()].copy_from_slice(values);
    upload(
        ctx,
        &all.iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
        tag,
    )
}

fn read_guarded(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<f32>> {
    let mut values = vec![0.0f32; buffer.bytes() / 4];
    unsafe {
        ctx.memcpy_d2h(values.as_mut_ptr().cast(), buffer.as_ptr(), buffer.bytes())?;
    }
    assert!(values[..8]
        .iter()
        .chain(&values[values.len() - 8..])
        .all(|v| v.to_bits() == GUARD_BITS));
    Ok(values[8..values.len() - 8].to_vec())
}

fn data(buffer: &DeviceBuffer) -> *mut std::ffi::c_void {
    unsafe { buffer.as_mut_ptr().cast::<f32>().add(8).cast() }
}

fn assert_near(actual: f32, expected: f64) {
    assert!(actual.is_finite() && expected.is_finite());
    assert!(
        (f64::from(actual) - expected).abs() < 1e-3 + 2e-5 * expected.abs(),
        "{actual} vs {expected}"
    );
}

fn compare_case(
    ctx: &CudaContext,
    rows: usize,
    k: usize,
    hidden: usize,
    offset: usize,
) -> Result<()> {
    let shape = Q8ActivationShape::new(rows, k, k + 32)?;
    let input = fixture::make_input(rows, k);
    let blocks = quantize_reference(&input, shape)?;
    let reconstructed: Vec<f32> = blocks
        .chunks_exact(shape.padded_columns() / 32)
        .flat_map(|row| row.iter().flat_map(|b| b.reconstruct()).take(k))
        .collect();
    let x = guarded_f32(ctx, &input, "q8-1-ffn:input")?;
    let quantized_all = vec![0xa5u8; shape.bytes() + 32];
    let quantized = upload(ctx, &quantized_all, "q8-1-ffn:activation")?;
    let activation = unsafe { quantized.as_mut_ptr().cast::<u8>().add(16).cast() };
    unsafe {
        ctx.quantize_q8_1_async(data(&x), activation, shape)?;
    }
    ctx.synchronize_stream(CudaStream::Decode)?;
    let mut gate = fixture::make_column_blocks(k, hidden);
    let mut up = fixture::make_column_blocks(k, hidden);
    for (index, block) in gate.chunks_exact_mut(34).enumerate() {
        if index % 7 == 0 {
            block[..2].fill(0);
        }
    }
    for block in up.chunks_exact_mut(34) {
        block[2..].reverse();
    }
    let wrap = |bytes: &[u8]| {
        let mut all = vec![0x5au8; bytes.len() + offset + 16];
        all[offset..offset + bytes.len()].copy_from_slice(bytes);
        all
    };
    let gate_all = wrap(&gate);
    let up_all = wrap(&up);
    let g = upload(ctx, &gate_all, "q8-1-ffn:gate")?;
    let u = upload(ctx, &up_all, "q8-1-ffn:up")?;
    let gate_pointer = unsafe { g.as_ptr().cast::<u8>().add(offset).cast() };
    let up_pointer = unsafe { u.as_ptr().cast::<u8>().add(offset).cast() };
    let output = guarded_f32(ctx, &vec![0.0; rows * hidden], "q8-1-ffn:output")?;
    let gate_output = guarded_f32(ctx, &vec![0.0; rows * hidden], "q8-1-ffn:gate-output")?;
    let up_output = guarded_f32(ctx, &vec![0.0; rows * hidden], "q8-1-ffn:up-output")?;
    let scratch = CudaContext::total_device_bytes();
    for launch in Q8FfnQ81Launch::ALL {
        let before = profile::snapshot();
        unsafe {
            ctx.ffn_q8_1_async(
                activation,
                gate_pointer,
                up_pointer,
                data(&output),
                Some((data(&gate_output), data(&up_output))),
                hidden as i32,
                shape,
                launch,
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
        let actual = read_guarded(ctx, &output)?;
        let actual_gate = read_guarded(ctx, &gate_output)?;
        let actual_up = read_guarded(ctx, &up_output)?;
        for row in 0..rows {
            for col in 0..hidden {
                let g = fixture::project_cpu(&reconstructed, &gate, row, col, k);
                let u = fixture::project_cpu(&reconstructed, &up, row, col, k);
                assert_near(actual_gate[row * hidden + col], g);
                assert_near(actual_up[row * hidden + col], u);
                assert_near(actual[row * hidden + col], g / (1.0 + (-g).exp()) * u);
            }
        }
        // Diagnostic capture must not change the timed kernel's numerical result.
        for _ in 0..8 {
            unsafe {
                ctx.ffn_q8_1_async(
                    activation,
                    gate_pointer,
                    up_pointer,
                    data(&output),
                    None,
                    hidden as i32,
                    shape,
                    launch,
                )?;
            }
        }
        ctx.synchronize_stream(CudaStream::Decode)?;
        assert!(read_guarded(ctx, &output)?
            .iter()
            .zip(&actual)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        assert_eq!(CudaContext::total_device_bytes(), scratch);
    }
    assert_eq!(read_guarded(ctx, &x)?, input);
    for (buffer, expected) in [(&g, gate_all), (&u, up_all)] {
        let mut actual = vec![0u8; buffer.bytes()];
        unsafe {
            ctx.memcpy_d2h(actual.as_mut_ptr().cast(), buffer.as_ptr(), actual.len())?;
        }
        assert_eq!(actual, expected);
    }
    let mut encoded = vec![0u8; quantized.bytes()];
    unsafe {
        ctx.memcpy_d2h(
            encoded.as_mut_ptr().cast(),
            quantized.as_ptr(),
            encoded.len(),
        )?;
    }
    assert_eq!(&encoded[..16], &[0xa5; 16]);
    assert_eq!(&encoded[16 + shape.bytes()..], &[0xa5; 16]);
    assert_eq!(
        encoded[16..16 + shape.bytes()],
        blocks.iter().flat_map(|b| b.to_bytes()).collect::<Vec<_>>()
    );
    Ok(())
}

/// Both fixed geometries match scalar CPU gate/up/SwiGLU on the quantized
/// reference inputs, preserve guards and share one encoded input for gate/up.
/// Distinct padded rows, signed weights and two-byte Q8_0 bases protect layout.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_q8_1_ffn_matches_quantized_cpu_and_reuses_storage() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for (k, hidden, offset) in [(32, 1, 0), (96, 37, 2), (2048, 257, 0), (11008, 33, 2)] {
        for rows in [1, 4] {
            compare_case(&ctx, rows, k, hidden, offset)?;
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Unsupported dimensions/alignment/diagnostic spans fail before enqueue;
/// no alternative backend, allocation or synchronization hides the error.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_q8_1_ffn_rejects_invalid_arguments_without_gpu_work() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let input = DeviceBuffer::new_tagged(&ctx, 256, "q8-1-ffn:invalid-input")?;
    let weight = DeviceBuffer::new_tagged(&ctx, 256, "q8-1-ffn:invalid-weight")?;
    let output = DeviceBuffer::new_tagged(&ctx, 256, "q8-1-ffn:invalid-output")?;
    let shape = Q8ActivationShape::new(1, 32, 32)?;
    let before = profile::snapshot();
    for launch in Q8FfnQ81Launch::ALL {
        unsafe {
            for (pointer, hidden, shape) in [
                (input.as_ptr(), 0, shape),
                (std::ptr::null(), 1, shape),
                (input.as_ptr().cast::<u8>().add(2).cast(), 1, shape),
                (input.as_ptr(), 1, Q8ActivationShape::new(1, 33, 64)?),
            ] {
                assert!(ctx
                    .ffn_q8_1_async(
                        pointer,
                        weight.as_ptr(),
                        weight.as_ptr(),
                        output.as_mut_ptr(),
                        None,
                        hidden,
                        shape,
                        launch
                    )
                    .is_err());
            }
            assert!(ctx
                .ffn_q8_1_async(
                    input.as_ptr(),
                    weight.as_ptr().cast::<u8>().add(1).cast(),
                    weight.as_ptr(),
                    output.as_mut_ptr(),
                    None,
                    1,
                    shape,
                    launch
                )
                .is_err());
            assert!(ctx
                .ffn_q8_1_async(
                    input.as_ptr(),
                    weight.as_ptr(),
                    weight.as_ptr(),
                    output.as_mut_ptr(),
                    Some((output.as_mut_ptr(), std::ptr::null_mut())),
                    1,
                    shape,
                    launch
                )
                .is_err());
            assert!(ctx
                .ffn_q8_1_async(
                    input.as_ptr(),
                    weight.as_ptr(),
                    weight.as_ptr(),
                    input.as_mut_ptr(),
                    None,
                    1,
                    shape,
                    launch
                )
                .is_err());
        }
    }
    assert_eq!(
        profile::counter_delta(&before, &profile::snapshot()),
        Default::default()
    );
    Ok(())
}

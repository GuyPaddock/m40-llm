#![cfg(all(feature = "cuda", nvcc))]
#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;

use anyhow::Result;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    evaluation::quantized_fixture::decode_q4_values,
    profile,
};

const GUARD: f32 = 12345.25;

fn make_weights(h: usize, k: usize, seed: usize) -> Vec<u8> {
    let mut raw = Vec::new();
    for block in 0..h * k / 32 {
        raw.extend_from_slice(
            &half::f16::from_f32(((block + seed) % 7) as f32 / 256.0 - 0.015625).to_le_bytes(),
        );
        for i in 0..16 {
            raw.push(((i * 37 + block * 13 + seed * 19) % 256) as u8);
        }
    }
    raw
}

fn upload_bytes(ctx: &CudaContext, bytes: &[u8]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new_tagged(ctx, bytes.len(), "q4_fusion_test:buffer")?;
    unsafe {
        ctx.memcpy_h2d(buffer.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())?;
    }
    Ok(buffer)
}

fn upload_values(ctx: &CudaContext, values: &[f32]) -> Result<DeviceBuffer> {
    upload_bytes(
        ctx,
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn read_bytes(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<u8>> {
    let mut bytes = vec![0; buffer.bytes()];
    unsafe {
        ctx.memcpy_d2h(bytes.as_mut_ptr().cast(), buffer.as_ptr(), bytes.len())?;
    }
    Ok(bytes)
}

fn read_values(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<f32>> {
    Ok(read_bytes(ctx, buffer)?
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn check_case(ctx: &CudaContext, rows: usize, k: usize, h: usize) -> Result<()> {
    let gate_raw = make_weights(h, k, 1);
    let up_raw = make_weights(h, k, 7);
    let gate = upload_bytes(ctx, &gate_raw)?;
    let up = upload_bytes(ctx, &up_raw)?;
    let gate_values = decode_q4_values(&gate_raw)?;
    let up_values = decode_q4_values(&up_raw)?;
    let input: Vec<_> = (0..rows * k)
        .map(|i| ((i * 31 + i / k * 7) % 59) as f32 / 113.0 - 0.25)
        .collect();
    let expected: Vec<_> = input
        .chunks_exact(k)
        .flat_map(|row| {
            gate_values
                .chunks_exact(k)
                .zip(up_values.chunks_exact(k))
                .map(move |(g, u)| {
                    let gate: f64 = row.iter().zip(g).map(|(&x, &w)| f64::from(x) * w).sum();
                    let up: f64 = row.iter().zip(u).map(|(&x, &w)| f64::from(x) * w).sum();
                    gate / (1.0 + (-gate).exp()) * up
                })
        })
        .collect();
    let producer = upload_values(ctx, &input)?;
    let input_device = upload_values(ctx, &vec![GUARD; rows * k + 16])?;
    let input_pointer = unsafe { input_device.as_mut_ptr().cast::<f32>().add(8).cast() };
    let output = upload_values(ctx, &vec![GUARD; rows * h + 16])?;
    let output_pointer = unsafe { output.as_mut_ptr().cast::<f32>().add(8).cast() };
    let control = upload_values(ctx, &vec![GUARD; rows * h])?;
    // Populate the guarded input before collecting the legacy single-row control.
    unsafe {
        ctx.memcpy_h2d(input_pointer, input.as_ptr().cast(), input.len() * 4)?;
    }
    for row in 0..rows {
        unsafe {
            ctx.mlp_gate_up_swiglu_f32xq4_0_gguf_decode_async(
                input_pointer.cast::<f32>().add(row * k).cast(),
                gate.as_ptr(),
                up.as_ptr(),
                control.as_mut_ptr().cast::<f32>().add(row * h).cast(),
                h as i32,
                k as i32,
            )?;
        }
    }
    ctx.synchronize_stream(CudaStream::Decode)?;
    ctx.synchronize_stream(CudaStream::Prefill)?;
    let control_bits: Vec<_> = read_values(ctx, &control)?
        .iter()
        .map(|v| v.to_bits())
        .collect();
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
        ctx.stream_wait_for_stream(stream, producer_stream, "q4_fusion_test:producer")?;
        for _ in 0..2 {
            let before = profile::snapshot();
            let resident = CudaContext::total_device_bytes();
            unsafe {
                ctx.q4_fused_rows_async(
                    input_pointer,
                    gate.as_ptr(),
                    up.as_ptr(),
                    output_pointer,
                    rows,
                    h,
                    k,
                    stream,
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
                    + delta.stream_waits
                    + delta.cublas_calls,
                0
            );
            assert_eq!(resident, CudaContext::total_device_bytes());
            ctx.synchronize_stream(stream)?;
            let actual = read_values(ctx, &output)?;
            assert!(actual[..8]
                .iter()
                .chain(&actual[rows * h + 8..])
                .all(|&v| v == GUARD));
            let values = &actual[8..rows * h + 8];
            assert_eq!(
                values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                control_bits
            );
            for (i, (&a, &b)) in values.iter().zip(&expected).enumerate() {
                assert!(
                    a.is_finite() && (f64::from(a) - b).abs() < 1e-3 + 2e-5 * b.abs(),
                    "row output {i}: {a} vs {b}"
                );
            }
        }
    }
    let actual_input = read_values(ctx, &input_device)?;
    assert_eq!(&actual_input[8..rows * k + 8], &input);
    assert!(actual_input[..8]
        .iter()
        .chain(&actual_input[rows * k + 8..])
        .all(|&v| v == GUARD));
    assert_eq!(read_bytes(ctx, &gate)?, gate_raw);
    assert_eq!(read_bytes(ctx, &up)?, up_raw);
    Ok(())
}

/// Distinct multi-row outputs retain legacy single-row bits and match FP64
/// projections/SwiGLU. Guards, both stream directions and allocation reuse hold.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_fused_rows_match_single_row_and_fp64() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let before = CudaContext::total_device_bytes();
    for rows in [1, 2, 4, 8] {
        check_case(&ctx, rows, 64, 35)?;
    }
    check_case(&ctx, 2, 2048, 11008)?;
    assert_eq!(before, CudaContext::total_device_bytes());
    Ok(())
}

/// Invalid row counts, dimensions, pointers and output overlaps fail before
/// launch, preserving live output bytes and releasing all test allocations.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_fused_rows_reject_invalid_spans() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let resident = CudaContext::total_device_bytes();
    {
        let input = upload_values(&ctx, &[0.25; 64])?;
        let weight = upload_bytes(&ctx, &make_weights(35, 64, 1))?;
        let output = upload_values(&ctx, &[GUARD; 35])?;
        let before = profile::snapshot();
        for (rows, h, k) in [
            (0, 35, 64),
            (3, 35, 64),
            (9, 35, 64),
            (1, 0, 64),
            (1, 35, 0),
            (1, 35, 63),
            (1, usize::MAX, 64),
        ] {
            assert!(unsafe {
                ctx.q4_fused_rows_async(
                    input.as_ptr(),
                    weight.as_ptr(),
                    weight.as_ptr(),
                    output.as_mut_ptr(),
                    rows,
                    h,
                    k,
                    CudaStream::Decode,
                )
            }
            .is_err());
        }
        for pointer in [std::ptr::null(), (usize::MAX - 3) as *const _, unsafe {
            input.as_ptr().cast::<u8>().add(1).cast()
        }] {
            assert!(unsafe {
                ctx.q4_fused_rows_async(
                    pointer,
                    weight.as_ptr(),
                    weight.as_ptr(),
                    output.as_mut_ptr(),
                    1,
                    35,
                    64,
                    CudaStream::Decode,
                )
            }
            .is_err());
        }
        for pointer in [
            std::ptr::null_mut(),
            input.as_mut_ptr(),
            weight.as_mut_ptr(),
            unsafe { output.as_mut_ptr().cast::<u8>().add(1).cast() },
        ] {
            assert!(unsafe {
                ctx.q4_fused_rows_async(
                    input.as_ptr(),
                    weight.as_ptr(),
                    weight.as_ptr(),
                    pointer,
                    1,
                    35,
                    64,
                    CudaStream::Decode,
                )
            }
            .is_err());
        }
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()).launches,
            0
        );
        assert_eq!(read_values(&ctx, &output)?, vec![GUARD; 35]);
    }
    assert_eq!(resident, CudaContext::total_device_bytes());
    Ok(())
}

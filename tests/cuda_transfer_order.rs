#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/env_restore.rs"]
mod env_restore;

use anyhow::{ensure, Result};
use half::f16;
use m40_llm::cuda::{CudaContext, CudaStream, DeviceBuffer};

fn read(ctx: &CudaContext, buffer: &DeviceBuffer, elements: usize) -> Result<Vec<f32>> {
    let mut values = vec![0.0; elements];
    unsafe {
        ctx.memcpy_d2h(values.as_mut_ptr().cast(), buffer.as_ptr(), elements * 4)?;
    }
    Ok(values)
}

/// Pageable H2D must finish before returning: immediate consumers on decode
/// and prefill must see the new values, without a readback, allocation or
/// unrelated synchronization hiding setup races. Reuse changes every row.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_pageable_upload_is_ready_for_both_consumer_streams() -> Result<()> {
    let _scope = cuda_test_scope::lock_cuda();
    let _stream = env_restore::EnvRestore::set("M40LLM_Q8_DECODE_STREAM", "decode");
    let ctx = CudaContext::new_m40()?;
    for (rows, k) in [(4, 2048), (16, 11008), (64, 11008)] {
        let n = 17;
        let elements = rows * k;
        let input = DeviceBuffer::new_tagged(&ctx, elements * 4, "transfer:input")?;
        let zeros = DeviceBuffer::new_tagged(&ctx, elements * 4, "transfer:zeros")?;
        let decoded = DeviceBuffer::new_tagged(&ctx, elements * 4, "transfer:decoded")?;
        let projected = DeviceBuffer::new_tagged(&ctx, rows * n * 4, "transfer:projected")?;
        let mut weights = vec![0; n * (k / 32) * 34];
        for (index, block) in weights.chunks_exact_mut(34).enumerate() {
            let column = index / (k / 32);
            block[..2].copy_from_slice(&f16::from_f32(1.0 / 128.0).to_le_bytes());
            // Power-of-two scale and +/-1 payload keep all sums exact below
            // 2^24, so rounding drift cannot be confused with a copy race.
            block[2..].fill((column as i8 % 3 - 1) as u8);
        }
        let d_weights = DeviceBuffer::new_tagged(&ctx, weights.len(), "transfer:weights")?;
        let mut host = vec![0.0_f32; elements];
        unsafe {
            ctx.memcpy_h2d(
                d_weights.as_mut_ptr(),
                weights.as_ptr().cast(),
                weights.len(),
            )?;
            ctx.memcpy_h2d(zeros.as_mut_ptr(), host.as_ptr().cast(), elements * 4)?;
            ctx.memcpy_h2d(input.as_mut_ptr(), host.as_ptr().cast(), elements * 4)?;
        }
        // Fence initial fixtures only, before the repeated-copy regression.
        assert!(read(&ctx, &input, elements)?.iter().all(|&v| v == 0.0));
        for iteration in 1..=16 {
            for row in 0..rows {
                host[row * k..(row + 1) * k].fill((iteration * rows + row + 1) as f32 / 1024.0);
            }
            unsafe {
                ctx.memcpy_h2d(input.as_mut_ptr(), host.as_ptr().cast(), elements * 4)?;
                ctx.residual_add_f32_async(
                    input.as_ptr(),
                    zeros.as_ptr(),
                    decoded.as_mut_ptr(),
                    elements,
                )?;
                ctx.gemm_f32xq8_0_gguf_f32_async(
                    input.as_ptr(),
                    d_weights.as_ptr(),
                    projected.as_mut_ptr(),
                    rows as i32,
                    n as i32,
                    k as i32,
                )?;
            }
            ctx.synchronize_stream(CudaStream::Decode)?;
            ctx.synchronize_stream(CudaStream::Prefill)?;
            let actual = read(&ctx, &decoded, elements)?;
            for (index, (&actual, &expected)) in actual.iter().zip(&host).enumerate() {
                ensure!(actual == expected,
                    "decode consumed stale H2D: rows={rows} K={k} iteration={iteration} index={index} actual={actual} expected={expected}");
            }
            let actual = read(&ctx, &projected, rows * n)?;
            for row in 0..rows {
                for column in 0..n {
                    let expected =
                        k as f32 * host[row * k] * (column as i32 % 3 - 1) as f32 / 128.0;
                    let actual = actual[row * n + column];
                    ensure!(actual == expected,
                    "prefill consumed stale H2D: rows={rows} K={k} iteration={iteration} row={row} col={column} actual={actual} expected={expected}");
                }
            }
        }
    }
    Ok(())
}

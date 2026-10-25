#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/env_restore.rs"]
mod env_restore;

use anyhow::{ensure, Result};
use half::f16;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer, ExactOldBacking, KVCache},
    kv_compression::KvRepresentativePolicy,
    profile,
};

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

/// Native registered/fallback weight upload must return consumable storage;
/// no allocation/readback may intervene before either stream's first read.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_native_weight_upload_is_ready_for_both_streams() -> Result<()> {
    let _scope = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    for bytes in [65536, 704512, 2818048] {
        let decode = DeviceBuffer::new_tagged(&ctx, bytes, "weights-ready:decode")?;
        let prefill = DeviceBuffer::new_tagged(&ctx, bytes, "weights-ready:prefill")?;
        let source: Vec<u8> = (0..bytes)
            .map(|i| (i.wrapping_mul(13) ^ (i >> 7)) as u8)
            .collect();
        let pointer = ctx.upload_weights(&source)?;
        unsafe {
            ctx.memcpy_d2d_async(decode.as_mut_ptr(), pointer, bytes, CudaStream::Decode)?;
            ctx.memcpy_d2d_async(prefill.as_mut_ptr(), pointer, bytes, CudaStream::Prefill)?;
        }
        ctx.synchronize_stream(CudaStream::Decode)?;
        ctx.synchronize_stream(CudaStream::Prefill)?;
        for buffer in [decode, prefill] {
            let mut actual = vec![0_u8; bytes];
            unsafe {
                ctx.memcpy_d2h(actual.as_mut_ptr().cast(), buffer.as_ptr(), bytes)?;
            }
            assert_eq!(actual, source);
        }
    }
    Ok(())
}

/// Fresh dense and compressed caches must be initialized before immediate
/// append/attention, without reset, allocation or a default-stream readback.
/// Summary/count/ring/representative metadata must retain its initial state.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_native_kv_initialization_precedes_first_decode() -> Result<()> {
    let _scope = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    for dim in [64, 128] {
        let heads = 2;
        let q_heads = 4;
        let k = vec![0.25_f32; heads * dim];
        let v = vec![1.5_f32; heads * dim];
        let q = vec![0.0_f32; q_heads * dim];
        let d_k = DeviceBuffer::new_tagged(&ctx, k.len() * 4, "kv-ready:k")?;
        let d_v = DeviceBuffer::new_tagged(&ctx, v.len() * 4, "kv-ready:v")?;
        let d_q = DeviceBuffer::new_tagged(&ctx, q.len() * 4, "kv-ready:q")?;
        let output = DeviceBuffer::new_tagged(&ctx, q.len() * 4, "kv-ready:output")?;
        unsafe {
            ctx.memcpy_h2d(d_k.as_mut_ptr(), k.as_ptr().cast(), d_k.bytes())?;
            ctx.memcpy_h2d(d_v.as_mut_ptr(), v.as_ptr().cast(), d_v.bytes())?;
            ctx.memcpy_h2d(d_q.as_mut_ptr(), q.as_ptr().cast(), d_q.bytes())?;
        }
        for backing in [
            None,
            Some(ExactOldBacking::Dense),
            Some(ExactOldBacking::Q8),
            Some(ExactOldBacking::Fp16KQ4V),
        ] {
            // Other diagnostic compressed attention is head64-only. Exercise
            // head128 with its supported preferred direct mixed backend.
            if dim == 128 && matches!(backing, Some(ExactOldBacking::Dense | ExactOldBacking::Q8)) {
                continue;
            }
            let cache = if let Some(backing) = backing {
                KVCache::new_compressed_with_context(
                    &ctx,
                    8192,
                    2,
                    heads as u32,
                    dim as u32,
                    4096,
                    32,
                    8,
                    2,
                    KvRepresentativePolicy::Last,
                    backing,
                )?
            } else {
                KVCache::new_with_context(&ctx, 8192, 2, heads as u32, dim as u32)?
            };
            unsafe {
                cache.append_token_f32_rope_k_at_async(
                    &ctx,
                    0,
                    d_k.as_ptr(),
                    d_v.as_ptr(),
                    0,
                    0,
                    10000.0,
                    1.0,
                )?;
                if backing == Some(ExactOldBacking::Fp16KQ4V) {
                    cache.attention_last_token_f32_gqa_block_select_exact_fp16_k_q4_v_old_direct_async(
                        &ctx,0,d_q.as_ptr(),q_heads as u32,1,4096,32,8,output.as_mut_ptr())?;
                } else if backing.is_some() {
                    cache.attention_last_token_f32_gqa_compressed_recent_only_async(
                        &ctx,
                        0,
                        d_q.as_ptr(),
                        q_heads as u32,
                        1,
                        output.as_mut_ptr(),
                    )?;
                } else {
                    cache.attention_last_token_f32_gqa_async(
                        &ctx,
                        0,
                        d_q.as_ptr(),
                        q_heads as u32,
                        1,
                        output.as_mut_ptr(),
                    )?;
                }
            }
            ctx.synchronize_stream(CudaStream::Decode)?;
            assert!(read(&ctx, &output, q.len())?
                .iter()
                .all(|&value| value == 1.5));
            if backing.is_some() {
                let state = cache.debug_compressed_snapshot(&ctx, 0)?;
                assert_eq!(state.seq_len, 1);
                assert!(state.block_counts.iter().all(|&value| value == 0));
                for values in [&state.summary_k_acc, &state.summary_v_acc] {
                    assert!(values.iter().all(|&value| value == 0.0));
                }
                for (values, first) in [(&state.recent_k_f16, 0.25), (&state.recent_v_f16, 1.5)] {
                    assert!(values[..heads * dim]
                        .iter()
                        .all(|&value| value == f16::from_f32(first).to_bits()));
                    assert!(values[heads * dim..].iter().all(|&value| value == 0));
                }
                for values in [
                    &state.summary_k_f16,
                    &state.summary_v_f16,
                    &state.rep_k_f16,
                    &state.rep_v_f16,
                ] {
                    assert!(values.iter().all(|&value| value == 0));
                }
                assert!(state.rep_positions.iter().all(|&value| value == u32::MAX));
                let other = cache.debug_compressed_snapshot(&ctx, 1)?;
                assert_eq!(other.seq_len, 0);
                assert!(other
                    .recent_k_f16
                    .iter()
                    .chain(&other.recent_v_f16)
                    .all(|&value| value == 0));
                assert!(other.block_counts.iter().all(|&value| value == 0));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum LegacyAppendKind {
    Fp16,
    Fp32,
    RotaryFp32,
}

/// A queued decode producer must publish length before legacy host reads it.
/// Back-to-back appends behind a backlog must use distinct slots for odd/even
/// token widths and preserve both payloads; overflow must leave them intact.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_legacy_append_observes_queued_length_publication() -> Result<()> {
    let _scope = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let busy_host = vec![0.0_f32; 1 << 22];
    let busy = DeviceBuffer::new_tagged(&ctx, busy_host.len() * 4, "legacy:backlog")?;
    unsafe {
        ctx.memcpy_h2d(busy.as_mut_ptr(), busy_host.as_ptr().cast(), busy.bytes())?;
    }
    for dim in [3, 64, 128] {
        for kind in [
            LegacyAppendKind::Fp16,
            LegacyAppendKind::Fp32,
            LegacyAppendKind::RotaryFp32,
        ] {
            if dim == 3 && matches!(kind, LegacyAppendKind::RotaryFp32) {
                continue;
            }
            let values = [0.25_f32, 1.5, -0.75, 2.5];
            let mut inputs = Vec::new();
            for value in values {
                let data: Vec<u8> = if matches!(kind, LegacyAppendKind::Fp16) {
                    (0..dim)
                        .flat_map(|_| f16::from_f32(value).to_le_bytes())
                        .collect()
                } else {
                    (0..dim).flat_map(|_| value.to_le_bytes()).collect()
                };
                let buffer = DeviceBuffer::new_tagged(&ctx, data.len(), "legacy:input")?;
                unsafe {
                    ctx.memcpy_h2d(buffer.as_mut_ptr(), data.as_ptr().cast(), data.len())?;
                }
                inputs.push(buffer);
            }
            let cache = KVCache::new_with_context(&ctx, 2, 1, 1, dim as u32)?;
            for _ in 0..16 {
                unsafe {
                    ctx.residual_add_f32_async(
                        busy.as_ptr(),
                        busy.as_ptr(),
                        busy.as_mut_ptr(),
                        busy_host.len(),
                    )?;
                }
            }
            for token in 0..2 {
                unsafe {
                    enqueue_legacy(
                        kind,
                        &cache,
                        &ctx,
                        inputs[token * 2].as_ptr(),
                        inputs[token * 2 + 1].as_ptr(),
                    )?;
                }
            }
            ctx.synchronize_stream(CudaStream::Decode)?;
            for token in 0..2 {
                let mut k = vec![0_u16; dim];
                let mut v = vec![0_u16; dim];
                let rc = unsafe {
                    m40_llm::cuda::ffi_debug_read_kv_token(
                        &ctx,
                        &cache,
                        0,
                        token as u32,
                        k.as_mut_ptr().cast(),
                        v.as_mut_ptr().cast(),
                    )
                };
                assert_eq!(rc, 0);
                assert!(
                    k.iter()
                        .all(|&actual| actual == f16::from_f32(values[token * 2]).to_bits()),
                    "K overwritten: {kind:?} dim={dim} token={token}"
                );
                assert!(
                    v.iter()
                        .all(|&actual| actual == f16::from_f32(values[token * 2 + 1]).to_bits()),
                    "V overwritten: {kind:?} dim={dim} token={token}"
                );
            }
            unsafe {
                assert!(
                    enqueue_legacy(kind, &cache, &ctx, inputs[0].as_ptr(), inputs[1].as_ptr())
                        .is_err()
                );
            }
        }
    }
    Ok(())
}

/// Dispatch only the legacy API under test; buffers are device-ready and kept
/// alive by the caller through completion. Rotary position intentionally stays
/// zero, independently of the append slot.
unsafe fn enqueue_legacy(
    kind: LegacyAppendKind,
    cache: &KVCache,
    ctx: &CudaContext,
    k: *const std::ffi::c_void,
    v: *const std::ffi::c_void,
) -> Result<()> {
    match kind {
        LegacyAppendKind::Fp16 => cache.append_token(ctx, 0, k, v),
        LegacyAppendKind::Fp32 => cache.append_token_f32_async(ctx, 0, k, v),
        LegacyAppendKind::RotaryFp32 => {
            cache.append_token_f32_rope_k_async(ctx, 0, k, v, 0, 10000.0, 1.0)
        }
    }
}

/// Production explicit-position append must stay enqueue-only on head64 and
/// head128: length publication adds no host copies, allocations or stream waits.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_explicit_position_append_stays_enqueue_only() -> Result<()> {
    let _scope = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    for dim in [64, 128] {
        let k = vec![0.0_f32; 2 * dim];
        let v = vec![1.5_f32; 2 * dim];
        let d_k = DeviceBuffer::new_tagged(&ctx, k.len() * 4, "at:k")?;
        let d_v = DeviceBuffer::new_tagged(&ctx, v.len() * 4, "at:v")?;
        unsafe {
            ctx.memcpy_h2d(d_k.as_mut_ptr(), k.as_ptr().cast(), d_k.bytes())?;
            ctx.memcpy_h2d(d_v.as_mut_ptr(), v.as_ptr().cast(), d_v.bytes())?;
        }
        let cache = KVCache::new_with_context(&ctx, 8, 1, 2, dim as u32)?;
        let before = profile::snapshot();
        for position in 0..2 {
            unsafe {
                cache.append_token_f32_rope_k_at_async(
                    &ctx,
                    0,
                    d_k.as_ptr(),
                    d_v.as_ptr(),
                    position,
                    position,
                    10000.0,
                    1.0,
                )?;
            }
        }
        let delta = profile::counter_delta(&before, &profile::snapshot());
        assert_eq!(delta.launches, 2);
        assert_eq!(delta.stream_syncs, 0);
        assert_eq!(delta.stream_waits, 0);
        assert_eq!(delta.h2d_copies + delta.d2h_copies, 0);
        assert_eq!(delta.device_allocs + delta.device_frees, 0);
        ctx.synchronize_stream(CudaStream::Decode)?;
    }
    Ok(())
}

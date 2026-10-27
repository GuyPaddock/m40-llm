#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{
        ffi_debug_read_kv_token, CudaContext, CudaStream, DeviceBuffer, ExactOldBacking, KVCache,
        ROPE_LAYOUT_ADJACENT, ROPE_LAYOUT_NEOX,
    },
    kv_compression::KvRepresentativePolicy,
};
use serde_json::{json, Value};
use std::{fs::OpenOptions, io::Write};

const WINDOW: u32 = 17;
const BLOCK: u32 = 4;
const SLOTS: u32 = 4;
const KV_HEADS: u32 = 2;
const HEAD_DIM: u32 = 128;
const CAPACITY: u32 = 2 * WINDOW + 1;

/// Independent physical slots represent two layers and two request sequences.
fn compressed_cache(ctx: &CudaContext) -> Result<KVCache> {
    KVCache::new_compressed_with_context(
        ctx,
        CAPACITY,
        SLOTS,
        KV_HEADS,
        HEAD_DIM,
        WINDOW,
        BLOCK,
        8,
        0,
        KvRepresentativePolicy::Last,
        ExactOldBacking::Fp16KQ4V,
    )
}

/// Read actual stored dense FP16 words, not an independently rounded estimate.
fn read_dense_token(
    ctx: &CudaContext,
    cache: &KVCache,
    slot: u32,
    position: u32,
) -> Result<(Vec<u16>, Vec<u16>)> {
    let mut k = vec![0u16; (KV_HEADS * HEAD_DIM) as usize];
    let mut v = k.clone();
    let rc = unsafe {
        ffi_debug_read_kv_token(
            ctx,
            cache,
            slot,
            position,
            k.as_mut_ptr().cast(),
            v.as_mut_ptr().cast(),
        )
    };
    ensure!(rc == 0, "dense debug read failed: {rc}");
    Ok((k, v))
}

/// Optional create-new evidence prevents overwriting a previous hardware run.
fn record_evidence(value: Value) -> Result<()> {
    if let Ok(path) = std::env::var("M40LLM_NO_OLD_REPORT") {
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        serde_json::to_writer(&mut file, &value)?;
        writeln!(file)?;
        file.flush()?;
    }
    eprintln!("{value}");
    Ok(())
}

/// Verify the recent ring's real stride, absolute positions and slot isolation
/// for sequential append and dense-prefix reconstruction before kernel reuse.
#[test]
#[ignore = "explicit M40 layout gate; unavailable hardware is an error"]
fn test_recent_ring_physical_layout() -> Result<()> {
    let ctx = cuda_env::ctx_m40()?;
    cuda_env::require_sm52(&ctx)?;
    let dense = KVCache::new_with_context(&ctx, CAPACITY, SLOTS, KV_HEADS, HEAD_DIM)?;
    let sequential = compressed_cache(&ctx)?;
    let rebuilt = compressed_cache(&ctx)?;
    let elements = (KV_HEADS * HEAD_DIM) as usize;
    let bytes = elements * size_of::<f32>();
    let input_k = DeviceBuffer::new_tagged(&ctx, bytes, "no-old:input-k")?;
    let input_v = DeviceBuffer::new_tagged(&ctx, bytes, "no-old:input-v")?;
    let lengths = [
        1,
        15,
        WINDOW - 1,
        WINDOW,
        WINDOW + 1,
        20,
        21,
        22,
        33,
        34,
        35,
    ];
    let mut cases = Vec::new();
    for (epoch, layout) in [ROPE_LAYOUT_ADJACENT, ROPE_LAYOUT_NEOX]
        .into_iter()
        .enumerate()
    {
        dense.reset(&ctx)?;
        sequential.reset(&ctx)?;
        rebuilt.reset(&ctx)?;
        for position in 0..CAPACITY {
            for slot in 0..SLOTS {
                let k: Vec<f32> = (0..elements)
                    .map(|i| {
                        ((i + position as usize * 13 + slot as usize * 19 + epoch * 37) as f32
                            * 0.017)
                            .sin()
                    })
                    .collect();
                let v: Vec<f32> = k
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| x * 0.7 + i as f32 * 0.001)
                    .collect();
                unsafe {
                    ctx.memcpy_h2d(input_k.as_mut_ptr(), k.as_ptr().cast(), bytes)?;
                    ctx.memcpy_h2d(input_v.as_mut_ptr(), v.as_ptr().cast(), bytes)?;
                    for cache in [&dense, &sequential] {
                        cache.append_token_f32_rope_k_at_layout_async(
                            &ctx,
                            slot,
                            input_k.as_ptr(),
                            input_v.as_ptr(),
                            position,
                            position,
                            10_000.0,
                            1.0,
                            layout,
                        )?;
                    }
                }
                ctx.synchronize_stream(CudaStream::Decode)?;
            }
            let length = position + 1;
            if !lengths.contains(&length) {
                continue;
            }
            rebuilt.build_compressed_from_dense(&ctx, &dense, length)?;
            for slot in 0..SLOTS {
                let a = sequential.debug_compressed_snapshot(&ctx, slot)?;
                let b = rebuilt.debug_compressed_snapshot(&ctx, slot)?;
                ensure!(
                    a.seq_len == length && b.seq_len == length,
                    "wrong published length"
                );
                ensure!(
                    dense.debug_sequence_length(&ctx, slot)? == length,
                    "dense length mismatch"
                );
                for absolute in length.saturating_sub(WINDOW)..length {
                    let ring = (absolute % WINDOW) as usize;
                    let (k, v) = read_dense_token(&ctx, &dense, slot, absolute)?;
                    let range = ring * elements..(ring + 1) * elements;
                    for snapshot in [&a, &b] {
                        ensure!(
                            snapshot.recent_k_f16[range.clone()] == k,
                            "K layout mismatch slot={slot} absolute={absolute}"
                        );
                        ensure!(
                            snapshot.recent_v_f16[range.clone()] == v,
                            "V layout mismatch slot={slot} absolute={absolute}"
                        );
                    }
                }
                cases.push(
                    json!({"epoch":epoch,"rope_layout":layout,"physical_slot":slot,"length":length,
                    "recent_start":length.saturating_sub(WINDOW),"recent_end_exclusive":length,
                    "sequential_matches_dense_bits":true,"rebuilt_matches_dense_bits":true}),
                );
            }
        }
    }
    record_evidence(
        json!({"kind":"no_old_layout_gate","window":WINDOW,"block_size":BLOCK,
        "head_dim":HEAD_DIM,"kv_heads":KV_HEADS,"capacity":CAPACITY,"slots":SLOTS,
        "is_layout_gate_passed":true,"cases":cases}),
    )
}

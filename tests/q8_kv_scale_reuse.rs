#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/guarded_bytes.rs"]
mod guarded_bytes;

use anyhow::{ensure, Result};
use guarded_bytes::GuardedBytes;
use half::f16;
use m40_llm::{
    cuda::{CudaContext, CudaStream, ExactBlockStagingPtrs, ExactOldBacking, KVCache},
    kv_compression::KvRepresentativePolicy,
};

const LENGTH: usize = 129;
const HEAD: usize = 64;
const KV_HEADS: usize = 2;
const Q_HEADS: usize = 4;
const SEQUENCES: usize = 4;

fn encode_floats(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn read_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn create_history(identity: usize, is_key: bool) -> Vec<f32> {
    (0..LENGTH * KV_HEADS * HEAD)
        .map(|index| {
            let value = if is_key {
                8.0 + (index / (KV_HEADS * HEAD * 32)) as f32 * 4.0
                    + ((index * 17 + identity * 11) % 61) as f32 / 16.0
            } else {
                ((index * 13 + identity * 7) % 97) as f32 / 65536.0 - 0.0007
            };
            f16::from_f32(value).to_f32()
        })
        .collect()
}

fn reconstruct_head(values: &[f32], is_old: bool) -> Vec<f32> {
    let maximum = values.iter().fold(0.0_f32, |a, b| a.max(b.abs()));
    let scale = if maximum > 0.0 {
        maximum / 127.0
    } else {
        1.0 / 127.0
    };
    values
        .iter()
        .map(|&value| {
            if is_old {
                f16::from_f32((value / scale).round_ties_even().clamp(-127.0, 127.0) * scale)
                    .to_f32()
            } else {
                value
            }
        })
        .collect()
}

fn check_sequence(
    ctx: &CudaContext,
    cache: &KVCache,
    sequence: usize,
    keys: &[f32],
    values: &[f32],
) -> Result<()> {
    let query = vec![0.001_f32; Q_HEADS * HEAD];
    let query_bytes = encode_floats(&query);
    let query_buffer = GuardedBytes::new(ctx, &query_bytes, 32)?;
    let output = GuardedBytes::new(ctx, &vec![0xa5; query_bytes.len()], 32)?;
    let direct = GuardedBytes::new(ctx, &vec![0xa5; query_bytes.len()], 32)?;
    let plane_bytes = Q_HEADS * LENGTH * HEAD * 2;
    let staged_keys = GuardedBytes::new(ctx, &vec![0xa5; plane_bytes], 32)?;
    let staged_values = GuardedBytes::new(ctx, &vec![0xa5; plane_bytes], 32)?;
    let positions = GuardedBytes::new(ctx, &vec![0xa5; Q_HEADS * LENGTH * 4], 32)?;
    let counts = GuardedBytes::new(ctx, &[0xa5; Q_HEADS * 4], 32)?;
    let staging = ExactBlockStagingPtrs {
        staged_k: staged_keys.pointer(),
        staged_v: staged_values.pointer(),
        staged_positions: positions.pointer(),
        staged_counts: counts.pointer(),
        capacity_tokens: LENGTH as u32,
    };
    unsafe {
        cache.attention_last_token_f32_gqa_block_select_exact_staged_q8_old_with_buffers_async(
            ctx,
            sequence as u32,
            query_buffer.pointer(),
            Q_HEADS as u32,
            LENGTH as u32,
            1,
            32,
            4,
            staging,
            output.pointer(),
        )?;
        cache.attention_last_token_f32_gqa_block_select_exact_q8_old_direct_async(
            ctx,
            sequence as u32,
            query_buffer.pointer(),
            Q_HEADS as u32,
            LENGTH as u32,
            1,
            32,
            4,
            direct.pointer(),
        )?;
    }
    ctx.synchronize_stream(CudaStream::Decode)?;
    assert_eq!(query_buffer.read(ctx)?, query_bytes);
    assert_eq!(read_words(&counts.read(ctx)?), vec![LENGTH as u32; Q_HEADS]);
    let selected = read_words(&positions.read(ctx)?);
    let key_bytes = staged_keys.read(ctx)?;
    let value_bytes = staged_values.read(ctx)?;
    for head in 0..Q_HEADS {
        let mut seen = [false; LENGTH];
        for row in 0..LENGTH {
            let token = selected[head * LENGTH + row] as usize;
            ensure!(
                token < LENGTH && !seen[token],
                "invalid/duplicate staged token"
            );
            seen[token] = true;
            let start = (token * KV_HEADS + head / (Q_HEADS / KV_HEADS)) * HEAD;
            for (source, actual) in [(keys, &key_bytes), (values, &value_bytes)] {
                let expected = reconstruct_head(&source[start..start + HEAD], token < LENGTH - 1);
                for (channel, value) in expected.into_iter().enumerate() {
                    let offset = ((head * LENGTH + row) * HEAD + channel) * 2;
                    let bits = u16::from_le_bytes([actual[offset], actual[offset + 1]]);
                    assert_eq!(
                        bits,
                        f16::from_f32(value).to_bits(),
                        "sequence={sequence} head={head} token={token} channel={channel}"
                    );
                }
            }
        }
        assert!(seen.iter().all(|&value| value));
    }
    let staged_output = read_words(&output.read(ctx)?);
    let direct_output = read_words(&direct.read(ctx)?);
    for (staged, direct) in staged_output.into_iter().zip(direct_output) {
        let staged = f32::from_bits(staged);
        let direct = f32::from_bits(direct);
        ensure!(
            staged.is_finite() && direct.is_finite() && (staged - direct).abs() < 1e-6,
            "staged/direct output differs"
        );
    }
    Ok(())
}

/// Large K and tiny V maxima must retain independent scales across both
/// active warps. Validate every reconstructed old value, recent values and
/// attention outputs across sequence slots, ring wraparound and reset/reuse.
#[test]
#[ignore = "requires serial M40 hardware execution"]
fn test_q8_eviction_preserves_independent_scales() -> Result<()> {
    for key in [
        "M40LLM_KV_BLOCK_SELECT_POLICY",
        "M40LLM_KV_SELECTED_BLOCK_ORDER",
        "M40LLM_KV_ANCHOR_BLOCKS",
        "M40LLM_KV_FORCE_INCLUDE_BLOCKS",
        "M40LLM_KV_FORCE_EXCLUDE_BLOCKS",
    ] {
        ensure!(
            std::env::var_os(key).is_none(),
            "unset {key} for this fixture"
        );
    }
    let ctx = CudaContext::new_m40()?;
    let before = CudaContext::total_device_bytes();
    {
        let cache = KVCache::new_compressed_with_context(
            &ctx,
            LENGTH as u32,
            SEQUENCES as u32,
            KV_HEADS as u32,
            HEAD as u32,
            1,
            32,
            4,
            0,
            KvRepresentativePolicy::Last,
            ExactOldBacking::Q8,
        )?;
        for lifetime in 0..4 {
            cache.reset(&ctx)?;
            for sequence in 0..SEQUENCES {
                let keys = create_history(lifetime * SEQUENCES + sequence, true);
                let values = create_history(lifetime * SEQUENCES + sequence, false);
                for token in 0..LENGTH {
                    let start = token * KV_HEADS * HEAD;
                    let end = start + KV_HEADS * HEAD;
                    let key_bytes = encode_floats(&keys[start..end]);
                    let value_bytes = encode_floats(&values[start..end]);
                    let key_buffer = GuardedBytes::new(&ctx, &key_bytes, 32)?;
                    let value_buffer = GuardedBytes::new(&ctx, &value_bytes, 32)?;
                    unsafe {
                        cache.append_token_f32_rope_k_at_async(
                            &ctx,
                            sequence as u32,
                            key_buffer.pointer(),
                            value_buffer.pointer(),
                            token as u32,
                            token as u32,
                            10000.0,
                            0.0,
                        )?;
                    }
                    ctx.synchronize_stream(CudaStream::Decode)?;
                    assert_eq!(key_buffer.read(&ctx)?, key_bytes);
                    assert_eq!(value_buffer.read(&ctx)?, value_bytes);
                }
            }
            for sequence in [3, 1, 0, 2] {
                check_sequence(
                    &ctx,
                    &cache,
                    sequence,
                    &create_history(lifetime * SEQUENCES + sequence, true),
                    &create_history(lifetime * SEQUENCES + sequence, false),
                )?;
            }
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), before);
    Ok(())
}

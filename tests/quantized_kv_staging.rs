#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;
#[path = "common/guarded_bytes.rs"]
mod guarded_bytes;

use anyhow::{ensure, Result};
use guarded_bytes::GuardedBytes;
use half::f16;
use m40_llm::cuda::{
    CudaContext, CudaStream, DeviceBuffer, ExactBlockStagingPtrs, ExactOldBacking, KVCache,
};
use m40_llm::kv_compression::KvRepresentativePolicy;

const Q_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const RECENT: usize = 5;
const BLOCK: usize = 4;
const TOP: usize = 2;
const LENGTH: usize = 23;

struct TokenHistory {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl TokenHistory {
    fn build(head_dim: usize, identity: usize) -> Self {
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for token in 0..LENGTH {
            keys.push(
                (0..KV_HEADS * head_dim)
                    .map(|index| {
                        let head = index / head_dim;
                        // Separated block means avoid making a selection tie a
                        // floating-point reduction-order test.
                        let value = (token / BLOCK) as f32 * 0.07
                            + head as f32 * 0.1
                            + identity as f32 * 0.003
                            + (index % 11) as f32 * 0.0003;
                        f16::from_f32(value).to_f32()
                    })
                    .collect(),
            );
            values.push(
                (0..KV_HEADS * head_dim)
                    .map(|index| {
                        let sample = (token * 193 + index * 71 + identity * 29) % 997;
                        f16::from_f32((sample as f32 - 498.0) * 0.0017).to_f32()
                    })
                    .collect(),
            );
        }
        Self { keys, values }
    }

    fn select_positions(&self, query: &[f32], head: usize, length: usize) -> Vec<usize> {
        let head_dim = query.len() / Q_HEADS;
        let kv_head = head / (Q_HEADS / KV_HEADS);
        let old = length.saturating_sub(RECENT);
        let mut scores: Vec<(usize, f64)> = (0..old.div_ceil(BLOCK))
            .map(|block| {
                let start = block * BLOCK;
                let end = (start + BLOCK).min(old);
                let sum: f64 = (start..end)
                    .flat_map(|token| {
                        (0..head_dim).map(move |channel| {
                            f64::from(query[head * head_dim + channel])
                                * f64::from(self.keys[token][kv_head * head_dim + channel])
                        })
                    })
                    .sum();
                (block, sum / (end - start) as f64)
            })
            .collect();
        scores.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
        scores
            .iter()
            .take(TOP)
            .flat_map(|&(block, _)| block * BLOCK..((block + 1) * BLOCK).min(old))
            .chain(old..length)
            .collect()
    }

    fn reconstruct_values(&self, head_dim: usize, length: usize) -> Vec<Vec<f32>> {
        let mut values = self.values[..length].to_vec();
        for token in values.iter_mut().take(length.saturating_sub(RECENT)) {
            for head in token.chunks_exact_mut(head_dim) {
                let maximum = head
                    .iter()
                    .fold(0.0_f32, |value, next| value.max(next.abs()));
                let scale = if maximum > 0.0 {
                    maximum / 7.0
                } else {
                    1.0 / 7.0
                };
                for value in head {
                    let code = (*value / scale).round_ties_even().clamp(-7.0, 7.0);
                    *value = f16::from_f32(code * scale).to_f32();
                }
            }
        }
        values
    }
}

fn encode_floats(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn decode_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn build_query(head_dim: usize, identity: usize) -> Vec<f32> {
    (0..Q_HEADS * head_dim)
        .map(|index| {
            let sign = if (index / head_dim).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            sign * (0.03 + (index % 7) as f32 * 0.001 + identity as f32 * 0.0002)
        })
        .collect()
}

/// Check the new head128 staged control and retained head64 specialization
/// against independent selected positions, reconstructed FP16 bytes and FP64
/// attention. Distinct physical slots represent two sequences in two layers;
/// reset/reuse, partial old blocks and recent-ring wraparound must preserve
/// isolation. Prefill-stream input copies explicitly precede decode consumers.
#[test]
#[ignore = "requires serial M40 hardware execution"]
fn test_mixed_staging_preserves_selected_cache_and_attention() -> Result<()> {
    for key in [
        "M40LLM_KV_SELECTED_BLOCK_ORDER",
        "M40LLM_KV_BLOCK_SELECT_POLICY",
        "M40LLM_KV_ANCHOR_BLOCKS",
        "M40LLM_KV_FORCE_INCLUDE_BLOCKS",
        "M40LLM_KV_FORCE_EXCLUDE_BLOCKS",
    ] {
        ensure!(
            std::env::var_os(key).is_none(),
            "unset {key} for this fixture"
        );
    }
    let ctx = cuda_env::ctx_m40()?;
    cuda_env::require_sm52(&ctx)?;
    let before = CudaContext::total_device_bytes();
    for head_dim in [64, 128] {
        run_cache_lifetimes(&ctx, head_dim)?;
        assert_eq!(CudaContext::total_device_bytes(), before);
    }
    Ok(())
}

fn run_cache_lifetimes(ctx: &CudaContext, head_dim: usize) -> Result<()> {
    let cache = KVCache::new_compressed_with_context(
        ctx,
        LENGTH as u32,
        4,
        KV_HEADS as u32,
        head_dim as u32,
        RECENT as u32,
        BLOCK as u32,
        TOP as u32,
        0,
        KvRepresentativePolicy::Last,
        ExactOldBacking::Fp16KQ4V,
    )?;
    let bytes = KV_HEADS * head_dim * 4;
    let keys = DeviceBuffer::new_tagged(ctx, bytes, "kv_staging_test:keys")?;
    let values = DeviceBuffer::new_tagged(ctx, bytes, "kv_staging_test:values")?;
    for lifetime in 0..2 {
        cache.reset(ctx)?;
        let histories: Vec<_> = (0..4)
            .map(|slot| TokenHistory::build(head_dim, lifetime * 4 + slot))
            .collect();
        for token in 0..LENGTH {
            for &slot in &[2, 0, 3, 1] {
                let history = &histories[slot];
                unsafe {
                    // The reused host-upload destinations have no unfinished
                    // readers: drain after each append below.
                    ctx.memcpy_h2d(
                        keys.as_mut_ptr(),
                        history.keys[token].as_ptr().cast(),
                        bytes,
                    )?;
                    ctx.memcpy_h2d(
                        values.as_mut_ptr(),
                        history.values[token].as_ptr().cast(),
                        bytes,
                    )?;
                    cache.append_token_f32_rope_k_at_async(
                        ctx,
                        slot as u32,
                        keys.as_ptr(),
                        values.as_ptr(),
                        token as u32,
                        token as u32,
                        10000.0,
                        0.0,
                    )?;
                }
                ctx.synchronize_stream(CudaStream::Decode)?;
            }
            if [1, 5, 9, 14, LENGTH].contains(&(token + 1)) {
                for &slot in &[3, 1, 0, 2] {
                    check_attention(
                        ctx,
                        &cache,
                        &histories[slot],
                        head_dim,
                        slot,
                        token + 1,
                        lifetime * 4 + slot,
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_attention(
    ctx: &CudaContext,
    cache: &KVCache,
    history: &TokenHistory,
    head_dim: usize,
    slot: usize,
    length: usize,
    identity: usize,
) -> Result<()> {
    let query = build_query(head_dim, identity);
    let query_bytes = encode_floats(&query);
    let query_source = GuardedBytes::new(ctx, &query_bytes, 32)?;
    let query_device = GuardedBytes::new(ctx, &vec![0; query_bytes.len()], 32)?;
    let output = GuardedBytes::new(ctx, &vec![0; query_bytes.len()], 32)?;
    let direct = GuardedBytes::new(ctx, &vec![0; query_bytes.len()], 32)?;
    let old = length.saturating_sub(RECENT);
    let capacity = length.min(RECENT) + old.min(TOP * BLOCK);
    let stage_bytes = Q_HEADS * capacity * head_dim * 2;
    let stage_k = GuardedBytes::new(ctx, &vec![0xa5; stage_bytes], 32)?;
    let stage_v = GuardedBytes::new(ctx, &vec![0xa5; stage_bytes], 32)?;
    let positions = GuardedBytes::new(ctx, &vec![0xa5; Q_HEADS * capacity * 4], 32)?;
    let counts = GuardedBytes::new(ctx, &[0xa5; Q_HEADS * 4], 32)?;
    let staging = ExactBlockStagingPtrs {
        staged_k: stage_k.pointer(),
        staged_v: stage_v.pointer(),
        staged_positions: positions.pointer(),
        staged_counts: counts.pointer(),
        capacity_tokens: capacity as u32,
    };
    unsafe {
        ctx.memcpy_d2d_async(
            query_device.pointer(),
            query_source.pointer(),
            query_bytes.len(),
            CudaStream::Prefill,
        )?;
    }
    ctx.stream_wait_for_stream(CudaStream::Decode, CudaStream::Prefill, "staged_test_query")?;
    let live_bytes = CudaContext::total_device_bytes();
    for _ in 0..2 {
        unsafe {
            cache.attention_last_token_f32_gqa_block_select_exact_staged_fp16_k_q4_v_old_with_buffers_async(
                ctx, slot as u32, query_device.pointer(), Q_HEADS as u32,
                length as u32, RECENT as u32, BLOCK as u32, TOP as u32,
                staging, output.pointer(),
            )?;
            cache.attention_last_token_f32_gqa_block_select_exact_fp16_k_q4_v_old_direct_async(
                ctx,
                slot as u32,
                query_device.pointer(),
                Q_HEADS as u32,
                length as u32,
                RECENT as u32,
                BLOCK as u32,
                TOP as u32,
                direct.pointer(),
            )?;
        }
        ctx.synchronize_stream(CudaStream::Decode)?;
        assert_eq!(CudaContext::total_device_bytes(), live_bytes);
        assert_eq!(query_device.read(ctx)?, query_bytes);
        assert_eq!(query_source.read(ctx)?, query_bytes);
        let counts = decode_words(&counts.read(ctx)?);
        let positions = decode_words(&positions.read(ctx)?);
        let keys = stage_k.read(ctx)?;
        let values = stage_v.read(ctx)?;
        let actual = decode_words(&output.read(ctx)?);
        let direct = decode_words(&direct.read(ctx)?);
        let reconstructed = history.reconstruct_values(head_dim, length);
        for (head, &count) in counts.iter().enumerate() {
            let selected = history.select_positions(&query, head, length);
            assert_eq!(count as usize, selected.len());
            let start = head * capacity;
            assert_eq!(
                &positions[start..start + selected.len()],
                selected
                    .iter()
                    .map(|&value| value as u32)
                    .collect::<Vec<_>>()
            );
            check_selected_values(
                history,
                &reconstructed,
                &selected,
                head,
                head_dim,
                capacity,
                &keys,
                &values,
            )?;
            let expected = calculate_attention(history, &reconstructed, &query, head, &selected);
            for (channel, expected) in expected.iter().enumerate() {
                let index = head * head_dim + channel;
                let staged = f32::from_bits(actual[index]);
                let direct = f32::from_bits(direct[index]);
                let tolerance = 1e-5 * (1.0 + expected.abs());
                ensure!(staged.is_finite() && direct.is_finite());
                assert!(
                    (f64::from(staged) - expected).abs() <= tolerance,
                    "staged slot={slot} len={length} head={head} channel={channel}"
                );
                assert!(
                    (f64::from(direct) - expected).abs() <= tolerance,
                    "direct slot={slot} len={length} head={head} channel={channel}"
                );
                assert!((staged - direct).abs() <= 1e-5);
            }
        }
    }
    // Bad capacity must reject before touching the output or staging owners.
    let saved_output = output.read(ctx)?;
    let invalid = ExactBlockStagingPtrs {
        capacity_tokens: 0,
        ..staging
    };
    assert!(unsafe {
        cache.attention_last_token_f32_gqa_block_select_exact_staged_fp16_k_q4_v_old_with_buffers_async(
            ctx, slot as u32, query_device.pointer(), Q_HEADS as u32,
            length as u32, RECENT as u32, BLOCK as u32, TOP as u32,
            invalid, output.pointer(),
        )
    }.is_err());
    ctx.synchronize_stream(CudaStream::Decode)?;
    assert_eq!(output.read(ctx)?, saved_output);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_selected_values(
    history: &TokenHistory,
    reconstructed: &[Vec<f32>],
    selected: &[usize],
    head: usize,
    head_dim: usize,
    capacity: usize,
    keys: &[u8],
    values: &[u8],
) -> Result<()> {
    let kv_head = head / (Q_HEADS / KV_HEADS);
    for (index, &token) in selected.iter().enumerate() {
        for channel in 0..head_dim {
            let source = kv_head * head_dim + channel;
            let offset = ((head * capacity + index) * head_dim + channel) * 2;
            ensure!(
                &keys[offset..offset + 2]
                    == f16::from_f32(history.keys[token][source])
                        .to_bits()
                        .to_le_bytes()
            );
            ensure!(
                &values[offset..offset + 2]
                    == f16::from_f32(reconstructed[token][source])
                        .to_bits()
                        .to_le_bytes()
            );
        }
    }
    Ok(())
}

fn calculate_attention(
    history: &TokenHistory,
    reconstructed: &[Vec<f32>],
    query: &[f32],
    head: usize,
    selected: &[usize],
) -> Vec<f64> {
    let head_dim = query.len() / Q_HEADS;
    let kv_head = head / (Q_HEADS / KV_HEADS);
    let scores: Vec<f64> = selected
        .iter()
        .map(|&token| {
            (0..head_dim)
                .map(|channel| {
                    f64::from(query[head * head_dim + channel])
                        * f64::from(history.keys[token][kv_head * head_dim + channel])
                })
                .sum::<f64>()
                / (head_dim as f64).sqrt()
        })
        .collect();
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
    let total: f64 = weights.iter().sum();
    (0..head_dim)
        .map(|channel| {
            selected
                .iter()
                .zip(&weights)
                .map(|(&token, weight)| {
                    weight * f64::from(reconstructed[token][kv_head * head_dim + channel])
                })
                .sum::<f64>()
                / total
        })
        .collect()
}

#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;
#[path = "common/env_restore.rs"]
mod env_restore;
#[path = "common/q8_model.rs"]
#[allow(dead_code)]
mod fixture;

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

/// The opt-in path must remain read-only, agree numerically with dense and
/// original attention, and stop being eligible at the first partial old block.
#[test]
#[ignore = "explicit M40 no-old adapter gate"]
fn test_no_old_attention_and_rollover() -> Result<()> {
    use m40_llm::cuda::{Head128AttentionConfig, Head128AttentionWorkspace};
    let _policy = env_restore::EnvRestore::set("M40LLM_KV_BLOCK_SELECT_POLICY", "topk");
    let _mode = env_restore::EnvRestore::set("M40LLM_KV_NO_OLD_ATTENTION", "dense-recent");
    let ctx = cuda_env::ctx_m40()?;
    cuda_env::require_sm52(&ctx)?;
    let dense = KVCache::new_with_context(&ctx, CAPACITY, SLOTS, KV_HEADS, HEAD_DIM)?;
    let cache = compressed_cache(&ctx)?;
    let heads = 4;
    cache.prepare_no_old_attention(&ctx, heads)?;
    let partials = Head128AttentionWorkspace::new(&ctx, heads)?;
    let bytes = (KV_HEADS * HEAD_DIM) as usize * size_of::<f32>();
    let k = DeviceBuffer::new_tagged(&ctx, bytes, "no-old:k")?;
    let v = DeviceBuffer::new_tagged(&ctx, bytes, "no-old:v")?;
    let q_bytes = (heads * HEAD_DIM) as usize * size_of::<f32>();
    let q = DeviceBuffer::new_tagged(&ctx, q_bytes, "no-old:q")?;
    let direct = DeviceBuffer::new_tagged(&ctx, q_bytes, "no-old:direct")?;
    let parallel = DeviceBuffer::new_tagged(&ctx, q_bytes, "no-old:parallel")?;
    let reference = DeviceBuffer::new_tagged(&ctx, q_bytes, "no-old:reference")?;
    let queries: Vec<f32> = (0..q_bytes / 4).map(|i| (i as f32 * 0.023).sin()).collect();
    unsafe {
        ctx.memcpy_h2d(q.as_mut_ptr(), queries.as_ptr().cast(), q_bytes)?;
    }
    let mut cases = Vec::new();
    for position in 0..CAPACITY {
        for slot in 0..SLOTS {
            let values: Vec<f32> = (0..bytes / 4)
                .map(|i| ((i + position as usize * 7 + slot as usize * 23) as f32 * 0.031).cos())
                .collect();
            unsafe {
                ctx.memcpy_h2d(k.as_mut_ptr(), values.as_ptr().cast(), bytes)?;
                ctx.memcpy_h2d(v.as_mut_ptr(), values.as_ptr().cast(), bytes)?;
                for kv in [&dense, &cache] {
                    kv.append_token_f32_rope_k_at_layout_async(
                        &ctx,
                        slot,
                        k.as_ptr(),
                        v.as_ptr(),
                        position,
                        position,
                        10_000.0,
                        1.0,
                        ROPE_LAYOUT_NEOX,
                    )?;
                }
            }
            ctx.synchronize_stream(CudaStream::Decode)?;
            let n = position + 1;
            let before = cache.debug_compressed_snapshot(&ctx, slot)?;
            unsafe {
                cache
                    .attention_last_token_f32_gqa_block_select_exact_fp16_k_q4_v_old_direct_async(
                        &ctx,
                        slot,
                        q.as_ptr(),
                        heads,
                        n,
                        WINDOW,
                        BLOCK,
                        8,
                        direct.as_mut_ptr(),
                    )?;
                let used = cache.try_enqueue_no_old_attention(
                    &ctx,
                    slot,
                    q.as_ptr(),
                    heads,
                    n,
                    WINDOW,
                    parallel.as_mut_ptr(),
                )?;
                ensure!(used == (n <= WINDOW), "wrong rollover eligibility at N={n}");
                if !used {
                    cache.attention_last_token_f32_gqa_block_select_exact_fp16_k_q4_v_old_direct_async(
                        &ctx, slot, q.as_ptr(), heads, n, WINDOW, BLOCK, 8, parallel.as_mut_ptr(),
                    )?;
                } else {
                    dense.attention_head128_parallel_async(
                        &ctx,
                        slot,
                        q.as_ptr(),
                        heads,
                        n,
                        reference.as_mut_ptr(),
                        Head128AttentionConfig {
                            partitions: 8,
                            threads: 256,
                        },
                        Some(&partials),
                    )?;
                }
                ctx.synchronize_stream(CudaStream::Decode)?;
            }
            let after = cache.debug_compressed_snapshot(&ctx, slot)?;
            ensure!(before == after, "attention mutated compressed state");
            let mut a = vec![0.0f32; q_bytes / 4];
            let mut b = a.clone();
            let mut c = a.clone();
            unsafe {
                ctx.memcpy_d2h(a.as_mut_ptr().cast(), direct.as_ptr(), q_bytes)?;
                ctx.memcpy_d2h(b.as_mut_ptr().cast(), parallel.as_ptr(), q_bytes)?;
                if n <= WINDOW {
                    ctx.memcpy_d2h(c.as_mut_ptr().cast(), reference.as_ptr(), q_bytes)?;
                }
            }
            let max_diff = a
                .iter()
                .zip(&b)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            ensure!(
                max_diff < 1e-3,
                "attention parity failed at N={n}: {max_diff}"
            );
            if n <= WINDOW {
                ensure!(
                    b.iter().zip(&c).all(|(b, c)| b.to_bits() == c.to_bits()),
                    "borrowed stride differs from dense"
                );
            } else {
                ensure!(
                    a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "rollover fallback changed attention"
                );
            }
            cases.push(json!({"length":n,"slot":slot,"used_fast_path":n<=WINDOW,"max_abs_diff":max_diff,"cache_bits_unchanged":true}));
        }
    }
    record_evidence(json!({"kind":"no_old_primitive_gate","is_gate_passed":true,"cases":cases}))
}

/// Full-layer numerical parity does not excuse changed valid FP16 cache bits.
/// Save the first failing checkpoint before rejecting the experiment.
#[test]
#[ignore = "explicit M40 full-model cache-state gate; may reject candidate"]
fn test_no_old_full_forward_cache_gate() -> Result<()> {
    use m40_llm::{
        decode_session::DecodeSession,
        infer::{
            execution_weights::Q8ExecutionConfig, q8_prefill::Q8PrefillBackend, LoadedModel,
            Q8AttentionProjectionBackend, Q8FfnConfig,
        },
        kv_compression::{KvCompressionConfig, ScopedRuntimeConfig},
    };
    let _env: Vec<_> = [
        ("M40LLM_PROJECTION_BACKEND", "large-model"),
        ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
        ("M40LLM_DECODE_GRAPH", "0"),
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_KV_BLOCK_SELECT_POLICY", "topk"),
        ("M40LLM_ATTN_HEAD128_KERNEL", "native"),
    ]
    .into_iter()
    .map(|(k, v)| env_restore::EnvRestore::set(k, v))
    .collect();
    let config = KvCompressionConfig {
        recent_window: WINDOW,
        block_size: BLOCK,
        ..KvCompressionConfig::default()
    };
    let _scope = ScopedRuntimeConfig::new(config.clone());
    let make_model = || -> Result<LoadedModel> {
        let (gguf, bytes) = fixture::build_prefill_fixture(true);
        LoadedModel::from_gguf_with_q8_runtime(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig::parse("split", Some("6144"), Some("gate-up,down,vocabulary"))?,
            Q8FfnConfig::joint_shared_scales(),
            Q8PrefillBackend::Native,
            Q8AttentionProjectionBackend::PackedWarp256,
        )
    };
    let mut a = make_model()?;
    let mut b = make_model()?;
    {
        let _off = env_restore::EnvRestore::set("M40LLM_KV_NO_OLD_ATTENTION", "off");
        a.allocate_compressed_kv_cache_for_layers(CAPACITY, &config)?;
    }
    {
        let _on = env_restore::EnvRestore::set("M40LLM_KV_NO_OLD_ATTENTION", "dense-recent");
        b.allocate_compressed_kv_cache_for_layers(CAPACITY, &config)?;
    }
    let mut x = DecodeSession::new(
        &a,
        256,
        true,
        "no-old-control",
        "no-old:a-x",
        "no-old:a-out",
    )?;
    let mut y = DecodeSession::new(
        &b,
        256,
        true,
        "no-old-candidate",
        "no-old:b-x",
        "no-old:b-out",
    )?;
    let ids: Vec<u32> = (0..CAPACITY).map(|i| 1 + (i * 7) % 319).collect();
    let mut cases = Vec::new();
    for n in 1..=CAPACITY as usize {
        let expected = x.logits_for_ids(&ids[..n], |_| {})?;
        let actual = y.logits_for_ids(&ids[..n], |_| {})?;
        let max_diff = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean_diff = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / expected.len() as f64;
        let top = |values: &[f32]| {
            values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap()
        };
        let mut mismatches = Vec::new();
        for layer in 0..2 {
            let sa = a
                .kv_cache
                .as_ref()
                .unwrap()
                .debug_compressed_snapshot(&a.cuda, layer)?;
            let sb = b
                .kv_cache
                .as_ref()
                .unwrap()
                .debug_compressed_snapshot(&b.cuda, layer)?;
            let k = sa
                .recent_k_f16
                .iter()
                .zip(&sb.recent_k_f16)
                .filter(|(a, b)| a != b)
                .count();
            let v = sa
                .recent_v_f16
                .iter()
                .zip(&sb.recent_v_f16)
                .filter(|(a, b)| a != b)
                .count();
            mismatches.push(json!({"layer":layer,"recent_k_mismatches":k,"recent_v_mismatches":v,"snapshot_equal":sa==sb}));
        }
        let cache_equal = mismatches.iter().all(|m| m["snapshot_equal"] == true);
        let token_equal = top(&expected) == top(&actual);
        cases.push(json!({"length":n,"max_abs_logit_diff":max_diff,"mean_abs_logit_diff":mean_diff,
            "logit_bits_equal":expected.iter().zip(&actual).all(|(a,b)|a.to_bits()==b.to_bits()),
            "control_greedy_id":top(&expected),"candidate_greedy_id":top(&actual),"layers":mismatches}));
        if !cache_equal || !token_equal || max_diff >= 1e-3 {
            record_evidence(
                json!({"kind":"no_old_full_forward_gate","is_gate_passed":false,
                "construction":"sequential","first_failing_length":n,"cache_equal":cache_equal,"token_equal":token_equal,"cases":cases}),
            )?;
            anyhow::bail!("no-old correctness gate failed at N={n}: cache_equal={cache_equal}, token_equal={token_equal}, max_diff={max_diff}");
        }
    }
    record_evidence(
        json!({"kind":"no_old_full_forward_gate","is_gate_passed":true,"construction":"sequential","cases":cases}),
    )
}

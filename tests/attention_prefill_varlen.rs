#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use m40_llm::infer::{BatchMetadata, BatchSequence};
use std::ffi::c_void;

fn f32s_to_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
        .collect()
}

fn cpu_prefill_gqa_varlen(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    meta: &BatchMetadata,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; meta.total_q_tokens() as usize * q_heads * head_dim];
    let group = q_heads / kv_heads;
    let inv_sqrt = 1.0f32 / (head_dim as f32).sqrt();

    for (seq_idx, seq) in meta.sequences().iter().enumerate() {
        let offsets = meta.offsets()[seq_idx];
        for q_idx in 0..seq.query_len as usize {
            let causal_end = seq.kv_len as usize - seq.query_len as usize + q_idx;
            for qh in 0..q_heads {
                let kvh = qh / group;
                let q_base = ((offsets.q_offset as usize + q_idx) * q_heads + qh) * head_dim;
                let mut scores = Vec::with_capacity(causal_end + 1);
                let mut max_score = f32::NEG_INFINITY;
                for t in 0..=causal_end {
                    let k_base = ((offsets.kv_offset as usize + t) * kv_heads + kvh) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_base + d] * k[k_base + d];
                    }
                    let score = dot * inv_sqrt;
                    max_score = max_score.max(score);
                    scores.push(score);
                }
                let denom: f32 = scores.iter().map(|score| (*score - max_score).exp()).sum();
                let denom = if denom > 0.0 { denom } else { 1.0 };
                let out_base = ((offsets.q_offset as usize + q_idx) * q_heads + qh) * head_dim;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (t, score) in scores.iter().enumerate() {
                        let v_base = ((offsets.kv_offset as usize + t) * kv_heads + kvh) * head_dim;
                        acc += ((*score - max_score).exp() / denom) * v[v_base + d];
                    }
                    out[out_base + d] = acc;
                }
            }
        }
    }

    out
}

#[test]
fn packed_varlen_prefill_gqa_matches_cpu_reference() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let q_heads = 4usize;
    let kv_heads = 2usize;
    let head_dim = 64usize;
    let meta = BatchMetadata::new(vec![
        BatchSequence {
            seq_len: 3,
            query_len: 3,
            kv_len: 3,
        },
        BatchSequence {
            seq_len: 5,
            query_len: 2,
            kv_len: 5,
        },
        BatchSequence {
            seq_len: 4,
            query_len: 1,
            kv_len: 4,
        },
    ])?;

    let q_len = meta.total_q_tokens() as usize * q_heads * head_dim;
    let kv_len = meta.total_kv_tokens() as usize * kv_heads * head_dim;
    let q: Vec<f32> = (0..q_len)
        .map(|i| ((i * 17 % 251) as f32) * 0.0011 - 0.13)
        .collect();
    let k: Vec<f32> = (0..kv_len)
        .map(|i| ((i * 23 % 263) as f32) * 0.0009 - 0.09)
        .collect();
    let v: Vec<f32> = (0..kv_len)
        .map(|i| ((i * 29 % 269) as f32) * 0.0013 - 0.17)
        .collect();
    let expected = cpu_prefill_gqa_varlen(&q, &k, &v, &meta, q_heads, kv_heads, head_dim);

    let q_lens: Vec<u32> = meta.sequences().iter().map(|seq| seq.query_len).collect();
    let kv_lens: Vec<u32> = meta.sequences().iter().map(|seq| seq.kv_len).collect();
    let bytes_q = q.len() * std::mem::size_of::<f32>();
    let bytes_kv = k.len() * std::mem::size_of::<f32>();
    let bytes_out = expected.len() * std::mem::size_of::<f32>();
    let bytes_offsets = meta.sequences().len() * std::mem::size_of::<u32>();

    let d_q = ctx.device_malloc(bytes_q)?;
    let d_k = ctx.device_malloc(bytes_kv)?;
    let d_v = ctx.device_malloc(bytes_kv)?;
    let d_q_offsets = ctx.device_malloc(bytes_offsets)?;
    let d_kv_offsets = ctx.device_malloc(bytes_offsets)?;
    let d_q_lens = ctx.device_malloc(bytes_offsets)?;
    let d_kv_lens = ctx.device_malloc(bytes_offsets)?;
    let d_out = ctx.device_malloc(bytes_out)?;

    unsafe {
        ctx.memcpy_h2d(d_q, f32s_to_bytes(&q).as_ptr() as *const c_void, bytes_q)?;
        ctx.memcpy_h2d(d_k, f32s_to_bytes(&k).as_ptr() as *const c_void, bytes_kv)?;
        ctx.memcpy_h2d(d_v, f32s_to_bytes(&v).as_ptr() as *const c_void, bytes_kv)?;
        ctx.memcpy_h2d(
            d_q_offsets,
            meta.q_offsets().as_ptr() as *const c_void,
            bytes_offsets,
        )?;
        ctx.memcpy_h2d(
            d_kv_offsets,
            meta.kv_offsets().as_ptr() as *const c_void,
            bytes_offsets,
        )?;
        ctx.memcpy_h2d(d_q_lens, q_lens.as_ptr() as *const c_void, bytes_offsets)?;
        ctx.memcpy_h2d(d_kv_lens, kv_lens.as_ptr() as *const c_void, bytes_offsets)?;
        ctx.attention_prefill_f32_gqa_varlen_head64(
            d_q as *const c_void,
            d_k as *const c_void,
            d_v as *const c_void,
            d_q_offsets as *const u32,
            d_kv_offsets as *const u32,
            d_q_lens as *const u32,
            d_kv_lens as *const u32,
            meta.sequences().len() as u32,
            q_heads as u32,
            kv_heads as u32,
            d_out,
        )?;

        let mut got_bytes = vec![0u8; bytes_out];
        ctx.memcpy_d2h(got_bytes.as_mut_ptr() as *mut c_void, d_out, bytes_out)?;
        let got = bytes_to_f32s(&got_bytes);
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            let diff = (g - e).abs();
            assert!(diff <= 2e-4, "mismatch at {i}: got {g}, expected {e}");
        }

        ctx.device_free(d_q)?;
        ctx.device_free(d_k)?;
        ctx.device_free(d_v)?;
        ctx.device_free(d_q_offsets)?;
        ctx.device_free(d_kv_offsets)?;
        ctx.device_free(d_q_lens)?;
        ctx.device_free(d_kv_lens)?;
        ctx.device_free(d_out)?;
    }

    Ok(())
}

#![cfg(not(feature = "cuda"))]

use anyhow::Result;
use m40_llm::cuda::{CudaContext, KVCache};
use std::ffi::c_void;

fn cpu_last_token_attention(
    q: &[f32],             // [num_heads*head_dim]
    k_tokens: &[Vec<f32>], // seq_len entries, each [num_heads*head_dim]
    v_tokens: &[Vec<f32>], // seq_len entries, each [num_heads*head_dim]
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let dim = num_heads * head_dim;
    assert_eq!(q.len(), dim);
    for (k_t, v_t) in k_tokens.iter().zip(v_tokens.iter()) {
        assert_eq!(k_t.len(), dim);
        assert_eq!(v_t.len(), dim);
    }
    let inv_sqrt = 1.0f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; dim];

    for h in 0..num_heads {
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        // pass 1: max
        let mut max_s = f32::NEG_INFINITY;
        for k_t in k_tokens.iter() {
            let k_base = &k_t[h * head_dim..(h + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * k_base[d];
            }
            let s = dot * inv_sqrt;
            if s > max_s {
                max_s = s;
            }
        }
        // pass 2: denom
        let mut denom = 0.0f32;
        let mut scores: Vec<f32> = Vec::with_capacity(k_tokens.len());
        for k_t in k_tokens.iter() {
            let k_base = &k_t[h * head_dim..(h + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * k_base[d];
            }
            let s = dot * inv_sqrt;
            let e = (s - max_s).exp();
            scores.push(e);
            denom += e;
        }
        if denom == 0.0 {
            denom = 1.0;
        }
        // pass 3: out
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for (t, v_t) in v_tokens.iter().enumerate() {
                let prob = scores[t] / denom;
                let v_base = &v_t[h * head_dim..(h + 1) * head_dim];
                acc += prob * v_base[d];
            }
            out[h * head_dim + d] = acc;
        }
    }
    out
}

#[test]
fn attention_last_token_cpu_matches_ref() -> Result<()> {
    let ctx = CudaContext::new(0)?; // non-CUDA stub

    let max_seq_len = 8u32;
    let max_batch_size = 1u32;
    let num_heads = 3u32;
    let head_dim = 5u32; // odd to exercise loops
    let dim = (num_heads * head_dim) as usize;

    let kv = KVCache::new_with_context(&ctx, max_seq_len, max_batch_size, num_heads, head_dim)?;

    // Build a short sequence
    let seq_id = 0u32;
    let seq_len = 4u32;

    // Construct tokens and append via FP32 API (will be stored as f16 internally)
    let mut k_tokens_rounded: Vec<Vec<f32>> = Vec::new();
    let mut v_tokens_rounded: Vec<Vec<f32>> = Vec::new();
    for t in 0..seq_len as usize {
        let mut k = Vec::with_capacity(dim);
        let mut v = Vec::with_capacity(dim);
        for i in 0..dim {
            k.push(((t * dim + i) as f32) * 0.01 - 1.0);
            v.push(((t * dim + (dim - 1 - i)) as f32) * 0.02 - 2.0);
        }
        // Append to KV (host pointers are accepted in non-CUDA build)
        kv.append_token_f32(
            &ctx,
            seq_id,
            k.as_ptr() as *const c_void,
            v.as_ptr() as *const c_void,
        )?;
        // Mirror library f16 rounding for expected values
        let k_rounded: Vec<f32> = k.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect();
        let v_rounded: Vec<f32> = v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect();
        k_tokens_rounded.push(k_rounded);
        v_tokens_rounded.push(v_rounded);
    }

    // Q for the last token (FP32)
    let mut q = Vec::with_capacity(dim);
    for i in 0..dim {
        q.push((i as f32) * 0.003 - 0.5);
    }

    // Run library attention (CPU path) and get output
    let mut out = vec![0f32; dim];
    kv.attention_last_token_f32(
        &ctx,
        seq_id,
        q.as_ptr() as *const c_void,
        seq_len,
        out.as_mut_ptr() as *mut c_void,
    )?;

    // CPU reference using f16-rounded K/V
    let out_cpu = cpu_last_token_attention(
        &q,
        &k_tokens_rounded,
        &v_tokens_rounded,
        num_heads as usize,
        head_dim as usize,
    );

    for i in 0..dim {
        let diff = (out[i] - out_cpu[i]).abs();
        assert!(
            diff < 1e-6,
            "mismatch at {}: lib={} cpu={}",
            i,
            out[i],
            out_cpu[i]
        );
    }

    Ok(())
}

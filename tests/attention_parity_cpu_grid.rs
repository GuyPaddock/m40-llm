#![cfg(not(feature = "cuda"))]

use anyhow::Result;
use m40_llm::cuda::{CudaContext, KVCache};
use std::ffi::c_void;

fn cpu_last_token_attention(
    q: &[f32],                // [num_heads*head_dim]
    k_tokens: &Vec<Vec<f32>>, // seq_len entries, each [num_heads*head_dim]
    v_tokens: &Vec<Vec<f32>>, // seq_len entries, each [num_heads*head_dim]
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let dim = num_heads * head_dim;
    assert_eq!(q.len(), dim);
    for t in 0..k_tokens.len() {
        assert_eq!(k_tokens[t].len(), dim);
        assert_eq!(v_tokens[t].len(), dim);
    }
    let inv_sqrt = 1.0f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; dim];

    for h in 0..num_heads {
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        // pass 1: max
        let mut max_s = f32::NEG_INFINITY;
        for t in 0..k_tokens.len() {
            let k_base = &k_tokens[t][h * head_dim..(h + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * k_base[d];
            }
            let s = dot * inv_sqrt;
            if s > max_s {
                max_s = s;
            }
        }
        // pass 2: denom and scores
        let mut denom = 0.0f32;
        let mut scores = vec![0.0f32; k_tokens.len()];
        for t in 0..k_tokens.len() {
            let k_base = &k_tokens[t][h * head_dim..(h + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * k_base[d];
            }
            let s = dot * inv_sqrt;
            let e = (s - max_s).exp();
            scores[t] = e;
            denom += e;
        }
        if denom == 0.0 {
            denom = 1.0;
        }
        // pass 3: out
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for t in 0..k_tokens.len() {
                let prob = scores[t] / denom;
                let v_base = &v_tokens[t][h * head_dim..(h + 1) * head_dim];
                acc += prob * v_base[d];
            }
            out[h * head_dim + d] = acc;
        }
    }
    out
}

#[test]
fn attention_last_token_cpu_parity_grid() -> Result<()> {
    let ctx = CudaContext::new(0)?; // non-CUDA stub

    // Grids (keep modest to avoid long test time)
    let head_dims = [1u32, 5, 7, 8, 16, 31, 32, 63, 64];
    let num_heads_list = [1u32, 2, 3, 4];
    let seq_lens = [1u32, 2, 3, 4, 7, 8, 16];

    for &num_heads in &num_heads_list {
        for &head_dim in &head_dims {
            let dim = (num_heads * head_dim) as usize;
            // pick a max_seq_len >= max(seq_lens)
            let max_seq_len = *seq_lens.iter().max().unwrap();

            for &seq_len in &seq_lens {
                // Fresh KV per seq_len to avoid carrying over previous tokens
                let kv = KVCache::new_with_context(&ctx, max_seq_len, 1, num_heads, head_dim)?;

                // Prepare and append seq_len tokens
                let mut k_tokens_rounded: Vec<Vec<f32>> = Vec::new();
                let mut v_tokens_rounded: Vec<Vec<f32>> = Vec::new();
                for t in 0..seq_len as usize {
                    // Deterministic but varied data
                    let mut k = Vec::with_capacity(dim);
                    let mut v = Vec::with_capacity(dim);
                    for i in 0..dim {
                        k.push(((t * dim + i) as f32) * 0.01 - 1.0);
                        v.push(((t * dim + (dim - 1 - i)) as f32) * 0.02 - 2.0);
                    }
                    // Append FP32 which is stored as f16 internally
                    kv.append_token_f32(
                        &ctx,
                        0,
                        k.as_ptr() as *const c_void,
                        v.as_ptr() as *const c_void,
                    )?;
                    // Mirror f16 rounding for reference
                    let k_r: Vec<f32> =
                        k.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect();
                    let v_r: Vec<f32> =
                        v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect();
                    k_tokens_rounded.push(k_r);
                    v_tokens_rounded.push(v_r);
                }

                // Query vector
                let mut q = Vec::with_capacity(dim);
                for i in 0..dim {
                    q.push((i as f32) * 0.003 - 0.5);
                }

                // Library (CPU path)
                let mut out = vec![0f32; dim];
                kv.attention_last_token_f32(
                    &ctx,
                    0,
                    q.as_ptr() as *const c_void,
                    seq_len,
                    out.as_mut_ptr() as *mut c_void,
                )?;

                // Reference
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
                        "mismatch (heads={},dim={},seq_len={},i={}) lib={} cpu={}",
                        num_heads,
                        head_dim,
                        seq_len,
                        i,
                        out[i],
                        out_cpu[i]
                    );
                }
            }
        }
    }

    Ok(())
}

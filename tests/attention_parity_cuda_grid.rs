#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use m40_llm::cuda::KVCache;
use std::ffi::c_void;

fn cast_f32_to_f16_then_back(vals: &[f32]) -> Vec<f32> {
    vals.iter()
        .map(|&x| half::f16::from_f32(x).to_f32())
        .collect()
}

fn cpu_last_token_attention(
    q: &[f32],             // [num_heads*head_dim]
    k_tokens: &[Vec<f32>], // seq_len entries, each [num_heads*head_dim]
    v_tokens: &[Vec<f32>], // seq_len entries, each [num_heads*head_dim]
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let dim = num_heads * head_dim;
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
            for (e, v_t) in scores.iter().zip(v_tokens.iter()) {
                let prob = *e / denom;
                let v_base = &v_t[h * head_dim..(h + 1) * head_dim];
                acc += prob * v_base[d];
            }
            out[h * head_dim + d] = acc;
        }
    }
    out
}

#[test]
fn attention_last_token_cuda_parity_grid() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let head_dims = [1u32, 5, 7, 8, 16, 31, 32];
    let num_heads_list = [1u32, 3, 4, 8];
    let seq_lens = [1u32, 2, 3, 4, 8, 16];

    let max_seq_len = *seq_lens.iter().max().unwrap();

    for &num_heads in &num_heads_list {
        for &head_dim in &head_dims {
            let dim = (num_heads * head_dim) as usize;

            for &seq_len in &seq_lens {
                let kv = KVCache::new_with_context(&ctx, max_seq_len, 1, num_heads, head_dim)?;

                // Build tokens on host with f16->f32 cast to match device storage
                let mut k_tokens_f32: Vec<Vec<f32>> = Vec::new();
                let mut v_tokens_f32: Vec<Vec<f32>> = Vec::new();
                for t in 0..seq_len as usize {
                    let mut k = Vec::with_capacity(dim);
                    let mut v = Vec::with_capacity(dim);
                    for i in 0..dim {
                        k.push(((t * dim + i) as f32) * 0.01 - 1.0);
                        v.push(((t * dim + (dim - 1 - i)) as f32) * 0.02 - 2.0);
                    }
                    let k_cast = cast_f32_to_f16_then_back(&k);
                    let v_cast = cast_f32_to_f16_then_back(&v);
                    k_tokens_f32.push(k_cast.clone());
                    v_tokens_f32.push(v_cast.clone());

                    // Upload this token via append_token_f32
                    let bytes = dim * std::mem::size_of::<f32>();
                    let d_k = ctx.device_malloc(bytes)?;
                    let d_v = ctx.device_malloc(bytes)?;
                    unsafe {
                        ctx.memcpy_h2d(d_k, k_cast.as_ptr() as *const c_void, bytes)?;
                        ctx.memcpy_h2d(d_v, v_cast.as_ptr() as *const c_void, bytes)?;
                        kv.append_token_f32(&ctx, 0, d_k as *const c_void, d_v as *const c_void)?;
                        ctx.device_free(d_k)?;
                        ctx.device_free(d_v)?;
                    }
                }

                // Q and output
                let mut q = Vec::with_capacity(dim);
                for i in 0..dim {
                    q.push((i as f32) * 0.003 - 0.5);
                }
                let bytes_f32 = dim * std::mem::size_of::<f32>();
                let d_q = ctx.device_malloc(bytes_f32)?;
                let d_out = ctx.device_malloc(bytes_f32)?;
                unsafe {
                    ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_f32)?;
                    kv.attention_last_token_f32(&ctx, 0, d_q as *const c_void, seq_len, d_out)?;
                }
                let mut out_gpu = vec![0f32; dim];
                unsafe {
                    ctx.memcpy_d2h(
                        out_gpu.as_mut_ptr() as *mut c_void,
                        d_out as *const c_void,
                        bytes_f32,
                    )?;
                    ctx.device_free(d_q)?;
                    ctx.device_free(d_out)?;
                }

                // CPU ref
                let out_cpu = cpu_last_token_attention(
                    &q,
                    &k_tokens_f32,
                    &v_tokens_f32,
                    num_heads as usize,
                    head_dim as usize,
                );

                for i in 0..dim {
                    let a = out_cpu[i];
                    let b = out_gpu[i];
                    let diff = (a - b).abs();
                    assert!(
                        diff < 1e-3,
                        "mismatch (heads={},dim={},seq_len={},i={}) cpu={} gpu={} diff={}",
                        num_heads,
                        head_dim,
                        seq_len,
                        i,
                        a,
                        b,
                        diff
                    );
                }
            }
        }
    }

    Ok(())
}

#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use m40_llm::cuda::KVCache;
use m40_llm::kv_compression::{set_runtime_config, KvCompressMode, KvCompressionConfig};
use std::ffi::c_void;

fn cast_f32_to_f16_then_back(vals: &[f32]) -> Vec<f32> {
    vals.iter()
        .map(|&x| half::f16::from_f32(x).to_f32())
        .collect()
}

#[test]
fn attention_block_select_exact_matches_dense_when_all_old_blocks_selected() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let q_heads = 4u32;
    let kv_heads = 2u32;
    let head_dim = 64u32;
    let seq_len = 16u32;
    let recent_window = 4u32;
    let block_size = 4u32;
    let top_blocks = 3u32;
    let q_dim = (q_heads * head_dim) as usize;
    let kv_dim = (kv_heads * head_dim) as usize;
    let kv = KVCache::new_with_context(&ctx, seq_len, 1, kv_heads, head_dim)?;

    for t in 0..seq_len as usize {
        let k: Vec<f32> = (0..kv_dim)
            .map(|i| ((t * 31 + i * 7) as f32) * 0.002 - 0.4)
            .collect();
        let v: Vec<f32> = (0..kv_dim)
            .map(|i| ((t * 17 + kv_dim - i) as f32) * 0.003 - 0.2)
            .collect();
        let bytes = kv_dim * std::mem::size_of::<f32>();
        let d_k = ctx.device_malloc(bytes)?;
        let d_v = ctx.device_malloc(bytes)?;
        unsafe {
            ctx.memcpy_h2d(d_k, k.as_ptr() as *const c_void, bytes)?;
            ctx.memcpy_h2d(d_v, v.as_ptr() as *const c_void, bytes)?;
            kv.append_token_f32(&ctx, 0, d_k as *const c_void, d_v as *const c_void)?;
            ctx.device_free(d_k)?;
            ctx.device_free(d_v)?;
        }
    }

    let q: Vec<f32> = (0..q_dim).map(|i| (i as f32) * 0.001 - 0.1).collect();
    let bytes_q = q_dim * std::mem::size_of::<f32>();
    let d_q = ctx.device_malloc(bytes_q)?;
    let d_dense = ctx.device_malloc(bytes_q)?;
    let d_sparse = ctx.device_malloc(bytes_q)?;
    unsafe {
        ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_q)?;
        kv.attention_last_token_f32_gqa(&ctx, 0, d_q as *const c_void, q_heads, seq_len, d_dense)?;
        kv.attention_last_token_f32_gqa_block_select_exact_async(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            recent_window,
            block_size,
            top_blocks,
            d_sparse,
        )?;
        ctx.synchronize_stream(m40_llm::cuda::CudaStream::Decode)?;
    }

    let mut dense = vec![0f32; q_dim];
    let mut sparse = vec![0f32; q_dim];
    unsafe {
        ctx.memcpy_d2h(
            dense.as_mut_ptr() as *mut c_void,
            d_dense as *const c_void,
            bytes_q,
        )?;
        ctx.memcpy_d2h(
            sparse.as_mut_ptr() as *mut c_void,
            d_sparse as *const c_void,
            bytes_q,
        )?;
        ctx.device_free(d_q)?;
        ctx.device_free(d_dense)?;
        ctx.device_free(d_sparse)?;
    }

    for (idx, (&a, &b)) in dense.iter().zip(&sparse).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "block-select-exact mismatch at {idx}: dense={a} sparse={b}"
        );
    }

    set_runtime_config(KvCompressionConfig {
        mode: KvCompressMode::Off,
        ..Default::default()
    });
    Ok(())
}

#[test]
fn attention_block_summary_lossy_is_finite_and_deterministic() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let q_heads = 4u32;
    let kv_heads = 2u32;
    let head_dim = 64u32;
    let seq_len = 20u32;
    let recent_window = 4u32;
    let block_size = 4u32;
    let q_dim = (q_heads * head_dim) as usize;
    let kv_dim = (kv_heads * head_dim) as usize;
    let kv = KVCache::new_with_context(&ctx, seq_len, 1, kv_heads, head_dim)?;

    for t in 0..seq_len as usize {
        let k: Vec<f32> = (0..kv_dim)
            .map(|i| ((t * 13 + i * 5) as f32).sin() * 0.02)
            .collect();
        let v: Vec<f32> = (0..kv_dim)
            .map(|i| ((t * 19 + i * 3) as f32).cos() * 0.03)
            .collect();
        let bytes = kv_dim * std::mem::size_of::<f32>();
        let d_k = ctx.device_malloc(bytes)?;
        let d_v = ctx.device_malloc(bytes)?;
        unsafe {
            ctx.memcpy_h2d(d_k, k.as_ptr() as *const c_void, bytes)?;
            ctx.memcpy_h2d(d_v, v.as_ptr() as *const c_void, bytes)?;
            kv.append_token_f32(&ctx, 0, d_k as *const c_void, d_v as *const c_void)?;
            ctx.device_free(d_k)?;
            ctx.device_free(d_v)?;
        }
    }

    let q: Vec<f32> = (0..q_dim).map(|i| (i as f32).sin() * 0.01).collect();
    let bytes_q = q_dim * std::mem::size_of::<f32>();
    let d_q = ctx.device_malloc(bytes_q)?;
    let d_summary_a = ctx.device_malloc(bytes_q)?;
    let d_summary_b = ctx.device_malloc(bytes_q)?;
    let d_select_lossy = ctx.device_malloc(bytes_q)?;
    unsafe {
        ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_q)?;
        kv.attention_last_token_f32_gqa_block_summary_lossy_async(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            recent_window,
            block_size,
            0,
            d_summary_a,
        )?;
        kv.attention_last_token_f32_gqa_block_summary_lossy_async(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            recent_window,
            block_size,
            0,
            d_summary_b,
        )?;
        kv.attention_last_token_f32_gqa_block_summary_lossy_async(
            &ctx,
            0,
            d_q as *const c_void,
            q_heads,
            seq_len,
            recent_window,
            block_size,
            2,
            d_select_lossy,
        )?;
        ctx.synchronize_stream(m40_llm::cuda::CudaStream::Decode)?;
    }

    let mut summary_a = vec![0f32; q_dim];
    let mut summary_b = vec![0f32; q_dim];
    let mut select_lossy = vec![0f32; q_dim];
    unsafe {
        ctx.memcpy_d2h(
            summary_a.as_mut_ptr() as *mut c_void,
            d_summary_a as *const c_void,
            bytes_q,
        )?;
        ctx.memcpy_d2h(
            summary_b.as_mut_ptr() as *mut c_void,
            d_summary_b as *const c_void,
            bytes_q,
        )?;
        ctx.memcpy_d2h(
            select_lossy.as_mut_ptr() as *mut c_void,
            d_select_lossy as *const c_void,
            bytes_q,
        )?;
        ctx.device_free(d_q)?;
        ctx.device_free(d_summary_a)?;
        ctx.device_free(d_summary_b)?;
        ctx.device_free(d_select_lossy)?;
    }

    for idx in 0..q_dim {
        assert!(
            summary_a[idx].is_finite(),
            "summary output {idx} is not finite"
        );
        assert!(
            select_lossy[idx].is_finite(),
            "select-lossy output {idx} is not finite"
        );
        assert!(
            (summary_a[idx] - summary_b[idx]).abs() < 1e-6,
            "summary output not deterministic at {idx}: {} vs {}",
            summary_a[idx],
            summary_b[idx]
        );
    }
    Ok(())
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

fn cpu_last_token_attention_gqa(
    q: &[f32],             // [q_heads*head_dim]
    k_tokens: &[Vec<f32>], // seq_len entries, each [kv_heads*head_dim]
    v_tokens: &[Vec<f32>], // seq_len entries, each [kv_heads*head_dim]
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(q_heads % kv_heads, 0);
    let dim = q_heads * head_dim;
    let inv_sqrt = 1.0f32 / (head_dim as f32).sqrt();
    let group = q_heads / kv_heads;
    let mut out = vec![0.0f32; dim];

    for qh_idx in 0..q_heads {
        let kvh_idx = qh_idx / group;
        let qh = &q[qh_idx * head_dim..(qh_idx + 1) * head_dim];
        let mut max_s = f32::NEG_INFINITY;
        for k_t in k_tokens.iter() {
            let k_base = &k_t[kvh_idx * head_dim..(kvh_idx + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * k_base[d];
            }
            max_s = max_s.max(dot * inv_sqrt);
        }

        let mut denom = 0.0f32;
        let mut scores: Vec<f32> = Vec::with_capacity(k_tokens.len());
        for k_t in k_tokens.iter() {
            let k_base = &k_t[kvh_idx * head_dim..(kvh_idx + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * k_base[d];
            }
            let e = (dot * inv_sqrt - max_s).exp();
            scores.push(e);
            denom += e;
        }
        if denom == 0.0 {
            denom = 1.0;
        }

        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for (e, v_t) in scores.iter().zip(v_tokens.iter()) {
                let v_base = &v_t[kvh_idx * head_dim..(kvh_idx + 1) * head_dim];
                acc += (*e / denom) * v_base[d];
            }
            out[qh_idx * head_dim + d] = acc;
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

#[test]
fn attention_last_token_cuda_gqa_parity_grid() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let head_dims = [1u32, 5, 8, 16, 32];
    let head_pairs = [(2u32, 1u32), (4, 1), (8, 2), (32, 4)];
    let seq_lens = [1u32, 2, 3, 4, 8, 16];
    let max_seq_len = *seq_lens.iter().max().unwrap();

    for &(q_heads, kv_heads) in &head_pairs {
        for &head_dim in &head_dims {
            let q_dim = (q_heads * head_dim) as usize;
            let kv_dim = (kv_heads * head_dim) as usize;

            for &seq_len in &seq_lens {
                let kv = KVCache::new_with_context(&ctx, max_seq_len, 1, kv_heads, head_dim)?;

                let mut k_tokens_f32: Vec<Vec<f32>> = Vec::new();
                let mut v_tokens_f32: Vec<Vec<f32>> = Vec::new();
                for t in 0..seq_len as usize {
                    let mut k = Vec::with_capacity(kv_dim);
                    let mut v = Vec::with_capacity(kv_dim);
                    for i in 0..kv_dim {
                        k.push(((t * kv_dim + i) as f32) * 0.011 - 0.7);
                        v.push(((t * kv_dim + (kv_dim - 1 - i)) as f32) * 0.017 - 1.3);
                    }
                    let k_cast = cast_f32_to_f16_then_back(&k);
                    let v_cast = cast_f32_to_f16_then_back(&v);
                    k_tokens_f32.push(k_cast.clone());
                    v_tokens_f32.push(v_cast.clone());

                    let bytes = kv_dim * std::mem::size_of::<f32>();
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

                let mut q = Vec::with_capacity(q_dim);
                for i in 0..q_dim {
                    q.push((i as f32) * 0.004 - 0.25);
                }
                let bytes_f32 = q_dim * std::mem::size_of::<f32>();
                let d_q = ctx.device_malloc(bytes_f32)?;
                let d_out = ctx.device_malloc(bytes_f32)?;
                unsafe {
                    ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_f32)?;
                    kv.attention_last_token_f32_gqa(
                        &ctx,
                        0,
                        d_q as *const c_void,
                        q_heads,
                        seq_len,
                        d_out,
                    )?;
                }
                let mut out_gpu = vec![0f32; q_dim];
                unsafe {
                    ctx.memcpy_d2h(
                        out_gpu.as_mut_ptr() as *mut c_void,
                        d_out as *const c_void,
                        bytes_f32,
                    )?;
                    ctx.device_free(d_q)?;
                    ctx.device_free(d_out)?;
                }

                let out_cpu = cpu_last_token_attention_gqa(
                    &q,
                    &k_tokens_f32,
                    &v_tokens_f32,
                    q_heads as usize,
                    kv_heads as usize,
                    head_dim as usize,
                );

                for i in 0..q_dim {
                    let a = out_cpu[i];
                    let b = out_gpu[i];
                    let diff = (a - b).abs();
                    assert!(
                        diff < 1e-3,
                        "GQA mismatch (q_heads={},kv_heads={},dim={},seq_len={},i={}) cpu={} gpu={} diff={}",
                        q_heads,
                        kv_heads,
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

#[test]
fn attention_last_token_cuda_gqa_head64_long_parity() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let q_heads = 32u32;
    let kv_heads = 4u32;
    let head_dim = 64u32;
    let seq_lens = [128u32, 512];
    let max_seq_len = *seq_lens.iter().max().unwrap();
    let q_dim = (q_heads * head_dim) as usize;
    let kv_dim = (kv_heads * head_dim) as usize;

    for &seq_len in &seq_lens {
        let kv = KVCache::new_with_context(&ctx, max_seq_len, 1, kv_heads, head_dim)?;
        let mut k_tokens_f32: Vec<Vec<f32>> = Vec::new();
        let mut v_tokens_f32: Vec<Vec<f32>> = Vec::new();

        let bytes = kv_dim * std::mem::size_of::<f32>();
        let d_k = ctx.device_malloc(bytes)?;
        let d_v = ctx.device_malloc(bytes)?;
        for t in 0..seq_len as usize {
            let mut k = Vec::with_capacity(kv_dim);
            let mut v = Vec::with_capacity(kv_dim);
            for i in 0..kv_dim {
                k.push(((t * kv_dim + i) as f32) * 0.0003 - 0.25);
                v.push(((t * kv_dim + (kv_dim - 1 - i)) as f32) * 0.0002 - 0.1);
            }
            let k_cast = cast_f32_to_f16_then_back(&k);
            let v_cast = cast_f32_to_f16_then_back(&v);
            k_tokens_f32.push(k_cast.clone());
            v_tokens_f32.push(v_cast.clone());

            unsafe {
                ctx.memcpy_h2d(d_k, k_cast.as_ptr() as *const c_void, bytes)?;
                ctx.memcpy_h2d(d_v, v_cast.as_ptr() as *const c_void, bytes)?;
                kv.append_token_f32(&ctx, 0, d_k as *const c_void, d_v as *const c_void)?;
            }
        }

        let q: Vec<f32> = (0..q_dim).map(|i| (i as f32) * 0.0004 - 0.2).collect();
        let bytes_q = q_dim * std::mem::size_of::<f32>();
        let d_q = ctx.device_malloc(bytes_q)?;
        let d_out = ctx.device_malloc(bytes_q)?;
        unsafe {
            ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes_q)?;
            kv.attention_last_token_f32_gqa(
                &ctx,
                0,
                d_q as *const c_void,
                q_heads,
                seq_len,
                d_out,
            )?;
        }
        let mut out_gpu = vec![0f32; q_dim];
        unsafe {
            ctx.memcpy_d2h(
                out_gpu.as_mut_ptr() as *mut c_void,
                d_out as *const c_void,
                bytes_q,
            )?;
            ctx.device_free(d_q)?;
            ctx.device_free(d_out)?;
            ctx.device_free(d_k)?;
            ctx.device_free(d_v)?;
        }

        let out_cpu = cpu_last_token_attention_gqa(
            &q,
            &k_tokens_f32,
            &v_tokens_f32,
            q_heads as usize,
            kv_heads as usize,
            head_dim as usize,
        );

        for i in 0..q_dim {
            let diff = (out_cpu[i] - out_gpu[i]).abs();
            assert!(
                diff < 1e-3,
                "GQA head64 mismatch (seq_len={}, i={}) cpu={} gpu={} diff={}",
                seq_len,
                i,
                out_cpu[i],
                out_gpu[i],
                diff
            );
        }
    }

    Ok(())
}

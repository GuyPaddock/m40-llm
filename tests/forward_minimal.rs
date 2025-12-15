#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use m40_llm::gguf::GgufModel;
use m40_llm::infer::{LoadedModel, ModelConfig};
use std::collections::HashMap;
use std::ffi::c_void;

fn f32s_to_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn f32s_to_halves_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bits = half::f16::from_f32(v).to_bits();
        out.push((bits & 0xFF) as u8);
        out.push((bits >> 8) as u8);
    }
    out
}

#[test]
fn forward_one_token_minimal_smoke() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    // Small toy dims where num_heads * head_dim = d_model
    let d_model = 8i32;
    let num_heads = 2u32;
    let head_dim = 4u32;
    assert_eq!(num_heads * head_dim, d_model as u32);
    let hidden_dim = 16i32;

    // Build a minimal LoadedModel with a KV cache
    let kv = m40_llm::cuda::KVCache::new_with_context(&ctx, 8, 1, num_heads, head_dim)?;
    let model_config = ModelConfig {
        architecture: "llama".into(),
        block_count: 1,
        context_length: 8,
        embedding_length: d_model as u32,
        feed_forward_length: hidden_dim as u32,
        attention_head_count: num_heads,
        attention_head_count_kv: num_heads,
        attention_key_length: head_dim,
        layer_norm_epsilon: 1e-5,
        rope_freq_base: 10_000.0,
        rope_freq_scale: 1.0,
        vocab_size: 1,
    };
    let mut lm = LoadedModel {
        gguf: GgufModel::new(0),
        cuda: ctx.clone(),
        kv_cache: Some(kv),
        device_tensors: HashMap::new(),
        weights_len: 0,
        #[cfg(feature = "cuda")]
        d_weights_base: std::ptr::null_mut(),
        host_weights: Vec::new(),
        model_config,
        #[cfg(feature = "gguf_ext")]
        typed_config: gguf_llms::model::ModelConfig {
            architecture: "llama".into(),
            block_count: 1,
            context_length: 8,
            embedding_length: d_model as u32,
            feed_forward_length: hidden_dim as u32,
            attention_head_count: num_heads,
            attention_head_count_kv: None,
            attention_key_length: Some(head_dim),
            layer_norm_epsilon: None,
            rope_freq_base: None,
        },
    };

    // Input x (1 x d_model)
    let x: Vec<f32> = (0..d_model as usize)
        .map(|i| (i as f32) * 0.01 - 0.1)
        .collect();
    let d_x = lm.cuda.device_malloc((d_model * 4) as usize)?;
    unsafe {
        lm.cuda.memcpy_h2d(
            d_x,
            f32s_to_bytes(&x).as_ptr() as *const c_void,
            (d_model * 4) as usize,
        )?;
    }

    // Weights: all row-major
    // Q/K/V: K x Nq with K=d_model, Nq=d_model
    let wq: Vec<f32> = (0..(d_model * d_model) as usize)
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();
    let wk: Vec<f32> = (0..(d_model * d_model) as usize)
        .map(|i| ((i as f32) * 0.02).cos())
        .collect();
    let wv: Vec<f32> = (0..(d_model * d_model) as usize)
        .map(|i| ((i as f32) * 0.03).tan().atan())
        .collect();
    let wo: Vec<f32> = (0..(d_model * d_model) as usize)
        .map(|i| ((i as f32) * 0.04).sin())
        .collect();

    let d_wq = lm.cuda.device_malloc((d_model * d_model * 2) as usize)?;
    let d_wk = lm.cuda.device_malloc((d_model * d_model * 2) as usize)?;
    let d_wv = lm.cuda.device_malloc((d_model * d_model * 2) as usize)?;
    let d_wo = lm.cuda.device_malloc((d_model * d_model * 2) as usize)?;

    unsafe {
        lm.cuda.memcpy_h2d(
            d_wq,
            f32s_to_halves_bytes(&wq).as_ptr() as *const c_void,
            (d_model * d_model * 2) as usize,
        )?;
        lm.cuda.memcpy_h2d(
            d_wk,
            f32s_to_halves_bytes(&wk).as_ptr() as *const c_void,
            (d_model * d_model * 2) as usize,
        )?;
        lm.cuda.memcpy_h2d(
            d_wv,
            f32s_to_halves_bytes(&wv).as_ptr() as *const c_void,
            (d_model * d_model * 2) as usize,
        )?;
        lm.cuda.memcpy_h2d(
            d_wo,
            f32s_to_halves_bytes(&wo).as_ptr() as *const c_void,
            (d_model * d_model * 2) as usize,
        )?;
    }

    // MLP weights: gate/up [K,H], down [H,N]
    let w_gate: Vec<f32> = (0..(d_model * hidden_dim) as usize)
        .map(|i| ((i as f32) * 0.015).sin())
        .collect();
    let w_up: Vec<f32> = (0..(d_model * hidden_dim) as usize)
        .map(|i| ((i as f32) * 0.017).cos())
        .collect();
    let w_down: Vec<f32> = (0..(hidden_dim * d_model) as usize)
        .map(|i| ((i as f32) * 0.019).sin())
        .collect();

    let d_w_gate = lm.cuda.device_malloc((d_model * hidden_dim * 2) as usize)?;
    let d_w_up = lm.cuda.device_malloc((d_model * hidden_dim * 2) as usize)?;
    let d_w_down = lm.cuda.device_malloc((hidden_dim * d_model * 2) as usize)?;

    unsafe {
        lm.cuda.memcpy_h2d(
            d_w_gate,
            f32s_to_halves_bytes(&w_gate).as_ptr() as *const c_void,
            (d_model * hidden_dim * 2) as usize,
        )?;
        lm.cuda.memcpy_h2d(
            d_w_up,
            f32s_to_halves_bytes(&w_up).as_ptr() as *const c_void,
            (d_model * hidden_dim * 2) as usize,
        )?;
        lm.cuda.memcpy_h2d(
            d_w_down,
            f32s_to_halves_bytes(&w_down).as_ptr() as *const c_void,
            (hidden_dim * d_model * 2) as usize,
        )?;
    }

    let d_out = lm.cuda.device_malloc((d_model * 4) as usize)?;

    unsafe {
        lm.forward_one_token_minimal(
            d_x as *const c_void,
            d_model,
            d_wq as *const c_void,
            d_wk as *const c_void,
            d_wv as *const c_void,
            d_wo as *const c_void,
            d_w_gate as *const c_void,
            d_w_up as *const c_void,
            d_w_down as *const c_void,
            hidden_dim,
            0,
            1,
            d_out,
        )?;
    }

    // Read back and do a minimal sanity check (finite values)
    let mut out_host = vec![0u8; (d_model * 4) as usize];
    unsafe {
        lm.cuda.memcpy_d2h(
            out_host.as_mut_ptr() as *mut c_void,
            d_out as *const c_void,
            out_host.len(),
        )?;
    }
    let out_vals: Vec<f32> = out_host
        .chunks_exact(4)
        .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
        .collect();
    assert_eq!(out_vals.len(), d_model as usize);
    for (i, v) in out_vals.iter().enumerate() {
        assert!(v.is_finite(), "non-finite at {}: {}", i, v);
    }

    unsafe {
        lm.cuda.device_free(d_x)?;
        lm.cuda.device_free(d_wq)?;
        lm.cuda.device_free(d_wk)?;
        lm.cuda.device_free(d_wv)?;
        lm.cuda.device_free(d_wo)?;
        lm.cuda.device_free(d_w_gate)?;
        lm.cuda.device_free(d_w_up)?;
        lm.cuda.device_free(d_w_down)?;
        lm.cuda.device_free(d_out)?;
    }

    // Drop model (frees KV cache)
    lm.kv_cache.take();

    Ok(())
}

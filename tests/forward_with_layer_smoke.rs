#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use m40_llm::gguf::{GgmlDType, GgufModel, GgufScalar, GgufTensor, GgufValue};
use m40_llm::infer::LoadedModel;
use std::ffi::c_void;

fn halves_from_f32(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bits = half::f16::from_f32(v).to_bits();
        out.push((bits & 0xFF) as u8);
        out.push((bits >> 8) as u8);
    }
    out
}

#[test]
fn forward_one_token_with_layer_smoke() -> Result<()> {
    let ctx = cuda_env::ctx_m40()?;
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    // Tiny dims ensuring num_heads * head_dim = d_model
    let d_model = 8usize;
    let hidden = 16usize;
    let num_heads = 2u32;
    let head_dim = 4u32;

    // Build a minimal GGUF in-memory model with required tensors
    let mut gg = GgufModel::new(0);
    gg.metadata.insert(
        "llama.embedding_length".to_string(),
        GgufValue::Scalar(GgufScalar::U32(d_model as u32)),
    );

    // Token embeddings: [vocab, d_model]
    let vocab = 32u64;
    gg.tensors.push(GgufTensor {
        name: "tok_embeddings.weight".into(),
        dtype: GgmlDType::F16,
        shape: vec![vocab, d_model as u64],
        offset: 0,
    });

    // Layer 0 weights in layers.* scheme
    gg.tensors.push(GgufTensor {
        name: "layers.0.attention.wq.weight".into(),
        dtype: GgmlDType::F16,
        shape: vec![d_model as u64, d_model as u64],
        offset: 0,
    });
    gg.tensors.push(GgufTensor {
        name: "layers.0.attention.wk.weight".into(),
        dtype: GgmlDType::F16,
        shape: vec![d_model as u64, d_model as u64],
        offset: 0,
    });
    gg.tensors.push(GgufTensor {
        name: "layers.0.attention.wv.weight".into(),
        dtype: GgmlDType::F16,
        shape: vec![d_model as u64, d_model as u64],
        offset: 0,
    });
    gg.tensors.push(GgufTensor {
        name: "layers.0.attention.wo.weight".into(),
        dtype: GgmlDType::F16,
        shape: vec![d_model as u64, d_model as u64],
        offset: 0,
    });

    gg.tensors.push(GgufTensor {
        name: "layers.0.feed_forward.w3.weight".into(),
        dtype: GgmlDType::F16,
        shape: vec![d_model as u64, hidden as u64],
        offset: 0,
    });
    gg.tensors.push(GgufTensor {
        name: "layers.0.feed_forward.w1.weight".into(),
        dtype: GgmlDType::F16,
        shape: vec![d_model as u64, hidden as u64],
        offset: 0,
    });
    gg.tensors.push(GgufTensor {
        name: "layers.0.feed_forward.w2.weight".into(),
        dtype: GgmlDType::F16,
        shape: vec![hidden as u64, d_model as u64],
        offset: 0,
    });

    // Prepare a contiguous weights blob with simple patterns; record offsets into gg
    let mut weights: Vec<u8> = Vec::new();
    let mut push_tensor = |name: &str, n_elems: usize, gen: &mut dyn FnMut(usize) -> f32| {
        let off = weights.len() as u64;
        if let Some(t) = gg.tensors.iter_mut().find(|t| t.name == name) {
            t.offset = off;
        }
        let mut vals = Vec::with_capacity(n_elems);
        for i in 0..n_elems {
            vals.push(gen(i));
        }
        weights.extend_from_slice(&halves_from_f32(&vals));
    };

    // tok_embeddings: vocab * d_model
    push_tensor(
        "tok_embeddings.weight",
        (vocab as usize) * d_model,
        &mut |i| ((i as f32) * 0.001).sin(),
    );

    // Attention weights: each d_model*d_model
    let sq = d_model * d_model;
    push_tensor("layers.0.attention.wq.weight", sq, &mut |i| {
        ((i as f32) * 0.01).sin()
    });
    push_tensor("layers.0.attention.wk.weight", sq, &mut |i| {
        ((i as f32) * 0.02).cos()
    });
    push_tensor("layers.0.attention.wv.weight", sq, &mut |i| {
        ((i as f32) * 0.03).tan().atan()
    });
    push_tensor("layers.0.attention.wo.weight", sq, &mut |i| {
        ((i as f32) * 0.04).sin()
    });

    // MLP weights
    push_tensor(
        "layers.0.feed_forward.w3.weight",
        d_model * hidden,
        &mut |i| ((i as f32) * 0.015).sin(),
    );
    push_tensor(
        "layers.0.feed_forward.w1.weight",
        d_model * hidden,
        &mut |i| ((i as f32) * 0.017).cos(),
    );
    push_tensor(
        "layers.0.feed_forward.w2.weight",
        hidden * d_model,
        &mut |i| ((i as f32) * 0.019).sin(),
    );

    let mut lm = LoadedModel::from_gguf(gg, weights, -1)?; // auto-select device

    // Allocate KV cache with matching heads
    // Allocate KV cache with explicit heads and head_dim to match d_model
    lm.allocate_kv_cache_with_layout(8, 1, num_heads, head_dim)?;

    // Load embedding for token 0
    let d_x = lm.cuda.device_malloc(d_model * 4)?;
    unsafe {
        lm.load_token_embedding_f16_to_f32(0, d_x)?;
    }

    // Output buffer
    let d_out = lm.cuda.device_malloc(d_model * 4)?;

    // Run one layer
    unsafe {
        let _ = lm.forward_one_token_with_layer(d_x as *const c_void, 0, 0, 1, d_out)?;
    }

    // Read back and assert finiteness
    let mut out_host = vec![0u8; d_model * 4];
    unsafe {
        lm.cuda
            .memcpy_d2h(out_host.as_mut_ptr() as *mut c_void, d_out, d_model * 4)?;
    }
    let out_vals: Vec<f32> = out_host
        .chunks_exact(4)
        .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
        .collect();
    assert_eq!(out_vals.len(), d_model);
    for (i, v) in out_vals.iter().enumerate() {
        assert!(v.is_finite(), "non-finite at {}: {}", i, v);
    }

    unsafe {
        lm.cuda.device_free(d_x)?;
        lm.cuda.device_free(d_out)?;
    }

    Ok(())
}

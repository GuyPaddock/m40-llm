#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;
#[path = "common/tiny_gguf.rs"]
mod tiny_gguf;

use anyhow::Result;
use m40_llm::cuda::CudaStream;
use m40_llm::decode_session::DecodeSession;
use m40_llm::gguf::{GgmlDType, GgufModel, GgufScalar, GgufTensor, GgufValue};
use m40_llm::infer::LoadedModel;
use m40_llm::kv_compression::{
    set_runtime_config, KvCompressMode, KvCompressionConfig, KvRepresentativePolicy,
    ScopedRuntimeConfig,
};
use m40_llm::profile;
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
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    profile::reset();

    // Tiny dims ensuring num_heads * head_dim = d_model
    let d_model = 8usize;
    let hidden = 16usize;
    let num_heads = 2u32;
    let head_dim = 4u32;

    // Build a minimal GGUF in-memory model with required tensors
    let mut gg = GgufModel::new(0);
    gg.metadata.insert(
        "general.architecture".to_string(),
        GgufValue::Scalar(GgufScalar::Str("llama".to_string())),
    );
    gg.metadata.insert(
        "llama.embedding_length".to_string(),
        GgufValue::Scalar(GgufScalar::U32(d_model as u32)),
    );
    gg.metadata.insert(
        "llama.attention.head_count".to_string(),
        GgufValue::Scalar(GgufScalar::U32(num_heads)),
    );
    gg.metadata.insert(
        "llama.block_count".to_string(),
        GgufValue::Scalar(GgufScalar::U32(1)),
    );
    gg.metadata.insert(
        "llama.context_length".to_string(),
        GgufValue::Scalar(GgufScalar::U32(16)),
    );
    gg.metadata.insert(
        "llama.feed_forward_length".to_string(),
        GgufValue::Scalar(GgufScalar::U32(hidden as u32)),
    );
    gg.metadata.insert(
        "llama.layer_norm_epsilon".to_string(),
        GgufValue::Scalar(GgufScalar::F32(1e-6)),
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
    lm.allocate_kv_cache_with_layout(8, 2, num_heads, head_dim)?;

    // Load embedding for token 0
    let d_x = lm.cuda.device_malloc(d_model * 4)?;
    unsafe {
        lm.load_token_embedding_f16_to_f32(0, d_x)?;
    }

    // Output buffer
    let d_out = lm.cuda.device_malloc(d_model * 4)?;

    // Run one layer
    unsafe {
        lm.forward_one_token_with_layer(d_x as *const c_void, 0, 0, 1, d_out)?;
    }
    let bytes_after_first = m40_llm::cuda::CudaContext::total_device_bytes();
    unsafe {
        lm.forward_one_token_with_layer(d_out as *const c_void, 0, 0, 2, d_x)?;
    }
    unsafe {
        lm.forward_one_token_with_layer(d_x as *const c_void, 0, 1, 1, d_out)?;
        lm.load_token_embedding_f16_to_f32(0, d_x)?;
    }
    let bytes_after_second = m40_llm::cuda::CudaContext::total_device_bytes();
    assert_eq!(
        bytes_after_second, bytes_after_first,
        "forward workspace should be reused for identical dimensions"
    );

    // Read back and assert finiteness
    let mut out_host = vec![0u8; d_model * 4];
    unsafe {
        lm.cuda
            .memcpy_d2h(out_host.as_mut_ptr() as *mut c_void, d_x, d_model * 4)?;
    }
    let out_vals: Vec<f32> = out_host
        .chunks_exact(4)
        .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
        .collect();
    assert_eq!(out_vals.len(), d_model);
    for (i, v) in out_vals.iter().enumerate() {
        assert!(v.is_finite(), "non-finite at {}: {}", i, v);
    }
    let snapshot = profile::snapshot();
    let mlp_waits = snapshot
        .by_op
        .get("mlp_gate_up_to_swiglu")
        .map(|counts| counts.stream_waits)
        .unwrap_or_default();
    assert!(
        mlp_waits >= 1,
        "forward must wait for async MLP gate/up GEMMs before SwiGLU reads their outputs"
    );

    unsafe {
        lm.cuda.device_free(d_x)?;
        lm.cuda.device_free(d_out)?;
    }

    Ok(())
}

#[test]
fn forward_batched_decode_uses_packed_attention() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    profile::reset();

    let cfg = tiny_gguf::TinyGgufConfig {
        vocab: 256,
        d_model: 128,
        hidden: 16,
        head_count: 2,
        block_count: 2,
        context_length: 16,
    };
    let (gg, weights) = tiny_gguf::make_identity_tiny_gguf(cfg.clone());
    let mut lm = LoadedModel::from_gguf(gg, weights, -1)?;
    lm.allocate_kv_cache_with_layout(16, 4, 2, 64)?;

    let d_x0 = lm.cuda.device_malloc(128 * 4)?;
    let d_x1 = lm.cuda.device_malloc(128 * 4)?;
    let d_out0 = lm.cuda.device_malloc(128 * 4)?;
    let d_out1 = lm.cuda.device_malloc(128 * 4)?;
    unsafe {
        lm.load_token_embedding_f16_to_f32(1, d_x0)?;
        lm.load_token_embedding_f16_to_f32(2, d_x1)?;
        lm.forward_one_token_all_layers_batched_for_sequences(&[
            m40_llm::infer::ForwardBatchItem {
                d_x_f32: d_x0,
                sequence_id: 0,
                seq_len: 1,
                d_out_f32: d_out0,
            },
            m40_llm::infer::ForwardBatchItem {
                d_x_f32: d_x1,
                sequence_id: 1,
                seq_len: 1,
                d_out_f32: d_out1,
            },
        ])?;
        lm.cuda.synchronize_stream(CudaStream::Decode)?;
        lm.cuda.synchronize_stream(CudaStream::Prefill)?;
    }

    let snapshot = profile::snapshot();
    let batched_attention_launches = snapshot
        .by_op
        .get("attention_last_token_f32_gqa_batched")
        .map(|counts| counts.launches)
        .unwrap_or_default();
    assert!(
        batched_attention_launches >= 1,
        "batched decode forward should use packed GQA attention"
    );

    let mut out_host = vec![0u8; 128 * 4];
    unsafe {
        lm.cuda
            .memcpy_d2h(out_host.as_mut_ptr() as *mut c_void, d_out0, 128 * 4)?;
    }
    let out_vals: Vec<f32> = out_host
        .chunks_exact(4)
        .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
        .collect();
    assert!(out_vals.iter().all(|v| v.is_finite()));

    unsafe {
        lm.cuda.device_free(d_x0)?;
        lm.cuda.device_free(d_x1)?;
        lm.cuda.device_free(d_out0)?;
        lm.cuda.device_free(d_out1)?;
    }

    Ok(())
}

#[test]
fn forward_batched_decode_supports_head128_attention() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    profile::reset();

    let cfg = tiny_gguf::TinyGgufConfig {
        vocab: 256,
        d_model: 128,
        hidden: 16,
        head_count: 1,
        block_count: 2,
        context_length: 16,
    };
    let (gg, weights) = tiny_gguf::make_identity_tiny_gguf(cfg);
    let mut lm = LoadedModel::from_gguf(gg, weights, -1)?;
    lm.allocate_kv_cache_with_layout(16, 4, 1, 128)?;

    let d_x0 = lm.cuda.device_malloc(128 * 4)?;
    let d_x1 = lm.cuda.device_malloc(128 * 4)?;
    let d_out0 = lm.cuda.device_malloc(128 * 4)?;
    let d_out1 = lm.cuda.device_malloc(128 * 4)?;
    unsafe {
        lm.load_token_embedding_f16_to_f32(1, d_x0)?;
        lm.load_token_embedding_f16_to_f32(2, d_x1)?;
        lm.forward_one_token_all_layers_batched_for_sequences(&[
            m40_llm::infer::ForwardBatchItem {
                d_x_f32: d_x0,
                sequence_id: 0,
                seq_len: 1,
                d_out_f32: d_out0,
            },
            m40_llm::infer::ForwardBatchItem {
                d_x_f32: d_x1,
                sequence_id: 1,
                seq_len: 1,
                d_out_f32: d_out1,
            },
        ])?;
        lm.cuda.synchronize_stream(CudaStream::Decode)?;
        lm.cuda.synchronize_stream(CudaStream::Prefill)?;
    }

    let snapshot = profile::snapshot();
    let batched_attention_launches = snapshot
        .by_op
        .get("attention_last_token_f32_gqa_batched")
        .map(|counts| counts.launches)
        .unwrap_or_default();
    assert!(
        batched_attention_launches >= 1,
        "head128 batched decode forward should use packed GQA attention"
    );

    let mut out_host = vec![0u8; 128 * 4];
    unsafe {
        lm.cuda
            .memcpy_d2h(out_host.as_mut_ptr() as *mut c_void, d_out0, 128 * 4)?;
    }
    let out_vals: Vec<f32> = out_host
        .chunks_exact(4)
        .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
        .collect();
    assert!(out_vals.iter().all(|v| v.is_finite()));

    unsafe {
        lm.cuda.device_free(d_x0)?;
        lm.cuda.device_free(d_x1)?;
        lm.cuda.device_free(d_out0)?;
        lm.cuda.device_free(d_out1)?;
    }

    Ok(())
}

#[test]
fn forward_batched_prefill_uses_varlen_attention() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    profile::reset();

    let cfg = tiny_gguf::TinyGgufConfig {
        vocab: 256,
        d_model: 128,
        hidden: 16,
        head_count: 2,
        block_count: 2,
        context_length: 16,
    };
    let (gg, weights) = tiny_gguf::make_identity_tiny_gguf(cfg.clone());
    let (gg_seq, weights_seq) = tiny_gguf::make_identity_tiny_gguf(cfg);
    let mut lm = LoadedModel::from_gguf(gg, weights, -1)?;
    lm.allocate_kv_cache_with_layout(16, 4, 2, 64)?;
    let mut lm_seq = LoadedModel::from_gguf(gg_seq, weights_seq, -1)?;
    lm_seq.allocate_kv_cache_with_layout(16, 4, 2, 64)?;

    let d_out0 = lm.cuda.device_malloc(128 * 4)?;
    let d_out1 = lm.cuda.device_malloc(128 * 4)?;
    let d_seq_x = lm_seq.cuda.device_malloc(128 * 4)?;
    let d_seq_out0 = lm_seq.cuda.device_malloc(128 * 4)?;
    let d_seq_out1 = lm_seq.cuda.device_malloc(128 * 4)?;
    let seq0 = [1, 2, 3];
    let seq1 = [4, 5];
    unsafe {
        lm.forward_prefill_all_layers_varlen_for_sequences(&[
            m40_llm::infer::ForwardPrefillSequence {
                token_ids: &seq0,
                sequence_id: 0,
                d_out_f32: d_out0,
            },
            m40_llm::infer::ForwardPrefillSequence {
                token_ids: &seq1,
                sequence_id: 1,
                d_out_f32: d_out1,
            },
        ])?;
        lm.cuda.synchronize_stream(CudaStream::Decode)?;
        lm.cuda.synchronize_stream(CudaStream::Prefill)?;

        for (idx, &tok) in seq0.iter().enumerate() {
            lm_seq.load_token_embedding_f16_to_f32(tok as u64, d_seq_x)?;
            lm_seq.forward_one_token_all_layers_for_sequence(
                d_seq_x,
                0,
                (idx + 1) as u32,
                d_seq_out0,
            )?;
        }
        for (idx, &tok) in seq1.iter().enumerate() {
            lm_seq.load_token_embedding_f16_to_f32(tok as u64, d_seq_x)?;
            lm_seq.forward_one_token_all_layers_for_sequence(
                d_seq_x,
                1,
                (idx + 1) as u32,
                d_seq_out1,
            )?;
        }
        lm_seq.cuda.synchronize_stream(CudaStream::Decode)?;
        lm_seq.cuda.synchronize_stream(CudaStream::Prefill)?;
    }

    let snapshot = profile::snapshot();
    let prefill_attention_launches = snapshot
        .by_op
        .get("attention_prefill_f32_gqa_varlen")
        .map(|counts| counts.launches)
        .unwrap_or_default();
    assert!(
        prefill_attention_launches >= 1,
        "batched prefill forward should use packed varlen GQA attention"
    );

    let mut out_host = vec![0u8; 128 * 4];
    let mut seq_host = vec![0u8; 128 * 4];
    unsafe {
        lm.cuda
            .memcpy_d2h(out_host.as_mut_ptr() as *mut c_void, d_out0, 128 * 4)?;
        lm_seq
            .cuda
            .memcpy_d2h(seq_host.as_mut_ptr() as *mut c_void, d_seq_out0, 128 * 4)?;
        lm.cuda.device_free(d_out0)?;
        lm.cuda.device_free(d_out1)?;
        lm_seq.cuda.device_free(d_seq_x)?;
        lm_seq.cuda.device_free(d_seq_out0)?;
        lm_seq.cuda.device_free(d_seq_out1)?;
    }
    let out_vals: Vec<f32> = out_host
        .chunks_exact(4)
        .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
        .collect();
    assert!(out_vals.iter().all(|v| v.is_finite()));
    let seq_vals: Vec<f32> = seq_host
        .chunks_exact(4)
        .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
        .collect();
    for (idx, (packed, sequential)) in out_vals.iter().zip(seq_vals.iter()).enumerate() {
        let diff = (packed - sequential).abs();
        assert!(
            diff <= 2e-3,
            "packed prefill differs from sequential decode at {idx}: packed={packed} sequential={sequential} diff={diff}"
        );
    }

    Ok(())
}

fn assert_close_logits(lhs: &[f32], rhs: &[f32], tol: f32) {
    assert_eq!(lhs.len(), rhs.len());
    for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff <= tol,
            "logit mismatch at {idx}: lhs={a} rhs={b} diff={diff} tol={tol}"
        );
    }
}

fn run_packed_prefill_logit_parity() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let cfg = tiny_gguf::TinyGgufConfig {
        vocab: 256,
        d_model: 128,
        hidden: 16,
        head_count: 2,
        block_count: 2,
        context_length: 32,
    };
    let (gg_seq, weights_seq) = tiny_gguf::make_identity_tiny_gguf(cfg.clone());
    let (gg_packed, weights_packed) = tiny_gguf::make_identity_tiny_gguf(cfg.clone());
    let mut sequential = LoadedModel::from_gguf(gg_seq, weights_seq, -1)?;
    let mut packed = LoadedModel::from_gguf(gg_packed, weights_packed, -1)?;
    sequential.allocate_kv_cache_with_layout(cfg.context_length, cfg.block_count, 2, 64)?;
    packed.allocate_kv_cache_with_layout(cfg.context_length, cfg.block_count, 2, 64)?;

    let ids = [3, 4, 5, 6];
    let mut sequential_session = DecodeSession::new_for_sequence(
        &sequential,
        0,
        cfg.d_model,
        true,
        "test_seq_prefill",
        "test_seq_prefill:x",
        "test_seq_prefill:out",
    )?;
    let mut packed_session = DecodeSession::new_for_sequence(
        &packed,
        0,
        cfg.d_model,
        true,
        "test_packed_prefill",
        "test_packed_prefill:x",
        "test_packed_prefill:out",
    )?;
    let sequential_logits = sequential_session.logits_for_ids(&ids, |_| {})?;
    let packed_logits = packed_session.logits_for_packed_prefix_then_ids(&ids, |_| {})?;
    sequential.cuda.synchronize_stream(CudaStream::Decode)?;
    sequential.cuda.synchronize_stream(CudaStream::Prefill)?;
    packed.cuda.synchronize_stream(CudaStream::Decode)?;
    packed.cuda.synchronize_stream(CudaStream::Prefill)?;
    assert_close_logits(&packed_logits, &sequential_logits, 2e-3);
    assert_eq!(packed_session.processed_len(), ids.len());
    Ok(())
}

#[test]
fn packed_prefill_logits_match_sequential_dense() -> Result<()> {
    run_packed_prefill_logit_parity()
}

#[test]
fn packed_prefill_logits_match_sequential_with_qkv_biases() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let cfg = tiny_gguf::TinyGgufConfig {
        vocab: 256,
        d_model: 128,
        hidden: 16,
        head_count: 2,
        block_count: 2,
        context_length: 32,
    };
    let (gg_seq, weights_seq) = tiny_gguf::make_attention_bias_tiny_gguf(cfg.clone());
    let (gg_packed, weights_packed) = tiny_gguf::make_attention_bias_tiny_gguf(cfg.clone());
    let mut sequential = LoadedModel::from_gguf(gg_seq, weights_seq, -1)?;
    let mut packed = LoadedModel::from_gguf(gg_packed, weights_packed, -1)?;
    sequential.allocate_kv_cache_with_layout(cfg.context_length, cfg.block_count, 2, 64)?;
    packed.allocate_kv_cache_with_layout(cfg.context_length, cfg.block_count, 2, 64)?;

    let ids = [3, 4, 5, 6];
    let mut sequential_session = DecodeSession::new_for_sequence(
        &sequential,
        0,
        cfg.d_model,
        true,
        "test_seq_bias_prefill",
        "test_seq_bias_prefill:x",
        "test_seq_bias_prefill:out",
    )?;
    let mut packed_session = DecodeSession::new_for_sequence(
        &packed,
        0,
        cfg.d_model,
        true,
        "test_packed_bias_prefill",
        "test_packed_bias_prefill:x",
        "test_packed_bias_prefill:out",
    )?;
    let sequential_logits = sequential_session.logits_for_ids(&ids, |_| {})?;
    let packed_logits = packed_session.logits_for_packed_prefix_then_ids(&ids, |_| {})?;
    sequential.cuda.synchronize_stream(CudaStream::Decode)?;
    sequential.cuda.synchronize_stream(CudaStream::Prefill)?;
    packed.cuda.synchronize_stream(CudaStream::Decode)?;
    packed.cuda.synchronize_stream(CudaStream::Prefill)?;
    assert_close_logits(&packed_logits, &sequential_logits, 2e-3);
    assert!(
        packed_logits.iter().any(|value| value.abs() > 0.01),
        "biased attention fixture should produce non-zero logits"
    );
    assert_eq!(packed_session.processed_len(), ids.len());
    Ok(())
}

#[test]
fn qwen_head128_packed_prefill_logits_match_sequential_with_qkv_biases() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let cfg = tiny_gguf::TinyGgufConfig {
        vocab: 256,
        d_model: 128,
        hidden: 16,
        head_count: 1,
        block_count: 2,
        context_length: 32,
    };
    let _runtime_guard = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    let (gg_seq, weights_seq) = tiny_gguf::make_qwen2_attention_bias_tiny_gguf(cfg.clone());
    let (gg_packed, weights_packed) = tiny_gguf::make_qwen2_attention_bias_tiny_gguf(cfg.clone());
    let mut sequential = LoadedModel::from_gguf(gg_seq, weights_seq, -1)?;
    let mut packed = LoadedModel::from_gguf(gg_packed, weights_packed, -1)?;
    sequential.allocate_kv_cache_with_layout(cfg.context_length, cfg.block_count, 1, 128)?;
    packed.allocate_kv_cache_with_layout(cfg.context_length, cfg.block_count, 1, 128)?;

    assert_eq!(sequential.model_config.architecture, "qwen2");
    assert_eq!(sequential.model_config.attention_key_length, 128);
    assert_eq!(sequential.model_config.rope_layout_code(), 1);

    let ids = [3, 4, 5, 6];
    let mut sequential_session = DecodeSession::new_for_sequence(
        &sequential,
        0,
        cfg.d_model,
        true,
        "test_seq_qwen_head128_prefill",
        "test_seq_qwen_head128_prefill:x",
        "test_seq_qwen_head128_prefill:out",
    )?;
    let mut packed_session = DecodeSession::new_for_sequence(
        &packed,
        0,
        cfg.d_model,
        true,
        "test_packed_qwen_head128_prefill",
        "test_packed_qwen_head128_prefill:x",
        "test_packed_qwen_head128_prefill:out",
    )?;
    let sequential_logits = sequential_session.logits_for_ids(&ids, |_| {})?;
    let packed_logits = packed_session.logits_for_packed_prefix_then_ids(&ids, |_| {})?;
    sequential.cuda.synchronize_stream(CudaStream::Decode)?;
    sequential.cuda.synchronize_stream(CudaStream::Prefill)?;
    packed.cuda.synchronize_stream(CudaStream::Decode)?;
    packed.cuda.synchronize_stream(CudaStream::Prefill)?;
    assert_close_logits(&packed_logits, &sequential_logits, 2e-3);
    assert!(
        packed_logits.iter().any(|value| value.abs() > 0.01),
        "Qwen-style biased attention fixture should produce non-zero logits"
    );
    assert_eq!(packed_session.processed_len(), ids.len());
    Ok(())
}

#[test]
fn qwen_head128_batched_prefill_matches_sequential_with_qkv_biases() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    profile::reset();

    let cfg = tiny_gguf::TinyGgufConfig {
        vocab: 256,
        d_model: 128,
        hidden: 16,
        head_count: 1,
        block_count: 2,
        context_length: 32,
    };
    let _runtime_guard = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    let (gg, weights) = tiny_gguf::make_qwen2_attention_bias_tiny_gguf(cfg.clone());
    let (gg_seq, weights_seq) = tiny_gguf::make_qwen2_attention_bias_tiny_gguf(cfg.clone());
    let mut lm = LoadedModel::from_gguf(gg, weights, -1)?;
    lm.allocate_kv_cache_with_layout(cfg.context_length, cfg.block_count * 2, 1, 128)?;
    let mut lm_seq = LoadedModel::from_gguf(gg_seq, weights_seq, -1)?;
    lm_seq.allocate_kv_cache_with_layout(cfg.context_length, cfg.block_count * 2, 1, 128)?;

    let d_out0 = lm.cuda.device_malloc(128 * 4)?;
    let d_out1 = lm.cuda.device_malloc(128 * 4)?;
    let d_seq_x = lm_seq.cuda.device_malloc(128 * 4)?;
    let d_seq_out0 = lm_seq.cuda.device_malloc(128 * 4)?;
    let d_seq_out1 = lm_seq.cuda.device_malloc(128 * 4)?;
    let seq0 = [1, 2, 3, 4];
    let seq1 = [5, 6];
    unsafe {
        lm.forward_prefill_all_layers_varlen_for_sequences(&[
            m40_llm::infer::ForwardPrefillSequence {
                token_ids: &seq0,
                sequence_id: 0,
                d_out_f32: d_out0,
            },
            m40_llm::infer::ForwardPrefillSequence {
                token_ids: &seq1,
                sequence_id: 1,
                d_out_f32: d_out1,
            },
        ])?;
        lm.cuda.synchronize_stream(CudaStream::Decode)?;
        lm.cuda.synchronize_stream(CudaStream::Prefill)?;

        for (idx, &tok) in seq0.iter().enumerate() {
            lm_seq.load_token_embedding_f16_to_f32(tok as u64, d_seq_x)?;
            lm_seq.forward_one_token_all_layers_for_sequence(
                d_seq_x,
                0,
                (idx + 1) as u32,
                d_seq_out0,
            )?;
        }
        for (idx, &tok) in seq1.iter().enumerate() {
            lm_seq.load_token_embedding_f16_to_f32(tok as u64, d_seq_x)?;
            lm_seq.forward_one_token_all_layers_for_sequence(
                d_seq_x,
                1,
                (idx + 1) as u32,
                d_seq_out1,
            )?;
        }
        lm_seq.cuda.synchronize_stream(CudaStream::Decode)?;
        lm_seq.cuda.synchronize_stream(CudaStream::Prefill)?;
    }

    let snapshot = profile::snapshot();
    let prefill_attention_launches = snapshot
        .by_op
        .get("attention_prefill_f32_gqa_varlen")
        .map(|counts| counts.launches)
        .unwrap_or_default();
    assert!(
        prefill_attention_launches >= 1,
        "Qwen head128 batched prefill should use packed varlen GQA attention"
    );

    let mut out0_host = vec![0u8; 128 * 4];
    let mut out1_host = vec![0u8; 128 * 4];
    let mut seq0_host = vec![0u8; 128 * 4];
    let mut seq1_host = vec![0u8; 128 * 4];
    unsafe {
        lm.cuda
            .memcpy_d2h(out0_host.as_mut_ptr() as *mut c_void, d_out0, 128 * 4)?;
        lm.cuda
            .memcpy_d2h(out1_host.as_mut_ptr() as *mut c_void, d_out1, 128 * 4)?;
        lm_seq
            .cuda
            .memcpy_d2h(seq0_host.as_mut_ptr() as *mut c_void, d_seq_out0, 128 * 4)?;
        lm_seq
            .cuda
            .memcpy_d2h(seq1_host.as_mut_ptr() as *mut c_void, d_seq_out1, 128 * 4)?;
        lm.cuda.device_free(d_out0)?;
        lm.cuda.device_free(d_out1)?;
        lm_seq.cuda.device_free(d_seq_x)?;
        lm_seq.cuda.device_free(d_seq_out0)?;
        lm_seq.cuda.device_free(d_seq_out1)?;
    }

    for (label, packed_bytes, sequential_bytes) in [
        ("seq0", out0_host.as_slice(), seq0_host.as_slice()),
        ("seq1", out1_host.as_slice(), seq1_host.as_slice()),
    ] {
        let packed_vals: Vec<f32> = packed_bytes
            .chunks_exact(4)
            .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
            .collect();
        let seq_vals: Vec<f32> = sequential_bytes
            .chunks_exact(4)
            .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
            .collect();
        for (idx, (packed, sequential)) in packed_vals.iter().zip(seq_vals.iter()).enumerate() {
            let diff = (packed - sequential).abs();
            assert!(
                diff <= 2e-3,
                "{label} Qwen head128 packed prefill differs from sequential decode at {idx}: packed={packed} sequential={sequential} diff={diff}"
            );
        }
    }

    Ok(())
}

fn run_compressed_chunked_prefill_logit_parity(mode: KvCompressMode) -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    struct ConfigGuard;
    impl Drop for ConfigGuard {
        fn drop(&mut self) {
            set_runtime_config(KvCompressionConfig::dense_reference());
        }
    }

    let config = KvCompressionConfig {
        mode,
        recent_window: 4,
        block_size: 2,
        top_blocks: 2,
        representatives: 0,
        representative_policy: Default::default(),
        ..KvCompressionConfig::for_mode(mode)
    };
    set_runtime_config(config.clone());
    let _guard = ConfigGuard;

    let cfg = tiny_gguf::TinyGgufConfig {
        vocab: 256,
        d_model: 128,
        hidden: 16,
        head_count: 2,
        block_count: 2,
        context_length: 16,
    };
    let (gg_seq, weights_seq) = tiny_gguf::make_identity_tiny_gguf(cfg.clone());
    let (gg_chunked, weights_chunked) = tiny_gguf::make_identity_tiny_gguf(cfg.clone());
    let mut sequential = LoadedModel::from_gguf(gg_seq, weights_seq, -1)?;
    let mut chunked = LoadedModel::from_gguf(gg_chunked, weights_chunked, -1)?;
    sequential.allocate_compressed_kv_cache_for_layers(cfg.context_length, &config)?;
    chunked.allocate_compressed_kv_cache_for_layers(cfg.context_length, &config)?;

    let ids = [3, 4, 5, 6, 7, 8, 9, 10, 11];
    let mut sequential_session = DecodeSession::new_for_sequence(
        &sequential,
        0,
        cfg.d_model,
        true,
        "test_seq_compressed_prefill",
        "test_seq_compressed_prefill:x",
        "test_seq_compressed_prefill:out",
    )?;
    let mut chunked_session = DecodeSession::new_for_sequence(
        &chunked,
        0,
        cfg.d_model,
        true,
        "test_chunked_compressed_prefill",
        "test_chunked_compressed_prefill:x",
        "test_chunked_compressed_prefill:out",
    )?;
    let sequential_logits = sequential_session.logits_for_ids(&ids, |_| {})?;
    let chunked_logits =
        chunked_session.logits_for_compressed_chunked_prefill_ids(&ids, 3, |_| {})?;
    sequential.cuda.synchronize_stream(CudaStream::Decode)?;
    sequential.cuda.synchronize_stream(CudaStream::Prefill)?;
    chunked.cuda.synchronize_stream(CudaStream::Decode)?;
    chunked.cuda.synchronize_stream(CudaStream::Prefill)?;

    assert_close_logits(&chunked_logits, &sequential_logits, 2e-3);
    assert_eq!(chunked_session.processed_len(), ids.len());
    assert_eq!(sequential_session.processed_len(), ids.len());

    let sequential_kv = sequential
        .kv_cache
        .as_ref()
        .expect("sequential compressed kv cache");
    let chunked_kv = chunked
        .kv_cache
        .as_ref()
        .expect("chunked compressed kv cache");
    for physical_seq in 0..cfg.block_count {
        let seq_snapshot =
            sequential_kv.debug_compressed_snapshot(&sequential.cuda, physical_seq)?;
        let chunk_snapshot = chunked_kv.debug_compressed_snapshot(&chunked.cuda, physical_seq)?;
        assert_eq!(
            chunk_snapshot, seq_snapshot,
            "compressed KV snapshot mismatch for physical sequence {physical_seq}"
        );
    }
    Ok(())
}

fn run_packed_then_compress_prefill_logit_parity(
    mode: KvCompressMode,
    representatives: u32,
    representative_policy: KvRepresentativePolicy,
) -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    struct ConfigGuard;
    impl Drop for ConfigGuard {
        fn drop(&mut self) {
            set_runtime_config(KvCompressionConfig::dense_reference());
        }
    }

    let config = KvCompressionConfig {
        mode,
        recent_window: 4,
        block_size: 2,
        top_blocks: 2,
        representatives,
        representative_policy,
        ..KvCompressionConfig::for_mode(mode)
    };
    set_runtime_config(config.clone());
    let _guard = ConfigGuard;

    let cfg = tiny_gguf::TinyGgufConfig {
        vocab: 256,
        d_model: 128,
        hidden: 16,
        head_count: 2,
        block_count: 2,
        context_length: 16,
    };
    let (gg_seq, weights_seq) = tiny_gguf::make_identity_tiny_gguf(cfg.clone());
    let (gg_packed, weights_packed) = tiny_gguf::make_identity_tiny_gguf(cfg.clone());
    let mut sequential = LoadedModel::from_gguf(gg_seq, weights_seq, -1)?;
    let mut packed = LoadedModel::from_gguf(gg_packed, weights_packed, -1)?;
    sequential.allocate_compressed_kv_cache_for_layers(cfg.context_length, &config)?;
    packed.allocate_compressed_kv_cache_for_layers(cfg.context_length, &config)?;

    let ids = [3, 4, 5, 6, 7, 8, 9, 10, 11];
    let mut sequential_session = DecodeSession::new_for_sequence(
        &sequential,
        0,
        cfg.d_model,
        true,
        "test_seq_packed_then_compress",
        "test_seq_packed_then_compress:x",
        "test_seq_packed_then_compress:out",
    )?;
    let mut packed_session = DecodeSession::new_for_sequence(
        &packed,
        0,
        cfg.d_model,
        true,
        "test_packed_then_compress",
        "test_packed_then_compress:x",
        "test_packed_then_compress:out",
    )?;
    let sequential_logits = sequential_session.logits_for_ids(&ids, |_| {})?;
    let (packed_logits, temp_bytes) =
        packed_session.logits_for_packed_then_compress_prefill_ids(&ids, |_| {})?;
    sequential.cuda.synchronize_stream(CudaStream::Decode)?;
    sequential.cuda.synchronize_stream(CudaStream::Prefill)?;
    packed.cuda.synchronize_stream(CudaStream::Decode)?;
    packed.cuda.synchronize_stream(CudaStream::Prefill)?;

    assert!(temp_bytes > 0);
    assert_close_logits(&packed_logits, &sequential_logits, 2e-3);
    assert_eq!(packed_session.processed_len(), ids.len());
    assert_eq!(sequential_session.processed_len(), ids.len());

    let sequential_kv = sequential
        .kv_cache
        .as_ref()
        .expect("sequential compressed kv cache");
    let packed_kv = packed
        .kv_cache
        .as_ref()
        .expect("packed compressed kv cache");
    for physical_seq in 0..cfg.block_count {
        let seq_snapshot =
            sequential_kv.debug_compressed_snapshot(&sequential.cuda, physical_seq)?;
        let packed_snapshot = packed_kv.debug_compressed_snapshot(&packed.cuda, physical_seq)?;
        assert_eq!(
            packed_snapshot, seq_snapshot,
            "packed-then-compress KV snapshot mismatch for physical sequence {physical_seq}"
        );
    }
    Ok(())
}

#[test]
fn compressed_chunked_prefill_matches_sequential_block_summary() -> Result<()> {
    run_compressed_chunked_prefill_logit_parity(KvCompressMode::BlockSummary)
}

#[test]
fn compressed_chunked_prefill_matches_sequential_block_select_lossy() -> Result<()> {
    run_compressed_chunked_prefill_logit_parity(KvCompressMode::BlockSelectLossy)
}

#[test]
fn packed_then_compress_prefill_matches_sequential_block_summary() -> Result<()> {
    run_packed_then_compress_prefill_logit_parity(
        KvCompressMode::BlockSummary,
        0,
        KvRepresentativePolicy::Last,
    )
}

#[test]
fn packed_then_compress_prefill_matches_sequential_block_select_lossy() -> Result<()> {
    run_packed_then_compress_prefill_logit_parity(
        KvCompressMode::BlockSelectLossy,
        0,
        KvRepresentativePolicy::Last,
    )
}

#[test]
fn packed_then_compress_prefill_matches_sequential_block_summary_last_reps() -> Result<()> {
    run_packed_then_compress_prefill_logit_parity(
        KvCompressMode::BlockSummary,
        2,
        KvRepresentativePolicy::Last,
    )
}

#[test]
fn packed_then_compress_prefill_matches_sequential_block_summary_stride_reps() -> Result<()> {
    run_packed_then_compress_prefill_logit_parity(
        KvCompressMode::BlockSummary,
        2,
        KvRepresentativePolicy::Stride,
    )
}

#[test]
fn packed_then_compress_prefill_matches_sequential_block_select_lossy_last_reps() -> Result<()> {
    run_packed_then_compress_prefill_logit_parity(
        KvCompressMode::BlockSelectLossy,
        2,
        KvRepresentativePolicy::Last,
    )
}

#[test]
fn packed_then_compress_prefill_matches_sequential_block_select_lossy_stride_reps() -> Result<()> {
    run_packed_then_compress_prefill_logit_parity(
        KvCompressMode::BlockSelectLossy,
        2,
        KvRepresentativePolicy::Stride,
    )
}

#[test]
fn cuda_graph_replays_forward_one_token_with_layer() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let cfg = tiny_gguf::TinyGgufConfig {
        vocab: 32,
        d_model: 8,
        hidden: 16,
        head_count: 2,
        block_count: 1,
        context_length: 16,
    };
    let (gguf, weights) = tiny_gguf::make_identity_tiny_gguf(cfg.clone());
    let mut lm = LoadedModel::from_gguf(gguf, weights, -1)?;
    let head_dim = cfg.d_model as u32 / cfg.head_count;
    lm.allocate_kv_cache_with_layout(8, 2, cfg.head_count, head_dim)?;

    let bytes = cfg.d_model * std::mem::size_of::<f32>();
    let d_x_normal = lm.cuda.device_malloc(bytes)?;
    let d_x_graph = lm.cuda.device_malloc(bytes)?;
    let d_out_normal = lm.cuda.device_malloc(bytes)?;
    let d_out_graph = lm.cuda.device_malloc(bytes)?;
    unsafe {
        lm.load_token_embedding_f16_to_f32(3, d_x_normal)?;
        lm.load_token_embedding_f16_to_f32(3, d_x_graph)?;

        lm.forward_one_token_with_layer(d_x_normal as *const c_void, 0, 0, 1, d_out_normal)?;
        lm.cuda.synchronize_stream(CudaStream::Decode)?;
        lm.cuda.synchronize_stream(CudaStream::Prefill)?;

        let graph = lm.cuda.capture_graph(CudaStream::Decode, || {
            lm.forward_one_token_with_layer(d_x_graph as *const c_void, 0, 1, 1, d_out_graph)
        })?;
        graph.launch(CudaStream::Decode)?;
        lm.cuda.synchronize_stream(CudaStream::Decode)?;
        lm.cuda.synchronize_stream(CudaStream::Prefill)?;

        let mut normal_bytes = vec![0u8; bytes];
        let mut graph_bytes = vec![0u8; bytes];
        lm.cuda.memcpy_d2h(
            normal_bytes.as_mut_ptr() as *mut c_void,
            d_out_normal,
            bytes,
        )?;
        lm.cuda
            .memcpy_d2h(graph_bytes.as_mut_ptr() as *mut c_void, d_out_graph, bytes)?;
        for (i, (normal, graph)) in normal_bytes
            .chunks_exact(4)
            .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
            .zip(
                graph_bytes
                    .chunks_exact(4)
                    .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]])),
            )
            .enumerate()
        {
            assert!(
                (normal - graph).abs() < 1e-5,
                "graph forward mismatch at {i}: normal={normal}, graph={graph}"
            );
        }

        lm.cuda.device_free(d_x_normal)?;
        lm.cuda.device_free(d_x_graph)?;
        lm.cuda.device_free(d_out_normal)?;
        lm.cuda.device_free(d_out_graph)?;
    }

    Ok(())
}

#[test]
fn decode_session_uses_one_layer_graph_when_enabled() -> Result<()> {
    struct EnvGuard;
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("M40LLM_DECODE_GRAPH");
        }
    }

    std::env::set_var("M40LLM_DECODE_GRAPH", "1");
    let _guard = EnvGuard;
    profile::reset();

    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let cfg = tiny_gguf::TinyGgufConfig {
        vocab: 32,
        d_model: 8,
        hidden: 16,
        head_count: 2,
        block_count: 1,
        context_length: 16,
    };
    let (gguf, weights) = tiny_gguf::make_identity_tiny_gguf(cfg.clone());
    let mut lm = LoadedModel::from_gguf(gguf, weights, -1)?;
    let head_dim = cfg.d_model as u32 / cfg.head_count;
    lm.allocate_kv_cache_with_layout(8, 2, cfg.head_count, head_dim)?;

    let mut session = DecodeSession::new_for_sequence(
        &lm,
        0,
        cfg.d_model,
        true,
        "test_decode_graph",
        "test_decode_graph:x",
        "test_decode_graph:out",
    )?;
    let logits = session.logits_for_ids(&[3, 4], |_| {})?;
    assert_eq!(logits.len(), cfg.vocab);

    let snapshot = profile::snapshot();
    let graph_launches = snapshot
        .by_op
        .get("cuda_graph_launch")
        .map(|counts| counts.launches)
        .unwrap_or_default();
    assert!(
        graph_launches >= 2,
        "expected DecodeSession graph replay launches, got {graph_launches}"
    );

    Ok(())
}

#[test]
fn decode_session_uses_multilayer_graph_when_enabled() -> Result<()> {
    struct EnvGuard;
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("M40LLM_DECODE_GRAPH");
        }
    }

    std::env::set_var("M40LLM_DECODE_GRAPH", "1");
    let _guard = EnvGuard;
    profile::reset();

    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let cfg = tiny_gguf::TinyGgufConfig {
        vocab: 32,
        d_model: 8,
        hidden: 16,
        head_count: 2,
        block_count: 2,
        context_length: 16,
    };
    let (gguf, weights) = tiny_gguf::make_identity_tiny_gguf(cfg.clone());
    let mut lm = LoadedModel::from_gguf(gguf, weights, -1)?;
    let head_dim = cfg.d_model as u32 / cfg.head_count;
    lm.allocate_kv_cache_with_layout(8, cfg.block_count, cfg.head_count, head_dim)?;

    let mut session = DecodeSession::new_for_sequence(
        &lm,
        0,
        cfg.d_model,
        true,
        "test_decode_multilayer_graph",
        "test_decode_multilayer_graph:x",
        "test_decode_multilayer_graph:out",
    )?;
    let logits = session.logits_for_ids(&[3, 4], |_| {})?;
    assert_eq!(logits.len(), cfg.vocab);

    let snapshot = profile::snapshot();
    let graph_launches = snapshot
        .by_op
        .get("cuda_graph_launch")
        .map(|counts| counts.launches)
        .unwrap_or_default();
    assert!(
        graph_launches >= 2,
        "expected multi-layer DecodeSession graph replay launches, got {graph_launches}"
    );

    Ok(())
}

#[test]
fn decode_session_graph_diagnostic_sync_records_timed_replay() -> Result<()> {
    struct EnvGuard;
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("M40LLM_DECODE_GRAPH");
            std::env::remove_var("M40LLM_DECODE_GRAPH_DIAG_SYNC");
            std::env::remove_var("M40LLM_DECODE_GRAPH_DIAG_MAX_MS");
        }
    }

    std::env::set_var("M40LLM_DECODE_GRAPH", "1");
    std::env::set_var("M40LLM_DECODE_GRAPH_DIAG_SYNC", "1");
    let _guard = EnvGuard;
    profile::reset();

    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let cfg = tiny_gguf::TinyGgufConfig {
        vocab: 32,
        d_model: 8,
        hidden: 16,
        head_count: 2,
        block_count: 1,
        context_length: 16,
    };
    let (gguf, weights) = tiny_gguf::make_identity_tiny_gguf(cfg.clone());
    let mut lm = LoadedModel::from_gguf(gguf, weights, -1)?;
    let head_dim = cfg.d_model as u32 / cfg.head_count;
    lm.allocate_kv_cache_with_layout(8, 2, cfg.head_count, head_dim)?;

    let mut session = DecodeSession::new_for_sequence(
        &lm,
        0,
        cfg.d_model,
        true,
        "test_decode_graph_diag",
        "test_decode_graph_diag:x",
        "test_decode_graph_diag:out",
    )?;
    let logits = session.logits_for_ids(&[3, 4], |_| {})?;
    assert_eq!(logits.len(), cfg.vocab);

    let snapshot = profile::snapshot();
    let timed_syncs = snapshot
        .by_op
        .get("cuda_graph_launch_timed_sync")
        .map(|counts| counts.stream_syncs)
        .unwrap_or_default();
    assert!(
        timed_syncs >= 2,
        "expected timed graph launch sync for each replay, got {timed_syncs}"
    );

    let graph_waits = snapshot
        .by_op
        .get("hidden_to_logits_stream")
        .map(|counts| counts.stream_waits)
        .unwrap_or_default();
    assert!(
        graph_waits >= 2,
        "expected explicit hidden->logits stream wait after graph replay, got {graph_waits}"
    );

    Ok(())
}

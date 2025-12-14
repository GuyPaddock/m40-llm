#![cfg(not(feature = "cuda"))]

use anyhow::Result;
use m40_llm::gguf::{GgufModel, GgufScalar, GgufValue};
use m40_llm::infer::LoadedModel;
use std::ffi::c_void;

fn minimal_gguf() -> GgufModel {
    use GgufScalar as S;
    use GgufValue as V;

    let mut gguf = GgufModel::new(0);
    gguf.metadata.insert(
        "general.architecture".into(),
        V::Scalar(S::Str("llama".into())),
    );
    gguf.metadata
        .insert("llama.embedding_length".into(), V::Scalar(S::U32(512)));
    gguf.metadata
        .insert("llama.attention.head_count".into(), V::Scalar(S::U32(8)));
    gguf.metadata
        .insert("llama.block_count".into(), V::Scalar(S::U32(1)));
    gguf.metadata
        .insert("llama.context_length".into(), V::Scalar(S::U32(128)));
    gguf.metadata
        .insert("llama.feed_forward_length".into(), V::Scalar(S::U32(2048)));
    gguf.metadata
        .insert("llama.vocab_size".into(), V::Scalar(S::U32(32_000)));
    gguf
}

#[test]
fn test_model_loading() -> Result<()> {
    let gguf = minimal_gguf();
    let gguf_bytes = vec![];
    let _model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;
    // LoadedModel no longer exposes raw device pointer; construction succeeds in non-CUDA path
    Ok(())
}

#[test]
fn test_kv_cache_allocation() -> Result<()> {
    let gguf = minimal_gguf();
    let gguf_bytes = vec![];
    let mut model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;
    model.allocate_kv_cache(128, 8)?;
    assert!(model.kv_cache.is_some());
    Ok(())
}

#[test]
fn test_attention_operation() -> Result<()> {
    let gguf = minimal_gguf();
    let gguf_bytes = vec![];
    let mut model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;
    // Allocate KV cache with standard layout (8 heads, 64 dim per head)
    model.allocate_kv_cache(128, 8)?;
    // Provide valid input/output buffers
    let dim = 8 * 64; // must match allocate_kv_cache layout
    let q: Vec<f32> = vec![0.0; dim as usize];
    let mut out: Vec<f32> = vec![0.0; dim as usize];
    unsafe {
        model.run_attention(
            q.as_ptr() as *const c_void,
            out.as_mut_ptr() as *mut c_void,
            0,
            4,
            dim as u32,
            8,
            64,
        )?;
    }
    Ok(())
}

#[test]
fn test_mlp_operation() -> Result<()> {
    let gguf = minimal_gguf();
    let gguf_bytes = vec![];
    let model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;
    let input = std::ptr::null::<c_void>();
    let output = std::ptr::null_mut::<c_void>();
    model.run_mlp(input, output, 8, 512, 2048)?;
    Ok(())
}

#[test]
fn test_rms_norm_operation() -> Result<()> {
    let gguf = minimal_gguf();
    let gguf_bytes = vec![];
    let model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;
    let input = std::ptr::null::<c_void>();
    let output = std::ptr::null_mut::<c_void>();
    unsafe {
        model.run_rms_norm(input, output, 128, 512, 1e-5)?;
    }
    Ok(())
}

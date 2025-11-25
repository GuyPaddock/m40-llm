#![cfg(not(feature = "cuda"))]

use anyhow::Result;
use m40_llm::gguf::GgufModel;
use m40_llm::infer::LoadedModel;
use std::ffi::c_void;

#[test]
fn test_model_loading() -> Result<()> {
    let gguf = GgufModel::new(0);
    let gguf_bytes = vec![];
    let _model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;
    // LoadedModel no longer exposes raw device pointer; construction succeeds in non-CUDA path
    Ok(())
}

#[test]
fn test_kv_cache_allocation() -> Result<()> {
    let gguf = GgufModel::new(0);
    let gguf_bytes = vec![];
    let mut model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;
    model.allocate_kv_cache(128, 8)?;
    assert!(model.kv_cache.is_some());
    Ok(())
}

#[test]
fn test_attention_operation() -> Result<()> {
    let gguf = GgufModel::new(0);
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
    let gguf = GgufModel::new(0);
    let gguf_bytes = vec![];
    let model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;
    let input = std::ptr::null::<c_void>();
    let output = std::ptr::null_mut::<c_void>();
    model.run_mlp(input, output, 8, 512, 2048)?;
    Ok(())
}

#[test]
fn test_rms_norm_operation() -> Result<()> {
    let gguf = GgufModel::new(0);
    let gguf_bytes = vec![];
    let model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;
    let input = std::ptr::null::<c_void>();
    let output = std::ptr::null_mut::<c_void>();
    model.run_rms_norm(input, output, 128, 512, 1e-5)?;
    Ok(())
}

#![cfg(not(feature = "cuda"))]

use anyhow::Result;
use m40_llm::gguf::GgufModel;
use m40_llm::infer::LoadedModel;
use std::ffi::c_void;

#[test]
fn test_model_loading() -> Result<()> {
    let gguf = GgufModel::new(0);
    let gguf_bytes = vec![];
    let model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;
    assert_eq!(model.d_data_base, std::ptr::null_mut::<c_void>());
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
    let model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;
    let input = 0 as *const c_void;
    let output = 0 as *mut c_void;
    model.run_attention(input, output, 0, 128, 512, 8, 64)?;
    Ok(())
}

#[test]
fn test_mlp_operation() -> Result<()> {
    let gguf = GgufModel::new(0);
    let gguf_bytes = vec![];
    let model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;
    let input = 0 as *const c_void;
    let output = 0 as *mut c_void;
    model.run_mlp(input, output, 8, 512, 2048)?;
    Ok(())
}

#[test]
fn test_rms_norm_operation() -> Result<()> {
    let gguf = GgufModel::new(0);
    let gguf_bytes = vec![];
    let model = LoadedModel::from_gguf(gguf, gguf_bytes, 0)?;
    let input = 0 as *const c_void;
    let output = 0 as *mut c_void;
    model.run_rms_norm(input, output, 128, 512, 1e-5)?;
    Ok(())
}

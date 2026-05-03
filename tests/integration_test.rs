#![cfg(all(feature = "cuda", nvcc))]

use anyhow::Result;
use m40_llm::gguf::{GgufModel, GgufScalar, GgufValue};
use m40_llm::infer::LoadedModel;
use std::ffi::c_void;

mod cuda_env;

fn cuda_device_or_skip() -> Option<i32> {
    let ctx = cuda_env::ctx_m40_or_skip()?;
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return None;
    }
    Some(-1)
}

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
    let Some(device_id) = cuda_device_or_skip() else {
        return Ok(());
    };
    let gguf = minimal_gguf();
    let gguf_bytes = vec![];
    let _model = LoadedModel::from_gguf(gguf, gguf_bytes, device_id)?;
    // LoadedModel no longer exposes raw device pointer; construction succeeds in non-CUDA path
    Ok(())
}

#[test]
fn test_kv_cache_allocation() -> Result<()> {
    let Some(device_id) = cuda_device_or_skip() else {
        return Ok(());
    };
    let gguf = minimal_gguf();
    let gguf_bytes = vec![];
    let mut model = LoadedModel::from_gguf(gguf, gguf_bytes, device_id)?;
    model.allocate_kv_cache(128, 8)?;
    assert!(model.kv_cache.is_some());
    Ok(())
}

#[test]
fn test_attention_operation() -> Result<()> {
    let Some(device_id) = cuda_device_or_skip() else {
        return Ok(());
    };
    let gguf = minimal_gguf();
    let gguf_bytes = vec![];
    let mut model = LoadedModel::from_gguf(gguf, gguf_bytes, device_id)?;
    // Allocate KV cache with standard layout (8 heads, 64 dim per head)
    model.allocate_kv_cache(128, 8)?;
    let dim = 8 * 64; // must match allocate_kv_cache layout
    let q: Vec<f32> = vec![1.0; dim as usize];
    let k: Vec<f32> = vec![0.5; dim as usize];
    let v: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001 + 1.0).collect();
    model.append_kv_token_f32_from_host(0, &k, &v)?;
    let mut out = vec![0.0f32; dim as usize];
    unsafe {
        let bytes = dim as usize * std::mem::size_of::<f32>();
        let d_q = model.cuda.device_malloc(bytes)?;
        let d_out = model.cuda.device_malloc(bytes)?;
        model
            .cuda
            .memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes)?;
        model.run_attention(d_q as *const c_void, d_out, 0, 1, dim as u32, 8, 64)?;
        model
            .cuda
            .memcpy_d2h(out.as_mut_ptr() as *mut c_void, d_out, bytes)?;
        model.cuda.device_free(d_q)?;
        model.cuda.device_free(d_out)?;
    }
    assert!(!out.iter().all(|&x| x == 0.0)); // Verify computation occurred
    Ok(())
}

#[test]
fn test_mlp_operation() -> Result<()> {
    let Some(device_id) = cuda_device_or_skip() else {
        return Ok(());
    };
    let gguf = minimal_gguf();
    let gguf_bytes = vec![];
    let model = LoadedModel::from_gguf(gguf, gguf_bytes, device_id)?;
    let input = std::ptr::null::<c_void>();
    let output = std::ptr::null_mut::<c_void>();
    model.run_mlp(input, output, 8, 512, 2048)?;
    Ok(())
}

#[test]
fn test_rms_norm_operation() -> Result<()> {
    let Some(device_id) = cuda_device_or_skip() else {
        return Ok(());
    };
    let gguf = minimal_gguf();
    let gguf_bytes = vec![];
    let model = LoadedModel::from_gguf(gguf, gguf_bytes, device_id)?;

    // Initialize host data
    let host_input = vec![1.0; 512];
    let mut host_output = vec![0.0; 512];

    unsafe {
        let elem_size = std::mem::size_of::<f32>();
        let bytes = 512 * elem_size;
        let d_input = model.cuda.device_malloc(bytes)?;
        let d_output = model.cuda.device_malloc(bytes)?;

        // Copy input to device
        let copy_size = 512 * std::mem::size_of::<f32>();
        model
            .cuda
            .memcpy_h2d(d_input, host_input.as_ptr() as *const c_void, copy_size)?;

        // Run kernel
        model.run_rms_norm(
            d_input, d_output, 1,   // seq_len/rows
            512, // dim
            1e-5,
        )?;

        // Copy back results
        let copy_size = 512 * std::mem::size_of::<f32>();
        model
            .cuda
            .memcpy_d2h(host_output.as_mut_ptr() as *mut c_void, d_output, copy_size)?;

        assert!(!host_output.iter().all(|&x| x == 0.0)); // Verify output was modified

        // Free device memory
        model.cuda.device_free(d_input)?;
        model.cuda.device_free(d_output)?;
    }
    Ok(())
}

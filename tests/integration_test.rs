#![cfg(feature = "cuda")]

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
    let q: Vec<f32> = vec![1.0; dim as usize]; // Initialize with non-zero values
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
    assert!(!out.iter().all(|&x| x == 0.0)); // Verify computation occurred
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

    // Initialize host data
    let host_input = vec![1.0; 512];
    let mut host_output = vec![0.0; 512];

    unsafe {
        // Allocate device buffers with validation
        let elem_size = std::mem::size_of::<f32>();
        let bytes = 512 * elem_size;
        println!("[DEBUG] Allocating {} bytes ({} elements)", bytes, 512);

        let d_input = model.cuda.device_malloc(bytes)?;
        eprintln!("[DEBUG] Allocating {} bytes for input", bytes);
        let d_output = model.cuda.device_malloc(bytes)?;

        // Verify allocations succeeded
        println!(
            "[DEBUG] Device pointers - input: {:?}, output: {:?}",
            d_input, d_output
        );
        assert!(!d_input.is_null(), "Failed to allocate input buffer");
        assert!(!d_output.is_null(), "Failed to allocate output buffer");

        // Validate device pointers before copy
        println!(
            "[DEBUG] d_input ptr: {:?}, size: {} bytes",
            d_input,
            512 * std::mem::size_of::<f32>()
        );
        assert!(!d_input.is_null(), "Device input pointer null");

        // Copy input to device
        let copy_size = 512 * std::mem::size_of::<f32>();
        eprintln!(
            "[DEBUG] Pre-memcpy_d2h validation - d_output: {:?}, host_output: {:?}, bytes: {}",
            d_output,
            host_output.as_ptr(),
            bytes
        );
        eprintln!(
            "[DEBUG] Memory alignment - host: {}, device: {}",
            host_output.as_ptr() as usize % 16,
            d_output as usize % 16
        );
        let res = model
            .cuda
            .memcpy_h2d(d_input, host_input.as_ptr() as *const c_void, copy_size);
        println!("[DEBUG] h2d copy result: {:?}", res);
        res?;

        // Validate device pointers before kernel
        println!(
            "[DEBUG] Pre-kernel pointers - input: {:?}, output: {:?}",
            d_input, d_output
        );

        // Run kernel
        println!("[DEBUG] Calling RMS norm with rows=1, dim=512, eps=1e-5");
        let kernel_res = model.run_rms_norm(
            d_input, d_output, 1,   // seq_len/rows
            512, // dim
            1e-5,
        );
        println!(
            "[DEBUG] Kernel execution successful: {:?}",
            kernel_res.is_ok()
        );
        println!("[DEBUG] Kernel result: {:?}", kernel_res);
        kernel_res?;

        // Validate device pointers after kernel
        println!(
            "[DEBUG] Post-kernel pointers - input: {:?}, output: {:?}",
            d_input, d_output
        );

        // Validate before d2h copy
        println!(
            "[DEBUG] d_output ptr: {:?}, size: {} bytes",
            d_output,
            512 * std::mem::size_of::<f32>()
        );
        assert!(!d_output.is_null(), "Device output pointer null");

        // Copy back results
        let copy_size = 512 * std::mem::size_of::<f32>();
        let res =
            model
                .cuda
                .memcpy_d2h(d_output, host_output.as_mut_ptr() as *mut c_void, copy_size);
        println!("[DEBUG] d2h copy result: {:?}", res);
        res?;

        assert!(!host_output.iter().all(|&x| x == 0.0)); // Verify output was modified

        // Free device memory
        model.cuda.device_free(d_input)?;
        model.cuda.device_free(d_output)?;
    }
    Ok(())
}

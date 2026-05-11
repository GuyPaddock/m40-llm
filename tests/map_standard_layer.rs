use anyhow::Result;
use m40_llm::gguf::{GgmlDType, GgufModel, GgufScalar, GgufValue};
use m40_llm::infer::{DeviceTensorView, LoadedModel, ModelConfig};
use std::collections::HashMap;

fn make_model_with_layer(
    layer: usize,
    d_model: usize,
    hidden: usize,
    f16_weights: bool,
) -> Result<LoadedModel> {
    fn strides_from(shape: &[u64]) -> Vec<usize> {
        let mut out = Vec::with_capacity(shape.len());
        let mut stride = 1usize;
        for &dim in shape.iter().rev() {
            out.push(stride);
            stride *= dim as usize;
        }
        out.reverse();
        out
    }
    fn make_view(dtype: GgmlDType, shape: Vec<u64>) -> DeviceTensorView {
        let strides = strides_from(&shape);
        DeviceTensorView {
            dtype,
            shape,
            strides,
            byte_offset: 0,
            nbytes: 0,
            #[cfg(feature = "cuda")]
            dptr: std::ptr::null_mut(),
        }
    }
    // Minimal GGUF backing; only used for metadata if present
    let mut gguf = GgufModel::new(0);
    gguf.metadata.insert(
        "general.architecture".into(),
        GgufValue::Scalar(GgufScalar::Str("llama".into())),
    );
    gguf.metadata.insert(
        "llama.embedding_length".into(),
        GgufValue::Scalar(GgufScalar::U32(d_model as u32)),
    );
    gguf.metadata.insert(
        "llama.attention.head_count".into(),
        GgufValue::Scalar(GgufScalar::U32(1)),
    );
    gguf.metadata.insert(
        "llama.feed_forward_length".into(),
        GgufValue::Scalar(GgufScalar::U32(hidden as u32)),
    );
    gguf.metadata.insert(
        "llama.block_count".into(),
        GgufValue::Scalar(GgufScalar::U32((layer + 1) as u32)),
    );
    gguf.metadata.insert(
        "llama.context_length".into(),
        GgufValue::Scalar(GgufScalar::U32(16)),
    );
    gguf.metadata.insert(
        "llama.vocab_size".into(),
        GgufValue::Scalar(GgufScalar::U32(1024)),
    );
    let cuda = match m40_llm::cuda::CudaContext::new(-1) {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };

    let mut device_tensors: HashMap<String, DeviceTensorView> = HashMap::new();
    // Embeddings: [vocab, d_model]
    device_tensors.insert(
        "tok_embeddings.weight".into(),
        make_view(GgmlDType::F16, vec![1024, d_model as u64]),
    );
    device_tensors.insert(
        "output.weight".into(),
        make_view(GgmlDType::F16, vec![d_model as u64, 1024]),
    );

    let wt = if f16_weights {
        GgmlDType::F16
    } else {
        GgmlDType::F32
    };

    // Attention weights: [d_model, d_model]
    for (key, shape) in [
        (
            format!("layers.{layer}.attention.wq.weight"),
            vec![d_model as u64, d_model as u64],
        ),
        (
            format!("layers.{layer}.attention.wk.weight"),
            vec![d_model as u64, d_model as u64],
        ),
        (
            format!("layers.{layer}.attention.wv.weight"),
            vec![d_model as u64, d_model as u64],
        ),
        (
            format!("layers.{layer}.attention.wo.weight"),
            vec![d_model as u64, d_model as u64],
        ),
    ] {
        device_tensors.insert(key, make_view(wt, shape));
    }

    // MLP: gate/up [d_model, hidden], down [hidden, d_model]
    for (key, shape) in [
        (
            format!("layers.{layer}.feed_forward.w3.weight"),
            vec![d_model as u64, hidden as u64],
        ),
        (
            format!("layers.{layer}.feed_forward.w1.weight"),
            vec![d_model as u64, hidden as u64],
        ),
        (
            format!("layers.{layer}.feed_forward.w2.weight"),
            vec![hidden as u64, d_model as u64],
        ),
    ] {
        device_tensors.insert(key, make_view(wt, shape));
    }

    let model_config = ModelConfig::from_metadata(&gguf.metadata, &gguf.tensors).unwrap();
    Ok(LoadedModel {
        gguf,
        cuda,
        kv_cache: None,
        #[cfg(feature = "cuda")]
        forward_workspace: std::sync::Mutex::new(None),
        #[cfg(feature = "cuda")]
        materialized_weights: std::sync::Mutex::new(std::collections::HashMap::new()),
        device_tensors,
        weights_len: 0,
        #[cfg(feature = "cuda")]
        d_weights_base: std::ptr::null_mut(),
        host_weights: Vec::new(),
        model_config,
        #[cfg(feature = "gguf_ext")]
        typed_config: gguf_llms::model::ModelConfig {
            architecture: "llama".into(),
            block_count: 1,
            context_length: 0,
            embedding_length: d_model as u32,
            feed_forward_length: hidden as u32,
            attention_head_count: 1,
            attention_head_count_kv: None,
            attention_key_length: Some((d_model as u32) / 1),
            layer_norm_epsilon: None,
            rope_freq_base: None,
        },
    })
}

#[test]
fn map_standard_layer_ok() -> Result<()> {
    let lm = match make_model_with_layer(0, 32, 64, true) {
        Ok(lm) => lm,
        Err(e) => {
            eprintln!("skipping: {}", e);
            return Ok(());
        }
    };
    let mapped = lm.map_standard_layer(0).expect("should map successfully");
    assert_eq!(mapped.d_model, 32);
    assert_eq!(mapped.hidden_dim, 64);
    Ok(())
}

#[test]
fn map_standard_layer_dtype_guard() -> Result<()> {
    // f16_weights=false -> weights are F32; map should fail
    let lm = match make_model_with_layer(1, 16, 32, false) {
        Ok(lm) => lm,
        Err(e) => {
            eprintln!("skipping: {}", e);
            return Ok(());
        }
    };
    let err = lm.map_standard_layer(1).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("expected F16"), "unexpected error: {msg}");
    Ok(())
}

#[test]
fn map_standard_layer_shape_guard_down() -> Result<()> {
    // Build valid model then corrupt w_down shape to mismatch hidden/d_model
    let mut lm = match make_model_with_layer(2, 24, 48, true) {
        Ok(lm) => lm,
        Err(e) => {
            eprintln!("skipping: {}", e);
            return Ok(());
        }
    };
    let bad_key = "layers.2.feed_forward.w2.weight".to_string();
    let entry = lm
        .device_tensors
        .get_mut(&bad_key)
        .expect("down weight present");
    entry.shape = vec![47, 25]; // wrong
    let err = lm.map_standard_layer(2).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("w_down shape invalid"),
        "unexpected error: {msg}"
    );
    Ok(())
}

#[test]
fn map_standard_layer_embed_dtype_guard() -> Result<()> {
    // Corrupt embeddings dtype to F32; should fail early
    let mut lm = match make_model_with_layer(3, 32, 64, true) {
        Ok(lm) => lm,
        Err(e) => {
            eprintln!("skipping: {}", e);
            return Ok(());
        }
    };
    let entry = lm
        .device_tensors
        .get_mut("tok_embeddings.weight")
        .expect("embeddings present");
    entry.dtype = GgmlDType::F32;
    let err = lm.map_standard_layer(3).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("tok_embeddings.weight expected F16"),
        "unexpected error: {msg}"
    );
    Ok(())
}

#[test]
fn map_standard_layer_wq_shape_guard() -> Result<()> {
    // Corrupt wq shape first dim to not equal d_model
    let mut lm = match make_model_with_layer(4, 32, 64, true) {
        Ok(lm) => lm,
        Err(e) => {
            eprintln!("skipping: {}", e);
            return Ok(());
        }
    };
    let key = "layers.4.attention.wq.weight".to_string();
    let entry = lm.device_tensors.get_mut(&key).expect("wq present");
    entry.shape = vec![31, 32];
    let err = lm.map_standard_layer(4).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("wq shape invalid"), "unexpected error: {msg}");
    Ok(())
}

#[test]
fn map_standard_layer_gate_shape_guard() -> Result<()> {
    // Corrupt w_gate shape second dim to 0 (invalid H)
    let mut lm = match make_model_with_layer(5, 32, 64, true) {
        Ok(lm) => lm,
        Err(e) => {
            eprintln!("skipping: {}", e);
            return Ok(());
        }
    };
    let key = "layers.5.feed_forward.w3.weight".to_string();
    let entry = lm.device_tensors.get_mut(&key).expect("w_gate present");
    entry.shape = vec![32, 0];
    let err = lm.map_standard_layer(5).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("w_gate shape invalid"),
        "unexpected error: {msg}"
    );
    Ok(())
}

#[test]
fn validate_full_layer_decode_requires_lm_head() -> Result<()> {
    let mut lm = match make_model_with_layer(0, 32, 64, true) {
        Ok(lm) => lm,
        Err(e) => {
            eprintln!("skipping: {}", e);
            return Ok(());
        }
    };
    lm.allocate_kv_cache_for_layers(16)?;
    lm.device_tensors.remove("output.weight");

    let err = lm.validate_full_layer_decode().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("lm_head tensor not found"),
        "unexpected error: {msg}"
    );
    Ok(())
}

#[test]
fn validate_full_layer_decode_requires_layer_kv_slots() -> Result<()> {
    let mut lm = match make_model_with_layer(0, 32, 64, true) {
        Ok(lm) => lm,
        Err(e) => {
            eprintln!("skipping: {}", e);
            return Ok(());
        }
    };
    lm.allocate_kv_cache(16, 0)?;

    let err = lm.validate_full_layer_decode().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("full-layer decode needs"),
        "unexpected error: {msg}"
    );
    Ok(())
}

#[test]
fn kv_layer_sequence_mapping_uses_sequence_major_slots() -> Result<()> {
    let mut lm = match make_model_with_layer(2, 32, 64, true) {
        Ok(lm) => lm,
        Err(e) => {
            eprintln!("skipping: {}", e);
            return Ok(());
        }
    };
    lm.allocate_kv_cache_for_layer_sequences(16, 3)?;

    assert_eq!(lm.kv_cache_logical_sequence_capacity(), 3);
    assert_eq!(lm.kv_cache_physical_slot_for_layer_sequence(0, 0)?, 0);
    assert_eq!(lm.kv_cache_physical_slot_for_layer_sequence(2, 0)?, 2);
    assert_eq!(lm.kv_cache_physical_slot_for_layer_sequence(0, 1)?, 3);
    assert_eq!(lm.kv_cache_physical_slot_for_layer_sequence(2, 2)?, 8);

    let err = lm
        .kv_cache_physical_slot_for_layer_sequence(0, 3)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("sequence_id 3 out of range"),
        "unexpected error: {msg}"
    );
    Ok(())
}

#[cfg(all(feature = "cuda", nvcc))]
#[test]
fn kv_layer_sequence_append_uses_distinct_physical_slots() -> Result<()> {
    use half::f16;
    use m40_llm::cuda::ffi_debug_read_kv_token;
    use std::ffi::c_void;

    let mut lm = match make_model_with_layer(1, 4, 8, true) {
        Ok(lm) => lm,
        Err(e) => {
            eprintln!("skipping: {}", e);
            return Ok(());
        }
    };
    lm.allocate_kv_cache_for_layer_sequences(4, 2)?;
    let physical_slot = lm.kv_cache_physical_slot_for_layer_sequence(1, 1)?;
    assert_eq!(physical_slot, 3);

    let k = [1.25f32, 2.5, 3.75, 5.0];
    let v = [10.5f32, 11.5, 12.5, 13.5];
    let bytes = k.len() * std::mem::size_of::<f32>();
    let d_k = lm.cuda.device_malloc(bytes)?;
    let d_v = lm.cuda.device_malloc(bytes)?;
    unsafe {
        lm.cuda
            .memcpy_h2d(d_k, k.as_ptr() as *const c_void, bytes)?;
        lm.cuda
            .memcpy_h2d(d_v, v.as_ptr() as *const c_void, bytes)?;
        lm.append_kv_token_f32_for_layer(1, 1, 0, d_k as *const c_void, d_v as *const c_void)?;
    }

    let mut k_back = vec![0u8; k.len() * std::mem::size_of::<u16>()];
    let mut v_back = vec![0u8; v.len() * std::mem::size_of::<u16>()];
    unsafe {
        ffi_debug_read_kv_token(
            &lm.cuda,
            lm.kv_cache.as_ref().expect("kv cache"),
            physical_slot,
            0,
            k_back.as_mut_ptr(),
            v_back.as_mut_ptr(),
        );
    }

    for (idx, expected) in k.iter().enumerate() {
        let bits = u16::from_le_bytes([k_back[idx * 2], k_back[idx * 2 + 1]]);
        assert_eq!(bits, f16::from_f32(*expected).to_bits());
    }
    for (idx, expected) in v.iter().enumerate() {
        let bits = u16::from_le_bytes([v_back[idx * 2], v_back[idx * 2 + 1]]);
        assert_eq!(bits, f16::from_f32(*expected).to_bits());
    }

    unsafe {
        lm.cuda.device_free(d_k)?;
        lm.cuda.device_free(d_v)?;
    }
    Ok(())
}

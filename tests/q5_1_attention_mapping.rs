use m40_llm::gguf::{GgmlDType, GgufModel};
use m40_llm::infer::{DeviceTensorView, LoadedModel};
use std::collections::HashMap;

fn device_tensor(dtype: GgmlDType, shape: Vec<u64>) -> DeviceTensorView {
    let elem_bytes = match dtype {
        GgmlDType::F16 => 2,
        GgmlDType::Q5_1 => {
            // Q5_1 stores blocks of 32 values with a fixed 6-byte stride in tests
            let n = *shape.get(1).unwrap_or(&0) as usize;
            let rows = *shape.get(0).unwrap_or(&0) as usize;
            let blocks = (n + 31) / 32;
            return DeviceTensorView {
                dtype,
                shape,
                byte_offset: 0,
                nbytes: blocks * 6 * rows,
                #[cfg(feature = "cuda")]
                dptr: std::ptr::null_mut(),
            };
        }
        _ => 0,
    };

    let nbytes = (shape.iter().product::<u64>() as usize).saturating_mul(elem_bytes);
    DeviceTensorView {
        dtype,
        shape,
        byte_offset: 0,
        nbytes,
        #[cfg(feature = "cuda")]
        dptr: std::ptr::null_mut(),
    }
}

fn make_model_with_q5_1_attention_layer(layer: usize, d_model: usize) -> LoadedModel {
    // Minimal GGUF backing; only used for metadata if present
    let gguf = GgufModel::new(0);
    let cuda = m40_llm::cuda::CudaContext::new(-1).unwrap();

    let mut device_tensors: HashMap<String, DeviceTensorView> = HashMap::new();
    // Embeddings: [vocab, d_model] - use F16 for embeddings
    device_tensors.insert(
        "tok_embeddings.weight".into(),
        device_tensor(GgmlDType::F16, vec![1024, d_model as u64]),
    );

    // Attention weights: [d_model, d_model] - use Q5_1 for attention
    let q5_1 = GgmlDType::Q5_1; // Q5_1 quantization

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
        device_tensors.insert(key, device_tensor(q5_1, shape));
    }

    // MLP weights: use F16 for MLP (not testing quantization here)
    let wt_f16 = GgmlDType::F16;

    // MLP: gate/up [d_model, hidden], down [hidden, d_model]
    let hidden_dim = d_model * 2; // arbitrary
    for (key, shape) in [
        (
            format!("layers.{layer}.feed_forward.w3.weight"),
            vec![d_model as u64, hidden_dim as u64],
        ),
        (
            format!("layers.{layer}.feed_forward.w1.weight"),
            vec![d_model as u64, hidden_dim as u64],
        ),
        (
            format!("layers.{layer}.feed_forward.w2.weight"),
            vec![hidden_dim as u64, d_model as u64],
        ),
    ] {
        device_tensors.insert(key, device_tensor(wt_f16, shape));
    }

    LoadedModel {
        gguf,
        cuda,
        kv_cache: None,
        device_tensors,
        #[cfg(feature = "cuda")]
        d_weights_base: std::ptr::null_mut(),
        #[cfg(not(feature = "cuda"))]
        host_weights: Vec::new(),
        #[cfg(feature = "gguf_ext")]
        typed_config: None,
    }
}

#[test]
fn q5_1_attention_mapping_ok() {
    let lm = make_model_with_q5_1_attention_layer(0, 32);
    // This should succeed - Q5_1 is already supported
    let mapped = lm
        .map_standard_layer(0)
        .expect("should map Q5_1 attention layer successfully");
    assert_eq!(mapped.d_model, 32);
    // hidden_dim should match what we configured in the test
    assert_eq!(mapped.hidden_dim, 32 * 2);
}

#[test]
fn q5_1_attention_mapping_accepts_f16_attention() {
    let mut lm = make_model_with_q5_1_attention_layer(1, 16);
    // Overwrite attention weights to F16 to ensure the legacy path still works deterministically
    for suffix in ["wq", "wk", "wv", "wo"] {
        let key = format!("layers.1.attention.{suffix}.weight");
        let entry = lm
            .device_tensors
            .get_mut(&key)
            .expect("attention weight present");
        entry.dtype = GgmlDType::F16;
        entry.nbytes = (entry.shape.iter().product::<u64>() * 2) as usize;
    }

    let mapped = lm.map_standard_layer(1).expect("should map f16 attention");
    assert_eq!(mapped.d_model, 16);
    assert_eq!(mapped.hidden_dim, 16 * 2);
}

#[test]
fn q5_1_attention_shape_guard() {
    // Corrupt wq shape first dim to not equal d_model
    let mut lm = make_model_with_q5_1_attention_layer(2, 32);
    let key = "layers.2.attention.wq.weight".to_string();
    let entry = lm.device_tensors.get_mut(&key).expect("wq present");
    entry.shape = vec![31, 32];
    let err = lm.map_standard_layer(2).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("wq shape invalid"), "unexpected error: {msg}");
}

#[test]
fn q5_1_attention_embed_dtype_guard() {
    // Corrupt embeddings dtype to F32; should fail early
    let mut lm = make_model_with_q5_1_attention_layer(3, 32);
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
}

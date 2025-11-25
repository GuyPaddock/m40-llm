use m40_llm::gguf::{GgmlDType, GgufModel};
use m40_llm::infer::{DeviceTensorView, LoadedModel};
use std::collections::HashMap;

fn make_model_with_layer(
    layer: usize,
    d_model: usize,
    hidden: usize,
    f16_weights: bool,
) -> LoadedModel {
    // Minimal GGUF backing; only used for metadata if present
    let gguf = GgufModel::new(0);
    let cuda = m40_llm::cuda::CudaContext::new(-1).unwrap();

    let mut device_tensors: HashMap<String, DeviceTensorView> = HashMap::new();
    // Embeddings: [vocab, d_model]
    device_tensors.insert(
        "tok_embeddings.weight".into(),
        DeviceTensorView {
            dtype: GgmlDType::F16,
            shape: vec![1024, d_model as u64],
            byte_offset: 0,
            nbytes: 0,
            #[cfg(feature = "cuda")]
            dptr: std::ptr::null_mut(),
        },
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
        device_tensors.insert(
            key,
            DeviceTensorView {
                dtype: wt,
                shape,
                byte_offset: 0,
                nbytes: 0,
                #[cfg(feature = "cuda")]
                dptr: std::ptr::null_mut(),
            },
        );
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
        device_tensors.insert(
            key,
            DeviceTensorView {
                dtype: wt,
                shape,
                byte_offset: 0,
                nbytes: 0,
                #[cfg(feature = "cuda")]
                dptr: std::ptr::null_mut(),
            },
        );
    }

    LoadedModel {
        gguf,
        cuda,
        kv_cache: None,
        device_tensors,
        #[cfg(feature = "cuda")]
        d_weights_base: std::ptr::null_mut(),
    }
}

#[test]
fn map_standard_layer_ok() {
    let lm = make_model_with_layer(0, 32, 64, true);
    let mapped = lm.map_standard_layer(0).expect("should map successfully");
    assert_eq!(mapped.d_model, 32);
    assert_eq!(mapped.hidden_dim, 64);
}

#[test]
fn map_standard_layer_dtype_guard() {
    // f16_weights=false -> weights are F32; map should fail
    let lm = make_model_with_layer(1, 16, 32, false);
    let err = lm.map_standard_layer(1).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("expected F16"), "unexpected error: {msg}");
}

#[test]
fn map_standard_layer_shape_guard_down() {
    // Build valid model then corrupt w_down shape to mismatch hidden/d_model
    let mut lm = make_model_with_layer(2, 24, 48, true);
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
}

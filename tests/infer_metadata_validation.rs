use m40_llm::gguf::{GgmlDType, GgufModel, GgufScalar, GgufValue};
use m40_llm::infer::{DeviceTensorView, LoadedModel, ModelConfig};
use std::collections::HashMap;

fn base_model_with_emb(vocab: usize, d_model: usize) -> LoadedModel {
    let mut gguf = GgufModel::new(0);
    // minimal llama markers
    gguf.metadata.insert(
        "general.architecture".into(),
        GgufValue::Scalar(GgufScalar::Str("llama".into())),
    );
    // ensure head_count divides d_model so typed/shape checks are fine
    gguf.metadata.insert(
        "llama.attention.head_count".into(),
        GgufValue::Scalar(GgufScalar::U32(1)),
    );
    gguf.metadata.insert(
        "llama.block_count".into(),
        GgufValue::Scalar(GgufScalar::U32(4)),
    );
    gguf.metadata.insert(
        "llama.context_length".into(),
        GgufValue::Scalar(GgufScalar::U32(16)),
    );
    gguf.metadata.insert(
        "llama.embedding_length".into(),
        GgufValue::Scalar(GgufScalar::U32(d_model as u32)),
    );
    gguf.metadata.insert(
        "llama.feed_forward_length".into(),
        GgufValue::Scalar(GgufScalar::U32((2 * d_model) as u32)),
    );
    gguf.metadata.insert(
        "llama.vocab_size".into(),
        GgufValue::Scalar(GgufScalar::U32(vocab as u32)),
    );

    let cuda = m40_llm::cuda::CudaContext::new(-1).unwrap();

    let mut device_tensors: HashMap<String, DeviceTensorView> = HashMap::new();
    device_tensors.insert(
        "tok_embeddings.weight".into(),
        DeviceTensorView {
            dtype: GgmlDType::F16,
            shape: vec![vocab as u64, d_model as u64],
            byte_offset: 0,
            nbytes: 0,
            #[cfg(feature = "cuda")]
            dptr: std::ptr::null_mut(),
        },
    );
    // a single layer's minimal required tensors to make map_standard_layer reach validations
    for key in [
        "layers.0.attention.wq.weight",
        "layers.0.attention.wk.weight",
        "layers.0.attention.wv.weight",
        "layers.0.attention.wo.weight",
        "layers.0.feed_forward.w3.weight",
        "layers.0.feed_forward.w1.weight",
        "layers.0.feed_forward.w2.weight",
    ] {
        let shape = if key.ends_with("w2.weight") {
            vec![2 * d_model as u64, d_model as u64]
        } else if key.ends_with("w3.weight") || key.ends_with("w1.weight") {
            vec![d_model as u64, 2 * d_model as u64]
        } else {
            vec![d_model as u64, d_model as u64]
        };
        device_tensors.insert(
            key.to_string(),
            DeviceTensorView {
                dtype: GgmlDType::F16,
                shape,
                byte_offset: 0,
                nbytes: 0,
                #[cfg(feature = "cuda")]
                dptr: std::ptr::null_mut(),
            },
        );
    }

    let model_config = ModelConfig::from_metadata(&gguf.metadata, &gguf.tensors).unwrap();
    LoadedModel {
        gguf,
        cuda,
        kv_cache: None,
        device_tensors,
        #[cfg(feature = "cuda")]
        d_weights_base: std::ptr::null_mut(),
        #[cfg(not(feature = "cuda"))]
        host_weights: Vec::new(),
        model_config,
        #[cfg(feature = "gguf_ext")]
        typed_config: gguf_llms::model::ModelConfig {
            architecture: "llama".into(),
            block_count: 4,
            context_length: 0,
            embedding_length: d_model as u32,
            feed_forward_length: (2 * d_model) as u32,
            attention_head_count: 1,
            attention_head_count_kv: None,
            attention_key_length: Some((d_model as u32) / 1),
            layer_norm_epsilon: None,
            rope_freq_base: None,
        },
    }
}

#[test]
fn vocab_size_must_match_embeddings_rows_when_present() {
    let mut lm = base_model_with_emb(100, 32);
    // set conflicting vocab_size metadata
    lm.gguf.metadata.insert(
        "llama.vocab_size".into(),
        GgufValue::Scalar(GgufScalar::U32(99)),
    );
    lm.model_config = ModelConfig::from_metadata(&lm.gguf.metadata, &lm.gguf.tensors).unwrap();
    let err = lm.map_standard_layer(0).unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("vocab_size meta"), "unexpected: {}", msg);
}

#[test]
fn layer_index_checked_against_block_count() {
    let mut lm = base_model_with_emb(100, 32);
    lm.gguf.metadata.insert(
        "llama.vocab_size".into(),
        GgufValue::Scalar(GgufScalar::U32(100)),
    );
    lm.gguf.metadata.insert(
        "llama.block_count".into(),
        GgufValue::Scalar(GgufScalar::U32(1)),
    );
    lm.model_config = ModelConfig::from_metadata(&lm.gguf.metadata, &lm.gguf.tensors).unwrap();
    // mapping layer 1 should fail (only 0 valid)
    let err = lm.map_standard_layer(1).unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("out of range"), "unexpected: {}", msg);
}

#[test]
fn context_length_zero_rejected_when_present() {
    let mut lm = base_model_with_emb(100, 32);
    lm.gguf.metadata.insert(
        "llama.context_length".into(),
        GgufValue::Scalar(GgufScalar::U32(0)),
    );
    let err = ModelConfig::from_metadata(&lm.gguf.metadata, &lm.gguf.tensors).unwrap_err();
    let msg = format!("{}", err);
    assert!(
        msg.contains("context_length must be > 0"),
        "unexpected: {}",
        msg
    );
}

#[test]
fn rope_params_must_be_finite_and_positive_when_present() {
    let mut lm = base_model_with_emb(100, 32);
    lm.gguf.metadata.insert(
        "llama.rope.freq_base".into(),
        GgufValue::Scalar(GgufScalar::F32(0.0)),
    );
    let err = ModelConfig::from_metadata(&lm.gguf.metadata, &lm.gguf.tensors).unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("rope_freq_base"), "unexpected: {}", msg);

    // fix base; set bad scale
    let mut lm2 = base_model_with_emb(100, 32);
    lm2.gguf.metadata.insert(
        "llama.rope.freq_scale".into(),
        GgufValue::Scalar(GgufScalar::F32(-1.0)),
    );
    let err2 = ModelConfig::from_metadata(&lm2.gguf.metadata, &lm2.gguf.tensors).unwrap_err();
    let msg2 = format!("{}", err2);
    assert!(msg2.contains("rope_freq_scale"), "unexpected: {}", msg2);
}

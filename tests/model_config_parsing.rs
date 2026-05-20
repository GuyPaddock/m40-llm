use m40_llm::gguf::{GgmlDType, GgufScalar, GgufTensor, GgufValue};
use m40_llm::infer::ModelConfig;
use std::collections::HashMap;

#[cfg(feature = "gguf_ext")]
use gguf_llms::model::ModelConfig as LlmsModelConfig;

fn base_metadata() -> HashMap<String, GgufValue> {
    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        GgufValue::Scalar(GgufScalar::Str("llama".into())),
    );
    meta.insert(
        "llama.embedding_length".into(),
        GgufValue::Scalar(GgufScalar::U32(16)),
    );
    meta.insert(
        "llama.attention.head_count".into(),
        GgufValue::Scalar(GgufScalar::U32(4)),
    );
    meta.insert(
        "llama.block_count".into(),
        GgufValue::Scalar(GgufScalar::U32(2)),
    );
    meta.insert(
        "llama.context_length".into(),
        GgufValue::Scalar(GgufScalar::U32(16)),
    );
    meta.insert(
        "llama.feed_forward_length".into(),
        GgufValue::Scalar(GgufScalar::U32(64)),
    );
    meta.insert(
        "llama.vocab_size".into(),
        GgufValue::Scalar(GgufScalar::U32(128)),
    );
    meta
}

#[test]
fn model_config_parses_required_fields() {
    let meta = base_metadata();
    let cfg = ModelConfig::from_metadata(&meta, &[]).expect("config should parse");
    assert_eq!(cfg.embedding_length, 16);
    assert_eq!(cfg.attention_head_count, 4);
    assert_eq!(cfg.feed_forward_length, 64);
}

#[test]
fn model_config_missing_head_count_fails() {
    let mut meta = base_metadata();
    meta.remove("llama.attention.head_count");
    let err = ModelConfig::from_metadata(&meta, &[]).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("head_count"));
}

#[test]
fn model_config_rejects_non_divisible_d_model() {
    let mut meta = HashMap::new();
    meta.extend(base_metadata());
    meta.insert(
        "llama.embedding_length".into(),
        GgufValue::Scalar(GgufScalar::U32(10)),
    );
    meta.insert(
        "llama.attention.head_count".into(),
        GgufValue::Scalar(GgufScalar::U32(3)),
    );
    let err = ModelConfig::from_metadata(&meta, &[]).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("attention_head_count"));
}

#[test]
fn model_config_respects_kv_head_override() {
    let mut meta = base_metadata();
    meta.insert(
        "llama.attention.head_count_kv".into(),
        GgufValue::Scalar(GgufScalar::U32(2)),
    );
    let cfg = ModelConfig::from_metadata(&meta, &[]).expect("config should parse");
    assert_eq!(cfg.attention_head_count, 4);
    assert_eq!(cfg.attention_head_count_kv, 2);
}

#[test]
fn model_config_parses_qwen2_metadata_aliases() {
    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        GgufValue::Scalar(GgufScalar::Str("qwen2".into())),
    );
    meta.insert(
        "qwen2.embedding_length".into(),
        GgufValue::Scalar(GgufScalar::U32(24)),
    );
    meta.insert(
        "qwen2.attention.head_count".into(),
        GgufValue::Scalar(GgufScalar::U32(6)),
    );
    meta.insert(
        "qwen2.attention.head_count_kv".into(),
        GgufValue::Scalar(GgufScalar::U32(2)),
    );
    meta.insert(
        "qwen2.block_count".into(),
        GgufValue::Scalar(GgufScalar::U32(3)),
    );
    meta.insert(
        "qwen2.context_length".into(),
        GgufValue::Scalar(GgufScalar::U32(32)),
    );
    meta.insert(
        "qwen2.feed_forward_length".into(),
        GgufValue::Scalar(GgufScalar::U32(96)),
    );
    meta.insert(
        "qwen2.vocab_size".into(),
        GgufValue::Scalar(GgufScalar::U32(256)),
    );
    meta.insert(
        "qwen2.rope.freq_base".into(),
        GgufValue::Scalar(GgufScalar::F32(1_000_000.0)),
    );
    meta.insert(
        "qwen2.attention.layer_norm_rms_epsilon".into(),
        GgufValue::Scalar(GgufScalar::F32(1e-6)),
    );

    let cfg = ModelConfig::from_metadata(&meta, &[]).expect("qwen2 config should parse");
    assert_eq!(cfg.architecture, "qwen2");
    assert_eq!(cfg.embedding_length, 24);
    assert_eq!(cfg.attention_head_count, 6);
    assert_eq!(cfg.attention_head_count_kv, 2);
    assert_eq!(cfg.attention_key_length, 4);
    assert_eq!(cfg.feed_forward_length, 96);
    assert_eq!(cfg.vocab_size, 256);
    assert_eq!(cfg.rope_freq_base, 1_000_000.0);
    assert_eq!(cfg.layer_norm_epsilon, 1e-6);
}

#[test]
fn model_config_derives_kv_heads_from_k_tensor() {
    let meta = base_metadata();
    let tensors = vec![GgufTensor {
        name: "blk.0.attn_k.weight".into(),
        dtype: GgmlDType::F16,
        shape: vec![16, 8],
        offset: 0,
    }];
    let cfg = ModelConfig::from_metadata(&meta, &tensors).expect("config should parse");
    assert_eq!(cfg.attention_head_count, 4);
    assert_eq!(cfg.attention_key_length, 4);
    assert_eq!(cfg.attention_head_count_kv, 2);
}

#[test]
fn model_config_rejects_mismatched_attention_key_length() {
    let mut meta = base_metadata();
    meta.insert(
        "llama.attention.key_length".into(),
        GgufValue::Scalar(GgufScalar::U32(5)),
    );
    let err = ModelConfig::from_metadata(&meta, &[]).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("attention_key_length"));
}

#[cfg(feature = "gguf_ext")]
#[test]
fn model_config_prefers_typed_feed_forward_length() {
    let mut meta = base_metadata();
    // Keep metadata feed_forward_length smaller than typed to ensure typed wins.
    meta.insert(
        "llama.feed_forward_length".into(),
        GgufValue::Scalar(GgufScalar::U32(32)),
    );
    let typed = LlmsModelConfig {
        architecture: "llama".into(),
        block_count: 3,
        context_length: 16,
        embedding_length: 16,
        feed_forward_length: 96,
        attention_head_count: 4,
        attention_head_count_kv: Some(2),
        attention_key_length: Some(4),
        layer_norm_epsilon: Some(1e-5),
        rope_freq_base: Some(10_000.0),
    };
    let cfg = ModelConfig::from_typed(&typed, &meta, &[]).expect("typed config should parse");
    assert_eq!(cfg.feed_forward_length, typed.feed_forward_length);
    assert_eq!(
        cfg.attention_head_count_kv,
        typed.attention_head_count_kv.unwrap()
    );
}

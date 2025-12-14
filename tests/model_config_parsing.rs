use m40_llm::gguf::{GgufScalar, GgufValue};
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

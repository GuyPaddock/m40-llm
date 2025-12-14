use m40_llm::gguf::{GgufScalar, GgufValue};
use m40_llm::infer::ModelConfig;
use std::collections::HashMap;

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
    meta.insert(
        "general.architecture".into(),
        GgufValue::Scalar(GgufScalar::Str("llama".into())),
    );
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
    assert!(msg.contains("not divisible"));
}

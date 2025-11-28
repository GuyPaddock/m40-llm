use m40_llm::gguf::GgufValue;
use m40_llm::tokenizer::{Tokenizer, TokenizerKind};
use std::collections::HashMap;

#[test]
fn detection_prefers_sp_when_sentencepiece_present() {
    let mut meta: HashMap<String, GgufValue> = HashMap::new();
    meta.insert(
        "tokenizer.ggml.model".to_string(),
        GgufValue::Scalar(m40_llm::gguf::GgufScalar::Str("spm".to_string())),
    );
    meta.insert(
        "sentencepiece.model".to_string(),
        GgufValue::Scalar(m40_llm::gguf::GgufScalar::Str("dummy".to_string())),
    );
    let t = Tokenizer::from_gguf_metadata(&meta).unwrap();
    assert!(matches!(t.kind(), TokenizerKind::SentencePiece));
}

#[test]
fn detection_sees_bpe_keys() {
    let mut meta: HashMap<String, GgufValue> = HashMap::new();
    meta.insert(
        "tokenizer.ggml.model".to_string(),
        GgufValue::Scalar(m40_llm::gguf::GgufScalar::Str("bpe".to_string())),
    );
    meta.insert(
        "tokenizer.ggml.bpe_merges".to_string(),
        GgufValue::Scalar(m40_llm::gguf::GgufScalar::Str("dummy_merges".to_string())),
    );
    let t = Tokenizer::from_gguf_metadata(&meta).unwrap();
    assert!(matches!(t.kind(), TokenizerKind::Bpe));
}

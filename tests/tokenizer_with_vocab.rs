use std::collections::HashMap;

use m40_llm::gguf::{GgufScalar, GgufValue};
use m40_llm::tokenizer::Tokenizer;

fn make_array(values: &[&str]) -> GgufValue {
    GgufValue::Array(
        values
            .iter()
            .map(|v| GgufScalar::Str((*v).to_string()))
            .collect(),
    )
}

#[test]
fn bpe_roundtrip_from_metadata() {
    let mut meta: HashMap<String, GgufValue> = HashMap::new();
    meta.insert(
        "tokenizer.ggml.model".to_string(),
        GgufValue::Scalar(GgufScalar::Str("bpe".to_string())),
    );

    let tokens = vec![
        " ", "h", "e", "l", "o", "w", "r", "d", "he", "hel", "hell", "hello", " w", " wo", " wor",
        " worl", " world",
    ];
    meta.insert("tokenizer.ggml.tokens".to_string(), make_array(&tokens));

    let merges = vec![
        "h e", "he l", "hel l", "hell o", "  w", " w o", " wo r", " wor l", " worl d",
    ];
    meta.insert("tokenizer.ggml.bpe_merges".to_string(), make_array(&merges));

    let tokenizer = Tokenizer::from_gguf_metadata(&meta).expect("tokenizer");
    let ids = tokenizer.encode("hello world").expect("encode");
    assert_eq!(ids, vec![11u32, 16u32]);

    let text = tokenizer.decode(&ids).expect("decode");
    assert_eq!(text, "hello world");
}

#[test]
fn sentencepiece_roundtrip_from_metadata() {
    let mut meta: HashMap<String, GgufValue> = HashMap::new();
    meta.insert(
        "tokenizer.ggml.model".to_string(),
        GgufValue::Scalar(GgufScalar::Str("spm".to_string())),
    );
    let tokens = vec!["▁hello", "▁world", "hello", "world"];
    meta.insert("tokenizer.ggml.tokens".to_string(), make_array(&tokens));

    let tokenizer = Tokenizer::from_gguf_metadata(&meta).expect("tokenizer");
    let ids = tokenizer.encode("hello world").expect("encode");
    assert_eq!(ids, vec![0u32, 1u32]);

    let text = tokenizer.decode(&ids).expect("decode");
    assert_eq!(text, "hello world");
}

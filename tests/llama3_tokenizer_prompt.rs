use std::collections::HashMap;

use m40_llm::decode::StoppingCriteria;
use m40_llm::generate::{prepare_prompt, PromptFormat};
use m40_llm::gguf::{GgufScalar, GgufValue};
use m40_llm::tokenizer::{Tokenizer, TokenizerKind};

fn make_array(values: Vec<String>) -> GgufValue {
    GgufValue::Array(values.into_iter().map(GgufScalar::Str).collect())
}

fn llama3_meta() -> HashMap<String, GgufValue> {
    let mut meta = HashMap::new();
    meta.insert(
        "tokenizer.ggml.model".to_string(),
        GgufValue::Scalar(GgufScalar::Str("gpt2".to_string())),
    );
    meta.insert(
        "tokenizer.ggml.pre".to_string(),
        GgufValue::Scalar(GgufScalar::Str("llama-bpe".to_string())),
    );
    meta.insert(
        "llama.vocab_size".to_string(),
        GgufValue::Scalar(GgufScalar::U32(128_256)),
    );
    meta.insert(
        "tokenizer.ggml.bos_token_id".to_string(),
        GgufValue::Scalar(GgufScalar::U32(128_000)),
    );
    meta.insert(
        "tokenizer.ggml.eos_token_id".to_string(),
        GgufValue::Scalar(GgufScalar::U32(128_009)),
    );
    meta.insert(
        "tokenizer.chat_template".to_string(),
        GgufValue::Scalar(GgufScalar::Str("llama3".to_string())),
    );
    let mut tokens = vec![String::new(); 128_256];
    tokens[128_000] = "<|begin_of_text|>".to_string();
    tokens[128_001] = "<|end_of_text|>".to_string();
    tokens[128_006] = "<|start_header_id|>".to_string();
    tokens[128_007] = "<|end_header_id|>".to_string();
    tokens[128_009] = "<|eot_id|>".to_string();
    meta.insert("tokenizer.ggml.tokens".to_string(), make_array(tokens));
    meta
}

#[test]
fn llama3_metadata_selects_tiktoken_backend() {
    let tokenizer = Tokenizer::from_gguf_metadata(&llama3_meta()).expect("tokenizer");
    assert!(matches!(tokenizer.kind(), TokenizerKind::Llama3));
    assert!(tokenizer.has_chat_template());
    assert_eq!(tokenizer.encode("Hello").unwrap(), vec![9906]);
    assert_eq!(tokenizer.encode(" Hello").unwrap(), vec![22691]);
    assert_eq!(tokenizer.token_id("<|begin_of_text|>"), Some(128_000));
    assert_eq!(tokenizer.token_id("<|start_header_id|>"), Some(128_006));
    assert_eq!(tokenizer.token_id("<|end_header_id|>"), Some(128_007));
    assert_eq!(tokenizer.token_id("<|eot_id|>"), Some(128_009));
}

#[test]
fn llama3_chat_prompt_uses_control_tokens_once() {
    let tokenizer = Tokenizer::from_gguf_metadata(&llama3_meta()).expect("tokenizer");
    let (prompt, add_bos) = prepare_prompt(&tokenizer, "Hello", PromptFormat::Llama3Chat);
    assert!(!add_bos);

    let ids = tokenizer
        .encode_with_specials(&prompt, add_bos, false)
        .expect("encode prompt");
    assert_eq!(ids.first().copied(), Some(128_000));
    assert!(ids.contains(&128_006));
    assert!(ids.contains(&128_007));
    assert!(ids.contains(&128_009));
}

#[test]
fn auto_prompt_format_selects_llama3_chat_for_unformatted_prompt() {
    let tokenizer = Tokenizer::from_gguf_metadata(&llama3_meta()).expect("tokenizer");
    let (prompt, add_bos) = prepare_prompt(&tokenizer, "Hello", PromptFormat::Auto);
    assert!(!add_bos);
    assert!(prompt.starts_with("<|begin_of_text|><|start_header_id|>user"));
    assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
}

#[test]
fn auto_prompt_format_preserves_preformatted_llama3_without_extra_bos() {
    let tokenizer = Tokenizer::from_gguf_metadata(&llama3_meta()).expect("tokenizer");
    let preformatted = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    let (prompt, add_bos) = prepare_prompt(&tokenizer, preformatted, PromptFormat::Auto);
    assert_eq!(prompt, preformatted);
    assert!(!add_bos);

    let ids = tokenizer
        .encode_with_specials(&prompt, add_bos, false)
        .expect("encode prompt");
    assert_eq!(ids.first().copied(), Some(128_000));
    assert_eq!(ids.iter().filter(|&&id| id == 128_000).count(), 1);
}

#[test]
fn raw_preformatted_llama3_does_not_add_extra_bos() {
    let tokenizer = Tokenizer::from_gguf_metadata(&llama3_meta()).expect("tokenizer");
    let preformatted = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello";
    let (prompt, add_bos) = prepare_prompt(&tokenizer, preformatted, PromptFormat::Raw);
    assert_eq!(prompt, preformatted);
    assert!(!add_bos);
}

#[test]
fn llama3_stop_ids_include_end_of_turn() {
    let tokenizer = Tokenizer::from_gguf_metadata(&llama3_meta()).expect("tokenizer");
    let stopping = StoppingCriteria::with_stop_ids(Some(8), tokenizer.stop_ids());
    assert!(stopping.should_stop(&[128_009]));
    assert!(stopping.should_stop(&[128_001]));
}

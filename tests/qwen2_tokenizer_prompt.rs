use std::collections::HashMap;

use m40_llm::decode::StoppingCriteria;
use m40_llm::generate::{prepare_prompt, PromptFormat};
use m40_llm::gguf::{GgufScalar, GgufValue};
use m40_llm::tokenizer::{Tokenizer, TokenizerKind};

fn make_array(values: Vec<String>) -> GgufValue {
    GgufValue::Array(values.into_iter().map(GgufScalar::Str).collect())
}

fn qwen2_meta() -> HashMap<String, GgufValue> {
    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".to_string(),
        GgufValue::Scalar(GgufScalar::Str("qwen2".to_string())),
    );
    meta.insert(
        "tokenizer.ggml.model".to_string(),
        GgufValue::Scalar(GgufScalar::Str("gpt2".to_string())),
    );
    meta.insert(
        "tokenizer.ggml.pre".to_string(),
        GgufValue::Scalar(GgufScalar::Str("qwen2".to_string())),
    );
    meta.insert(
        "qwen2.vocab_size".to_string(),
        GgufValue::Scalar(GgufScalar::U32(151_646)),
    );
    meta.insert(
        "tokenizer.ggml.eos_token_id".to_string(),
        GgufValue::Scalar(GgufScalar::U32(151_645)),
    );
    meta.insert(
        "tokenizer.chat_template".to_string(),
        GgufValue::Scalar(GgufScalar::Str("qwen2".to_string())),
    );
    let mut tokens = vec![String::new(); 151_646];
    tokens[151_643] = "<|endoftext|>".to_string();
    tokens[151_644] = "<|im_start|>".to_string();
    tokens[151_645] = "<|im_end|>".to_string();
    meta.insert("tokenizer.ggml.tokens".to_string(), make_array(tokens));
    meta
}

#[test]
fn qwen2_metadata_selects_qwen_tokenizer() {
    let tokenizer = Tokenizer::from_gguf_metadata(&qwen2_meta()).expect("tokenizer");
    assert!(matches!(tokenizer.kind(), TokenizerKind::Qwen2));
    assert!(tokenizer.has_chat_template());
    assert_eq!(tokenizer.token_id("<|im_start|>"), Some(151_644));
    assert_eq!(tokenizer.token_id("<|im_end|>"), Some(151_645));
}

#[test]
fn qwen2_chat_prompt_preserves_control_tokens() {
    let tokenizer = Tokenizer::from_gguf_metadata(&qwen2_meta()).expect("tokenizer");
    let (prompt, add_bos) = prepare_prompt(&tokenizer, "Hello", PromptFormat::QwenChat);
    assert!(!add_bos);
    assert!(prompt.starts_with("<|im_start|>user\n"));
    assert!(prompt.ends_with("<|im_start|>assistant\n"));

    let ids = tokenizer
        .encode_with_specials(&prompt, add_bos, false)
        .expect("encode prompt");
    assert_eq!(ids.first().copied(), Some(151_644));
    assert!(ids.contains(&151_645));
}

#[test]
fn auto_prompt_format_selects_qwen_chat_for_unformatted_prompt() {
    let tokenizer = Tokenizer::from_gguf_metadata(&qwen2_meta()).expect("tokenizer");
    let (prompt, add_bos) = prepare_prompt(&tokenizer, "Hello", PromptFormat::Auto);
    assert!(!add_bos);
    assert!(prompt.starts_with("<|im_start|>user\n"));
    assert!(prompt.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn qwen2_stop_ids_include_im_end() {
    let tokenizer = Tokenizer::from_gguf_metadata(&qwen2_meta()).expect("tokenizer");
    let stopping = StoppingCriteria::with_stop_ids(Some(8), tokenizer.stop_ids());
    assert!(stopping.should_stop(&[151_643]));
    assert!(stopping.should_stop(&[151_645]));
}

#[test]
fn qwen2_uses_tiktoken_backend_for_text_roundtrip() {
    let tokenizer = Tokenizer::from_gguf_metadata(&qwen2_meta()).expect("tokenizer");
    let text = "Hello, please answer with the word OK.";
    let ids = tokenizer.encode(text).expect("encode");
    assert_eq!(ids, tiktoken::get_encoding("qwen2").unwrap().encode(text));
    assert_eq!(tokenizer.decode(&ids).expect("decode"), text);
}

#[test]
fn qwen2_special_token_roundtrip_uses_real_ids() {
    let tokenizer = Tokenizer::from_gguf_metadata(&qwen2_meta()).expect("tokenizer");
    let text = "<|im_start|>user\nhello<|im_end|>";
    let ids = tokenizer.encode_allowing_specials(text).expect("encode");
    assert_eq!(ids.first().copied(), Some(151_644));
    assert_eq!(ids.last().copied(), Some(151_645));
    assert_eq!(tokenizer.decode(&ids).expect("decode"), text);
}

/// Preserve caller-supplied Qwen ChatML without adding a hidden BOS, even when
/// the model declares a BOS ID or the caller already included that token.
#[test]
fn test_preformatted_qwen_prompt_preserves_token_ids() {
    let mut metadata = qwen2_meta();
    metadata.insert(
        "tokenizer.ggml.bos_token_id".to_string(),
        GgufValue::Scalar(GgufScalar::U32(151_643)),
    );
    let tokenizer = Tokenizer::from_gguf_metadata(&metadata).expect("tokenizer");
    let chat = "<|im_start|>system\nYou are Qwen.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
    for preformatted in [chat.to_string(), format!("<|endoftext|>{chat}")] {
        let expected = tokenizer
            .encode_allowing_specials(&preformatted)
            .expect("encode original");
        for prompt_format in [PromptFormat::Raw, PromptFormat::Auto] {
            let (prepared, add_bos) = prepare_prompt(&tokenizer, &preformatted, prompt_format);
            assert!(!add_bos);
            assert_eq!(prepared, preformatted);
            let actual = tokenizer
                .encode_with_specials(&prepared, add_bos, false)
                .expect("encode prepared");
            assert_eq!(actual, expected);
        }
    }
}

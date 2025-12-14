use m40_llm::tokenizer::Tokenizer;
use std::collections::HashMap;

#[test]
fn byte_level_roundtrip_ascii() {
    let t = Tokenizer::byte_level();
    let s = "Hello, GPU!";
    let ids = t.encode(s).expect("encode");
    assert_eq!(ids.len(), s.len());
    let back = t.decode(&ids).expect("decode");
    assert_eq!(back, s);
}

#[test]
fn byte_level_roundtrip_utf8_multibyte() {
    let t = Tokenizer::byte_level();
    let s = "café ☕"; // contains multibyte UTF-8
    let ids = t.encode(s).expect("encode");
    let back = t.decode(&ids).expect("decode");
    assert_eq!(back, s);
}

#[test]
fn decode_replaces_out_of_range_with_qmark() {
    let t = Tokenizer::byte_level();
    let ids = vec![65u32, 300u32, 66u32];
    let back = t.decode(&ids).expect("decode");
    assert!(!back.contains('\0'));
    assert_eq!(back, "A?B");
}

#[test]
fn from_gguf_metadata_fallbacks_to_byte_level() {
    let meta: HashMap<String, m40_llm::gguf::GgufValue> = HashMap::new();
    let t = Tokenizer::from_gguf_metadata(&meta).expect("from_gguf_metadata");
    let ids = t.encode("ok").unwrap();
    assert_eq!(ids, vec![b'o' as u32, b'k' as u32]);
}

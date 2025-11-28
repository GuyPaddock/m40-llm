use m40_llm::tokenizer::Tokenizer;

#[test]
fn encode_with_bos_eos_when_ids_present() {
    let mut t = Tokenizer::byte_level();
    // simulate GGUF-derived specials
    t.set_bos_id(Some(1));
    t.set_eos_id(Some(2));

    let ids = t.encode_with_specials("hi", true, true).unwrap();
    assert_eq!(ids[0], 1);
    assert_eq!(*ids.last().unwrap(), 2);
}

#[test]
fn decode_ignores_non_content_specials() {
    let mut t = Tokenizer::byte_level();
    t.set_bos_id(Some(10));
    t.set_eos_id(Some(11));
    t.set_pad_id(Some(12));

    // <BOS> 'A' <PAD> 'B' <EOS>
    let ids = vec![10u32, b'A' as u32, 12u32, b'B' as u32, 11u32];
    let s = t.decode_ignoring_specials(&ids).unwrap();
    assert_eq!(s, "AB");
}

#[test]
fn is_special_and_strip_helpers() {
    let mut t = Tokenizer::byte_level();
    t.set_bos_id(Some(7));
    t.set_eos_id(Some(8));
    t.set_pad_id(Some(9));
    t.set_unk_id(Some(13));

    assert!(t.is_special(7));
    assert!(t.is_special(8));
    assert!(t.is_special(9));
    assert!(!t.is_special(13)); // UNK considered content-bearing

    let ids = vec![7u32, 65u32, 13u32, 9u32, 66u32, 8u32];
    let stripped = t.strip_non_content_specials(&ids);
    assert_eq!(stripped, vec![65u32, 13u32, 66u32]);
}

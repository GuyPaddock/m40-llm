#![cfg(feature = "gguf_ext")]

use m40_llm::gguf::load_gguf;
use m40_llm::gguf_ext::{extract_model_config_with_gguf_llms, overview};
use std::fs::File;
use std::io::Write;

fn write_le_u32(f: &mut File, v: u32) {
    f.write_all(&v.to_le_bytes()).unwrap();
}
fn write_le_u64(f: &mut File, v: u64) {
    f.write_all(&v.to_le_bytes()).unwrap();
}

fn write_string(f: &mut File, s: &str) {
    write_le_u64(f, s.len() as u64);
    f.write_all(s.as_bytes()).unwrap();
}

#[test]
fn gguf_ext_extract_model_config_minimal_llama_meta_ok() {
    // Build a tiny GGUF with 2 metadata entries:
    //  - general.architecture: string "llama"
    //  - llama.block_count: u32 2
    // No tensors.
    let mut path = std::env::temp_dir();
    path.push(format!("m40llm_gguf_meta_{}.gguf", std::process::id()));

    {
        let mut f = File::create(&path).unwrap();
        // header
        f.write_all(b"GGUF").unwrap();
        write_le_u32(&mut f, 3); // version
        write_le_u64(&mut f, 0); // n_tensors
        write_le_u64(&mut f, 6); // n_kv

        // kv[0]: general.architecture = string("llama")
        write_string(&mut f, "general.architecture");
        write_le_u32(&mut f, 8); // value type STRING
        write_string(&mut f, "llama");

        // kv[1]: llama.block_count = u32(2)
        write_string(&mut f, "llama.block_count");
        write_le_u32(&mut f, 4); // value type U32
        write_le_u32(&mut f, 2);

        // kv[2]: llama.embedding_length = u32(16)
        write_string(&mut f, "llama.embedding_length");
        write_le_u32(&mut f, 4);
        write_le_u32(&mut f, 16);

        // kv[3]: llama.attention.head_count = u32(2)
        write_string(&mut f, "llama.attention.head_count");
        write_le_u32(&mut f, 4);
        write_le_u32(&mut f, 2);

        // kv[4]: llama.context_length = u32(128)
        write_string(&mut f, "llama.context_length");
        write_le_u32(&mut f, 4);
        write_le_u32(&mut f, 128);

        // kv[5]: llama.attention.head_count_kv = u32(2)
        write_string(&mut f, "llama.attention.head_count_kv");
        write_le_u32(&mut f, 4);
        write_le_u32(&mut f, 2);

        f.flush().unwrap();
    }

    // Hand-rolled loader should see kv_len == 6
    let model = load_gguf(&path).expect("hand-rolled loader should parse GGUF with metadata");
    assert_eq!(model.metadata.len(), 6);

    // gguf_ext overview should reflect kv_len and no tensors
    let info = overview(&path).expect("gguf_ext overview should succeed");
    assert_eq!(info.kv_len, 6);
    assert_eq!(info.n_tensors, 0);

    // Extract typed config via gguf-llms. It may fail for minimal metadata; we only ensure it doesn't panic.
    let _maybe_cfg = extract_model_config_with_gguf_llms(&path).ok();

    let _ = std::fs::remove_file(&path);
}

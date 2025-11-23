#![cfg(feature = "gguf_ext")]

use m40_llm::gguf::load_gguf;
use m40_llm::gguf_ext::load_all_tensors_with_llms;
use std::fs::File;
use std::io::Write;

fn write_le_u32(f: &mut File, v: u32) {
    f.write_all(&v.to_le_bytes()).unwrap();
}
fn write_le_u64(f: &mut File, v: u64) {
    f.write_all(&v.to_le_bytes()).unwrap();
}

#[test]
fn gguf_ext_load_all_tensors_empty_file_ok() {
    // Construct a minimal valid GGUF file with no metadata and no tensors.
    // Layout: magic("GGUF"), version(u32), n_tensors(u64)=0, n_kv(u64)=0. No further data.
    let mut path = std::env::temp_dir();
    path.push(format!("m40llm_gguf_empty_{}.gguf", std::process::id()));

    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"GGUF").unwrap();
        write_le_u32(&mut f, 3); // arbitrary version
        write_le_u64(&mut f, 0); // n_tensors
        write_le_u64(&mut f, 0); // n_kv
        f.flush().unwrap();
    }

    // External loader via gguf-llms should produce empty tensor map
    let map = load_all_tensors_with_llms(&path).expect("load_all_tensors should succeed on empty");
    assert!(map.is_empty());

    // Hand-rolled loader should parse header and compute data_offset correctly
    let model = load_gguf(&path).expect("hand-rolled loader should parse empty GGUF");
    assert_eq!(model.data_offset, 4 + 4 + 8 + 8);

    let _ = std::fs::remove_file(&path);
}

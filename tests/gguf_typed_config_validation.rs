#![cfg(feature = "gguf_ext")]

use m40_llm::gguf::load_gguf;
use m40_llm::infer::LoadedModel;
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
fn gguf_ext_typed_config_divisibility_error_surfaces() {
    // embedding_length=10 with attention.head_count=3 should fail validation
    let mut path = std::env::temp_dir();
    path.push(format!(
        "m40llm_gguf_ext_invalid_{}.gguf",
        std::process::id()
    ));

    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"GGUF").unwrap();
        write_le_u32(&mut f, 3); // version
        write_le_u64(&mut f, 0); // n_tensors
                                 // metadata: arch, block_count, context_length, feed_forward_length, embedding_length, head_count
        write_le_u64(&mut f, 6);

        write_string(&mut f, "general.architecture");
        write_le_u32(&mut f, 8); // STRING
        write_string(&mut f, "llama");

        write_string(&mut f, "llama.block_count");
        write_le_u32(&mut f, 4); // U32
        write_le_u32(&mut f, 1);

        write_string(&mut f, "llama.context_length");
        write_le_u32(&mut f, 4); // U32
        write_le_u32(&mut f, 8);

        write_string(&mut f, "llama.feed_forward_length");
        write_le_u32(&mut f, 4); // U32
        write_le_u32(&mut f, 32);

        write_string(&mut f, "llama.embedding_length");
        write_le_u32(&mut f, 4); // U32
        write_le_u32(&mut f, 10);

        write_string(&mut f, "llama.attention.head_count");
        write_le_u32(&mut f, 4); // U32
        write_le_u32(&mut f, 3);

        f.flush().unwrap();
    }

    let gg = load_gguf(&path).expect("parse gguf");
    let bytes = std::fs::read(&path).unwrap();
    let err = match LoadedModel::from_gguf(gg, bytes, -1) {
        Ok(_) => panic!("from_gguf should have failed validation"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    eprintln!("validation error: {msg}");
    assert!(msg.contains("not divisible"));

    let _ = std::fs::remove_file(&path);
}

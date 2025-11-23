#![cfg(not(feature = "cuda"))]

use m40_llm::gguf::{load_gguf, GgufScalar, GgufValue};
use std::fs::File;
use std::io::Write;

fn write_le_u32(f: &mut File, v: u32) {
    f.write_all(&v.to_le_bytes()).unwrap();
}
fn write_le_u64(f: &mut File, v: u64) {
    f.write_all(&v.to_le_bytes()).unwrap();
}
fn write_le_f32(f: &mut File, v: f32) {
    f.write_all(&v.to_le_bytes()).unwrap();
}
fn write_string(f: &mut File, s: &str) {
    write_le_u64(f, s.len() as u64);
    f.write_all(s.as_bytes()).unwrap();
}

#[test]
fn gguf_metadata_llama_core_params() {
    // Build GGUF with core llama.* metadata and no tensors
    let mut path = std::env::temp_dir();
    path.push(format!("m40llm_gguf_meta_core_{}.gguf", std::process::id()));

    {
        let mut f = File::create(&path).unwrap();
        // header
        f.write_all(b"GGUF").unwrap();
        write_le_u32(&mut f, 3); // version
        write_le_u64(&mut f, 0); // n_tensors
        write_le_u64(&mut f, 8); // n_kv

        // general.architecture = "llama"
        write_string(&mut f, "general.architecture");
        write_le_u32(&mut f, 8); // STRING
        write_string(&mut f, "llama");

        // llama.block_count = u32(4)
        write_string(&mut f, "llama.block_count");
        write_le_u32(&mut f, 4); // U32
        write_le_u32(&mut f, 4);

        // llama.embedding_length = u32(32)
        write_string(&mut f, "llama.embedding_length");
        write_le_u32(&mut f, 4); // U32
        write_le_u32(&mut f, 32);

        // llama.attention.head_count = u32(4)
        write_string(&mut f, "llama.attention.head_count");
        write_le_u32(&mut f, 4);
        write_le_u32(&mut f, 4);

        // llama.attention.head_count_kv = u32(2)
        write_string(&mut f, "llama.attention.head_count_kv");
        write_le_u32(&mut f, 4);
        write_le_u32(&mut f, 2);

        // llama.context_length = u32(128)
        write_string(&mut f, "llama.context_length");
        write_le_u32(&mut f, 4);
        write_le_u32(&mut f, 128);

        // llama.rope.freq_base = f32(10000.0)
        write_string(&mut f, "llama.rope.freq_base");
        write_le_u32(&mut f, 6); // F32
        write_le_f32(&mut f, 10000.0);

        // llama.rope.freq_scale = f32(1.0)
        write_string(&mut f, "llama.rope.freq_scale");
        write_le_u32(&mut f, 6); // F32
        write_le_f32(&mut f, 1.0);

        f.flush().unwrap();
    }

    let gg = load_gguf(&path).expect("parse gguf");
    assert_eq!(gg.tensors.len(), 0);
    assert_eq!(gg.metadata.len(), 8);

    // Basic metadata assertions
    assert_eq!(
        gg.metadata
            .get("general.architecture")
            .and_then(GgufValue::as_str),
        Some("llama")
    );

    fn as_u32(v: &GgufValue) -> Option<u32> {
        match v {
            GgufValue::Scalar(GgufScalar::U32(x)) => Some(*x),
            GgufValue::Scalar(GgufScalar::U64(x)) => (*x).try_into().ok(),
            GgufValue::Scalar(GgufScalar::I32(x)) if *x >= 0 => Some(*x as u32),
            GgufValue::Scalar(GgufScalar::I64(x)) if *x >= 0 => (*x as u64).try_into().ok(),
            _ => None,
        }
    }
    fn as_f32(v: &GgufValue) -> Option<f32> {
        match v {
            GgufValue::Scalar(GgufScalar::F32(x)) => Some(*x),
            GgufValue::Scalar(GgufScalar::F64(x)) => Some(*x as f32),
            _ => None,
        }
    }

    let n_layer = as_u32(gg.metadata.get("llama.block_count").unwrap()).unwrap() as usize;
    let d_model = as_u32(gg.metadata.get("llama.embedding_length").unwrap()).unwrap() as usize;
    let n_head = as_u32(gg.metadata.get("llama.attention.head_count").unwrap()).unwrap() as usize;
    let n_head_kv =
        as_u32(gg.metadata.get("llama.attention.head_count_kv").unwrap()).unwrap() as usize;
    let context_len = as_u32(gg.metadata.get("llama.context_length").unwrap()).unwrap() as usize;
    let rope_base = as_f32(gg.metadata.get("llama.rope.freq_base").unwrap()).unwrap();
    let rope_scale = as_f32(gg.metadata.get("llama.rope.freq_scale").unwrap()).unwrap();

    assert_eq!(n_layer, 4);
    assert_eq!(d_model, 32);
    assert_eq!(n_head, 4);
    assert_eq!(n_head_kv, 2);
    assert_eq!(context_len, 128);
    assert!((rope_base - 10000.0).abs() < 1e-6);
    assert!((rope_scale - 1.0).abs() < 1e-6);

    // Derived: head_dim = d_model / n_head
    assert_eq!(d_model % n_head, 0);
    let head_dim = d_model / n_head;
    assert_eq!(head_dim, 8);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn gguf_metadata_vocab_fallback_from_tensor_shape() {
    // Build GGUF with no llama.vocab_size, but include token embedding tensor
    let mut path = std::env::temp_dir();
    path.push(format!(
        "m40llm_gguf_vocab_fallback_{}.gguf",
        std::process::id()
    ));

    let vocab: u64 = 17;
    let d_model: u64 = 32;

    {
        let mut f = File::create(&path).unwrap();
        // header
        f.write_all(b"GGUF").unwrap();
        write_le_u32(&mut f, 3); // version
        write_le_u64(&mut f, 1); // n_tensors
        write_le_u64(&mut f, 4); // n_kv

        // Minimal llama hparams (no vocab size key)
        write_string(&mut f, "general.architecture");
        write_le_u32(&mut f, 8); // STRING
        write_string(&mut f, "llama");

        write_string(&mut f, "llama.embedding_length");
        write_le_u32(&mut f, 4); // U32
        write_le_u32(&mut f, d_model as u32);

        write_string(&mut f, "llama.attention.head_count");
        write_le_u32(&mut f, 4);
        write_le_u32(&mut f, 4);

        write_string(&mut f, "llama.block_count");
        write_le_u32(&mut f, 4);
        write_le_u32(&mut f, 4);

        // tensor[0]: tok_embeddings.weight with shape [vocab, d_model], dtype F16 (1), offset 0
        write_string(&mut f, "tok_embeddings.weight");
        write_le_u32(&mut f, 2); // n_dims
        write_le_u64(&mut f, vocab);
        write_le_u64(&mut f, d_model);
        write_le_u32(&mut f, 1); // F16
        write_le_u64(&mut f, 0); // offset into data region

        // data region: fill with zeros of size vocab * d_model * sizeof(f16)
        let nbytes = (vocab * d_model) as usize * 2;
        f.write_all(&vec![0u8; nbytes]).unwrap();
        f.flush().unwrap();
    }

    let gg = load_gguf(&path).expect("parse gguf");
    assert_eq!(gg.tensors.len(), 1);
    let emb = gg
        .tensors
        .iter()
        .find(|t| t.name == "tok_embeddings.weight")
        .unwrap();
    assert_eq!(emb.shape, vec![vocab, d_model]);

    // Derive vocab from tensor shape[0]
    let derived_vocab = emb.shape[0] as usize;
    assert_eq!(derived_vocab, vocab as usize);

    // Sanity: head_dim is consistent with d_model and head_count
    let d_model_meta = match gg.metadata.get("llama.embedding_length").unwrap() {
        GgufValue::Scalar(GgufScalar::U32(x)) => *x as usize,
        _ => panic!("bad type"),
    };
    let n_head_meta = match gg.metadata.get("llama.attention.head_count").unwrap() {
        GgufValue::Scalar(GgufScalar::U32(x)) => *x as usize,
        _ => panic!("bad type"),
    };
    assert_eq!(d_model_meta % n_head_meta, 0);

    let _ = std::fs::remove_file(&path);
}

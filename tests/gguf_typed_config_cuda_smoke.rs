#![cfg(all(feature = "cuda", feature = "gguf_ext", nvcc))]

mod cuda_env;

use anyhow::Result;
use m40_llm::gguf::load_gguf;
use m40_llm::infer::LoadedModel;
use std::fs::File;
use std::io::{Read, Write};

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
fn gguf_typed_config_and_kv_layout_cuda_smoke() -> Result<()> {
    // Build a minimal-but-valid GGUF file on disk so gguf-llms can parse typed config
    let mut path = std::env::temp_dir();
    path.push(format!(
        "m40llm_gguf_typed_cuda_{}.gguf",
        std::process::id()
    ));

    let d_model = 32u64;
    let n_head = 4u64;
    let head_dim = d_model / n_head; // = 8
    let n_kv = n_head;
    let n_q = n_head * head_dim; // 32
    let n_k = n_kv * head_dim; // 32
    let n_v = n_kv * head_dim; // 32
    let h = 64u64;

    {
        let mut f = File::create(&path).unwrap();
        // header
        f.write_all(b"GGUF").unwrap();
        write_le_u32(&mut f, 3); // version
        let n_tensors = 8u64; // emb + q k v o + gate up down
        write_le_u64(&mut f, n_tensors);
        let n_kv = 6u64; // arch, block_count, context_length, embedding_length, feed_forward_length, head_count
        write_le_u64(&mut f, n_kv);

        // metadata
        write_string(&mut f, "general.architecture");
        write_le_u32(&mut f, 8); // STRING
        write_string(&mut f, "llama");

        write_string(&mut f, "llama.block_count");
        write_le_u32(&mut f, 4); // U32
        write_le_u32(&mut f, 1);

        write_string(&mut f, "llama.context_length");
        write_le_u32(&mut f, 4); // U32
        write_le_u32(&mut f, 16);

        write_string(&mut f, "llama.embedding_length");
        write_le_u32(&mut f, 4); // U32
        write_le_u32(&mut f, d_model as u32);

        write_string(&mut f, "llama.feed_forward_length");
        write_le_u32(&mut f, 4); // U32
        write_le_u32(&mut f, h as u32);

        write_string(&mut f, "llama.attention.head_count");
        write_le_u32(&mut f, 4); // U32
        write_le_u32(&mut f, n_head as u32);

        // tensor descriptors
        let mut off: u64 = 0;
        let emb_k = 128u64;
        let emb_n = d_model; // [vocab, d_model]
                             // emb
        write_string(&mut f, "tok_embeddings.weight");
        write_le_u32(&mut f, 2); // n_dims
        write_le_u64(&mut f, emb_k);
        write_le_u64(&mut f, emb_n);
        write_le_u32(&mut f, 1); // F16
        write_le_u64(&mut f, off);
        off += emb_k * emb_n * 2;

        // helper
        let mut tensor = |name: &str, k: u64, n: u64| {
            write_string(&mut f, name);
            write_le_u32(&mut f, 2);
            write_le_u64(&mut f, k);
            write_le_u64(&mut f, n);
            write_le_u32(&mut f, 1); // F16
            write_le_u64(&mut f, off);
            off += k * n * 2;
        };
        tensor("blk.0.attn_q.weight", d_model, n_q);
        tensor("blk.0.attn_k.weight", d_model, n_k);
        tensor("blk.0.attn_v.weight", d_model, n_v);
        tensor("blk.0.attn_output.weight", d_model, d_model);
        tensor("blk.0.ffn_gate.weight", d_model, h);
        tensor("blk.0.ffn_up.weight", d_model, h);
        tensor("blk.0.ffn_down.weight", h, d_model);

        // data region
        let data_bytes = off as usize;
        f.write_all(&vec![0u8; data_bytes]).unwrap();
        f.flush().unwrap();
    }

    // Parse with our hand-rolled loader and read bytes
    let gg = load_gguf(&path).expect("parse gguf");
    let mut file_bytes = Vec::new();
    {
        let mut f = File::open(&path).unwrap();
        f.read_to_end(&mut file_bytes).unwrap();
    }

    // Construct model; typed_config should be Some
    let mut lm = LoadedModel::from_gguf(gg, file_bytes, -1)?;
    let cfg = lm.typed_config.as_ref().expect("typed_config Some");
    assert_eq!(cfg.embedding_length as u64, d_model);
    assert_eq!(cfg.attention_head_count as u64, n_head);

    // Map and verify shapes
    let w = lm.map_standard_layer(0)?;
    assert_eq!(w.d_model as u64, d_model);

    // KV allocation pulls from typed_config
    lm.allocate_kv_cache(16, 1)?;
    let kv = lm.kv_cache.as_ref().unwrap();
    assert_eq!(kv.num_heads(), n_head as u32);
    assert_eq!(kv.head_dim(), head_dim as u32);

    let _ = std::fs::remove_file(&path);
    Ok(())
}

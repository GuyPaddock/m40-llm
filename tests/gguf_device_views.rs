#![cfg(not(feature = "cuda"))]
use m40_llm::gguf::{load_gguf, GgmlDType};
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
fn write_kv_str(f: &mut File, key: &str, val: &str) {
    write_string(f, key);
    write_le_u32(f, 8); // string type
    write_string(f, val);
}
fn write_kv_u32(f: &mut File, key: &str, val: u32) {
    write_string(f, key);
    write_le_u32(f, 4); // u32 type
    write_le_u32(f, val);
}
fn write_minimal_metadata(f: &mut File) {
    write_kv_str(f, "general.architecture", "llama");
    write_kv_u32(f, "llama.embedding_length", 4);
    write_kv_u32(f, "llama.attention.head_count", 1);
    write_kv_u32(f, "llama.block_count", 1);
    write_kv_u32(f, "llama.context_length", 16);
    write_kv_u32(f, "llama.feed_forward_length", 8);
    write_kv_u32(f, "llama.vocab_size", 32);
}

#[test]
fn gguf_device_views_basic_f16_size_and_shape() {
    // Build GGUF with 0 kvs and 1 tensor: name "W", shape [K=4, N=8], dtype F16, offset 0
    let mut path = std::env::temp_dir();
    path.push(format!("m40llm_gguf_tensor_{}.gguf", std::process::id()));

    let (k, n) = (4u64, 8u64);
    let dtype = 1u32; // F16 in our mapping

    {
        let mut f = File::create(&path).unwrap();
        // header
        f.write_all(b"GGUF").unwrap();
        write_le_u32(&mut f, 3); // version
        write_le_u64(&mut f, 1); // n_tensors
        write_le_u64(&mut f, 7); // n_kv

        // metadata
        write_minimal_metadata(&mut f);

        // tensor[0]
        write_string(&mut f, "W");
        write_le_u32(&mut f, 2); // n_dims
        write_le_u64(&mut f, k);
        write_le_u64(&mut f, n);
        write_le_u32(&mut f, dtype); // F16
        write_le_u64(&mut f, 0); // offset from data region

        // data region: fill with zeros of size bytes_per_elem * K * N
        let nbytes = (k * n) as usize * 2; // f16
        f.write_all(&vec![0u8; nbytes]).unwrap();
        f.flush().unwrap();
    }

    let gg = load_gguf(&path).expect("parse gguf");
    assert_eq!(gg.tensors.len(), 1);
    assert!(gg.data_offset > 0);

    // Read all bytes and construct LoadedModel (device  -1 auto)
    let bytes = std::fs::read(&path).unwrap();
    let lm = LoadedModel::from_gguf(gg, bytes, -1).expect("from_gguf");

    let w = lm.device_tensors.get("W").expect("W view exists");
    assert!(matches!(w.dtype, GgmlDType::F16));
    assert_eq!(w.shape, vec![k, n]);
    assert_eq!(w.byte_offset, 0);
    assert_eq!(w.nbytes, (k * n) as usize * 2);
    assert_eq!(w.strides, vec![n as usize, 1]);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn gguf_device_views_f32_size_and_shape() {
    // GGUF with 1 tensor F32 [3,5]
    let mut path = std::env::temp_dir();
    path.push(format!(
        "m40llm_gguf_tensor_f32_{}.gguf",
        std::process::id()
    ));

    let (k, n) = (3u64, 5u64);
    let dtype = 0u32; // F32 in our mapping

    {
        let mut f = File::create(&path).unwrap();
        // header
        f.write_all(b"GGUF").unwrap();
        write_le_u32(&mut f, 3); // version
        write_le_u64(&mut f, 1); // n_tensors
        write_le_u64(&mut f, 7); // n_kv

        write_minimal_metadata(&mut f);

        // tensor[0]
        write_string(&mut f, "W32");
        write_le_u32(&mut f, 2); // n_dims
        write_le_u64(&mut f, k);
        write_le_u64(&mut f, n);
        write_le_u32(&mut f, dtype); // F32
        write_le_u64(&mut f, 0); // offset from data region

        // data region: fill with zeros of size bytes_per_elem * K * N
        let nbytes = (k * n) as usize * 4; // f32
        f.write_all(&vec![0u8; nbytes]).unwrap();
        f.flush().unwrap();
    }

    let gg = load_gguf(&path).expect("parse gguf");
    assert_eq!(gg.tensors.len(), 1);
    assert!(gg.data_offset > 0);

    let bytes = std::fs::read(&path).unwrap();
    let lm = LoadedModel::from_gguf(gg, bytes, -1).expect("from_gguf");

    let w = lm.device_tensors.get("W32").expect("W32 view exists");
    assert!(matches!(w.dtype, GgmlDType::F32));
    assert_eq!(w.shape, vec![k, n]);
    assert_eq!(w.byte_offset, 0);
    assert_eq!(w.nbytes, (k * n) as usize * 4);
    assert_eq!(w.strides, vec![n as usize, 1]);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn gguf_device_views_reject_empty_shape() {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "m40llm_gguf_tensor_empty_shape_{}.gguf",
        std::process::id()
    ));

    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"GGUF").unwrap();
        write_le_u32(&mut f, 3);
        write_le_u64(&mut f, 1);
        write_le_u64(&mut f, 7);
        write_minimal_metadata(&mut f);

        write_string(&mut f, "empty");
        write_le_u32(&mut f, 0); // n_dims = 0
        write_le_u32(&mut f, 1); // dtype f16
        write_le_u64(&mut f, 0); // offset

        f.write_all(&[0u8; 2]).unwrap();
        f.flush().unwrap();
    }

    let gg = load_gguf(&path).expect("parse gguf");
    let bytes = std::fs::read(&path).unwrap();
    assert!(LoadedModel::from_gguf(gg, bytes, -1).is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn gguf_device_views_reject_zero_dim() {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "m40llm_gguf_tensor_zero_dim_{}.gguf",
        std::process::id()
    ));

    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"GGUF").unwrap();
        write_le_u32(&mut f, 3);
        write_le_u64(&mut f, 1);
        write_le_u64(&mut f, 7);
        write_minimal_metadata(&mut f);

        write_string(&mut f, "zero");
        write_le_u32(&mut f, 2);
        write_le_u64(&mut f, 4);
        write_le_u64(&mut f, 0); // zero dim
        write_le_u32(&mut f, 1); // dtype f16
        write_le_u64(&mut f, 0);

        f.write_all(&vec![0u8; 8]).unwrap();
        f.flush().unwrap();
    }

    let gg = load_gguf(&path).expect("parse gguf");
    let bytes = std::fs::read(&path).unwrap();
    assert!(LoadedModel::from_gguf(gg, bytes, -1).is_err());
    let _ = std::fs::remove_file(&path);
}

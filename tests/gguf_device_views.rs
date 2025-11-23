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
        write_le_u64(&mut f, 0); // n_kv

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
    assert_eq!(gg.data_offset > 0, true);

    // Read all bytes and construct LoadedModel (device  -1 auto)
    let bytes = std::fs::read(&path).unwrap();
    let lm = LoadedModel::from_gguf(gg, bytes, -1).expect("from_gguf");

    let w = lm.device_tensors.get("W").expect("W view exists");
    assert!(matches!(w.dtype, GgmlDType::F16));
    assert_eq!(w.shape, vec![k, n]);
    assert_eq!(w.byte_offset, 0);
    assert_eq!(w.nbytes, (k * n) as usize * 2);

    let _ = std::fs::remove_file(&path);
}

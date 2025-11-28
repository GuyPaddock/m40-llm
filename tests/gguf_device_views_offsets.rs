#![cfg(not(feature = "cuda"))]
use m40_llm::gguf::{load_gguf, GgmlDType};
use m40_llm::infer::LoadedModel;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

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

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "m40llm_{}_{}_{}.gguf",
        name,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    ));
    p
}

#[test]
fn gguf_device_views_nonzero_offsets_multiple_tensors_ok() {
    // Two tensors: A [2,4] F16 at offset 0, B [3,3] F16 at non-zero offset with padding
    let path = tmp_path("gguf_offsets_ok");
    let (a_shape, b_shape) = ((2u64, 4u64), (3u64, 3u64));
    let f16 = 1u32;

    let (a_nbytes, b_nbytes) = (
        (a_shape.0 * a_shape.1) as usize * 2,
        (b_shape.0 * b_shape.1) as usize * 2,
    );
    let a_off = 0u64;
    let pad = 16usize; // arbitrary padding between tensors
    let b_off = (a_off as usize + a_nbytes + pad) as u64;

    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"GGUF").unwrap();
        write_le_u32(&mut f, 3); // version
        write_le_u64(&mut f, 2); // n_tensors
        write_le_u64(&mut f, 0); // n_kv

        // Tensor A
        write_string(&mut f, "A");
        write_le_u32(&mut f, 2); // n_dims
        write_le_u64(&mut f, a_shape.0);
        write_le_u64(&mut f, a_shape.1);
        write_le_u32(&mut f, f16);
        write_le_u64(&mut f, a_off);

        // Tensor B
        write_string(&mut f, "B");
        write_le_u32(&mut f, 2);
        write_le_u64(&mut f, b_shape.0);
        write_le_u64(&mut f, b_shape.1);
        write_le_u32(&mut f, f16);
        write_le_u64(&mut f, b_off);

        // Data region: need to cover B's end
        let data_len = (b_off as usize) + b_nbytes;
        f.write_all(&vec![0u8; data_len]).unwrap(); // zeroed payload is fine
        f.flush().unwrap();
    }

    let gg = load_gguf(&path).expect("parse gguf");
    assert_eq!(gg.tensors.len(), 2);
    let bytes = std::fs::read(&path).unwrap();
    let lm = LoadedModel::from_gguf(gg, bytes, -1).expect("from_gguf ok");

    let a = lm.device_tensors.get("A").unwrap();
    assert!(matches!(a.dtype, GgmlDType::F16));
    assert_eq!(a.shape, vec![a_shape.0, a_shape.1]);
    assert_eq!(a.byte_offset, a_off);
    assert_eq!(a.nbytes, a_nbytes);

    let b = lm.device_tensors.get("B").unwrap();
    assert!(matches!(b.dtype, GgmlDType::F16));
    assert_eq!(b.shape, vec![b_shape.0, b_shape.1]);
    assert_eq!(b.byte_offset, b_off);
    assert_eq!(b.nbytes, b_nbytes);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn gguf_device_views_overflow_detected() {
    // One tensor that would overflow given too-short data region
    let path = tmp_path("gguf_overflow");
    let (k, n) = (4u64, 8u64);
    let f16 = 1u32;
    let nbytes_needed = (k * n) as usize * 2;

    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"GGUF").unwrap();
        write_le_u32(&mut f, 3);
        write_le_u64(&mut f, 1);
        write_le_u64(&mut f, 0);

        write_string(&mut f, "W");
        write_le_u32(&mut f, 2);
        write_le_u64(&mut f, k);
        write_le_u64(&mut f, n);
        write_le_u32(&mut f, f16);
        write_le_u64(&mut f, 0);

        // Intentionally short by 1 byte
        f.write_all(&vec![0u8; nbytes_needed - 1]).unwrap();
        f.flush().unwrap();
    }

    let gg = load_gguf(&path).expect("parse gguf");
    let bytes = std::fs::read(&path).unwrap();
    let err = LoadedModel::from_gguf(gg, bytes, -1)
        .err()
        .expect("should error");
    let msg = format!("{}", err);
    assert!(msg.contains("overflows weights blob"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn gguf_device_views_quantized_nbytes_zero() {
    // Quantized dtype gets nbytes = 0 in view; loader should not error
    let path = tmp_path("gguf_quant");
    let (k, n) = (2u64, 2u64);
    let q4_0 = 2u32; // per mapping in gguf.rs

    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"GGUF").unwrap();
        write_le_u32(&mut f, 3);
        write_le_u64(&mut f, 1);
        write_le_u64(&mut f, 0);

        write_string(&mut f, "Q");
        write_le_u32(&mut f, 2);
        write_le_u64(&mut f, k);
        write_le_u64(&mut f, n);
        write_le_u32(&mut f, q4_0);
        write_le_u64(&mut f, 0);

        // Provide small data region; loader doesn't validate unknown sizes
        f.write_all(&[0u8; 1]).unwrap();
        f.flush().unwrap();
    }

    let gg = load_gguf(&path).expect("parse gguf");
    let bytes = std::fs::read(&path).unwrap();
    let lm = LoadedModel::from_gguf(gg, bytes, -1).expect("from_gguf ok");

    let q = lm.device_tensors.get("Q").unwrap();
    assert_eq!(q.nbytes, 0);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn gguf_device_views_misaligned_offset_rejected() {
    // F32 requires 4-byte alignment; put at offset 2 to force misalignment
    let path = tmp_path("gguf_misaligned");
    let (k, n) = (1u64, 1u64);
    let f32_dtype = 0u32; // GgmlDType::F32 per mapping in gguf.rs

    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"GGUF").unwrap();
        write_le_u32(&mut f, 3);
        write_le_u64(&mut f, 1);
        write_le_u64(&mut f, 0);

        write_string(&mut f, "X");
        write_le_u32(&mut f, 2);
        write_le_u64(&mut f, k);
        write_le_u64(&mut f, n);
        write_le_u32(&mut f, f32_dtype);
        write_le_u64(&mut f, 2); // misaligned

        // Data region long enough but offset is misaligned
        f.write_all(&vec![0u8; 8]).unwrap();
        f.flush().unwrap();
    }

    let gg = load_gguf(&path).expect("parse gguf");
    let bytes = std::fs::read(&path).unwrap();
    let err = LoadedModel::from_gguf(gg, bytes, -1)
        .err()
        .expect("should error");
    let msg = format!("{}", err);
    assert!(msg.contains("misaligned"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn gguf_device_views_offset_beyond_end_rejected() {
    // Unknown-size dtype (simulate quantized) should still require offset < data_len
    let path = tmp_path("gguf_offset_beyond_end");
    let (k, n) = (1u64, 1u64);
    let q4_0 = 2u32; // treated as unknown nbytes in loader

    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"GGUF").unwrap();
        write_le_u32(&mut f, 3);
        write_le_u64(&mut f, 1);
        write_le_u64(&mut f, 0);

        write_string(&mut f, "Q");
        write_le_u32(&mut f, 2);
        write_le_u64(&mut f, k);
        write_le_u64(&mut f, n);
        write_le_u32(&mut f, q4_0);
        write_le_u64(&mut f, 4); // offset exactly at end of 4-byte data region

        // Provide 4 bytes only; offset==len should be rejected
        f.write_all(&vec![0u8; 4]).unwrap();
        f.flush().unwrap();
    }

    let gg = load_gguf(&path).expect("parse gguf");
    let bytes = std::fs::read(&path).unwrap();
    let err = LoadedModel::from_gguf(gg, bytes, -1)
        .err()
        .expect("should error");
    let msg = format!("{}", err);
    assert!(msg.contains("starts beyond end") || msg.contains("overflows weights blob"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn gguf_device_views_f16_misaligned_offset_rejected() {
    // F16 requires 2-byte alignment; put at offset 1 to force misalignment
    let path = tmp_path("gguf_f16_misaligned");
    let (k, n) = (1u64, 1u64);
    let f16 = 1u32; // GgmlDType::F16 per mapping

    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"GGUF").unwrap();
        write_le_u32(&mut f, 3);
        write_le_u64(&mut f, 1);
        write_le_u64(&mut f, 0);

        write_string(&mut f, "Y");
        write_le_u32(&mut f, 2);
        write_le_u64(&mut f, k);
        write_le_u64(&mut f, n);
        write_le_u32(&mut f, f16);
        write_le_u64(&mut f, 1); // misaligned for F16

        // Data region long enough but offset is misaligned
        f.write_all(&vec![0u8; 4]).unwrap();
        f.flush().unwrap();
    }

    let gg = load_gguf(&path).expect("parse gguf");
    let bytes = std::fs::read(&path).unwrap();
    let err = LoadedModel::from_gguf(gg, bytes, -1)
        .err()
        .expect("should error");
    let msg = format!("{}", err);
    assert!(msg.contains("misaligned"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn gguf_device_views_end_overflow_from_offset_rejected() {
    // Provide buffer length that is insufficient when starting from non-zero offset
    let path = tmp_path("gguf_end_overflow_from_offset");
    let (k, n) = (2u64, 2u64); // 4 elems F16 => 8 bytes needed
    let f16 = 1u32;
    let offset = 5u64; // 5 + 8 = 13, but we'll only provide 10 bytes

    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"GGUF").unwrap();
        write_le_u32(&mut f, 3);
        write_le_u64(&mut f, 1);
        write_le_u64(&mut f, 0);

        write_string(&mut f, "Z");
        write_le_u32(&mut f, 2);
        write_le_u64(&mut f, k);
        write_le_u64(&mut f, n);
        write_le_u32(&mut f, f16);
        write_le_u64(&mut f, offset);

        // Only 10 bytes in data region => overflow relative to offset
        f.write_all(&vec![0u8; 10]).unwrap();
        f.flush().unwrap();
    }

    let gg = load_gguf(&path).expect("parse gguf");
    let bytes = std::fs::read(&path).unwrap();
    let err = LoadedModel::from_gguf(gg, bytes, -1)
        .err()
        .expect("should error");
    let msg = format!("{}", err);
    assert!(msg.contains("overflows weights blob"));
    let _ = std::fs::remove_file(&path);
}

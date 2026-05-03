#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use half::f16;
use m40_llm::gguf::{GgmlDType, GgufModel, GgufScalar, GgufTensor, GgufValue};
use m40_llm::infer::LoadedModel;
use std::collections::HashMap;
use std::ffi::c_void;

fn metadata(vocab: usize, d_model: usize) -> HashMap<String, GgufValue> {
    use GgufScalar as S;
    let mut m = HashMap::new();
    m.insert(
        "general.architecture".into(),
        GgufValue::Scalar(S::Str("llama".into())),
    );
    m.insert(
        "llama.embedding_length".into(),
        GgufValue::Scalar(S::U32(d_model as u32)),
    );
    m.insert(
        "llama.attention.head_count".into(),
        GgufValue::Scalar(S::U32(1)),
    );
    m.insert("llama.block_count".into(), GgufValue::Scalar(S::U32(1)));
    m.insert("llama.context_length".into(), GgufValue::Scalar(S::U32(16)));
    m.insert(
        "llama.feed_forward_length".into(),
        GgufValue::Scalar(S::U32((d_model * 2) as u32)),
    );
    m.insert(
        "llama.vocab_size".into(),
        GgufValue::Scalar(S::U32(vocab as u32)),
    );
    m
}

fn model_with_embeddings(dtype: GgmlDType, vocab: usize, d_model: usize) -> GgufModel {
    let mut gguf = GgufModel::new(0);
    gguf.metadata = metadata(vocab, d_model);
    gguf.tensors.push(GgufTensor {
        name: "tok_embeddings.weight".into(),
        dtype,
        shape: vec![vocab as u64, d_model as u64],
        offset: 0,
    });
    gguf
}

fn read_device_f32(model: &LoadedModel, ptr: *const c_void, len: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; len * std::mem::size_of::<f32>()];
    unsafe {
        model
            .cuda
            .memcpy_d2h(bytes.as_mut_ptr() as *mut c_void, ptr, bytes.len())?;
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-3,
            "mismatch at {idx}: actual={a} expected={e}"
        );
    }
}

#[test]
fn f16_embedding_row_load_matches_host() -> Result<()> {
    let Some(ctx) = cuda_env::ctx_m40_or_skip() else {
        return Ok(());
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let vocab = 3;
    let d_model = 8;
    let token_id = 1u64;
    let values: Vec<f32> = (0..vocab * d_model)
        .map(|i| (i as f32 - 7.0) * 0.125)
        .collect();
    let mut weights = Vec::with_capacity(values.len() * 2);
    for &value in &values {
        weights.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
    }

    let model = LoadedModel::from_gguf(
        model_with_embeddings(GgmlDType::F16, vocab, d_model),
        weights,
        -1,
    )?;
    let out = model
        .cuda
        .device_malloc(d_model * std::mem::size_of::<f32>())?;
    unsafe {
        model.load_token_embedding_to_f32(token_id, out)?;
    }
    let actual = read_device_f32(&model, out as *const c_void, d_model)?;
    unsafe {
        model.cuda.device_free(out)?;
    }

    let expected: Vec<f32> = values[token_id as usize * d_model..(token_id as usize + 1) * d_model]
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect();
    assert_close(&actual, &expected);
    Ok(())
}

#[test]
fn q80_embedding_row_load_matches_host() -> Result<()> {
    let Some(ctx) = cuda_env::ctx_m40_or_skip() else {
        return Ok(());
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let vocab = 3;
    let d_model = 64;
    let token_id = 1u64;
    let blocks_per_row = d_model / 32;
    let mut weights = Vec::with_capacity(vocab * blocks_per_row * 34);
    let mut expected_rows = vec![vec![0f32; d_model]; vocab];

    for (row, expected) in expected_rows.iter_mut().enumerate() {
        for block in 0..blocks_per_row {
            let scale = f16::from_f32(0.25 + row as f32 * 0.125 + block as f32 * 0.0625);
            weights.extend_from_slice(&scale.to_bits().to_le_bytes());
            for idx in 0..32 {
                let q = ((row as i32 * 17 + block as i32 * 5 + idx as i32) % 23 - 11) as i8;
                weights.push(q as u8);
                expected[block * 32 + idx] = scale.to_f32() * q as f32;
            }
        }
    }

    let model = LoadedModel::from_gguf(
        model_with_embeddings(GgmlDType::Q8_0, vocab, d_model),
        weights,
        -1,
    )?;
    let out = model
        .cuda
        .device_malloc(d_model * std::mem::size_of::<f32>())?;
    unsafe {
        model.load_token_embedding_to_f32(token_id, out)?;
    }
    let actual = read_device_f32(&model, out as *const c_void, d_model)?;
    unsafe {
        model.cuda.device_free(out)?;
    }

    assert_close(&actual, &expected_rows[token_id as usize]);
    Ok(())
}

#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use half::f16;
use m40_llm::gguf::{GgmlDType, GgufModel, GgufScalar, GgufTensor, GgufValue};
use m40_llm::infer::LoadedModel;
use std::ffi::c_void;

fn halves_from_f32(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bits = f16::from_f32(v).to_bits();
        out.push((bits & 0xFF) as u8);
        out.push((bits >> 8) as u8);
    }
    out
}

fn halves_to_f32(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for i in (0..bytes.len()).step_by(2) {
        let bits = (bytes[i] as u16) | ((bytes[i + 1] as u16) << 8);
        out.push(f16::from_bits(bits).to_f32());
    }
    out
}

fn matmul_row_major(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    // a: [m x k], b: [k x n] -> c: [m x n]
    let mut c = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn rms_norm(x: &[f32], eps: f32) -> Vec<f32> {
    let d = x.len();
    let mut mean_sq = 0f32;
    for &v in x {
        mean_sq += v * v;
    }
    mean_sq /= d as f32;
    let s = 1.0f32 / (mean_sq + eps).sqrt();
    x.iter().map(|&v| v * s).collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[test]
fn forward_one_token_minimal_parity_toy() -> Result<()> {
    let ctx = cuda_env::ctx_m40()?;
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    // Tiny dims
    let d_model = 8usize;
    let hidden = 16usize;
    let num_heads = 2u32;
    let head_dim = 4u32;
    assert_eq!(num_heads * head_dim, d_model as u32);

    // Build GGUF skeleton
    let mut gg = GgufModel::new(0);
    gg.metadata.insert(
        "llama.embedding_length".to_string(),
        GgufValue::Scalar(GgufScalar::U32(d_model as u32)),
    );

    // Tensors
    let vocab = 32u64;
    gg.tensors.push(GgufTensor {
        name: "tok_embeddings.weight".into(),
        dtype: GgmlDType::F16,
        shape: vec![vocab, d_model as u64],
        offset: 0,
    });
    for (name, shape) in [
        (
            "layers.0.attention.wq.weight",
            vec![d_model as u64, d_model as u64],
        ),
        (
            "layers.0.attention.wk.weight",
            vec![d_model as u64, d_model as u64],
        ),
        (
            "layers.0.attention.wv.weight",
            vec![d_model as u64, d_model as u64],
        ),
        (
            "layers.0.attention.wo.weight",
            vec![d_model as u64, d_model as u64],
        ),
        (
            "layers.0.feed_forward.w3.weight",
            vec![d_model as u64, hidden as u64],
        ), // gate
        (
            "layers.0.feed_forward.w1.weight",
            vec![d_model as u64, hidden as u64],
        ), // up
        (
            "layers.0.feed_forward.w2.weight",
            vec![hidden as u64, d_model as u64],
        ), // down
    ] {
        gg.tensors.push(GgufTensor {
            name: name.into(),
            dtype: GgmlDType::F16,
            shape,
            offset: 0,
        });
    }

    // Build weights blob (F16) with deterministic patterns
    let mut weights: Vec<u8> = Vec::new();
    fn push_tensor_halves(
        gg: &mut GgufModel,
        weights: &mut Vec<u8>,
        tensor_name: &str,
        vals: &[f32],
    ) {
        let off = weights.len() as u64;
        if let Some(t) = gg.tensors.iter_mut().find(|t| t.name == tensor_name) {
            t.offset = off;
        }
        weights.extend_from_slice(&halves_from_f32(vals));
    }

    // Generate f32 sources for parity and convert to halves in weights blob
    let emb_f32: Vec<f32> = (0..(vocab as usize) * d_model)
        .map(|i| ((i as f32) * 0.001).sin())
        .collect();
    push_tensor_halves(&mut gg, &mut weights, "tok_embeddings.weight", &emb_f32);

    let sq = d_model * d_model;
    let wq_f32: Vec<f32> = (0..sq).map(|i| ((i as f32) * 0.01).sin()).collect();
    let wk_f32: Vec<f32> = (0..sq).map(|i| ((i as f32) * 0.02).cos()).collect();
    let wv_f32: Vec<f32> = (0..sq).map(|i| ((i as f32) * 0.03).tan().atan()).collect();
    let wo_f32: Vec<f32> = (0..sq).map(|i| ((i as f32) * 0.04).sin()).collect();
    push_tensor_halves(
        &mut gg,
        &mut weights,
        "layers.0.attention.wq.weight",
        &wq_f32,
    );
    push_tensor_halves(
        &mut gg,
        &mut weights,
        "layers.0.attention.wk.weight",
        &wk_f32,
    );
    push_tensor_halves(
        &mut gg,
        &mut weights,
        "layers.0.attention.wv.weight",
        &wv_f32,
    );
    push_tensor_halves(
        &mut gg,
        &mut weights,
        "layers.0.attention.wo.weight",
        &wo_f32,
    );

    let w_gate_f32: Vec<f32> = (0..(d_model * hidden))
        .map(|i| ((i as f32) * 0.015).sin())
        .collect();
    let w_up_f32: Vec<f32> = (0..(d_model * hidden))
        .map(|i| ((i as f32) * 0.017).cos())
        .collect();
    let w_down_f32: Vec<f32> = (0..(hidden * d_model))
        .map(|i| ((i as f32) * 0.019).sin())
        .collect();
    push_tensor_halves(
        &mut gg,
        &mut weights,
        "layers.0.feed_forward.w3.weight",
        &w_gate_f32,
    );
    push_tensor_halves(
        &mut gg,
        &mut weights,
        "layers.0.feed_forward.w1.weight",
        &w_up_f32,
    );
    push_tensor_halves(
        &mut gg,
        &mut weights,
        "layers.0.feed_forward.w2.weight",
        &w_down_f32,
    );

    // Load model and allocate KV
    let mut lm = LoadedModel::from_gguf(gg, weights.clone(), -1)?;
    lm.allocate_kv_cache_with_layout(8, 1, num_heads, head_dim)?;

    // Device input from embedding 0
    let d_x = lm.cuda.device_malloc(d_model * 4)?;
    unsafe {
        lm.load_token_embedding_f16_to_f32(0, d_x)?;
    }

    let d_out = lm.cuda.device_malloc(d_model * 4)?;
    unsafe {
        lm.forward_one_token_with_layer(d_x as *const c_void, 0, 0, 1, d_out)?;
    }

    // Read back device output
    let mut out_dev_bytes = vec![0u8; d_model * 4];
    unsafe {
        lm.cuda.memcpy_d2h(
            out_dev_bytes.as_mut_ptr() as *mut c_void,
            d_out,
            d_model * 4,
        )?;
    }
    let out_dev: Vec<f32> = out_dev_bytes
        .chunks_exact(4)
        .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
        .collect();

    // Build CPU reference from the same half-precision weights (dequantized to f32)
    // Helper to get a named tensor slice from weights blob and convert to f32
    let get_tensor_f32 = |name: &str,
                          rows: usize,
                          cols: usize,
                          weights_blob: &Vec<u8>,
                          lm: &LoadedModel|
     -> Vec<f32> {
        let t = lm.device_tensors.get(name).unwrap();
        let off = t.byte_offset as usize;
        let nbytes = rows * cols * 2;
        let slice = &weights_blob[off..off + nbytes];
        halves_to_f32(slice)
    };

    let x0 = {
        // tok_embeddings row 0
        let t = lm.device_tensors.get("tok_embeddings.weight").unwrap();
        let off = t.byte_offset as usize;
        let row_bytes = d_model * 2;
        let row = &weights[off..off + row_bytes];
        halves_to_f32(row)
    };

    let x_n = rms_norm(&x0, 1e-6);

    let wq = get_tensor_f32(
        "layers.0.attention.wq.weight",
        d_model,
        d_model,
        &weights,
        &lm,
    );
    let wk = get_tensor_f32(
        "layers.0.attention.wk.weight",
        d_model,
        d_model,
        &weights,
        &lm,
    );
    let wv = get_tensor_f32(
        "layers.0.attention.wv.weight",
        d_model,
        d_model,
        &weights,
        &lm,
    );
    let wo = get_tensor_f32(
        "layers.0.attention.wo.weight",
        d_model,
        d_model,
        &weights,
        &lm,
    );

    // Q, K, V (1 x d_model)
    let _q = matmul_row_major(&x_n, 1, d_model, &wq, d_model);
    let _k = matmul_row_major(&x_n, 1, d_model, &wk, d_model);
    let v = matmul_row_major(&x_n, 1, d_model, &wv, d_model);

    // With seq_len=1, attention context equals V
    let context = v;

    let y_attn = matmul_row_major(&context, 1, d_model, &wo, d_model);
    let x1: Vec<f32> = x0.iter().zip(y_attn.iter()).map(|(a, b)| a + b).collect();

    // Post-attn norm
    let x1n = rms_norm(&x1, 1e-6);

    let w_gate = get_tensor_f32(
        "layers.0.feed_forward.w3.weight",
        d_model,
        hidden,
        &weights,
        &lm,
    );
    let w_up = get_tensor_f32(
        "layers.0.feed_forward.w1.weight",
        d_model,
        hidden,
        &weights,
        &lm,
    );
    let w_down = get_tensor_f32(
        "layers.0.feed_forward.w2.weight",
        hidden,
        d_model,
        &weights,
        &lm,
    );

    let gate = matmul_row_major(&x1n, 1, d_model, &w_gate, hidden);
    let up = matmul_row_major(&x1n, 1, d_model, &w_up, hidden);
    let hidden_act: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| silu(*g) * *u)
        .collect();

    // hidden_act [1 x hidden] * w_down [hidden x d_model] => [1 x d_model]
    let y_mlp = matmul_row_major(&hidden_act, 1, hidden, &w_down, d_model);
    let y_ref: Vec<f32> = x1.iter().zip(y_mlp.iter()).map(|(a, b)| a + b).collect();

    assert_eq!(y_ref.len(), out_dev.len());
    for (i, (a, b)) in y_ref.iter().zip(out_dev.iter()).enumerate() {
        let diff = (*a - *b).abs();
        assert!(
            diff <= 1e-3,
            "mismatch at {}: got {:.6}, expect {:.6}, diff {:.6}",
            i,
            b,
            a,
            diff
        );
    }

    unsafe {
        lm.cuda.device_free(d_x)?;
        lm.cuda.device_free(d_out)?;
    }

    Ok(())
}

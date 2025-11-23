#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use m40_llm::gguf::GgufModel;
use m40_llm::infer::LoadedModel;
use std::collections::HashMap;
use std::ffi::c_void;

fn f32s_to_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn f32s_to_halves_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bits = half::f16::from_f32(v).to_bits();
        out.push((bits & 0xFF) as u8);
        out.push((bits >> 8) as u8);
    }
    out
}

fn bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
        .collect()
}

fn cpu_rowmajor_gemm_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
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

#[test]
fn mlp_gates_and_down_proj_f32xf16_f32_smoke() -> Result<()> {
    let ctx = cuda_env::ctx_m40()?;
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    // Dimensions: X [M,K], W_gate [K,H], W_up [K,H], W_down [H,N]
    let (m, k, h, n) = (3i32, 4i32, 5i32, 6i32);

    // Host data
    let x: Vec<f32> = (0..(m * k) as usize)
        .map(|i| (i as f32) * 0.07 - 0.2)
        .collect();
    let w_gate: Vec<f32> = (0..(k * h) as usize)
        .map(|i| ((i as f32) * 0.011).sin())
        .collect();
    let w_up: Vec<f32> = (0..(k * h) as usize)
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();
    let w_down: Vec<f32> = (0..(h * n) as usize)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();

    // CPU reference: gates and up
    let gate_ref = cpu_rowmajor_gemm_f32(&x, &w_gate, m as usize, h as usize, k as usize);
    let up_ref = cpu_rowmajor_gemm_f32(&x, &w_up, m as usize, h as usize, k as usize);

    // CPU hidden using simple SiLU gate (SwiGLU simplified for parity of matmuls)
    // We are only validating matmul parity; activation is not part of wrappers.
    // For down-proj reference, multiply (gate_ref elementwise sigmoid) * up_ref as hidden
    fn sigmoid(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }
    let hidden_ref: Vec<f32> = gate_ref
        .iter()
        .zip(up_ref.iter())
        .map(|(&g, &u)| (g * sigmoid(g)) * u)
        .collect();
    let y_ref = cpu_rowmajor_gemm_f32(&hidden_ref, &w_down, m as usize, n as usize, h as usize);

    // Device buffers
    let d_x = ctx.device_malloc((m * k * 4) as usize)?;
    let d_w_gate = ctx.device_malloc((k * h * 2) as usize)?;
    let d_w_up = ctx.device_malloc((k * h * 2) as usize)?;
    let d_w_down = ctx.device_malloc((h * n * 2) as usize)?;
    let d_gate = ctx.device_malloc((m * h * 4) as usize)?;
    let d_up = ctx.device_malloc((m * h * 4) as usize)?;
    let d_hidden = ctx.device_malloc((m * h * 4) as usize)?;
    let d_y = ctx.device_malloc((m * n * 4) as usize)?;

    unsafe {
        ctx.memcpy_h2d(
            d_x,
            f32s_to_bytes(&x).as_ptr() as *const c_void,
            (m * k * 4) as usize,
        )?;
        ctx.memcpy_h2d(
            d_w_gate,
            f32s_to_halves_bytes(&w_gate).as_ptr() as *const c_void,
            (k * h * 2) as usize,
        )?;
        ctx.memcpy_h2d(
            d_w_up,
            f32s_to_halves_bytes(&w_up).as_ptr() as *const c_void,
            (k * h * 2) as usize,
        )?;
        ctx.memcpy_h2d(
            d_w_down,
            f32s_to_halves_bytes(&w_down).as_ptr() as *const c_void,
            (h * n * 2) as usize,
        )?;
    }

    let lm = LoadedModel {
        gguf: GgufModel::new(0),
        cuda: ctx.clone(),
        kv_cache: None,
        device_tensors: HashMap::new(),
        #[cfg(feature = "cuda")]
        d_weights_base: std::ptr::null_mut(),
    };

    unsafe {
        lm.mlp_gates_f32xf16_f32(
            d_x as *const c_void,
            m,
            k,
            d_w_gate as *const c_void,
            d_w_up as *const c_void,
            h,
            d_gate,
            d_up,
        )?;

        // For parity we emulate the simple SiLU gating on host and copy to device
        // In real integration this would be a device kernel; here we focus on GEMM parity only.
        let hidden_host = hidden_ref.clone();
        ctx.memcpy_h2d(
            d_hidden,
            f32s_to_bytes(&hidden_host).as_ptr() as *const c_void,
            (m * h * 4) as usize,
        )?;

        lm.mlp_down_proj_f32xf16_f32(
            d_hidden as *const c_void,
            m,
            h,
            d_w_down as *const c_void,
            n,
            d_y,
        )?;
    }

    let mut h_gate = vec![0u8; (m * h * 4) as usize];
    let mut h_up = vec![0u8; (m * h * 4) as usize];
    let mut h_y = vec![0u8; (m * n * 4) as usize];
    unsafe {
        ctx.memcpy_d2h(
            h_gate.as_mut_ptr() as *mut c_void,
            d_gate as *const c_void,
            h_gate.len(),
        )?;
        ctx.memcpy_d2h(
            h_up.as_mut_ptr() as *mut c_void,
            d_up as *const c_void,
            h_up.len(),
        )?;
        ctx.memcpy_d2h(
            h_y.as_mut_ptr() as *mut c_void,
            d_y as *const c_void,
            h_y.len(),
        )?;
    }

    let gate_got = bytes_to_f32s(&h_gate);
    let up_got = bytes_to_f32s(&h_up);
    let y_got = bytes_to_f32s(&h_y);

    for (g, e) in gate_got.iter().zip(gate_ref.iter()) {
        assert!(
            (g - e).abs() <= 1e-3,
            "gate matmul mismatch: got {} expect {}",
            g,
            e
        );
    }
    for (g, e) in up_got.iter().zip(up_ref.iter()) {
        assert!(
            (g - e).abs() <= 1e-3,
            "up matmul mismatch: got {} expect {}",
            g,
            e
        );
    }
    for (g, e) in y_got.iter().zip(y_ref.iter()) {
        assert!(
            (g - e).abs() <= 1e-3,
            "down-proj mismatch: got {} expect {}",
            g,
            e
        );
    }

    unsafe {
        ctx.device_free(d_x)?;
        ctx.device_free(d_w_gate)?;
        ctx.device_free(d_w_up)?;
        ctx.device_free(d_w_down)?;
        ctx.device_free(d_gate)?;
        ctx.device_free(d_up)?;
        ctx.device_free(d_hidden)?;
        ctx.device_free(d_y)?;
    }

    Ok(())
}

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
fn qkv_project_f32xf16_f32_smoke() -> Result<()> {
    let ctx = cuda_env::ctx_m40()?;
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    // MxK input, three KxN* weights
    let (m, k) = (2i32, 3i32);
    let (nq, nk, nv) = (4i32, 5i32, 6i32);

    // Deterministic data
    let a: Vec<f32> = (0..(m * k) as usize)
        .map(|i| (i as f32) * 0.1 - 0.3)
        .collect();
    let wq: Vec<f32> = (0..(k * nq) as usize)
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();
    let wk: Vec<f32> = (0..(k * nk) as usize)
        .map(|i| ((i as f32) * 0.02).cos())
        .collect();
    let wv: Vec<f32> = (0..(k * nv) as usize)
        .map(|i| ((i as f32) * 0.03).tan().atan())
        .collect();

    let expect_q = cpu_rowmajor_gemm_f32(&a, &wq, m as usize, nq as usize, k as usize);
    let expect_k = cpu_rowmajor_gemm_f32(&a, &wk, m as usize, nk as usize, k as usize);
    let expect_v = cpu_rowmajor_gemm_f32(&a, &wv, m as usize, nv as usize, k as usize);

    // Device buffers
    let bytes_a = (m * k * 4) as usize;
    let dq = ctx.device_malloc((m * nq * 4) as usize)?;
    let dk = ctx.device_malloc((m * nk * 4) as usize)?;
    let dv = ctx.device_malloc((m * nv * 4) as usize)?;

    let da = ctx.device_malloc(bytes_a)?;
    let dwq = ctx.device_malloc((k * nq * 2) as usize)?;
    let dwk = ctx.device_malloc((k * nk * 2) as usize)?;
    let dwv = ctx.device_malloc((k * nv * 2) as usize)?;

    unsafe {
        ctx.memcpy_h2d(da, f32s_to_bytes(&a).as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(
            dwq,
            f32s_to_halves_bytes(&wq).as_ptr() as *const c_void,
            (k * nq * 2) as usize,
        )?;
        ctx.memcpy_h2d(
            dwk,
            f32s_to_halves_bytes(&wk).as_ptr() as *const c_void,
            (k * nk * 2) as usize,
        )?;
        ctx.memcpy_h2d(
            dwv,
            f32s_to_halves_bytes(&wv).as_ptr() as *const c_void,
            (k * nv * 2) as usize,
        )?;
    }

    // Minimal LoadedModel wrapping the context
    let lm = LoadedModel {
        gguf: GgufModel::new(0),
        cuda: ctx.clone(),
        kv_cache: None,
        device_tensors: HashMap::new(),
        #[cfg(feature = "cuda")]
        d_weights_base: std::ptr::null_mut(),
    };

    unsafe {
        lm.qkv_project_f32xf16_f32(
            da as *const c_void,
            m,
            k,
            dwq as *const c_void,
            nq,
            dwk as *const c_void,
            nk,
            dwv as *const c_void,
            nv,
            dq,
            dk,
            dv,
        )?;
    }

    let mut hq = vec![0u8; (m * nq * 4) as usize];
    let mut hk = vec![0u8; (m * nk * 4) as usize];
    let mut hv = vec![0u8; (m * nv * 4) as usize];
    unsafe {
        ctx.memcpy_d2h(
            hq.as_mut_ptr() as *mut c_void,
            dq as *const c_void,
            hq.len(),
        )?;
        ctx.memcpy_d2h(
            hk.as_mut_ptr() as *mut c_void,
            dk as *const c_void,
            hk.len(),
        )?;
        ctx.memcpy_d2h(
            hv.as_mut_ptr() as *mut c_void,
            dv as *const c_void,
            hv.len(),
        )?;
    }

    let got_q = bytes_to_f32s(&hq);
    let got_k = bytes_to_f32s(&hk);
    let got_v = bytes_to_f32s(&hv);

    for (g, e) in got_q.iter().zip(expect_q.iter()) {
        assert!((g - e).abs() <= 1e-3, "Q mismatch: got {} expect {}", g, e);
    }
    for (g, e) in got_k.iter().zip(expect_k.iter()) {
        assert!((g - e).abs() <= 1e-3, "K mismatch: got {} expect {}", g, e);
    }
    for (g, e) in got_v.iter().zip(expect_v.iter()) {
        assert!((g - e).abs() <= 1e-3, "V mismatch: got {} expect {}", g, e);
    }

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(dwq)?;
        ctx.device_free(dwk)?;
        ctx.device_free(dwv)?;
        ctx.device_free(dq)?;
        ctx.device_free(dk)?;
        ctx.device_free(dv)?;
    }

    Ok(())
}

#[test]
fn out_proj_f32xf16_f32_smoke() -> Result<()> {
    let ctx = cuda_env::ctx_m40()?;
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let (m, k, n) = (2i32, 3i32, 4i32);
    let a: Vec<f32> = (0..(m * k) as usize)
        .map(|i| (i as f32) * 0.05 - 0.1)
        .collect();
    let w: Vec<f32> = (0..(k * n) as usize)
        .map(|i| ((i as f32) * 0.015).sin())
        .collect();
    let expect = cpu_rowmajor_gemm_f32(&a, &w, m as usize, n as usize, k as usize);

    let da = ctx.device_malloc((m * k * 4) as usize)?;
    let dw = ctx.device_malloc((k * n * 2) as usize)?;
    let dc = ctx.device_malloc((m * n * 4) as usize)?;

    unsafe {
        ctx.memcpy_h2d(
            da,
            f32s_to_bytes(&a).as_ptr() as *const c_void,
            (m * k * 4) as usize,
        )?;
        ctx.memcpy_h2d(
            dw,
            f32s_to_halves_bytes(&w).as_ptr() as *const c_void,
            (k * n * 2) as usize,
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
        lm.out_proj_f32xf16_f32(da as *const c_void, dw as *const c_void, dc, m, n, k)?;
    }

    let mut hc = vec![0u8; (m * n * 4) as usize];
    unsafe {
        ctx.memcpy_d2h(
            hc.as_mut_ptr() as *mut c_void,
            dc as *const c_void,
            hc.len(),
        )?;
    }
    let got = bytes_to_f32s(&hc);

    for (g, e) in got.iter().zip(expect.iter()) {
        assert!(
            (g - e).abs() <= 1e-3,
            "out_proj mismatch: got {} expect {}",
            g,
            e
        );
    }

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(dw)?;
        ctx.device_free(dc)?;
    }

    Ok(())
}

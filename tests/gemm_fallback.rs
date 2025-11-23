#![cfg(all(feature = "cuda", nvcc))]

use anyhow::Result;
use m40_llm::cuda::CudaContext;
use std::ffi::c_void;

fn f32s_to_halves_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bits = half::f16::from_f32(v).to_bits();
        out.push((bits & 0xFF) as u8);
        out.push((bits >> 8) as u8);
    }
    out
}

fn halves_bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let lo = bytes[2 * i] as u16;
        let hi = (bytes[2 * i + 1] as u16) << 8;
        let bits = hi | lo;
        out.push(half::f16::from_bits(bits).to_f32());
    }
    out
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
fn gemm_fallback_square_2x2() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let (m, k, n) = (2i32, 2i32, 2i32);
    let a = [1.0f32, 2.0, 3.0, 4.0]; // 2x2
    let b = [5.0f32, 6.0, 7.0, 8.0]; // 2x2
    let expect = cpu_rowmajor_gemm_f32(&a, &b, m as usize, n as usize, k as usize);

    let ha = f32s_to_halves_bytes(&a);
    let hb = f32s_to_halves_bytes(&b);

    let bytes_a = (m * k * 2) as usize;
    let bytes_b = (k * n * 2) as usize;
    let bytes_c = (m * n * 2) as usize;

    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(bytes_b)?;
    let dc = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, ha.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, hb.as_ptr() as *const c_void, bytes_b)?;
        ctx.gemm_f16_f32(da as *const c_void, db as *const c_void, dc, m, n, k)?;
    }

    let mut hc = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(hc.as_mut_ptr() as *mut c_void, dc as *const c_void, bytes_c)?;
    }

    let got = halves_bytes_to_f32s(&hc);
    for (g, e) in got.iter().zip(expect.iter()) {
        let diff = (g - e).abs();
        assert!(diff <= 0.75, "got {:?} expect {:?}", got, expect);
    }

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc)?;
    }
    Ok(())
}

#[test]
fn gemm_fallback_rectangular_1x3x2() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let (m, k, n) = (1i32, 3i32, 2i32);
    let a = [1.0f32, 2.0, 3.0]; // 1x3
    let b = [4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0]; // 3x2 row-major
    let expect = cpu_rowmajor_gemm_f32(&a, &b, m as usize, n as usize, k as usize);

    let ha = f32s_to_halves_bytes(&a);
    let hb = f32s_to_halves_bytes(&b);

    let bytes_a = (m * k * 2) as usize;
    let bytes_b = (k * n * 2) as usize;
    let bytes_c = (m * n * 2) as usize;

    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(bytes_b)?;
    let dc = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, ha.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, hb.as_ptr() as *const c_void, bytes_b)?;
        ctx.gemm_f16_f32(da as *const c_void, db as *const c_void, dc, m, n, k)?;
    }

    let mut hc = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(hc.as_mut_ptr() as *mut c_void, dc as *const c_void, bytes_c)?;
    }

    let got = halves_bytes_to_f32s(&hc);
    for (g, e) in got.iter().zip(expect.iter()) {
        let diff = (g - e).abs();
        assert!(diff <= 0.75, "got {:?} expect {:?}", got, expect);
    }

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc)?;
    }
    Ok(())
}

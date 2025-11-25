#![cfg(all(feature = "cuda", nvcc, have_cublas_header))]

mod cuda_env;

use anyhow::Result;
use std::ffi::c_void;

// Helper to convert f32 to IEEE-754 half bits via host conversion
fn f32_to_f16_bits(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

fn make_half_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bits = f32_to_f16_bits(v);
        out.push((bits & 0xFF) as u8);
        out.push((bits >> 8) as u8);
    }
    out
}

#[test]
fn gemm_f16_storage_f32_compute_rowmajor_shapes() -> Result<()> {
    // Validate our layout convention for GEMM wrapper against a simple known case.
    // A[MxK] row-major, B[KxN] row-major, C[MxN] row-major where we expect
    // cublasGemmEx call with (N, M, K) and leading dims (N, K, N) as in kernels.cu.

    // Case: M=2, K=2, N=2
    // A = [ 1 2 ; 3 4 ]
    // B = [ 5 6 ; 7 8 ]
    // C = A * B = [ 19 22 ; 43 50 ]

    let ctx = cuda_env::ctx_m40()?;
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let m = 2;
    let k = 2;
    let n = 2;
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];

    let ha = make_half_bytes(&a);
    let hb = make_half_bytes(&b);

    let bytes_a = (m * k * 2) as usize;
    let bytes_b = (k * n * 2) as usize;
    let bytes_c = (m * n * 2) as usize;

    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(bytes_b)?;
    let dc = ctx.device_malloc(bytes_c)?;

    ctx.memcpy_h2d(da, ha.as_ptr() as *const c_void, bytes_a)?;
    ctx.memcpy_h2d(db, hb.as_ptr() as *const c_void, bytes_b)?;

    ctx.gemm_f16_f32(da as *const c_void, db as *const c_void, dc, m, n, k)?;

    // Copy back result in half, convert to f32, compare to expected
    let mut hc = vec![0u8; bytes_c];
    ctx.memcpy_d2h(hc.as_mut_ptr() as *mut c_void, dc as *const c_void, bytes_c)?;

    // Convert 4 half elements to f32
    let mut got = [0f32; 4];
    for i in 0..4 {
        let lo = hc[2 * i] as u16;
        let hi = (hc[2 * i + 1] as u16) << 8;
        let bits = hi | lo;
        got[i] = half::f16::from_bits(bits).to_f32();
    }

    let expect = [19.0f32, 22.0, 43.0, 50.0];
    for (g, e) in got.iter().zip(expect.iter()) {
        let diff = (g - e).abs();
        assert!(diff <= 0.75, "got {:?} expect {:?}", got, expect);
    }

    ctx.device_free(da)?;
    ctx.device_free(db)?;
    ctx.device_free(dc)?;
    Ok(())
}

#[test]
fn gemm_f16_storage_f32_compute_rectangular() -> Result<()> {
    // M=1, K=3, N=2
    // A = [1,2,3]
    // B = [[4,5],[6,7],[8,9]] row-major as [4,5,6,7,8,9]
    // C = A*B = [40, 46]
    let ctx = cuda_env::ctx_m40()?;
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let m = 1;
    let k = 3;
    let n = 2;
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0];

    let ha = make_half_bytes(&a);
    let hb = make_half_bytes(&b);

    let bytes_a = (m * k * 2) as usize;
    let bytes_b = (k * n * 2) as usize;
    let bytes_c = (m * n * 2) as usize;

    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(bytes_b)?;
    let dc = ctx.device_malloc(bytes_c)?;

    ctx.memcpy_h2d(da, ha.as_ptr() as *const c_void, bytes_a)?;
    ctx.memcpy_h2d(db, hb.as_ptr() as *const c_void, bytes_b)?;

    ctx.gemm_f16_f32(da as *const c_void, db as *const c_void, dc, m, n, k)?;

    let mut hc = vec![0u8; bytes_c];
    ctx.memcpy_d2h(hc.as_mut_ptr() as *mut c_void, dc as *const c_void, bytes_c)?;

    let mut got = vec![0f32; (m * n) as usize];
    for i in 0..got.len() {
        let lo = hc[2 * i] as u16;
        let hi = (hc[2 * i + 1] as u16) << 8;
        let bits = hi | lo;
        got[i] = half::f16::from_bits(bits).to_f32();
    }

    let expect = [40.0f32, 46.0];
    for (g, e) in got.iter().zip(expect.iter()) {
        let diff = (g - e).abs();
        assert!(diff <= 0.75, "got {:?} expect {:?}", got, expect);
    }

    ctx.device_free(da)?;
    ctx.device_free(db)?;
    ctx.device_free(dc)?;
    Ok(())
}

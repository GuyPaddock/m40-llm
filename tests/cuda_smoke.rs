#![cfg(all(feature = "cuda", nvcc))]

use anyhow::Result;
use m40_llm::cuda::{CudaContext, KVCache};
use std::ffi::c_void;

#[test]
fn cuda_context_and_memcpy() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let bytes = 16usize;
    let dptr = ctx.device_malloc(bytes)?;

    let host = [1u8; 16];
    ctx.memcpy_h2d(dptr, host.as_ptr() as *const c_void, bytes)?;

    let mut out = [0u8; 16];
    ctx.memcpy_d2h(
        out.as_mut_ptr() as *mut c_void,
        dptr as *const c_void,
        bytes,
    )?;
    assert_eq!(out, host);

    ctx.device_free(dptr)?;
    Ok(())
}

#[test]
fn kvcache_create_append() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let kv = KVCache::new_with_context(&ctx, 4, 1, 2, 4)?;

    // One token worth of K/V per head
    let elems_per_token = (2u32 * 4u32) as usize;
    let bytes = elems_per_token * 2; // __half bytes

    let hzeros = vec![0u8; bytes];
    let kd = ctx.device_malloc(bytes)?;
    let vd = ctx.device_malloc(bytes)?;

    ctx.memcpy_h2d(kd, hzeros.as_ptr() as *const c_void, bytes)?;
    ctx.memcpy_h2d(vd, hzeros.as_ptr() as *const c_void, bytes)?;

    kv.append_token(&ctx, 0, kd as *const c_void, vd as *const c_void)?;

    ctx.device_free(kd)?;
    ctx.device_free(vd)?;
    Ok(())
}

#[test]
fn gemm_call() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let m = 2;
    let n = 2;
    let k = 2;

    let elems_a = (m * k) as usize;
    let elems_b = (k * n) as usize;
    let elems_c = (m * n) as usize;

    let bytes_a = elems_a * 2; // __half
    let bytes_b = elems_b * 2; // __half
    let bytes_c = elems_c * 2; // __half

    let ha = vec![0u8; bytes_a];
    let hb = vec![0u8; bytes_b];

    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(bytes_b)?;
    let dc = ctx.device_malloc(bytes_c)?;

    ctx.memcpy_h2d(da, ha.as_ptr() as *const c_void, bytes_a)?;
    ctx.memcpy_h2d(db, hb.as_ptr() as *const c_void, bytes_b)?;

    ctx.gemm_f16_f32(da as *const c_void, db as *const c_void, dc, m, n, k)?;

    ctx.device_free(da)?;
    ctx.device_free(db)?;
    ctx.device_free(dc)?;
    Ok(())
}

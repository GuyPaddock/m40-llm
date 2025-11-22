#![cfg(all(feature = "cuda", nvcc))]

use anyhow::Result;
use m40_llm::cuda::{CudaContext, KVCache};
use std::ffi::c_void;

// CPU reference: f32 -> f16 (round to nearest even should match __float2half_rn for normal ranges)
fn cpu_f32_to_f16_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

#[test]
fn append_token_f32_cast_matches_cpu() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let num_heads = 3u32;
    let head_dim = 5u32; // odd to exercise tail path
    let kv = KVCache::new_with_context(&ctx, 2, 1, num_heads, head_dim)?;

    let elems = (num_heads * head_dim) as usize;
    let bytes_f32 = elems * std::mem::size_of::<f32>();
    let bytes_f16 = elems * 2;

    // Prepare host input
    let mut h_k_f32 = Vec::<f32>::with_capacity(elems);
    let mut h_v_f32 = Vec::<f32>::with_capacity(elems);
    for i in 0..elems {
        h_k_f32.push((i as f32) * 0.25 - 10.0);
        h_v_f32.push(((elems - i) as f32) * 0.125 - 5.0);
    }

    // Device buffers for f32 inputs
    let d_k_f32 = ctx.device_malloc(bytes_f32)?;
    let d_v_f32 = ctx.device_malloc(bytes_f32)?;
    unsafe {
        ctx.memcpy_h2d(d_k_f32, h_k_f32.as_ptr() as *const c_void, bytes_f32)?;
        ctx.memcpy_h2d(d_v_f32, h_v_f32.as_ptr() as *const c_void, bytes_f32)?;
    }

    // Append which performs device-side cast and store
    unsafe {
        kv.append_token_f32(&ctx, 0, d_k_f32 as *const c_void, d_v_f32 as *const c_void)?;
    }

    // Read back the stored token via debug API
    let mut out_k = vec![0u8; bytes_f16];
    let mut out_v = vec![0u8; bytes_f16];

    // SAFETY: test helper FFI
    unsafe {
        let rc = m40_llm::cuda::ffi_debug_read_kv_token(
            &ctx,
            &kv,
            0,
            0,
            out_k.as_mut_ptr(),
            out_v.as_mut_ptr(),
        );
        assert_eq!(rc, 0);
    }

    // Compare per element in f16 bitspace
    for i in 0..elems {
        let cpu_bits_k = cpu_f32_to_f16_bits(h_k_f32[i]);
        let cpu_bits_v = cpu_f32_to_f16_bits(h_v_f32[i]);
        let got_bits_k = u16::from_le_bytes([out_k[2 * i], out_k[2 * i + 1]]);
        let got_bits_v = u16::from_le_bytes([out_v[2 * i], out_v[2 * i + 1]]);
        assert_eq!(got_bits_k, cpu_bits_k, "k mismatch at {}", i);
        assert_eq!(got_bits_v, cpu_bits_v, "v mismatch at {}", i);
    }

    unsafe {
        ctx.device_free(d_k_f32)?;
        ctx.device_free(d_v_f32)?;
    }
    Ok(())
}

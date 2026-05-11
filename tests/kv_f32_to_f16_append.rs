#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use half::f16;
use m40_llm::cuda::{ffi_debug_read_kv_token, CudaStream, KVCache};
use std::ffi::c_void;

fn f32s_to_f16_bits(xs: &[f32]) -> Vec<u16> {
    xs.iter().map(|&x| f16::from_f32(x).to_bits()).collect()
}

fn f16_bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

#[test]
fn test_kvcache_append_token_f32_casts_and_stores_fp16() -> Result<()> {
    // Choose odd elems_per_token to exercise half2 tail and potential misalignment
    let num_heads: u32 = 1;
    let head_dim: u32 = 3; // elems_per_token = 3 (odd)
    let max_seq_len: u32 = 4;
    let max_batch_size: u32 = 1;

    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let kv = KVCache::new_with_context(&ctx, max_seq_len, max_batch_size, num_heads, head_dim)?;

    let elems_per_token = (num_heads as usize) * (head_dim as usize);

    // Prepare K/V input for token 0 (FP32 on device)
    let k0_host: Vec<f32> = (0..elems_per_token).map(|i| i as f32 + 0.25).collect();
    let v0_host: Vec<f32> = (0..elems_per_token)
        .map(|i| 100.0 + i as f32 + 0.5)
        .collect();

    // Device alloc and upload
    let k0_dev: *mut c_void = ctx.device_malloc(k0_host.len() * 4)?;
    let v0_dev: *mut c_void = ctx.device_malloc(v0_host.len() * 4)?;
    unsafe {
        ctx.memcpy_h2d(k0_dev, k0_host.as_ptr() as *const c_void, k0_host.len() * 4)?;
        ctx.memcpy_h2d(v0_dev, v0_host.as_ptr() as *const c_void, v0_host.len() * 4)?;
    }

    // Append seq 0, token 0
    unsafe {
        kv.append_token_f32(&ctx, 0, k0_dev as *const c_void, v0_dev as *const c_void)?;
    }

    // Read back token 0 as FP16
    let bytes = elems_per_token * 2;
    let mut k0_back = vec![0u8; bytes];
    let mut v0_back = vec![0u8; bytes];
    unsafe {
        ffi_debug_read_kv_token(&ctx, &kv, 0, 0, k0_back.as_mut_ptr(), v0_back.as_mut_ptr());
    }

    // Compare against CPU reference cast
    let k0_expected_bits = f32s_to_f16_bits(&k0_host);
    let v0_expected_bits = f32s_to_f16_bits(&v0_host);

    for i in 0..elems_per_token {
        let bits = u16::from_le_bytes([k0_back[2 * i], k0_back[2 * i + 1]]);
        assert_eq!(bits, k0_expected_bits[i], "K token0 element {} mismatch", i);
        let bits_v = u16::from_le_bytes([v0_back[2 * i], v0_back[2 * i + 1]]);
        assert_eq!(
            bits_v, v0_expected_bits[i],
            "V token0 element {} mismatch",
            i
        );
    }

    // Now append another token to exercise potential unaligned half2 path
    let k1_host: Vec<f32> = (0..elems_per_token)
        .map(|i| -10.0 + i as f32 * 0.125)
        .collect();
    let v1_host: Vec<f32> = (0..elems_per_token)
        .map(|i| std::f32::consts::PI + i as f32 * 0.25)
        .collect();

    unsafe {
        ctx.memcpy_h2d(k0_dev, k1_host.as_ptr() as *const c_void, k1_host.len() * 4)?;
        ctx.memcpy_h2d(v0_dev, v1_host.as_ptr() as *const c_void, v1_host.len() * 4)?;
        kv.append_token_f32_async(&ctx, 0, k0_dev as *const c_void, v0_dev as *const c_void)?;
        ctx.synchronize_stream(CudaStream::Decode)?;
    }

    let mut k1_back = vec![0u8; bytes];
    let mut v1_back = vec![0u8; bytes];
    unsafe {
        ffi_debug_read_kv_token(&ctx, &kv, 0, 1, k1_back.as_mut_ptr(), v1_back.as_mut_ptr());
    }

    let k1_expected_bits = f32s_to_f16_bits(&k1_host);
    let v1_expected_bits = f32s_to_f16_bits(&v1_host);

    for i in 0..elems_per_token {
        let bits = u16::from_le_bytes([k1_back[2 * i], k1_back[2 * i + 1]]);
        assert_eq!(bits, k1_expected_bits[i], "K token1 element {} mismatch", i);
        let bits_v = u16::from_le_bytes([v1_back[2 * i], v1_back[2 * i + 1]]);
        assert_eq!(
            bits_v, v1_expected_bits[i],
            "V token1 element {} mismatch",
            i
        );
    }

    // Cleanup
    unsafe {
        ctx.device_free(k0_dev)?;
        ctx.device_free(v0_dev)?;
    }

    Ok(())
}

#[test]
fn test_kvcache_append_token_f32_rope_k_matches_separate_rope_append() -> Result<()> {
    let num_heads: u32 = 2;
    let head_dim: u32 = 4;
    let max_seq_len: u32 = 4;
    let max_batch_size: u32 = 1;
    let past_len: u32 = 3;
    let freq_base = 10_000.0;
    let freq_scale = 1.0;

    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let kv_separate =
        KVCache::new_with_context(&ctx, max_seq_len, max_batch_size, num_heads, head_dim)?;
    let kv_fused =
        KVCache::new_with_context(&ctx, max_seq_len, max_batch_size, num_heads, head_dim)?;
    let elems_per_token = (num_heads as usize) * (head_dim as usize);
    let bytes_f32 = elems_per_token * std::mem::size_of::<f32>();

    let k_host: Vec<f32> = (0..elems_per_token)
        .map(|i| -0.75 + i as f32 * 0.375)
        .collect();
    let v_host: Vec<f32> = (0..elems_per_token)
        .map(|i| 2.5 + i as f32 * 0.25)
        .collect();

    let k_separate_dev: *mut c_void = ctx.device_malloc(bytes_f32)?;
    let k_fused_dev: *mut c_void = ctx.device_malloc(bytes_f32)?;
    let v_dev: *mut c_void = ctx.device_malloc(bytes_f32)?;
    unsafe {
        ctx.memcpy_h2d(k_separate_dev, k_host.as_ptr() as *const c_void, bytes_f32)?;
        ctx.memcpy_h2d(k_fused_dev, k_host.as_ptr() as *const c_void, bytes_f32)?;
        ctx.memcpy_h2d(v_dev, v_host.as_ptr() as *const c_void, bytes_f32)?;

        ctx.rope_f32_inplace(
            k_separate_dev,
            1,
            num_heads,
            head_dim,
            past_len,
            freq_base,
            freq_scale,
        )?;
        kv_separate.append_token_f32(
            &ctx,
            0,
            k_separate_dev as *const c_void,
            v_dev as *const c_void,
        )?;
        kv_fused.append_token_f32_rope_k(
            &ctx,
            0,
            k_fused_dev as *const c_void,
            v_dev as *const c_void,
            past_len,
            freq_base,
            freq_scale,
        )?;
    }

    let bytes_f16 = elems_per_token * std::mem::size_of::<f16>();
    let mut k_separate_back = vec![0u8; bytes_f16];
    let mut v_separate_back = vec![0u8; bytes_f16];
    let mut k_fused_back = vec![0u8; bytes_f16];
    let mut v_fused_back = vec![0u8; bytes_f16];
    unsafe {
        ffi_debug_read_kv_token(
            &ctx,
            &kv_separate,
            0,
            0,
            k_separate_back.as_mut_ptr(),
            v_separate_back.as_mut_ptr(),
        );
        ffi_debug_read_kv_token(
            &ctx,
            &kv_fused,
            0,
            0,
            k_fused_back.as_mut_ptr(),
            v_fused_back.as_mut_ptr(),
        );
    }

    let k_separate = f16_bytes_to_f32s(&k_separate_back);
    let v_separate = f16_bytes_to_f32s(&v_separate_back);
    let k_fused = f16_bytes_to_f32s(&k_fused_back);
    let v_fused = f16_bytes_to_f32s(&v_fused_back);
    for i in 0..elems_per_token {
        assert!(
            (k_separate[i] - k_fused[i]).abs() <= 1e-3,
            "K element {i} mismatch: separate={} fused={}",
            k_separate[i],
            k_fused[i]
        );
        assert!(
            (v_separate[i] - v_fused[i]).abs() <= 1e-3,
            "V element {i} mismatch: separate={} fused={}",
            v_separate[i],
            v_fused[i]
        );
    }

    unsafe {
        ctx.device_free(k_separate_dev)?;
        ctx.device_free(k_fused_dev)?;
        ctx.device_free(v_dev)?;
    }

    Ok(())
}

#[test]
fn test_kvcache_reset_restarts_append_position() -> Result<()> {
    let num_heads: u32 = 1;
    let head_dim: u32 = 3;
    let max_seq_len: u32 = 4;
    let max_batch_size: u32 = 1;

    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let kv = KVCache::new_with_context(&ctx, max_seq_len, max_batch_size, num_heads, head_dim)?;
    let elems_per_token = (num_heads as usize) * (head_dim as usize);

    let k_first: Vec<f32> = (0..elems_per_token).map(|i| i as f32 + 1.0).collect();
    let v_first: Vec<f32> = (0..elems_per_token).map(|i| i as f32 + 10.0).collect();
    let k_after_reset: Vec<f32> = (0..elems_per_token).map(|i| i as f32 + 100.0).collect();
    let v_after_reset: Vec<f32> = (0..elems_per_token).map(|i| i as f32 + 200.0).collect();

    let k_dev: *mut c_void = ctx.device_malloc(elems_per_token * 4)?;
    let v_dev: *mut c_void = ctx.device_malloc(elems_per_token * 4)?;
    unsafe {
        ctx.memcpy_h2d(
            k_dev,
            k_first.as_ptr() as *const c_void,
            elems_per_token * 4,
        )?;
        ctx.memcpy_h2d(
            v_dev,
            v_first.as_ptr() as *const c_void,
            elems_per_token * 4,
        )?;
        kv.append_token_f32(&ctx, 0, k_dev as *const c_void, v_dev as *const c_void)?;
    }

    kv.reset(&ctx)?;

    unsafe {
        ctx.memcpy_h2d(
            k_dev,
            k_after_reset.as_ptr() as *const c_void,
            elems_per_token * 4,
        )?;
        ctx.memcpy_h2d(
            v_dev,
            v_after_reset.as_ptr() as *const c_void,
            elems_per_token * 4,
        )?;
        kv.append_token_f32(&ctx, 0, k_dev as *const c_void, v_dev as *const c_void)?;
    }

    let bytes = elems_per_token * 2;
    let mut k_back = vec![0u8; bytes];
    let mut v_back = vec![0u8; bytes];
    unsafe {
        ffi_debug_read_kv_token(&ctx, &kv, 0, 0, k_back.as_mut_ptr(), v_back.as_mut_ptr());
    }

    let k_expected_bits = f32s_to_f16_bits(&k_after_reset);
    let v_expected_bits = f32s_to_f16_bits(&v_after_reset);
    for i in 0..elems_per_token {
        let bits = u16::from_le_bytes([k_back[2 * i], k_back[2 * i + 1]]);
        assert_eq!(
            bits, k_expected_bits[i],
            "K reset token0 element {} mismatch",
            i
        );
        let bits_v = u16::from_le_bytes([v_back[2 * i], v_back[2 * i + 1]]);
        assert_eq!(
            bits_v, v_expected_bits[i],
            "V reset token0 element {} mismatch",
            i
        );
    }

    unsafe {
        ctx.device_free(k_dev)?;
        ctx.device_free(v_dev)?;
    }

    Ok(())
}

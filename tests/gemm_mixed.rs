#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use m40_llm::cuda::CudaStream;
// CudaContext not used directly in this test
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

fn cpu_gguf_gemm_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn q8_0_gguf_bytes_from_dequantized(vals: &[f32], n: usize, k: usize) -> Vec<u8> {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let blocks_per_col = k.div_ceil(QK);
    let mut out = vec![0u8; n * blocks_per_col * BLOCK_BYTES];
    for col in 0..n {
        for block in 0..blocks_per_col {
            let start = block * QK;
            let end = (start + QK).min(k);
            let scale = vals[col * k + start..col * k + end]
                .iter()
                .fold(0.0f32, |acc, v| acc.max(v.abs()))
                / 127.0;
            let scale = if scale == 0.0 { 1.0 } else { scale };
            let base = (col * blocks_per_col + block) * BLOCK_BYTES;
            let scale_bits = half::f16::from_f32(scale).to_bits();
            out[base] = (scale_bits & 0xff) as u8;
            out[base + 1] = (scale_bits >> 8) as u8;
            for idx in 0..QK {
                let k_idx = start + idx;
                let q = if k_idx < k {
                    (vals[col * k + k_idx] / scale).round().clamp(-128.0, 127.0) as i8
                } else {
                    0
                };
                out[base + 2 + idx] = q as u8;
            }
        }
    }
    out
}

fn q8_0_gguf_dequantize(bytes: &[u8], n: usize, k: usize) -> Vec<f32> {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 34;
    let blocks_per_col = k.div_ceil(QK);
    let mut out = vec![0f32; n * k];
    for col in 0..n {
        for block in 0..blocks_per_col {
            let base = (col * blocks_per_col + block) * BLOCK_BYTES;
            let scale_bits = u16::from_le_bytes([bytes[base], bytes[base + 1]]);
            let scale = half::f16::from_bits(scale_bits).to_f32();
            for idx in 0..QK {
                let k_idx = block * QK + idx;
                if k_idx < k {
                    let q = bytes[base + 2 + idx] as i8;
                    out[col * k + k_idx] = f32::from(q) * scale;
                }
            }
        }
    }
    out
}

#[test]
fn gemm_f32xf16_f32_square_2x2() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let (m, k, n) = (2i32, 2i32, 2i32);
    let a = [1.0f32, 2.0, 3.0, 4.0]; // 2x2 f32
    let b = [5.0f32, 6.0, 7.0, 8.0]; // 2x2 f16
    let expect = cpu_rowmajor_gemm_f32(&a, &b, m as usize, n as usize, k as usize);

    let a_bytes = f32s_to_bytes(&a);
    let b_half = f32s_to_halves_bytes(&b);

    let bytes_a = (m * k * 4) as usize;
    let bytes_b = (k * n * 2) as usize;
    let bytes_c = (m * n * 4) as usize;

    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(bytes_b)?;
    let dc = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_half.as_ptr() as *const c_void, bytes_b)?;
        ctx.gemm_f32xf16_f32(da as *const c_void, db as *const c_void, dc, m, n, k)?;
    }

    let mut hc = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(hc.as_mut_ptr() as *mut c_void, dc as *const c_void, bytes_c)?;
    }

    let got = bytes_to_f32s(&hc);
    for (g, e) in got.iter().zip(expect.iter()) {
        let diff = (g - e).abs();
        assert!(diff <= 1e-3, "got {:?} expect {:?}", got, expect);
    }

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc)?;
    }
    Ok(())
}

#[test]
fn gemm_f32xf16_f32_rectangular_1x3x2() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let (m, k, n) = (1i32, 3i32, 2i32);
    let a = [1.0f32, 2.0, 3.0]; // 1x3 f32
    let b = [4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0]; // 3x2 f16 row-major
    let expect = cpu_rowmajor_gemm_f32(&a, &b, m as usize, n as usize, k as usize);

    let a_bytes = f32s_to_bytes(&a);
    let b_half = f32s_to_halves_bytes(&b);

    let bytes_a = (m * k * 4) as usize;
    let bytes_b = (k * n * 2) as usize;
    let bytes_c = (m * n * 4) as usize;

    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(bytes_b)?;
    let dc = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_half.as_ptr() as *const c_void, bytes_b)?;
        ctx.gemm_f32xf16_f32(da as *const c_void, db as *const c_void, dc, m, n, k)?;
    }

    let mut hc = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(hc.as_mut_ptr() as *mut c_void, dc as *const c_void, bytes_c)?;
    }

    let got = bytes_to_f32s(&hc);
    for (g, e) in got.iter().zip(expect.iter()) {
        let diff = (g - e).abs();
        assert!(diff <= 1e-3, "got {:?} expect {:?}", got, expect);
    }

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc)?;
    }
    Ok(())
}

#[test]
fn gemm_f32xf16_gguf_f32_rectangular_1x3x2() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let (m, k, n) = (1i32, 3i32, 2i32);
    let a = [1.0f32, 2.0, 3.0];
    // Logical weight columns are [4, 6, 8] and [5, 7, 9], stored K-fastest as GGUF does.
    let b = [4.0f32, 6.0, 8.0, 5.0, 7.0, 9.0];
    let expect = cpu_gguf_gemm_f32(&a, &b, m as usize, n as usize, k as usize);

    let a_bytes = f32s_to_bytes(&a);
    let b_half = f32s_to_halves_bytes(&b);

    let bytes_a = (m * k * 4) as usize;
    let bytes_b = (k * n * 2) as usize;
    let bytes_c = (m * n * 4) as usize;

    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(bytes_b)?;
    let dc = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_half.as_ptr() as *const c_void, bytes_b)?;
        ctx.gemm_f32xf16_gguf_f32(da as *const c_void, db as *const c_void, dc, m, n, k)?;
    }

    let mut hc = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(hc.as_mut_ptr() as *mut c_void, dc as *const c_void, bytes_c)?;
    }

    let got = bytes_to_f32s(&hc);
    for (g, e) in got.iter().zip(expect.iter()) {
        let diff = (g - e).abs();
        assert!(diff <= 1e-3, "got {:?} expect {:?}", got, expect);
    }

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc)?;
    }
    Ok(())
}

#[test]
fn gemm_f32xq8_0_gguf_f32_rectangular_2x35x3() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let (m, k, n) = (2i32, 35i32, 3i32);
    let a: Vec<f32> = (0..m * k)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.25)
        .collect();
    let b: Vec<f32> = (0..n * k)
        .map(|idx| ((idx % 11) as f32 - 5.0) * 0.125)
        .collect();
    let b_q8 = q8_0_gguf_bytes_from_dequantized(&b, n as usize, k as usize);
    let b_deq = q8_0_gguf_dequantize(&b_q8, n as usize, k as usize);
    let expect = cpu_gguf_gemm_f32(&a, &b_deq, m as usize, n as usize, k as usize);

    let a_bytes = f32s_to_bytes(&a);
    let bytes_a = (m * k * 4) as usize;
    let bytes_c = (m * n * 4) as usize;
    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(b_q8.len())?;
    let dc = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_q8.as_ptr() as *const c_void, b_q8.len())?;
        ctx.gemm_f32xq8_0_gguf_f32(da as *const c_void, db as *const c_void, dc, m, n, k)?;
    }

    let mut hc = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(
            hc.as_mut_ptr() as *mut c_void,
            dc as *const c_void,
            hc.len(),
        )?;
    }

    let got = bytes_to_f32s(&hc);
    for (g, e) in got.iter().zip(expect.iter()) {
        let diff = (g - e).abs();
        assert!(diff <= 1e-4, "got {:?} expect {:?}", got, expect);
    }

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc)?;
    }
    Ok(())
}

#[test]
fn gemm_f32xq8_0_blockloop_matches_cpu_tail_2x35x3() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let (m, k, n) = (2i32, 35i32, 3i32);
    let a: Vec<f32> = (0..m * k)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.25)
        .collect();
    let b: Vec<f32> = (0..n * k)
        .map(|idx| ((idx % 11) as f32 - 5.0) * 0.125)
        .collect();
    let b_q8 = q8_0_gguf_bytes_from_dequantized(&b, n as usize, k as usize);
    let b_deq = q8_0_gguf_dequantize(&b_q8, n as usize, k as usize);
    let expect = cpu_gguf_gemm_f32(&a, &b_deq, m as usize, n as usize, k as usize);

    let a_bytes = f32s_to_bytes(&a);
    let bytes_a = (m * k * 4) as usize;
    let bytes_c = (m * n * 4) as usize;
    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(b_q8.len())?;
    let dc = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_q8.as_ptr() as *const c_void, b_q8.len())?;
        ctx.gemm_f32xq8_0_gguf_f32_blockloop_async(
            da as *const c_void,
            db as *const c_void,
            dc,
            m,
            n,
            k,
        )?;
        ctx.synchronize_stream(CudaStream::Prefill)?;
    }

    let mut got_bytes = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(
            got_bytes.as_mut_ptr() as *mut c_void,
            dc as *const c_void,
            got_bytes.len(),
        )?;
    }
    let got = bytes_to_f32s(&got_bytes);
    for (g, e) in got.iter().zip(expect.iter()) {
        assert!((g - e).abs() <= 1e-4, "got {:?} expect {:?}", got, expect);
    }

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc)?;
    }
    Ok(())
}

#[test]
fn gemm_f32xq8_0_blockloop_matches_cpu_medium_5x96x65() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let (m, k, n) = (5i32, 96i32, 65i32);
    let a: Vec<f32> = (0..m * k)
        .map(|idx| ((idx % 17) as f32 - 8.0) * 0.03125)
        .collect();
    let b: Vec<f32> = (0..n * k)
        .map(|idx| ((idx % 23) as f32 - 11.0) * 0.015625)
        .collect();
    let b_q8 = q8_0_gguf_bytes_from_dequantized(&b, n as usize, k as usize);
    let b_deq = q8_0_gguf_dequantize(&b_q8, n as usize, k as usize);
    let expect = cpu_gguf_gemm_f32(&a, &b_deq, m as usize, n as usize, k as usize);

    let a_bytes = f32s_to_bytes(&a);
    let bytes_a = (m * k * 4) as usize;
    let bytes_c = (m * n * 4) as usize;
    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(b_q8.len())?;
    let dc = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_q8.as_ptr() as *const c_void, b_q8.len())?;
        ctx.gemm_f32xq8_0_gguf_f32_blockloop_async(
            da as *const c_void,
            db as *const c_void,
            dc,
            m,
            n,
            k,
        )?;
        ctx.synchronize_stream(CudaStream::Prefill)?;
    }

    let mut got_bytes = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(
            got_bytes.as_mut_ptr() as *mut c_void,
            dc as *const c_void,
            got_bytes.len(),
        )?;
    }
    let got = bytes_to_f32s(&got_bytes);
    for (g, e) in got.iter().zip(expect.iter()) {
        assert!((g - e).abs() <= 1e-4, "got {g} expect {e}");
    }

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc)?;
    }
    Ok(())
}

#[test]
fn gemm_f32xq8_0_blockloop_matches_generic_prefill_shape() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let (m, k, n) = (64i32, 2048i32, 2048i32);
    let a: Vec<f32> = (0..m * k)
        .map(|idx| ((idx % 17) as f32 - 8.0) * 0.03125)
        .collect();
    let b: Vec<f32> = (0..n * k)
        .map(|idx| ((idx % 23) as f32 - 11.0) * 0.015625)
        .collect();
    let b_q8 = q8_0_gguf_bytes_from_dequantized(&b, n as usize, k as usize);

    let a_bytes = f32s_to_bytes(&a);
    let bytes_a = (m * k * 4) as usize;
    let bytes_c = (m * n * 4) as usize;
    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(b_q8.len())?;
    let dc_generic = ctx.device_malloc(bytes_c)?;
    let dc_blockloop = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_q8.as_ptr() as *const c_void, b_q8.len())?;
        ctx.gemm_f32xq8_0_gguf_f32_generic_async(
            da as *const c_void,
            db as *const c_void,
            dc_generic,
            m,
            n,
            k,
        )?;
        ctx.gemm_f32xq8_0_gguf_f32_blockloop_async(
            da as *const c_void,
            db as *const c_void,
            dc_blockloop,
            m,
            n,
            k,
        )?;
        ctx.synchronize_stream(CudaStream::Prefill)?;
    }

    let mut generic_bytes = vec![0u8; bytes_c];
    let mut blockloop_bytes = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(
            generic_bytes.as_mut_ptr() as *mut c_void,
            dc_generic as *const c_void,
            generic_bytes.len(),
        )?;
        ctx.memcpy_d2h(
            blockloop_bytes.as_mut_ptr() as *mut c_void,
            dc_blockloop as *const c_void,
            blockloop_bytes.len(),
        )?;
    }
    let generic = bytes_to_f32s(&generic_bytes);
    let blockloop = bytes_to_f32s(&blockloop_bytes);
    for (g, b) in generic.iter().zip(blockloop.iter()) {
        assert!((g - b).abs() <= 1e-4, "generic got {g} blockloop got {b}");
    }

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc_generic)?;
        ctx.device_free(dc_blockloop)?;
    }
    Ok(())
}

#[test]
fn gemm_f32xq8_0_decode_kernel_matches_generic_1x2048x2048() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let (m, k, n) = (1i32, 2048i32, 2048i32);
    let a: Vec<f32> = (0..m * k)
        .map(|idx| ((idx % 17) as f32 - 8.0) * 0.03125)
        .collect();
    let b: Vec<f32> = (0..n * k)
        .map(|idx| ((idx % 23) as f32 - 11.0) * 0.015625)
        .collect();
    let b_q8 = q8_0_gguf_bytes_from_dequantized(&b, n as usize, k as usize);

    let a_bytes = f32s_to_bytes(&a);
    let bytes_a = (m * k * 4) as usize;
    let bytes_c = (m * n * 4) as usize;
    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(b_q8.len())?;
    let dc_generic = ctx.device_malloc(bytes_c)?;
    let dc_decode = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_q8.as_ptr() as *const c_void, b_q8.len())?;
        ctx.gemm_f32xq8_0_gguf_f32_generic_async(
            da as *const c_void,
            db as *const c_void,
            dc_generic,
            m,
            n,
            k,
        )?;
        ctx.gemm_f32xq8_0_gguf_f32_decode_async(
            da as *const c_void,
            db as *const c_void,
            dc_decode,
            m,
            n,
            k,
        )?;
        ctx.synchronize_stream(CudaStream::Prefill)?;
    }

    let mut generic_bytes = vec![0u8; bytes_c];
    let mut decode_bytes = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(
            generic_bytes.as_mut_ptr() as *mut c_void,
            dc_generic as *const c_void,
            generic_bytes.len(),
        )?;
        ctx.memcpy_d2h(
            decode_bytes.as_mut_ptr() as *mut c_void,
            dc_decode as *const c_void,
            decode_bytes.len(),
        )?;
    }
    let generic = bytes_to_f32s(&generic_bytes);
    let decode = bytes_to_f32s(&decode_bytes);
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for (g, d) in generic.iter().zip(decode.iter()) {
        // The decode kernel uses a parallel reduction, so large K rows can
        // legitimately differ from the scalar baseline by FP32 reduction order.
        let diff = (g - d).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }
    let mean_diff = sum_diff / generic.len() as f32;
    assert!(
        max_diff <= 2e-1 && mean_diff <= 2e-2,
        "decode mismatch max_diff={max_diff} mean_diff={mean_diff}"
    );

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc_generic)?;
        ctx.device_free(dc_decode)?;
    }
    Ok(())
}

#[test]
fn gemm_f32xq8_0_dispatch_handles_tail_k() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let (m, k, n) = (1i32, 35i32, 64i32);
    let a: Vec<f32> = (0..m * k)
        .map(|idx| ((idx % 7) as f32 - 3.0) * 0.25)
        .collect();
    let b: Vec<f32> = (0..n * k)
        .map(|idx| ((idx % 11) as f32 - 5.0) * 0.125)
        .collect();
    let b_q8 = q8_0_gguf_bytes_from_dequantized(&b, n as usize, k as usize);
    let b_deq = q8_0_gguf_dequantize(&b_q8, n as usize, k as usize);
    let expect = cpu_gguf_gemm_f32(&a, &b_deq, m as usize, n as usize, k as usize);

    let a_bytes = f32s_to_bytes(&a);
    let bytes_a = (m * k * 4) as usize;
    let bytes_c = (m * n * 4) as usize;
    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(b_q8.len())?;
    let dc = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_q8.as_ptr() as *const c_void, b_q8.len())?;
        ctx.gemm_f32xq8_0_gguf_f32(da as *const c_void, db as *const c_void, dc, m, n, k)?;
    }

    let mut got_bytes = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(
            got_bytes.as_mut_ptr() as *mut c_void,
            dc as *const c_void,
            got_bytes.len(),
        )?;
    }
    let got = bytes_to_f32s(&got_bytes);
    for (g, e) in got.iter().zip(expect.iter()) {
        assert!((g - e).abs() <= 1e-4, "got {g} expect {e}");
    }

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc)?;
    }
    Ok(())
}

#[test]
fn materialized_gguf_f32_sgemm_matches_cpu_rectangular_2x3x4() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let (m, k, n) = (2i32, 3i32, 4i32);
    let a = [1.0f32, -2.0, 3.0, 0.5, 4.0, -1.0];
    let b = [
        0.25f32, 0.5, 0.75, -1.0, -1.25, 1.5, 2.0, -0.5, 0.125, 3.0, 2.5, -2.0,
    ];
    let expect = cpu_gguf_gemm_f32(&a, &b, m as usize, n as usize, k as usize);

    let a_bytes = f32s_to_bytes(&a);
    let b_half = f32s_to_halves_bytes(&b);

    let bytes_a = (m * k * 4) as usize;
    let bytes_b = (k * n * 2) as usize;
    let bytes_bt = (k * n * 4) as usize;
    let bytes_c = (m * n * 4) as usize;

    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(bytes_b)?;
    let dbt = ctx.device_malloc(bytes_bt)?;
    let dc = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_half.as_ptr() as *const c_void, bytes_b)?;
        ctx.materialize_gguf_f16_to_f32_colmajor_nt(db as *const c_void, dbt, n, k)?;
        ctx.gemm_f32xf32_f32(da as *const c_void, dbt as *const c_void, dc, m, n, k)?;
    }

    let mut hc = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(hc.as_mut_ptr() as *mut c_void, dc as *const c_void, bytes_c)?;
    }

    let got = bytes_to_f32s(&hc);
    for (g, e) in got.iter().zip(expect.iter()) {
        let diff = (g - e).abs();
        assert!(diff <= 1e-3, "got {:?} expect {:?}", got, expect);
    }

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dbt)?;
        ctx.device_free(dc)?;
    }
    Ok(())
}

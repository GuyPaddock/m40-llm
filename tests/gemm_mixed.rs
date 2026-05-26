#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use m40_llm::cuda::CudaStream;
// CudaContext not used directly in this test
use std::ffi::c_void;

struct EnvRestore {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvRestore {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        std::env::set_var(key, value);
        Self { key, previous }
    }
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => std::env::set_var(self.key, value),
            None => std::env::remove_var(self.key),
        }
    }
}

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

fn q4_0_gguf_bytes_from_dequantized(vals: &[f32], n: usize, k: usize) -> Vec<u8> {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 18;
    let blocks_per_col = k.div_ceil(QK);
    let mut out = vec![0u8; n * blocks_per_col * BLOCK_BYTES];
    for col in 0..n {
        for block in 0..blocks_per_col {
            let start = block * QK;
            let end = (start + QK).min(k);
            let scale = vals[col * k + start..col * k + end]
                .iter()
                .fold(0.0f32, |acc, v| acc.max(v.abs()))
                / 7.0;
            let scale = if scale == 0.0 { 1.0 } else { scale };
            let base = (col * blocks_per_col + block) * BLOCK_BYTES;
            let scale_bits = half::f16::from_f32(scale).to_bits();
            out[base] = (scale_bits & 0xff) as u8;
            out[base + 1] = (scale_bits >> 8) as u8;
            for pair in 0..16 {
                let lo_idx = start + pair;
                let hi_idx = start + pair + 16;
                let lo = if lo_idx < k {
                    (vals[col * k + lo_idx] / scale).round().clamp(-8.0, 7.0) as i32 + 8
                } else {
                    8
                } as u8;
                let hi = if hi_idx < k {
                    (vals[col * k + hi_idx] / scale).round().clamp(-8.0, 7.0) as i32 + 8
                } else {
                    8
                } as u8;
                out[base + 2 + pair] = (lo & 0x0f) | ((hi & 0x0f) << 4);
            }
        }
    }
    out
}

fn q4_0_gguf_dequantize(bytes: &[u8], n: usize, k: usize) -> Vec<f32> {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 18;
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
                    let packed = bytes[base + 2 + (idx & 15)];
                    let q = if idx < 16 { packed & 0x0f } else { packed >> 4 };
                    out[col * k + k_idx] = (i32::from(q) - 8) as f32 * scale;
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
fn gemm_f32xq8_0_shared_activation_matches_cpu_medium_5x96x65() -> Result<()> {
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
        ctx.gemm_f32xq8_0_gguf_f32_shared_activation_async(
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
fn gemm_f32xq8_0_shared_activation_matches_cpu_tail_3x35x33() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let (m, k, n) = (3i32, 35i32, 33i32);
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
        ctx.gemm_f32xq8_0_gguf_f32_shared_activation_async(
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
fn gemm_f32xq8_0_shared_activation_matches_blockloop_prefill_shape() -> Result<()> {
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
    let dc_blockloop = ctx.device_malloc(bytes_c)?;
    let dc_shared = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_q8.as_ptr() as *const c_void, b_q8.len())?;
        ctx.gemm_f32xq8_0_gguf_f32_blockloop_async(
            da as *const c_void,
            db as *const c_void,
            dc_blockloop,
            m,
            n,
            k,
        )?;
        ctx.synchronize_stream(CudaStream::Prefill)?;
        ctx.gemm_f32xq8_0_gguf_f32_shared_activation_async(
            da as *const c_void,
            db as *const c_void,
            dc_shared,
            m,
            n,
            k,
        )?;
        ctx.synchronize_stream(CudaStream::Prefill)?;
    }

    let mut blockloop_bytes = vec![0u8; bytes_c];
    let mut shared_bytes = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(
            blockloop_bytes.as_mut_ptr() as *mut c_void,
            dc_blockloop as *const c_void,
            blockloop_bytes.len(),
        )?;
        ctx.memcpy_d2h(
            shared_bytes.as_mut_ptr() as *mut c_void,
            dc_shared as *const c_void,
            shared_bytes.len(),
        )?;
    }
    let blockloop = bytes_to_f32s(&blockloop_bytes);
    let shared = bytes_to_f32s(&shared_bytes);
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for (b, s) in blockloop.iter().zip(shared.iter()) {
        let diff = (b - s).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }
    let mean_diff = sum_diff / blockloop.len() as f32;
    assert!(
        max_diff <= 5e-1 && mean_diff <= 5e-3,
        "shared mismatch max_diff={max_diff} mean_diff={mean_diff}"
    );

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc_blockloop)?;
        ctx.device_free(dc_shared)?;
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
        ctx.synchronize_stream(CudaStream::Decode)?;
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
fn gemm_f32xq4_0_blockloop_matches_cpu_tail_2x35x3() -> Result<()> {
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
        .map(|idx| ((idx % 11) as f32 - 5.0) * 0.0625)
        .collect();
    let b: Vec<f32> = (0..n * k)
        .map(|idx| ((idx % 13) as f32 - 6.0) * 0.03125)
        .collect();
    let b_q4 = q4_0_gguf_bytes_from_dequantized(&b, n as usize, k as usize);
    let b_deq = q4_0_gguf_dequantize(&b_q4, n as usize, k as usize);
    let expect = cpu_gguf_gemm_f32(&a, &b_deq, m as usize, n as usize, k as usize);

    let a_bytes = f32s_to_bytes(&a);
    let bytes_a = (m * k * 4) as usize;
    let bytes_c = (m * n * 4) as usize;
    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(b_q4.len())?;
    let dc = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_q4.as_ptr() as *const c_void, b_q4.len())?;
        ctx.gemm_f32xq4_0_gguf_f32_blockloop_async(
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
    for (i, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 1e-4,
            "idx {i}: got {g}, expected {e}, diff {}",
            (g - e).abs()
        );
    }

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc)?;
    }
    Ok(())
}

#[test]
fn gemm_f32xq4_0_decode_matches_blockloop_1x2048x1024() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let (m, k, n) = (1i32, 2048i32, 1024i32);
    let a: Vec<f32> = (0..m * k)
        .map(|idx| ((idx % 17) as f32 - 8.0) * 0.03125)
        .collect();
    let b: Vec<f32> = (0..n * k)
        .map(|idx| ((idx % 23) as f32 - 11.0) * 0.015625)
        .collect();
    let b_q4 = q4_0_gguf_bytes_from_dequantized(&b, n as usize, k as usize);

    let a_bytes = f32s_to_bytes(&a);
    let bytes_a = (m * k * 4) as usize;
    let bytes_c = (m * n * 4) as usize;
    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(b_q4.len())?;
    let dc_blockloop = ctx.device_malloc(bytes_c)?;
    let dc_decode = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_q4.as_ptr() as *const c_void, b_q4.len())?;
        ctx.gemm_f32xq4_0_gguf_f32_blockloop_async(
            da as *const c_void,
            db as *const c_void,
            dc_blockloop,
            m,
            n,
            k,
        )?;
        ctx.gemm_f32xq4_0_gguf_f32_decode_async(
            da as *const c_void,
            db as *const c_void,
            dc_decode,
            m,
            n,
            k,
        )?;
        ctx.synchronize_stream(CudaStream::Prefill)?;
        ctx.synchronize_stream(CudaStream::Decode)?;
    }

    let mut blockloop_bytes = vec![0u8; bytes_c];
    let mut decode_bytes = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(
            blockloop_bytes.as_mut_ptr() as *mut c_void,
            dc_blockloop as *const c_void,
            blockloop_bytes.len(),
        )?;
        ctx.memcpy_d2h(
            decode_bytes.as_mut_ptr() as *mut c_void,
            dc_decode as *const c_void,
            decode_bytes.len(),
        )?;
    }
    let blockloop = bytes_to_f32s(&blockloop_bytes);
    let decode = bytes_to_f32s(&decode_bytes);
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for (b, d) in blockloop.iter().zip(decode.iter()) {
        let diff = (b - d).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }
    let mean_diff = sum_diff / blockloop.len() as f32;
    assert!(
        max_diff <= 3e-1 && mean_diff <= 5e-2,
        "q4 decode mismatch max_diff={max_diff} mean_diff={mean_diff}"
    );

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc_blockloop)?;
        ctx.device_free(dc_decode)?;
    }
    Ok(())
}

#[test]
fn gemm_f32xq8_0_decode_tiled2_matches_decode_large_k() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let (m, k, n) = (1i32, 4096i32, 1025i32);
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
    let dc_decode = ctx.device_malloc(bytes_c)?;
    let dc_tiled = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_q8.as_ptr() as *const c_void, b_q8.len())?;
        ctx.gemm_f32xq8_0_gguf_f32_decode_async(
            da as *const c_void,
            db as *const c_void,
            dc_decode,
            m,
            n,
            k,
        )?;
        ctx.gemm_f32xq8_0_gguf_f32_decode_tiled2_async(
            da as *const c_void,
            db as *const c_void,
            dc_tiled,
            m,
            n,
            k,
        )?;
        ctx.synchronize_stream(CudaStream::Prefill)?;
        ctx.synchronize_stream(CudaStream::Decode)?;
    }

    let mut decode_bytes = vec![0u8; bytes_c];
    let mut tiled_bytes = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(
            decode_bytes.as_mut_ptr() as *mut c_void,
            dc_decode as *const c_void,
            decode_bytes.len(),
        )?;
        ctx.memcpy_d2h(
            tiled_bytes.as_mut_ptr() as *mut c_void,
            dc_tiled as *const c_void,
            tiled_bytes.len(),
        )?;
    }
    let decode = bytes_to_f32s(&decode_bytes);
    let tiled = bytes_to_f32s(&tiled_bytes);
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for (d, t) in decode.iter().zip(tiled.iter()) {
        let diff = (d - t).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }
    let mean_diff = sum_diff / decode.len() as f32;
    assert!(
        max_diff <= 1e-5 && mean_diff <= 1e-6,
        "tiled decode mismatch max_diff={max_diff} mean_diff={mean_diff}"
    );

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc_decode)?;
        ctx.device_free(dc_tiled)?;
    }
    Ok(())
}

#[test]
fn gemm_f32xq8_0_decode_tiled2_matches_decode_vocab_shape() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let (m, k, n) = (1i32, 2048i32, 4097i32);
    let a: Vec<f32> = (0..m * k)
        .map(|idx| ((idx % 19) as f32 - 9.0) * 0.02734375)
        .collect();
    let b: Vec<f32> = (0..n * k)
        .map(|idx| ((idx % 29) as f32 - 14.0) * 0.013671875)
        .collect();
    let b_q8 = q8_0_gguf_bytes_from_dequantized(&b, n as usize, k as usize);

    let a_bytes = f32s_to_bytes(&a);
    let bytes_a = (m * k * 4) as usize;
    let bytes_c = (m * n * 4) as usize;
    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(b_q8.len())?;
    let dc_decode = ctx.device_malloc(bytes_c)?;
    let dc_tiled = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_q8.as_ptr() as *const c_void, b_q8.len())?;
        ctx.gemm_f32xq8_0_gguf_f32_decode_async(
            da as *const c_void,
            db as *const c_void,
            dc_decode,
            m,
            n,
            k,
        )?;
        ctx.gemm_f32xq8_0_gguf_f32_decode_tiled2_async(
            da as *const c_void,
            db as *const c_void,
            dc_tiled,
            m,
            n,
            k,
        )?;
        ctx.synchronize_stream(CudaStream::Prefill)?;
        ctx.synchronize_stream(CudaStream::Decode)?;
    }

    let mut decode_bytes = vec![0u8; bytes_c];
    let mut tiled_bytes = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(
            decode_bytes.as_mut_ptr() as *mut c_void,
            dc_decode as *const c_void,
            decode_bytes.len(),
        )?;
        ctx.memcpy_d2h(
            tiled_bytes.as_mut_ptr() as *mut c_void,
            dc_tiled as *const c_void,
            tiled_bytes.len(),
        )?;
    }
    let decode = bytes_to_f32s(&decode_bytes);
    let tiled = bytes_to_f32s(&tiled_bytes);
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for (d, t) in decode.iter().zip(tiled.iter()) {
        let diff = (d - t).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }
    let mean_diff = sum_diff / decode.len() as f32;
    assert!(
        max_diff <= 1e-5 && mean_diff <= 1e-6,
        "tiled vocab-shape decode mismatch max_diff={max_diff} mean_diff={mean_diff}"
    );

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc_decode)?;
        ctx.device_free(dc_tiled)?;
    }
    Ok(())
}

#[test]
fn gemm_f32xq8_0_decode_split4_matches_decode() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let (m, k, n) = (1i32, 2048i32, 1025i32);
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
    let dc_decode = ctx.device_malloc(bytes_c)?;
    let dc_split = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_q8.as_ptr() as *const c_void, b_q8.len())?;
        ctx.gemm_f32xq8_0_gguf_f32_decode_async(
            da as *const c_void,
            db as *const c_void,
            dc_decode,
            m,
            n,
            k,
        )?;
        {
            let _split = EnvRestore::set("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4");
            ctx.gemm_f32xq8_0_gguf_f32_decode_async(
                da as *const c_void,
                db as *const c_void,
                dc_split,
                m,
                n,
                k,
            )?;
        }
        ctx.synchronize_stream(CudaStream::Prefill)?;
        ctx.synchronize_stream(CudaStream::Decode)?;
    }

    let mut decode_bytes = vec![0u8; bytes_c];
    let mut split_bytes = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(
            decode_bytes.as_mut_ptr() as *mut c_void,
            dc_decode as *const c_void,
            decode_bytes.len(),
        )?;
        ctx.memcpy_d2h(
            split_bytes.as_mut_ptr() as *mut c_void,
            dc_split as *const c_void,
            split_bytes.len(),
        )?;
    }
    let decode = bytes_to_f32s(&decode_bytes);
    let split = bytes_to_f32s(&split_bytes);
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for (d, s) in decode.iter().zip(split.iter()) {
        let diff = (d - s).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }
    let mean_diff = sum_diff / decode.len() as f32;
    assert!(
        max_diff <= 1e-5 && mean_diff <= 1e-6,
        "split4 decode mismatch max_diff={max_diff} mean_diff={mean_diff}"
    );

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc_decode)?;
        ctx.device_free(dc_split)?;
    }
    Ok(())
}

#[test]
fn gemm_f32xq8_0_decode_coltile4_split4_matches_decode() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let (m, k, n) = (1i32, 2048i32, 1031i32);
    let a: Vec<f32> = (0..m * k)
        .map(|idx| ((idx % 19) as f32 - 9.0) * 0.025)
        .collect();
    let b: Vec<f32> = (0..n * k)
        .map(|idx| ((idx % 29) as f32 - 14.0) * 0.0125)
        .collect();
    let b_q8 = q8_0_gguf_bytes_from_dequantized(&b, n as usize, k as usize);

    let a_bytes = f32s_to_bytes(&a);
    let bytes_a = (m * k * 4) as usize;
    let bytes_c = (m * n * 4) as usize;
    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(b_q8.len())?;
    let dc_decode = ctx.device_malloc(bytes_c)?;
    let dc_tiled = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_q8.as_ptr() as *const c_void, b_q8.len())?;
        ctx.gemm_f32xq8_0_gguf_f32_decode_async(
            da as *const c_void,
            db as *const c_void,
            dc_decode,
            m,
            n,
            k,
        )?;
        {
            let _split = EnvRestore::set("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4");
            let _tile = EnvRestore::set("M40LLM_Q8_DECODE_COL_TILE", "4");
            ctx.gemm_f32xq8_0_gguf_f32_decode_async(
                da as *const c_void,
                db as *const c_void,
                dc_tiled,
                m,
                n,
                k,
            )?;
        }
        ctx.synchronize_stream(CudaStream::Prefill)?;
        ctx.synchronize_stream(CudaStream::Decode)?;
    }

    let mut decode_bytes = vec![0u8; bytes_c];
    let mut tiled_bytes = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(
            decode_bytes.as_mut_ptr() as *mut c_void,
            dc_decode as *const c_void,
            decode_bytes.len(),
        )?;
        ctx.memcpy_d2h(
            tiled_bytes.as_mut_ptr() as *mut c_void,
            dc_tiled as *const c_void,
            tiled_bytes.len(),
        )?;
    }
    let decode = bytes_to_f32s(&decode_bytes);
    let tiled = bytes_to_f32s(&tiled_bytes);
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for (d, t) in decode.iter().zip(tiled.iter()) {
        let diff = (d - t).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }
    let mean_diff = sum_diff / decode.len() as f32;
    assert!(
        max_diff <= 1e-5 && mean_diff <= 1e-6,
        "coltile4 split4 decode mismatch max_diff={max_diff} mean_diff={mean_diff}"
    );

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc_decode)?;
        ctx.device_free(dc_tiled)?;
    }
    Ok(())
}

#[test]
fn q8_0_mlp_gate_up_split4_matches_baseline() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let (k, h) = (2048i32, 257i32);
    let a: Vec<f32> = (0..k)
        .map(|idx| ((idx % 31) as f32 - 15.0) * 0.01953125)
        .collect();
    let gate: Vec<f32> = (0..h * k)
        .map(|idx| ((idx % 37) as f32 - 18.0) * 0.01171875)
        .collect();
    let up: Vec<f32> = (0..h * k)
        .map(|idx| ((idx % 41) as f32 - 20.0) * 0.0107421875)
        .collect();
    let gate_q8 = q8_0_gguf_bytes_from_dequantized(&gate, h as usize, k as usize);
    let up_q8 = q8_0_gguf_bytes_from_dequantized(&up, h as usize, k as usize);

    let a_bytes = f32s_to_bytes(&a);
    let bytes_a = (k * 4) as usize;
    let bytes_c = (h * 4) as usize;
    let da = ctx.device_malloc(bytes_a)?;
    let d_gate = ctx.device_malloc(gate_q8.len())?;
    let d_up = ctx.device_malloc(up_q8.len())?;
    let d_baseline = ctx.device_malloc(bytes_c)?;
    let d_split = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(d_gate, gate_q8.as_ptr() as *const c_void, gate_q8.len())?;
        ctx.memcpy_h2d(d_up, up_q8.as_ptr() as *const c_void, up_q8.len())?;
        ctx.mlp_gate_up_swiglu_f32xq8_0_gguf_decode_async(
            da as *const c_void,
            d_gate as *const c_void,
            d_up as *const c_void,
            d_baseline,
            h,
            k,
        )?;
        {
            let _split = EnvRestore::set("M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK", "4");
            ctx.mlp_gate_up_swiglu_f32xq8_0_gguf_decode_async(
                da as *const c_void,
                d_gate as *const c_void,
                d_up as *const c_void,
                d_split,
                h,
                k,
            )?;
        }
        ctx.synchronize_stream(CudaStream::Prefill)?;
        ctx.synchronize_stream(CudaStream::Decode)?;
    }

    let mut baseline_bytes = vec![0u8; bytes_c];
    let mut split_bytes = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(
            baseline_bytes.as_mut_ptr() as *mut c_void,
            d_baseline as *const c_void,
            baseline_bytes.len(),
        )?;
        ctx.memcpy_d2h(
            split_bytes.as_mut_ptr() as *mut c_void,
            d_split as *const c_void,
            split_bytes.len(),
        )?;
    }
    let baseline = bytes_to_f32s(&baseline_bytes);
    let split = bytes_to_f32s(&split_bytes);
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for (b, s) in baseline.iter().zip(split.iter()) {
        let diff = (b - s).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }
    let mean_diff = sum_diff / baseline.len() as f32;
    assert!(
        max_diff <= 1e-4 && mean_diff <= 1e-5,
        "q8 split4 MLP mismatch max_diff={max_diff} mean_diff={mean_diff}"
    );

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(d_gate)?;
        ctx.device_free(d_up)?;
        ctx.device_free(d_baseline)?;
        ctx.device_free(d_split)?;
    }
    Ok(())
}

#[test]
fn q8_0_qkv_split4_matches_baseline() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let k = 2048i32;
    let (nq, nk, nv) = (257i32, 65i32, 65i32);
    let a: Vec<f32> = (0..k)
        .map(|idx| ((idx % 29) as f32 - 14.0) * 0.015625)
        .collect();
    let wq: Vec<f32> = (0..nq * k)
        .map(|idx| ((idx % 31) as f32 - 15.0) * 0.0107421875)
        .collect();
    let wk: Vec<f32> = (0..nk * k)
        .map(|idx| ((idx % 37) as f32 - 18.0) * 0.009765625)
        .collect();
    let wv: Vec<f32> = (0..nv * k)
        .map(|idx| ((idx % 41) as f32 - 20.0) * 0.0087890625)
        .collect();
    let bq: Vec<f32> = (0..nq).map(|idx| ((idx % 7) as f32 - 3.0) * 0.01).collect();
    let bk: Vec<f32> = (0..nk).map(|idx| ((idx % 5) as f32 - 2.0) * 0.01).collect();
    let bv: Vec<f32> = (0..nv).map(|idx| ((idx % 3) as f32 - 1.0) * 0.01).collect();
    let wq_q8 = q8_0_gguf_bytes_from_dequantized(&wq, nq as usize, k as usize);
    let wk_q8 = q8_0_gguf_bytes_from_dequantized(&wk, nk as usize, k as usize);
    let wv_q8 = q8_0_gguf_bytes_from_dequantized(&wv, nv as usize, k as usize);

    let a_bytes = f32s_to_bytes(&a);
    let bq_bytes = f32s_to_bytes(&bq);
    let bk_bytes = f32s_to_bytes(&bk);
    let bv_bytes = f32s_to_bytes(&bv);
    let bytes_a = (k * 4) as usize;
    let bytes_q = (nq * 4) as usize;
    let bytes_k = (nk * 4) as usize;
    let bytes_v = (nv * 4) as usize;

    let da = ctx.device_malloc(bytes_a)?;
    let dwq = ctx.device_malloc(wq_q8.len())?;
    let dwk = ctx.device_malloc(wk_q8.len())?;
    let dwv = ctx.device_malloc(wv_q8.len())?;
    let dbq = ctx.device_malloc(bq_bytes.len())?;
    let dbk = ctx.device_malloc(bk_bytes.len())?;
    let dbv = ctx.device_malloc(bv_bytes.len())?;
    let dq_base = ctx.device_malloc(bytes_q)?;
    let dk_base = ctx.device_malloc(bytes_k)?;
    let dv_base = ctx.device_malloc(bytes_v)?;
    let dq_split = ctx.device_malloc(bytes_q)?;
    let dk_split = ctx.device_malloc(bytes_k)?;
    let dv_split = ctx.device_malloc(bytes_v)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(dwq, wq_q8.as_ptr() as *const c_void, wq_q8.len())?;
        ctx.memcpy_h2d(dwk, wk_q8.as_ptr() as *const c_void, wk_q8.len())?;
        ctx.memcpy_h2d(dwv, wv_q8.as_ptr() as *const c_void, wv_q8.len())?;
        ctx.memcpy_h2d(dbq, bq_bytes.as_ptr() as *const c_void, bq_bytes.len())?;
        ctx.memcpy_h2d(dbk, bk_bytes.as_ptr() as *const c_void, bk_bytes.len())?;
        ctx.memcpy_h2d(dbv, bv_bytes.as_ptr() as *const c_void, bv_bytes.len())?;

        ctx.qkv_f32xq8_0_gguf_decode_async(
            da as *const c_void,
            dwq as *const c_void,
            dwk as *const c_void,
            dwv as *const c_void,
            Some(dbq as *const c_void),
            Some(dbk as *const c_void),
            Some(dbv as *const c_void),
            dq_base,
            dk_base,
            dv_base,
            nq,
            nk,
            nv,
            k,
        )?;
        {
            let _split = EnvRestore::set("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4");
            ctx.qkv_f32xq8_0_gguf_decode_async(
                da as *const c_void,
                dwq as *const c_void,
                dwk as *const c_void,
                dwv as *const c_void,
                Some(dbq as *const c_void),
                Some(dbk as *const c_void),
                Some(dbv as *const c_void),
                dq_split,
                dk_split,
                dv_split,
                nq,
                nk,
                nv,
                k,
            )?;
        }
        ctx.synchronize_stream(CudaStream::Prefill)?;
        ctx.synchronize_stream(CudaStream::Decode)?;
    }

    let compare = |label: &str, base_ptr, split_ptr, bytes| -> Result<()> {
        let mut base_bytes = vec![0u8; bytes];
        let mut split_bytes = vec![0u8; bytes];
        unsafe {
            ctx.memcpy_d2h(
                base_bytes.as_mut_ptr() as *mut c_void,
                base_ptr as *const c_void,
                bytes,
            )?;
            ctx.memcpy_d2h(
                split_bytes.as_mut_ptr() as *mut c_void,
                split_ptr as *const c_void,
                bytes,
            )?;
        }
        let base = bytes_to_f32s(&base_bytes);
        let split = bytes_to_f32s(&split_bytes);
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f32;
        for (b, s) in base.iter().zip(split.iter()) {
            let diff = (b - s).abs();
            max_diff = max_diff.max(diff);
            sum_diff += diff;
        }
        let mean_diff = sum_diff / base.len() as f32;
        assert!(
            max_diff <= 1e-4 && mean_diff <= 1e-5,
            "q8 split4 QKV {label} mismatch max_diff={max_diff} mean_diff={mean_diff}"
        );
        Ok(())
    };
    compare("q", dq_base, dq_split, bytes_q)?;
    compare("k", dk_base, dk_split, bytes_k)?;
    compare("v", dv_base, dv_split, bytes_v)?;

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(dwq)?;
        ctx.device_free(dwk)?;
        ctx.device_free(dwv)?;
        ctx.device_free(dbq)?;
        ctx.device_free(dbk)?;
        ctx.device_free(dbv)?;
        ctx.device_free(dq_base)?;
        ctx.device_free(dk_base)?;
        ctx.device_free(dv_base)?;
        ctx.device_free(dq_split)?;
        ctx.device_free(dk_split)?;
        ctx.device_free(dv_split)?;
    }
    Ok(())
}

#[test]
fn q8_0_lm_head_argmax_matches_decode_logits_argmax() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }
    let (m, k, n) = (1i32, 2048i32, 4099i32);
    let a: Vec<f32> = (0..m * k)
        .map(|idx| ((idx % 31) as f32 - 15.0) * 0.01953125)
        .collect();
    let b: Vec<f32> = (0..n * k)
        .map(|idx| ((idx % 37) as f32 - 18.0) * 0.01171875)
        .collect();
    let b_q8 = q8_0_gguf_bytes_from_dequantized(&b, n as usize, k as usize);

    let a_bytes = f32s_to_bytes(&a);
    let bytes_a = (m * k * 4) as usize;
    let bytes_logits = (m * n * 4) as usize;
    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(b_q8.len())?;
    let d_logits = ctx.device_malloc(bytes_logits)?;
    let d_token_ref = ctx.device_malloc(std::mem::size_of::<u32>())?;
    let d_token_fused = ctx.device_malloc(std::mem::size_of::<u32>())?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_q8.as_ptr() as *const c_void, b_q8.len())?;
        ctx.gemm_f32xq8_0_gguf_f32_decode_async(
            da as *const c_void,
            db as *const c_void,
            d_logits,
            m,
            n,
            k,
        )?;
        ctx.argmax_f32_async(
            d_logits as *const c_void,
            n as u32,
            d_token_ref,
            CudaStream::Decode,
        )?;
        ctx.synchronize_stream(CudaStream::Decode)?;
        ctx.q8_0_lm_head_argmax_async(
            da as *const c_void,
            db as *const c_void,
            d_logits,
            d_token_fused,
            n,
            k,
            CudaStream::Decode,
        )?;
        ctx.synchronize_stream(CudaStream::Decode)?;
    }

    let mut ref_token = 0u32;
    let mut fused_token = 0u32;
    unsafe {
        ctx.memcpy_d2h(
            (&mut ref_token as *mut u32).cast::<c_void>(),
            d_token_ref as *const c_void,
            std::mem::size_of::<u32>(),
        )?;
        ctx.memcpy_d2h(
            (&mut fused_token as *mut u32).cast::<c_void>(),
            d_token_fused as *const c_void,
            std::mem::size_of::<u32>(),
        )?;
    }
    assert_eq!(fused_token, ref_token);

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(d_logits)?;
        ctx.device_free(d_token_ref)?;
        ctx.device_free(d_token_fused)?;
    }
    Ok(())
}

#[test]
fn gemm_f32xf16_gguf_decode_kernel_matches_sync_1x2048x2048() -> Result<()> {
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
    let b_f16 = f32s_to_halves_bytes(&b);

    let a_bytes = f32s_to_bytes(&a);
    let bytes_a = (m * k * 4) as usize;
    let bytes_c = (m * n * 4) as usize;
    let da = ctx.device_malloc(bytes_a)?;
    let db = ctx.device_malloc(b_f16.len())?;
    let dc_sync = ctx.device_malloc(bytes_c)?;
    let dc_decode = ctx.device_malloc(bytes_c)?;

    unsafe {
        ctx.memcpy_h2d(da, a_bytes.as_ptr() as *const c_void, bytes_a)?;
        ctx.memcpy_h2d(db, b_f16.as_ptr() as *const c_void, b_f16.len())?;
        ctx.gemm_f32xf16_gguf_f32(da as *const c_void, db as *const c_void, dc_sync, m, n, k)?;
        ctx.gemm_f32xf16_gguf_f32_decode_async(
            da as *const c_void,
            db as *const c_void,
            dc_decode,
            m,
            n,
            k,
        )?;
        ctx.synchronize_stream(CudaStream::Prefill)?;
    }

    let mut sync_bytes = vec![0u8; bytes_c];
    let mut decode_bytes = vec![0u8; bytes_c];
    unsafe {
        ctx.memcpy_d2h(
            sync_bytes.as_mut_ptr() as *mut c_void,
            dc_sync as *const c_void,
            sync_bytes.len(),
        )?;
        ctx.memcpy_d2h(
            decode_bytes.as_mut_ptr() as *mut c_void,
            dc_decode as *const c_void,
            decode_bytes.len(),
        )?;
    }
    let sync = bytes_to_f32s(&sync_bytes);
    let decode = bytes_to_f32s(&decode_bytes);
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for (s, d) in sync.iter().zip(decode.iter()) {
        let diff = (s - d).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }
    let mean_diff = sum_diff / sync.len() as f32;
    assert!(
        max_diff <= 5e-4 && mean_diff <= 5e-5,
        "f16 decode mismatch max_diff={max_diff} mean_diff={mean_diff}"
    );

    unsafe {
        ctx.device_free(da)?;
        ctx.device_free(db)?;
        ctx.device_free(dc_sync)?;
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

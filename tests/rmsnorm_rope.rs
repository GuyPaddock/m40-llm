#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use std::ffi::c_void;

fn cpu_rms_norm(input: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * dim];
    for r in 0..rows {
        let base = r * dim;
        let row = &input[base..base + dim];
        let mut sum = 0.0f32;
        for &v in row {
            sum += v * v;
        }
        let scale = (sum / dim as f32 + eps).sqrt().recip();
        for c in 0..dim {
            out[base + c] = row[c] * scale;
        }
    }
    out
}

fn cpu_rope(
    q: &mut [f32],
    k: &mut [f32],
    rows: usize,
    num_heads: usize,
    head_dim: usize,
    past_len: usize,
    freq_base: f32,
    freq_scale: f32,
) {
    let pairs_per_head = head_dim / 2;
    for row in 0..rows {
        let pos = (past_len + row) as f32 * freq_scale;
        for h in 0..num_heads {
            let head_offset = row * num_heads * head_dim + h * head_dim;
            for pair in 0..pairs_per_head {
                let idx = head_offset + 2 * pair;
                let theta = pos * freq_base.powf(-2.0f32 * pair as f32 / head_dim as f32);
                let c = theta.cos();
                let s = theta.sin();

                let q0 = q[idx];
                let q1 = q[idx + 1];
                q[idx] = q0 * c - q1 * s;
                q[idx + 1] = q0 * s + q1 * c;

                let k0 = k[idx];
                let k1 = k[idx + 1];
                k[idx] = k0 * c - k1 * s;
                k[idx + 1] = k0 * s + k1 * c;
            }
        }
    }
}

#[test]
fn rms_norm_kernel_matches_cpu() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(c) => c,
        None => return Ok(()),
    };

    let rows = 2usize;
    let dim = 8usize;
    let eps = 1e-5f32;
    let input: Vec<f32> = (0..rows * dim).map(|i| (i as f32) * 0.25 - 3.0).collect();
    let expected = cpu_rms_norm(&input, rows, dim, eps);

    let bytes = input.len() * std::mem::size_of::<f32>();
    let d_in = ctx.device_malloc(bytes)?;
    let d_out = ctx.device_malloc(bytes)?;
    unsafe {
        ctx.memcpy_h2d(d_in, input.as_ptr() as *const c_void, bytes)?;
        ctx.rms_norm_f32(d_in, d_out, rows as u32, dim as u32, eps)?;
        let mut out_bytes = vec![0u8; bytes];
        ctx.memcpy_d2h(out_bytes.as_mut_ptr() as *mut c_void, d_out, bytes)?;
        let mut actual = Vec::with_capacity(rows * dim);
        for chunk in out_bytes.chunks_exact(4) {
            actual.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "mismatch at {}: got {}, expected {}",
                i,
                a,
                b
            );
        }
    }
    unsafe {
        ctx.device_free(d_in)?;
        ctx.device_free(d_out)?;
    }
    Ok(())
}

#[test]
fn rope_kernel_matches_cpu() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(c) => c,
        None => return Ok(()),
    };

    let rows = 1usize;
    let num_heads = 2usize;
    let head_dim = 4usize;
    let past_len = 3usize;
    let freq_base = 10_000.0f32;
    let freq_scale = 1.0f32;

    let mut q: Vec<f32> = (0..rows * num_heads * head_dim)
        .map(|i| (i as f32) * 0.1 + 0.5)
        .collect();
    let mut k = q.clone();
    let mut expected_q = q.clone();
    let mut expected_k = k.clone();
    cpu_rope(
        &mut expected_q,
        &mut expected_k,
        rows,
        num_heads,
        head_dim,
        past_len,
        freq_base,
        freq_scale,
    );

    let bytes = q.len() * std::mem::size_of::<f32>();
    let d_q = ctx.device_malloc(bytes)?;
    let d_k = ctx.device_malloc(bytes)?;
    unsafe {
        ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, bytes)?;
        ctx.memcpy_h2d(d_k, k.as_ptr() as *const c_void, bytes)?;
        ctx.rope_f32(
            d_q,
            d_k,
            rows as u32,
            num_heads as u32,
            head_dim as u32,
            past_len as u32,
            freq_base,
            freq_scale,
        )?;

        let mut out_q = vec![0u8; bytes];
        let mut out_k = vec![0u8; bytes];
        ctx.memcpy_d2h(out_q.as_mut_ptr() as *mut c_void, d_q, bytes)?;
        ctx.memcpy_d2h(out_k.as_mut_ptr() as *mut c_void, d_k, bytes)?;

        let to_f32 = |buf: Vec<u8>| -> Vec<f32> {
            buf.chunks_exact(4)
                .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
                .collect()
        };
        let got_q = to_f32(out_q);
        let got_k = to_f32(out_k);

        for (i, (a, b)) in got_q.iter().zip(expected_q.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "q mismatch at {}: got {}, expected {}",
                i,
                a,
                b
            );
        }
        for (i, (a, b)) in got_k.iter().zip(expected_k.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "k mismatch at {}: got {}, expected {}",
                i,
                a,
                b
            );
        }
    }

    unsafe {
        ctx.device_free(d_q)?;
        ctx.device_free(d_k)?;
    }
    Ok(())
}

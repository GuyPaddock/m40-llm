#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use m40_llm::cuda::CudaStream;
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

fn cpu_rms_norm_weighted(
    input: &[f32],
    weight: &[f32],
    rows: usize,
    dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = cpu_rms_norm(input, rows, dim, eps);
    for r in 0..rows {
        for c in 0..dim {
            out[r * dim + c] *= weight[c];
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
fn weighted_rms_norm_kernel_matches_cpu_f32_weight() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(c) => c,
        None => return Ok(()),
    };

    let rows = 2usize;
    let dim = 8usize;
    let eps = 1e-5f32;
    let input: Vec<f32> = (0..rows * dim).map(|i| (i as f32) * 0.125 - 1.0).collect();
    let weight: Vec<f32> = (0..dim).map(|i| 0.5 + i as f32 * 0.1).collect();
    let expected = cpu_rms_norm_weighted(&input, &weight, rows, dim, eps);

    let input_bytes = input.len() * std::mem::size_of::<f32>();
    let weight_bytes = weight.len() * std::mem::size_of::<f32>();
    let d_in = ctx.device_malloc(input_bytes)?;
    let d_weight = ctx.device_malloc(weight_bytes)?;
    let d_out = ctx.device_malloc(input_bytes)?;
    unsafe {
        ctx.memcpy_h2d(d_in, input.as_ptr() as *const c_void, input_bytes)?;
        ctx.memcpy_h2d(d_weight, weight.as_ptr() as *const c_void, weight_bytes)?;
        ctx.rms_norm_f32_weighted(
            d_in,
            d_weight as *const c_void,
            d_out,
            rows as u32,
            dim as u32,
            eps,
            1,
        )?;
        let mut out_bytes = vec![0u8; input_bytes];
        ctx.memcpy_d2h(out_bytes.as_mut_ptr() as *mut c_void, d_out, input_bytes)?;
        let actual: Vec<f32> = out_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
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
        ctx.device_free(d_weight)?;
        ctx.device_free(d_out)?;
    }
    Ok(())
}

#[test]
fn weighted_rms_norm_ldg_experiment_matches_cpu_f32_weight() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(c) => c,
        None => return Ok(()),
    };

    let rows = 3usize;
    let dim = 64usize;
    let eps = 1e-5f32;
    let input: Vec<f32> = (0..rows * dim)
        .map(|i| ((i * 17 % 251) as f32) * 0.01 - 1.25)
        .collect();
    let weight: Vec<f32> = (0..dim).map(|i| 0.75 + i as f32 * 0.003).collect();
    let expected = cpu_rms_norm_weighted(&input, &weight, rows, dim, eps);

    let input_bytes = input.len() * std::mem::size_of::<f32>();
    let weight_bytes = weight.len() * std::mem::size_of::<f32>();
    let d_in = ctx.device_malloc(input_bytes)?;
    let d_weight = ctx.device_malloc(weight_bytes)?;
    let d_out = ctx.device_malloc(input_bytes)?;
    unsafe {
        ctx.memcpy_h2d(d_in, input.as_ptr() as *const c_void, input_bytes)?;
        ctx.memcpy_h2d(d_weight, weight.as_ptr() as *const c_void, weight_bytes)?;
        std::env::set_var("M40LLM_CACHE_EXPERIMENT", "ldg");
        let result = ctx.rms_norm_f32_weighted(
            d_in,
            d_weight as *const c_void,
            d_out,
            rows as u32,
            dim as u32,
            eps,
            1,
        );
        std::env::remove_var("M40LLM_CACHE_EXPERIMENT");
        result?;
        let mut out_bytes = vec![0u8; input_bytes];
        ctx.memcpy_d2h(out_bytes.as_mut_ptr() as *mut c_void, d_out, input_bytes)?;
        let actual: Vec<f32> = out_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
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
        ctx.device_free(d_weight)?;
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

    let q: Vec<f32> = (0..rows * num_heads * head_dim)
        .map(|i| (i as f32) * 0.1 + 0.5)
        .collect();
    let k = q.clone();
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

#[test]
fn async_norm_and_rope_wrappers_match_cpu() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(c) => c,
        None => return Ok(()),
    };

    let rows = 2usize;
    let dim = 8usize;
    let eps = 1e-5f32;
    let input: Vec<f32> = (0..rows * dim).map(|i| i as f32 * 0.2 - 1.0).collect();
    let weight: Vec<f32> = (0..dim).map(|i| 0.75 + i as f32 * 0.05).collect();
    let expected_rms = cpu_rms_norm(&input, rows, dim, eps);
    let expected_weighted = cpu_rms_norm_weighted(&input, &weight, rows, dim, eps);

    let input_bytes = input.len() * std::mem::size_of::<f32>();
    let weight_bytes = weight.len() * std::mem::size_of::<f32>();
    let d_in = ctx.device_malloc(input_bytes)?;
    let d_weight = ctx.device_malloc(weight_bytes)?;
    let d_rms = ctx.device_malloc(input_bytes)?;
    let d_weighted = ctx.device_malloc(input_bytes)?;
    unsafe {
        ctx.memcpy_h2d(d_in, input.as_ptr() as *const c_void, input_bytes)?;
        ctx.memcpy_h2d(d_weight, weight.as_ptr() as *const c_void, weight_bytes)?;
        ctx.rms_norm_f32_async(d_in, d_rms, rows as u32, dim as u32, eps)?;
        ctx.rms_norm_f32_weighted_async(
            d_in,
            d_weight as *const c_void,
            d_weighted,
            rows as u32,
            dim as u32,
            eps,
            1,
        )?;
        ctx.synchronize_stream(CudaStream::Decode)?;

        let mut rms_bytes = vec![0u8; input_bytes];
        let mut weighted_bytes = vec![0u8; input_bytes];
        ctx.memcpy_d2h(rms_bytes.as_mut_ptr() as *mut c_void, d_rms, input_bytes)?;
        ctx.memcpy_d2h(
            weighted_bytes.as_mut_ptr() as *mut c_void,
            d_weighted,
            input_bytes,
        )?;
        let actual_rms: Vec<f32> = rms_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let actual_weighted: Vec<f32> = weighted_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        for (i, (got, want)) in actual_rms.iter().zip(&expected_rms).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "async rms mismatch at {i}: got {got}, expected {want}"
            );
        }
        for (i, (got, want)) in actual_weighted.iter().zip(&expected_weighted).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "async weighted rms mismatch at {i}: got {got}, expected {want}"
            );
        }
    }

    let rope_rows = 1usize;
    let num_heads = 2usize;
    let head_dim = 4usize;
    let past_len = 5usize;
    let freq_base = 10_000.0f32;
    let freq_scale = 1.0f32;
    let q: Vec<f32> = (0..rope_rows * num_heads * head_dim)
        .map(|i| i as f32 * 0.125 - 0.25)
        .collect();
    let k = q.clone();
    let mut expected_q = q.clone();
    let mut expected_k = k.clone();
    cpu_rope(
        &mut expected_q,
        &mut expected_k,
        rope_rows,
        num_heads,
        head_dim,
        past_len,
        freq_base,
        freq_scale,
    );
    let rope_bytes = q.len() * std::mem::size_of::<f32>();
    let d_q = ctx.device_malloc(rope_bytes)?;
    let d_k = ctx.device_malloc(rope_bytes)?;
    let d_x = ctx.device_malloc(rope_bytes)?;
    let d_x_position_dev = ctx.device_malloc(rope_bytes)?;
    let d_past_len = ctx.device_malloc(std::mem::size_of::<u32>())?;
    unsafe {
        ctx.memcpy_h2d(d_q, q.as_ptr() as *const c_void, rope_bytes)?;
        ctx.memcpy_h2d(d_k, k.as_ptr() as *const c_void, rope_bytes)?;
        ctx.memcpy_h2d(d_x, q.as_ptr() as *const c_void, rope_bytes)?;
        ctx.memcpy_h2d(d_x_position_dev, q.as_ptr() as *const c_void, rope_bytes)?;
        ctx.memcpy_h2d(
            d_past_len,
            &(past_len as u32) as *const u32 as *const c_void,
            std::mem::size_of::<u32>(),
        )?;
        ctx.rope_f32_async(
            d_q,
            d_k,
            rope_rows as u32,
            num_heads as u32,
            head_dim as u32,
            past_len as u32,
            freq_base,
            freq_scale,
        )?;
        ctx.rope_f32_inplace_async(
            d_x,
            rope_rows as u32,
            num_heads as u32,
            head_dim as u32,
            past_len as u32,
            freq_base,
            freq_scale,
        )?;
        ctx.rope_f32_inplace_position_dev_async(
            d_x_position_dev,
            rope_rows as u32,
            num_heads as u32,
            head_dim as u32,
            d_past_len as *const u32,
            freq_base,
            freq_scale,
        )?;
        ctx.synchronize_stream(CudaStream::Decode)?;
        let mut q_bytes = vec![0u8; rope_bytes];
        let mut k_bytes = vec![0u8; rope_bytes];
        let mut x_bytes = vec![0u8; rope_bytes];
        let mut x_position_dev_bytes = vec![0u8; rope_bytes];
        ctx.memcpy_d2h(q_bytes.as_mut_ptr() as *mut c_void, d_q, rope_bytes)?;
        ctx.memcpy_d2h(k_bytes.as_mut_ptr() as *mut c_void, d_k, rope_bytes)?;
        ctx.memcpy_d2h(x_bytes.as_mut_ptr() as *mut c_void, d_x, rope_bytes)?;
        ctx.memcpy_d2h(
            x_position_dev_bytes.as_mut_ptr() as *mut c_void,
            d_x_position_dev,
            rope_bytes,
        )?;
        let actual_q: Vec<f32> = q_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let actual_k: Vec<f32> = k_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let actual_x: Vec<f32> = x_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let actual_x_position_dev: Vec<f32> = x_position_dev_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        for (i, (got, want)) in actual_q.iter().zip(&expected_q).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "async rope q mismatch at {i}: got {got}, expected {want}"
            );
        }
        for (i, (got, want)) in actual_k.iter().zip(&expected_k).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "async rope k mismatch at {i}: got {got}, expected {want}"
            );
        }
        for (i, (got, want)) in actual_x.iter().zip(&expected_q).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "async rope inplace mismatch at {i}: got {got}, expected {want}"
            );
        }
        for (i, (got, want)) in actual_x_position_dev.iter().zip(&expected_q).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "device-position async rope inplace mismatch at {i}: got {got}, expected {want}"
            );
        }

        ctx.device_free(d_in)?;
        ctx.device_free(d_weight)?;
        ctx.device_free(d_rms)?;
        ctx.device_free(d_weighted)?;
        ctx.device_free(d_q)?;
        ctx.device_free(d_k)?;
        ctx.device_free(d_x)?;
        ctx.device_free(d_x_position_dev)?;
        ctx.device_free(d_past_len)?;
    }
    Ok(())
}

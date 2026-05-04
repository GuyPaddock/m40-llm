#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use std::ffi::c_void;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn read_f32s(bytes: Vec<u8>) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[test]
fn residual_add_kernel_matches_cpu() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(c) => c,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("skipping: {e}");
        return Ok(());
    }

    let a: Vec<f32> = (0..513).map(|i| i as f32 * 0.125 - 12.0).collect();
    let b: Vec<f32> = (0..513).map(|i| 3.5 - i as f32 * 0.03125).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let bytes = a.len() * std::mem::size_of::<f32>();

    let d_a = ctx.device_malloc(bytes)?;
    let d_b = ctx.device_malloc(bytes)?;
    let d_out = ctx.device_malloc(bytes)?;
    unsafe {
        ctx.memcpy_h2d(d_a, f32_bytes(&a).as_ptr() as *const c_void, bytes)?;
        ctx.memcpy_h2d(d_b, f32_bytes(&b).as_ptr() as *const c_void, bytes)?;
        ctx.residual_add_f32(d_a, d_b, d_out, a.len())?;

        let mut out_bytes = vec![0u8; bytes];
        ctx.memcpy_d2h(out_bytes.as_mut_ptr() as *mut c_void, d_out, bytes)?;
        let actual = read_f32s(out_bytes);
        for (i, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "mismatch at {i}: got {got}, expected {want}"
            );
        }

        ctx.device_free(d_a)?;
        ctx.device_free(d_b)?;
        ctx.device_free(d_out)?;
    }
    Ok(())
}

#[test]
fn swiglu_kernel_matches_cpu() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(c) => c,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("skipping: {e}");
        return Ok(());
    }

    let gate: Vec<f32> = (0..257).map(|i| i as f32 * 0.05 - 5.0).collect();
    let up: Vec<f32> = (0..257).map(|i| 1.25 - i as f32 * 0.015).collect();
    let expected: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| {
            let silu = *g / (1.0 + (-*g).exp());
            silu * u
        })
        .collect();
    let bytes = gate.len() * std::mem::size_of::<f32>();

    let d_gate = ctx.device_malloc(bytes)?;
    let d_up = ctx.device_malloc(bytes)?;
    let d_out = ctx.device_malloc(bytes)?;
    unsafe {
        ctx.memcpy_h2d(d_gate, f32_bytes(&gate).as_ptr() as *const c_void, bytes)?;
        ctx.memcpy_h2d(d_up, f32_bytes(&up).as_ptr() as *const c_void, bytes)?;
        ctx.swiglu_f32(d_gate, d_up, d_out, gate.len())?;

        let mut out_bytes = vec![0u8; bytes];
        ctx.memcpy_d2h(out_bytes.as_mut_ptr() as *mut c_void, d_out, bytes)?;
        let actual = read_f32s(out_bytes);
        for (i, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "mismatch at {i}: got {got}, expected {want}"
            );
        }

        ctx.device_free(d_gate)?;
        ctx.device_free(d_up)?;
        ctx.device_free(d_out)?;
    }
    Ok(())
}

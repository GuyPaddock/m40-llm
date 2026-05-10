#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::{bail, Result};
use m40_llm::cuda::PersistentDecodeStatus;
use std::ffi::c_void;
use std::time::{Duration, Instant};

fn f32s_to_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|ch| f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]))
        .collect()
}

#[test]
fn persistent_decode_vec_lifecycle() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{}", e);
        return Ok(());
    }

    let input: Vec<f32> = (0..1024).map(|i| i as f32 * 0.01 - 2.0).collect();
    let bytes = input.len() * std::mem::size_of::<f32>();
    let d_in = ctx.device_malloc(bytes)?;
    let d_out = ctx.device_malloc(bytes)?;

    unsafe {
        ctx.memcpy_h2d(d_in, f32s_to_bytes(&input).as_ptr() as *const c_void, bytes)?;
    }

    ctx.start_persistent_decode()?;
    let command_id = unsafe {
        ctx.persistent_decode_submit_vec(
            d_in as *const c_void,
            d_out,
            input.len() as u32,
            1.001,
            0.125,
            8,
        )?
    };

    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let poll = ctx.persistent_decode_poll()?;
        if poll.status == PersistentDecodeStatus::Done && poll.command_id == command_id {
            break;
        }
        if Instant::now() > deadline {
            bail!("persistent decode command did not complete: {poll:?}");
        }
        std::thread::yield_now();
    }

    let mut out_bytes = vec![0u8; bytes];
    unsafe {
        ctx.memcpy_d2h(out_bytes.as_mut_ptr() as *mut c_void, d_out, bytes)?;
    }
    let got = bytes_to_f32s(&out_bytes);
    for (i, (&input, &got)) in input.iter().zip(got.iter()).enumerate() {
        let mut expected = input;
        for _ in 0..8 {
            expected = expected * 1.001 + 0.125;
        }
        let diff = (got - expected).abs();
        assert!(
            diff <= 1e-5,
            "mismatch at {i}: got {got}, expected {expected}, diff {diff}"
        );
    }

    ctx.stop_persistent_decode()?;
    ctx.stop_persistent_decode()?;
    unsafe {
        ctx.device_free(d_in)?;
        ctx.device_free(d_out)?;
    }
    Ok(())
}

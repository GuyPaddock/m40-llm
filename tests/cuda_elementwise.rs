#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use m40_llm::cuda::CudaStream;
use std::ffi::c_void;

fn f32s_to_halves_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(values) / 2);
    for value in values {
        out.extend_from_slice(&half::f16::from_f32(*value).to_bits().to_le_bytes());
    }
    out
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn cpu_gguf_gemm_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[row * k + kk] * half::f16::from_f32(b[col * k + kk]).to_f32();
            }
            c[row * n + col] = acc;
        }
    }
    c
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

#[test]
fn async_elementwise_wrappers_match_cpu() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(c) => c,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("skipping: {e}");
        return Ok(());
    }

    let a: Vec<f32> = (0..129).map(|i| i as f32 * 0.25 - 2.0).collect();
    let b: Vec<f32> = (0..129).map(|i| 1.5 - i as f32 * 0.125).collect();
    let expected_add: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let expected_swiglu: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(g, u)| {
            let silu = *g / (1.0 + (-*g).exp());
            silu * u
        })
        .collect();
    let bytes = a.len() * std::mem::size_of::<f32>();

    let d_a = ctx.device_malloc(bytes)?;
    let d_b = ctx.device_malloc(bytes)?;
    let d_add = ctx.device_malloc(bytes)?;
    let d_swiglu = ctx.device_malloc(bytes)?;
    unsafe {
        ctx.memcpy_h2d(d_a, f32_bytes(&a).as_ptr() as *const c_void, bytes)?;
        ctx.memcpy_h2d(d_b, f32_bytes(&b).as_ptr() as *const c_void, bytes)?;
        ctx.residual_add_f32_async(d_a, d_b, d_add, a.len())?;
        ctx.swiglu_f32_async(d_a, d_b, d_swiglu, a.len())?;
        ctx.synchronize_stream(CudaStream::Decode)?;

        let mut add_bytes = vec![0u8; bytes];
        let mut swiglu_bytes = vec![0u8; bytes];
        ctx.memcpy_d2h(add_bytes.as_mut_ptr() as *mut c_void, d_add, bytes)?;
        ctx.memcpy_d2h(swiglu_bytes.as_mut_ptr() as *mut c_void, d_swiglu, bytes)?;
        for (i, (got, want)) in read_f32s(add_bytes).iter().zip(&expected_add).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "async residual mismatch at {i}: got {got}, expected {want}"
            );
        }
        for (i, (got, want)) in read_f32s(swiglu_bytes)
            .iter()
            .zip(&expected_swiglu)
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-5,
                "async swiglu mismatch at {i}: got {got}, expected {want}"
            );
        }

        ctx.device_free(d_a)?;
        ctx.device_free(d_b)?;
        ctx.device_free(d_add)?;
        ctx.device_free(d_swiglu)?;
    }
    Ok(())
}

#[test]
fn cuda_graph_replays_decode_elementwise_work() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(c) => c,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("skipping: {e}");
        return Ok(());
    }

    let a: Vec<f32> = (0..257).map(|i| i as f32 * 0.03125 - 3.0).collect();
    let b: Vec<f32> = (0..257).map(|i| 2.0 - i as f32 * 0.015625).collect();
    let expected_add: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let expected_swiglu: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(g, u)| {
            let silu = *g / (1.0 + (-*g).exp());
            silu * u
        })
        .collect();
    let bytes = a.len() * std::mem::size_of::<f32>();

    let d_a = ctx.device_malloc(bytes)?;
    let d_b = ctx.device_malloc(bytes)?;
    let d_add = ctx.device_malloc(bytes)?;
    let d_swiglu = ctx.device_malloc(bytes)?;
    unsafe {
        ctx.memcpy_h2d(d_a, f32_bytes(&a).as_ptr() as *const c_void, bytes)?;
        ctx.memcpy_h2d(d_b, f32_bytes(&b).as_ptr() as *const c_void, bytes)?;

        let graph = ctx.capture_graph(CudaStream::Decode, || {
            ctx.residual_add_f32_async(d_a, d_b, d_add, a.len())?;
            ctx.swiglu_f32_async(d_a, d_b, d_swiglu, a.len())
        })?;
        graph.launch(CudaStream::Decode)?;
        ctx.synchronize_stream(CudaStream::Decode)?;

        let mut add_bytes = vec![0u8; bytes];
        let mut swiglu_bytes = vec![0u8; bytes];
        ctx.memcpy_d2h(add_bytes.as_mut_ptr() as *mut c_void, d_add, bytes)?;
        ctx.memcpy_d2h(swiglu_bytes.as_mut_ptr() as *mut c_void, d_swiglu, bytes)?;
        for (i, (got, want)) in read_f32s(add_bytes).iter().zip(&expected_add).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "graph residual mismatch at {i}: got {got}, expected {want}"
            );
        }
        for (i, (got, want)) in read_f32s(swiglu_bytes)
            .iter()
            .zip(&expected_swiglu)
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-5,
                "graph swiglu mismatch at {i}: got {got}, expected {want}"
            );
        }

        ctx.device_free(d_a)?;
        ctx.device_free(d_b)?;
        ctx.device_free(d_add)?;
        ctx.device_free(d_swiglu)?;
    }

    Ok(())
}

#[test]
fn cuda_graph_replays_one_layer_projection_gemms() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(c) => c,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("skipping: {e}");
        return Ok(());
    }

    let (m, k, n, hidden) = (1i32, 4i32, 4i32, 6i32);
    let x = [-0.75f32, 0.25, 1.5, -2.0];
    let wq: Vec<f32> = (0..(n * k)).map(|i| (i as f32 * 0.03125).sin()).collect();
    let wk: Vec<f32> = (0..(n * k)).map(|i| (i as f32 * 0.046875).cos()).collect();
    let wv: Vec<f32> = (0..(n * k)).map(|i| i as f32 * 0.02 - 0.3).collect();
    let wo: Vec<f32> = (0..(n * k)).map(|i| 0.25 - i as f32 * 0.015).collect();
    let w_gate: Vec<f32> = (0..(hidden * k))
        .map(|i| (i as f32 * 0.021).sin())
        .collect();
    let w_up: Vec<f32> = (0..(hidden * k))
        .map(|i| (i as f32 * 0.017).cos())
        .collect();
    let w_down: Vec<f32> = (0..(n * hidden))
        .map(|i| (i as f32 * 0.019).sin())
        .collect();

    let expected_q = cpu_gguf_gemm_f32(&x, &wq, m as usize, n as usize, k as usize);
    let expected_k = cpu_gguf_gemm_f32(&x, &wk, m as usize, n as usize, k as usize);
    let expected_v = cpu_gguf_gemm_f32(&x, &wv, m as usize, n as usize, k as usize);
    let expected_attn = cpu_gguf_gemm_f32(&expected_q, &wo, m as usize, n as usize, k as usize);
    let expected_gate = cpu_gguf_gemm_f32(&x, &w_gate, m as usize, hidden as usize, k as usize);
    let expected_up = cpu_gguf_gemm_f32(&x, &w_up, m as usize, hidden as usize, k as usize);
    let expected_down = cpu_gguf_gemm_f32(
        &expected_gate,
        &w_down,
        m as usize,
        n as usize,
        hidden as usize,
    );

    unsafe fn upload_materialized_weight(
        ctx: &m40_llm::cuda::CudaContext,
        weight: &[f32],
        n: i32,
        k: i32,
    ) -> Result<(*mut c_void, *mut c_void)> {
        let half = f32s_to_halves_bytes(weight);
        let bytes_half = half.len();
        let bytes_f32 = weight.len() * std::mem::size_of::<f32>();
        let d_half = ctx.device_malloc(bytes_half)?;
        let d_f32 = ctx.device_malloc(bytes_f32)?;
        ctx.memcpy_h2d(d_half, half.as_ptr() as *const c_void, bytes_half)?;
        ctx.materialize_gguf_f16_to_f32_colmajor_nt(d_half as *const c_void, d_f32, n, k)?;
        Ok((d_half, d_f32))
    }

    unsafe {
        let bytes_x = x.len() * std::mem::size_of::<f32>();
        let bytes_n = n as usize * std::mem::size_of::<f32>();
        let bytes_hidden = hidden as usize * std::mem::size_of::<f32>();
        let d_x = ctx.device_malloc(bytes_x)?;
        let d_q = ctx.device_malloc(bytes_n)?;
        let d_k = ctx.device_malloc(bytes_n)?;
        let d_v = ctx.device_malloc(bytes_n)?;
        let d_attn = ctx.device_malloc(bytes_n)?;
        let d_gate = ctx.device_malloc(bytes_hidden)?;
        let d_up = ctx.device_malloc(bytes_hidden)?;
        let d_down = ctx.device_malloc(bytes_n)?;
        ctx.memcpy_h2d(d_x, f32_bytes(&x).as_ptr() as *const c_void, bytes_x)?;

        let (d_wq_half, d_wq) = upload_materialized_weight(&ctx, &wq, n, k)?;
        let (d_wk_half, d_wk) = upload_materialized_weight(&ctx, &wk, n, k)?;
        let (d_wv_half, d_wv) = upload_materialized_weight(&ctx, &wv, n, k)?;
        let (d_wo_half, d_wo) = upload_materialized_weight(&ctx, &wo, n, k)?;
        let (d_w_gate_half, d_w_gate) = upload_materialized_weight(&ctx, &w_gate, hidden, k)?;
        let (d_w_up_half, d_w_up) = upload_materialized_weight(&ctx, &w_up, hidden, k)?;
        let (d_w_down_half, d_w_down) = upload_materialized_weight(&ctx, &w_down, n, hidden)?;

        ctx.gemm_f32xf32_f32_async(d_x as *const c_void, d_wq as *const c_void, d_q, m, n, k)?;
        ctx.synchronize_stream(CudaStream::Prefill)?;

        let graph = ctx.capture_graph(CudaStream::Prefill, || {
            ctx.gemm_f32xf32_f32_async(d_x as *const c_void, d_wq as *const c_void, d_q, m, n, k)?;
            ctx.gemm_f32xf32_f32_async(d_x as *const c_void, d_wk as *const c_void, d_k, m, n, k)?;
            ctx.gemm_f32xf32_f32_async(d_x as *const c_void, d_wv as *const c_void, d_v, m, n, k)?;
            ctx.gemm_f32xf32_f32_async(
                d_q as *const c_void,
                d_wo as *const c_void,
                d_attn,
                m,
                n,
                k,
            )?;
            ctx.gemm_f32xf32_f32_async(
                d_x as *const c_void,
                d_w_gate as *const c_void,
                d_gate,
                m,
                hidden,
                k,
            )?;
            ctx.gemm_f32xf32_f32_async(
                d_x as *const c_void,
                d_w_up as *const c_void,
                d_up,
                m,
                hidden,
                k,
            )?;
            ctx.gemm_f32xf32_f32_async(
                d_gate as *const c_void,
                d_w_down as *const c_void,
                d_down,
                m,
                n,
                hidden,
            )
        })?;
        graph.launch(CudaStream::Prefill)?;
        ctx.synchronize_stream(CudaStream::Prefill)?;

        let read = |ptr: *const c_void, bytes: usize| -> Result<Vec<f32>> {
            let mut out = vec![0u8; bytes];
            ctx.memcpy_d2h(out.as_mut_ptr() as *mut c_void, ptr, bytes)?;
            Ok(read_f32s(out))
        };
        let actual_q = read(d_q, bytes_n)?;
        let actual_k = read(d_k, bytes_n)?;
        let actual_v = read(d_v, bytes_n)?;
        let actual_attn = read(d_attn, bytes_n)?;
        let actual_gate = read(d_gate, bytes_hidden)?;
        let actual_up = read(d_up, bytes_hidden)?;
        let actual_down = read(d_down, bytes_n)?;

        for (name, actual, expected) in [
            ("q", actual_q, expected_q),
            ("k", actual_k, expected_k),
            ("v", actual_v, expected_v),
            ("attn", actual_attn, expected_attn),
            ("gate", actual_gate, expected_gate),
            ("up", actual_up, expected_up),
            ("down", actual_down, expected_down),
        ] {
            for (i, (got, want)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (got - want).abs() < 2e-3,
                    "graph {name} GEMM mismatch at {i}: got {got}, expected {want}"
                );
            }
        }

        for ptr in [
            d_x,
            d_q,
            d_k,
            d_v,
            d_attn,
            d_gate,
            d_up,
            d_down,
            d_wq_half,
            d_wq,
            d_wk_half,
            d_wk,
            d_wv_half,
            d_wv,
            d_wo_half,
            d_wo,
            d_w_gate_half,
            d_w_gate,
            d_w_up_half,
            d_w_up,
            d_w_down_half,
            d_w_down,
        ] {
            ctx.device_free(ptr)?;
        }
    }

    Ok(())
}

#[test]
fn stream_wait_allows_prefill_gemm_to_consume_decode_swiglu() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(c) => c,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("skipping: {e}");
        return Ok(());
    }

    let (m, k, n) = (1i32, 4i32, 3i32);
    let gate = [-1.25f32, -0.5, 0.75, 2.0];
    let up = [0.25f32, -1.5, 0.5, 1.25];
    let b = [
        0.5f32, -0.25, 1.5, -1.0, 0.75, 0.125, 1.25, -0.5, -0.75, 0.375, 2.0, -1.25,
    ];
    let swiglu_expected: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| {
            let silu = *g / (1.0 + (-*g).exp());
            silu * u
        })
        .collect();
    let expected = cpu_gguf_gemm_f32(&swiglu_expected, &b, m as usize, n as usize, k as usize);

    let gate_bytes = f32_bytes(&gate);
    let up_bytes = f32_bytes(&up);
    let b_half = f32s_to_halves_bytes(&b);
    let bytes_vec = gate.len() * std::mem::size_of::<f32>();
    let bytes_b_half = b_half.len();
    let bytes_b_f32 = b.len() * std::mem::size_of::<f32>();
    let bytes_c = expected.len() * std::mem::size_of::<f32>();

    let d_gate = ctx.device_malloc(bytes_vec)?;
    let d_up = ctx.device_malloc(bytes_vec)?;
    let d_swiglu = ctx.device_malloc(bytes_vec)?;
    let d_b_half = ctx.device_malloc(bytes_b_half)?;
    let d_b_f32 = ctx.device_malloc(bytes_b_f32)?;
    let d_c = ctx.device_malloc(bytes_c)?;
    unsafe {
        ctx.memcpy_h2d(d_gate, gate_bytes.as_ptr() as *const c_void, bytes_vec)?;
        ctx.memcpy_h2d(d_up, up_bytes.as_ptr() as *const c_void, bytes_vec)?;
        ctx.memcpy_h2d(d_b_half, b_half.as_ptr() as *const c_void, bytes_b_half)?;
        ctx.materialize_gguf_f16_to_f32_colmajor_nt(d_b_half as *const c_void, d_b_f32, n, k)?;

        ctx.swiglu_f32_async(d_gate, d_up, d_swiglu, gate.len())?;
        ctx.stream_wait_for_stream(
            CudaStream::Prefill,
            CudaStream::Decode,
            "test_swiglu_to_gemm",
        )?;
        ctx.gemm_f32xf32_f32(
            d_swiglu as *const c_void,
            d_b_f32 as *const c_void,
            d_c,
            m,
            n,
            k,
        )?;

        let mut c_bytes = vec![0u8; bytes_c];
        ctx.memcpy_d2h(c_bytes.as_mut_ptr() as *mut c_void, d_c, bytes_c)?;
        for (i, (got, want)) in read_f32s(c_bytes).iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "stream wait GEMM mismatch at {i}: got {got}, expected {want}"
            );
        }

        ctx.device_free(d_gate)?;
        ctx.device_free(d_up)?;
        ctx.device_free(d_swiglu)?;
        ctx.device_free(d_b_half)?;
        ctx.device_free(d_b_f32)?;
        ctx.device_free(d_c)?;
    }

    Ok(())
}

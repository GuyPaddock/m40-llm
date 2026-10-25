#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/env_restore.rs"]
mod env_restore;
#[path = "common/q8_model.rs"]
mod fixture;

use anyhow::{ensure, Result};
use env_restore::EnvRestore;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    infer::{
        execution_weights::{Q8ExecutionConfig, Q8ExecutionMode},
        LoadedModel,
    },
    profile,
};
use std::ffi::c_void;

const GATE: &str = "layers.0.feed_forward.w3.weight";
const UP: &str = "layers.0.feed_forward.w1.weight";

fn make_model(mode: Q8ExecutionMode, hidden: usize) -> Result<LoadedModel> {
    let (gguf, bytes) = if hidden == 128 {
        fixture::build(0)
    } else {
        fixture::build_projection_fixture(0, [64; 3], hidden)
    };
    LoadedModel::from_gguf_with_q8_execution(
        gguf,
        bytes,
        -1,
        Q8ExecutionConfig {
            mode,
            budget_bytes: 1024 * 1024,
        },
    )
}

unsafe fn enqueue(
    model: &LoadedModel,
    input: *const c_void,
    output: *mut c_void,
    rows: i32,
    hidden: i32,
    k: i32,
) -> Result<bool> {
    model.try_mlp_gate_up_swiglu_f32xf16_gguf_decode_async(
        input,
        rows,
        k,
        model.device_tensors[GATE].dptr,
        model.device_tensors[UP].dptr,
        hidden,
        output,
    )
}

fn run_mlp(
    model: &LoadedModel,
    hidden: usize,
    amplitude: f32,
    stream: CudaStream,
) -> Result<Vec<f32>> {
    let input: Vec<f32> = fixture::projection::make_input(1, 64)
        .into_iter()
        .map(|v| v * amplitude)
        .collect();
    let d_input = DeviceBuffer::new_tagged(&model.cuda, input.len() * 4, "mlp:input")?;
    let d_output = DeviceBuffer::new_tagged(&model.cuda, (hidden + 4) * 4, "mlp:output")?;
    let mut actual = vec![12345.0_f32; hidden + 4];
    unsafe {
        model
            .cuda
            .memcpy_h2d(d_input.as_mut_ptr(), input.as_ptr().cast(), d_input.bytes())?;
        model.cuda.memcpy_h2d(
            d_output.as_mut_ptr(),
            actual.as_ptr().cast(),
            d_output.bytes(),
        )?;
    }
    let before = profile::snapshot();
    ensure!(
        unsafe {
            enqueue(
                model,
                d_input.as_ptr(),
                d_output.as_mut_ptr(),
                1,
                hidden as i32,
                64,
            )?
        },
        "fused MLP should dispatch"
    );
    let after = profile::snapshot();
    let delta = profile::counter_delta(&before, &after);
    assert_eq!(delta.launches, 1);
    assert_eq!(delta.stream_syncs, 0);
    assert_eq!(delta.device_allocs, 0);
    assert_eq!(delta.device_frees, 0);
    assert_eq!(delta.h2d_copies, 0);
    assert_eq!(delta.d2h_copies, 0);
    let op = if model.q8_execution_stats().mode == Q8ExecutionMode::Off {
        "mlp_gate_up_swiglu_f32xq8_0_gguf_decode"
    } else {
        "mlp_gate_up_swiglu_f32xq8_separate"
    };
    assert_eq!(
        after.by_op.get(op).map_or(0, |v| v.launches)
            - before.by_op.get(op).map_or(0, |v| v.launches),
        1
    );
    model.cuda.synchronize_stream(stream)?;
    unsafe {
        model.cuda.memcpy_d2h(
            actual.as_mut_ptr().cast(),
            d_output.as_ptr(),
            d_output.bytes(),
        )?;
    }
    assert_eq!(&actual[hidden..], &[12345.0; 4]);
    actual.truncate(hidden);
    let source = |name| {
        let tensor = &model.device_tensors[name];
        &model.host_weights
            [tensor.byte_offset as usize..tensor.byte_offset as usize + tensor.nbytes]
    };
    let (gate_bytes, up_bytes) = (source(GATE), source(UP));
    assert_ne!(
        gate_bytes, up_bytes,
        "independent weights must detect swapped/aliased operands"
    );
    for (col, &value) in actual.iter().enumerate() {
        let gate = fixture::projection::project_cpu(&input, gate_bytes, 0, col, 64);
        let up = fixture::projection::project_cpu(&input, up_bytes, 0, col, 64);
        let expected = gate * (1.0 / (1.0 + (-gate).exp())) * up;
        ensure!(
            value.is_finite() && (f64::from(value) - expected).abs() < 1e-3 + expected.abs() * 1e-5,
            "SwiGLU CPU mismatch: hidden={hidden} column={col}"
        );
    }
    Ok(actual)
}

/// Split-plane gate/up/SwiGLU stays one allocation-free async launch with
/// independent operands, output guards, narrow/tail widths, and both streams.
/// CPU/native results match; invalid dimensions and non-decode calls do not launch.
#[test]
fn test_prepared_fused_mlp_matches_native_and_cpu() -> Result<()> {
    let _fused = EnvRestore::set("M40LLM_FUSED_Q8_MLP_SWIGLU", "1");
    let initial = CudaContext::total_device_bytes();
    for hidden in [1, 31, 65, 128] {
        let native = make_model(Q8ExecutionMode::Off, hidden)?;
        let props = native.cuda.current_device_props()?;
        ensure!(
            props.major == 5 && props.minor == 2,
            "MLP test requires sm_52"
        );
        for mode in [Q8ExecutionMode::Split, Q8ExecutionMode::Mixed] {
            let model = make_model(mode, hidden)?;
            for stream in [CudaStream::Prefill, CudaStream::Decode] {
                let _stream = EnvRestore::set(
                    "M40LLM_Q8_DECODE_STREAM",
                    if stream == CudaStream::Decode {
                        "decode"
                    } else {
                        "prefill"
                    },
                );
                for amplitude in [1.0, -3.0, 8.0] {
                    let expected = run_mlp(&native, hidden, amplitude, stream)?;
                    let actual = run_mlp(&model, hidden, amplitude, stream)?;
                    let (maximum, mean) = fixture::projection::measure_error(&actual, &expected);
                    ensure!(maximum < 1e-3 && mean < 1e-4, "native SwiGLU mismatch");
                }
            }
            let input = DeviceBuffer::new_tagged(&model.cuda, 64 * 4, "mlp:invalid-input")?;
            let output =
                DeviceBuffer::new_tagged(&model.cuda, (hidden + 4) * 4, "mlp:invalid-output")?;
            let before = profile::snapshot();
            assert!(unsafe {
                enqueue(
                    &model,
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    1,
                    hidden as i32,
                    32,
                )
            }
            .is_err());
            assert!(unsafe {
                enqueue(
                    &model,
                    input.as_ptr(),
                    std::ptr::null_mut(),
                    1,
                    hidden as i32,
                    64,
                )
            }
            .is_err());
            assert!(!unsafe {
                enqueue(
                    &model,
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    2,
                    hidden as i32,
                    64,
                )?
            });
            {
                let _disabled = EnvRestore::set("M40LLM_FUSED_Q8_MLP_SWIGLU", "0");
                assert!(!unsafe {
                    enqueue(
                        &model,
                        input.as_ptr(),
                        output.as_mut_ptr(),
                        1,
                        hidden as i32,
                        64,
                    )?
                });
            }
            assert_eq!(
                profile::counter_delta(&before, &profile::snapshot()).launches,
                0
            );
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

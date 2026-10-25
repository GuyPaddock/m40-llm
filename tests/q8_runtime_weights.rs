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

fn make_model(mode: Q8ExecutionMode, seed: u8) -> Result<LoadedModel> {
    let (gguf, bytes) = fixture::build(seed);
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

fn project(
    model: &LoadedModel,
    name: &str,
    rows: usize,
    stream: CudaStream,
    is_synchronous: bool,
) -> Result<Vec<f32>> {
    let tensor = model.device_tensors.get(name).unwrap();
    let (k, n) = (tensor.shape[0] as usize, tensor.shape[1] as usize);
    let input = fixture::projection::make_input(rows, k);
    let d_input = DeviceBuffer::new_tagged(&model.cuda, input.len() * 4, "runtime_q8:input")?;
    let d_output = DeviceBuffer::new_tagged(&model.cuda, rows * n * 4, "runtime_q8:output")?;
    unsafe {
        model
            .cuda
            .memcpy_h2d(d_input.as_mut_ptr(), input.as_ptr().cast(), d_input.bytes())?;
    }
    let before = profile::snapshot();
    unsafe {
        if is_synchronous {
            model.matmul_f32xf16_gguf_f32(
                d_input.as_ptr(),
                tensor.dptr,
                d_output.as_mut_ptr(),
                rows as i32,
                n as i32,
                k as i32,
            )?;
        } else {
            model.matmul_f32xf16_gguf_f32_async(
                d_input.as_ptr(),
                tensor.dptr,
                d_output.as_mut_ptr(),
                rows as i32,
                n as i32,
                k as i32,
            )?;
        }
    }
    let delta = profile::counter_delta(&before, &profile::snapshot());
    assert_eq!(delta.stream_syncs, u64::from(is_synchronous));
    assert_eq!(delta.device_allocs, 0);
    assert_eq!(delta.device_frees, 0);
    assert_eq!(delta.h2d_copies, 0);
    if !is_synchronous {
        model.cuda.synchronize_stream(stream)?;
    }
    let mut output = vec![0.0_f32; rows * n];
    unsafe {
        model.cuda.memcpy_d2h(
            output.as_mut_ptr().cast(),
            d_output.as_ptr(),
            d_output.bytes(),
        )?;
    }
    let bytes = &model.host_weights
        [tensor.byte_offset as usize..tensor.byte_offset as usize + tensor.nbytes];
    for (index, &value) in output.iter().enumerate() {
        let expected = fixture::projection::project_cpu(&input, bytes, index / n, index % n, k);
        ensure!(
            value.is_finite() && (f64::from(value) - expected).abs() < 1e-3 + expected.abs() * 1e-5,
            "runtime projection mismatch: {name} rows={rows}"
        );
    }
    Ok(output)
}

/// Preparation must be model-owned and complete before any forward work.
/// Split/mixed dispatch preserves native math, uses no per-call allocations,
/// retains explicit sync/async producer-stream behavior, rejects failed setup
/// without leaks, and never aliases another model's cache.
#[test]
fn test_runtime_cache_prepares_dispatches_and_releases_per_model() -> Result<()> {
    let _decode = EnvRestore::set("M40LLM_Q8_DECODE_STREAM", "decode");
    let _tiled = EnvRestore::set("M40LLM_Q8_DECODE_TILED_COLS", "1");
    let initial = CudaContext::total_device_bytes();
    {
        let native = make_model(Q8ExecutionMode::Off, 0)?;
        let props = native.cuda.current_device_props()?;
        ensure!(
            props.major == 5 && props.minor == 2,
            "runtime Q8 tests require M40"
        );
        assert!(native.q8_execution_stats().tensors.is_empty());
        assert!(!native.q8_execution_stats().has_prepared_vocabulary);
        for mode in [Q8ExecutionMode::Split, Q8ExecutionMode::Mixed] {
            let model = make_model(mode, 0)?;
            let stats = model.q8_execution_stats();
            assert!(stats.has_prepared_vocabulary);
            assert_eq!(
                stats.tensors.len(),
                if mode == Q8ExecutionMode::Mixed {
                    11
                } else {
                    8
                }
            );
            assert_eq!(stats.resident_bytes, stats.estimated_bytes);
            assert_eq!(
                stats.resident_bytes,
                stats
                    .tensors
                    .iter()
                    .map(|t| t.resident_bytes)
                    .sum::<usize>()
            );
            assert_eq!(
                stats.total_weight_payload_bytes,
                stats.source_weights_bytes + stats.resident_bytes
            );
            assert_eq!(model.host_weights, native.host_weights);
            for name in native.device_tensors.keys() {
                for rows in [1, 4, 64] {
                    let stream = if rows == 1 {
                        CudaStream::Decode
                    } else {
                        CudaStream::Prefill
                    };
                    let expected = project(&native, name, rows, stream, false)?;
                    let before = profile::snapshot();
                    let actual = project(&model, name, rows, stream, false)?;
                    let after = profile::snapshot();
                    let op = if mode == Q8ExecutionMode::Mixed
                        && rows == 64
                        && (name.contains("wq.") || name.contains("wo.") || name.contains("w2."))
                    {
                        "gemm_f32xq8_tiled_f32"
                    } else {
                        "gemm_f32xq8_separate_f32"
                    };
                    let old = before.by_op.get(op).map_or(0, |c| c.launches);
                    let new = after.by_op.get(op).map_or(0, |c| c.launches);
                    assert_eq!(new - old, 1);
                    let (maximum, mean) = fixture::projection::measure_error(&actual, &expected);
                    ensure!(maximum < 1e-3 && mean < 1e-4, "native vs prepared mismatch");
                }
            }
            for stream in [CudaStream::Prefill, CudaStream::Decode] {
                let _stream = EnvRestore::set(
                    "M40LLM_Q8_DECODE_STREAM",
                    if stream == CudaStream::Decode {
                        "decode"
                    } else {
                        "prefill"
                    },
                );
                let name = "layers.0.attention.wq.weight";
                let expected = project(&native, name, 1, stream, false)?;
                assert_eq!(
                    project(&model, name, 1, stream, false)?,
                    project(&model, name, 1, stream, true)?
                );
                let (maximum, mean) = fixture::projection::measure_error(
                    &project(&model, name, 1, stream, true)?,
                    &expected,
                );
                ensure!(maximum < 1e-3 && mean < 1e-4, "producer-stream mismatch");
            }
            let different = make_model(mode, 1)?;
            assert_ne!(
                different.q8_execution_stats().tensors[0]
                    .key
                    .source
                    .model_sha256(),
                model.q8_execution_stats().tensors[0]
                    .key
                    .source
                    .model_sha256()
            );
            let before = project(
                &different,
                "layers.0.attention.wq.weight",
                1,
                CudaStream::Decode,
                false,
            )?;
            drop(model);
            assert_eq!(
                project(
                    &different,
                    "layers.0.attention.wq.weight",
                    1,
                    CudaStream::Decode,
                    false
                )?,
                before
            );
        }
        let before = CudaContext::total_device_bytes();
        let (gguf, bytes) = fixture::build(0);
        assert!(LoadedModel::from_gguf_with_q8_execution(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig {
                mode: Q8ExecutionMode::Split,
                budget_bytes: 0
            }
        )
        .is_err());
        assert_eq!(CudaContext::total_device_bytes(), before);
        let (gguf, bytes) = fixture::build_f16();
        assert!(LoadedModel::from_gguf_with_q8_execution(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig {
                mode: Q8ExecutionMode::Split,
                budget_bytes: 1024 * 1024
            }
        )
        .is_err());
        assert_eq!(CudaContext::total_device_bytes(), before);
        let _upload = EnvRestore::set("M40LLM_ENABLE_NVCC", "0");
        let (gguf, bytes) = fixture::build(0);
        assert!(LoadedModel::from_gguf_with_q8_execution(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig {
                mode: Q8ExecutionMode::Split,
                budget_bytes: 1024 * 1024
            }
        )
        .is_err());
        assert_eq!(CudaContext::total_device_bytes(), before);
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

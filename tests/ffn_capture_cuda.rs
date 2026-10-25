#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/env_restore.rs"]
mod env_restore;
#[path = "common/q8_model.rs"]
#[allow(dead_code)]
mod fixture;

use anyhow::Result;
use env_restore::EnvRestore;
use m40_llm::{
    cuda::{ffi_debug_read_kv_token, CudaContext, CudaStream, DeviceBuffer},
    decode_session::DecodeSession,
    evaluation::{
        corpus::Corpus,
        ffn_capture::{FfnCaptureArtifact, FfnCapturePlan},
    },
    infer::{
        activation_sparsity::ActivationCalibrationPlan, execution_weights::Q8ExecutionConfig,
        ForwardBatchItem, ForwardPrefillSequence, LoadedModel,
    },
    kv_compression::{KvCompressMode, KvCompressionConfig, ScopedRuntimeConfig},
    profile,
};

fn configure() -> Vec<EnvRestore> {
    [
        ("M40LLM_ENABLE_NVCC", "1"),
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_Q8_DECODE_COL_TILE", "4"),
        ("M40LLM_PROJECTION_BACKEND", "large-model"),
        ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
        ("M40LLM_DECODE_GRAPH", "0"),
        ("M40LLM_FUSED_Q8_QKV", "1"),
        ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
        ("M40LLM_FORWARD_FINITE_LOG", "0"),
        ("M40LLM_PREFILL_CHUNK_SIZE", "0"),
    ]
    .into_iter()
    .map(|(key, value)| EnvRestore::set(key, value))
    .collect()
}

fn corpus() -> Result<Corpus> {
    Corpus::parse(include_str!("../data/research_quality/v1.json"))
}

fn plan() -> Result<FfnCapturePlan> {
    FfnCapturePlan::new(
        "1".repeat(64),
        "2".repeat(64),
        "3".repeat(64),
        &corpus()?,
        2,
        vec![0, 1],
        64,
        128,
        2,
    )
}

fn model() -> Result<LoadedModel> {
    let (gguf, weights) = fixture::build_with_layers(0, 2);
    LoadedModel::from_gguf_with_q8_execution(
        gguf,
        weights,
        -1,
        Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
    )
}

fn session(model: &LoadedModel) -> Result<DecodeSession> {
    DecodeSession::new_for_sequence_with_kv_config(
        model,
        0,
        KvCompressionConfig::dense_reference(),
        64,
        true,
        "ffn-capture-test",
        "ffn-capture:input",
        "ffn-capture:output",
    )
}

fn drain(model: &LoadedModel) -> Result<()> {
    model.cuda.synchronize_stream(CudaStream::Decode)?;
    model.cuda.synchronize_stream(CudaStream::Prefill)
}

fn compare_kv(a: &LoadedModel, b: &LoadedModel, tokens: u32) -> Result<()> {
    for model in [a, b] {
        drain(model)?;
    }
    for layer in 0..2 {
        for position in 0..tokens {
            let mut payloads = Vec::new();
            for model in [a, b] {
                let mut k = vec![0u8; 128];
                let mut v = vec![0u8; 128];
                let kv = model.kv_cache.as_ref().unwrap();
                assert_eq!(
                    unsafe {
                        ffi_debug_read_kv_token(
                            &model.cuda,
                            kv,
                            layer,
                            position,
                            k.as_mut_ptr(),
                            v.as_mut_ptr(),
                        )
                    },
                    0
                );
                payloads.push((k, v));
            }
            assert_eq!(
                payloads[0], payloads[1],
                "layer={layer} position={position}"
            );
        }
    }
    Ok(())
}

fn verify_replay(model: &LoadedModel, artifact: &FfnCaptureArtifact) -> Result<()> {
    let input = DeviceBuffer::new_tagged(&model.cuda, 64 * 4, "ffn-replay:input")?;
    let gate = DeviceBuffer::new_tagged(&model.cuda, 128 * 4, "ffn-replay:gate")?;
    let up = DeviceBuffer::new_tagged(&model.cuda, 128 * 4, "ffn-replay:up")?;
    let hidden = DeviceBuffer::new_tagged(&model.cuda, 128 * 4, "ffn-replay:hidden")?;
    for layer in &artifact.layers {
        let mapped = model.map_standard_layer(layer.source.layer_id)?;
        let w_gate = mapped.w_gate.dptr.cast_const();
        let w_up = mapped.w_up.dptr.cast_const();
        assert!(!w_gate.is_null() && !w_up.is_null());
        for row in 0..4 {
            let x = &layer.normalized_inputs[row * 64..(row + 1) * 64];
            let expected = &layer.swiglu_inputs[row * 128..(row + 1) * 128];
            unsafe {
                model
                    .cuda
                    .memcpy_h2d(input.as_mut_ptr(), x.as_ptr().cast(), input.bytes())?;
                if !model.try_mlp_gate_up_swiglu_f32xf16_gguf_decode_async(
                    input.as_ptr(),
                    1,
                    64,
                    w_gate,
                    w_up,
                    128,
                    hidden.as_mut_ptr(),
                )? {
                    model.mlp_gates_f32xf16_gguf_f32_async(
                        input.as_ptr(),
                        1,
                        64,
                        w_gate,
                        w_up,
                        128,
                        gate.as_mut_ptr(),
                        up.as_mut_ptr(),
                    )?;
                    drain(model)?;
                    model.cuda.swiglu_f32_async(
                        gate.as_ptr(),
                        up.as_ptr(),
                        hidden.as_mut_ptr(),
                        128,
                    )?;
                }
            }
            drain(model)?;
            let mut actual = vec![0.0f32; 128];
            unsafe {
                model.cuda.memcpy_d2h(
                    actual.as_mut_ptr().cast(),
                    hidden.as_ptr(),
                    hidden.bytes(),
                )?;
            }
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            let source_bytes =
                |source: &m40_llm::evaluation::ffn_capture::FfnTensorSource| -> Result<&[u8]> {
                    let start = source.byte_offset as usize;
                    Ok(&model.host_weights[start..start + source.byte_len()?])
                };
            for (column, &actual) in actual.iter().enumerate() {
                let gate = fixture::projection::project_cpu(
                    x,
                    source_bytes(&layer.source.gate)?,
                    0,
                    column,
                    64,
                );
                let up = fixture::projection::project_cpu(
                    x,
                    source_bytes(&layer.source.up)?,
                    0,
                    column,
                    64,
                );
                let expected = gate / (1.0 + (-gate).exp()) * up;
                assert!((f64::from(actual) - expected).abs() <= 2e-4 * (1.0 + expected.abs()));
            }
        }
    }
    Ok(())
}

/// Active capture preserves logits, KV bytes and kernel/allocation counts under
/// the prepared down/vocabulary control. Inactive capture adds no copies/syncs.
/// Paired vectors replay the same SwiGLU and agree with independent CPU math.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_ffn_capture_preserves_prepared_forward_and_replays_paired_inputs() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let _dense = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    for (fused, selected) in [("1", vec![0, 1]), ("0", vec![0, 1]), ("1", vec![1])] {
        let _fused = EnvRestore::set("M40LLM_FUSED_Q8_MLP_SWIGLU", fused);
        let mut a = model()?;
        let mut b = model()?;
        assert!(a.q8_execution_stats().resident_bytes > 0);
        let before = CudaContext::total_device_bytes();
        let mut capture_plan = plan()?;
        capture_plan.selected_layer_ids = selected.clone();
        a.begin_ffn_capture(capture_plan, 1_000_000)?;
        assert_eq!(CudaContext::total_device_bytes(), before);
        for model in [&mut a, &mut b] {
            model.allocate_kv_cache_for_layers(16)?;
        }
        // Warm both paths with capture configured but no eligible task.
        for model in [&a, &b] {
            let mut session = session(model)?;
            session.logits_for_ids(&[2, 3, 4], |_| {})?;
        }
        a.reset_kv_cache()?;
        b.reset_kv_cache()?;
        {
            let mut sa = session(&a)?;
            let mut sb = session(&b)?;
            let before = profile::snapshot();
            let actual = sa.logits_for_ids(&[2, 3, 4], |_| {})?;
            let active = profile::counter_delta(&before, &profile::snapshot());
            let before = profile::snapshot();
            let expected = sb.logits_for_ids(&[2, 3, 4], |_| {})?;
            let reference = profile::counter_delta(&before, &profile::snapshot());
            assert_eq!(
                active, reference,
                "inactive capture must not alter GPU activity"
            );
            assert_eq!(actual, expected);
        }
        for task in &corpus()?.calibration.tasks {
            a.reset_kv_cache()?;
            b.reset_kv_cache()?;
            a.begin_ffn_capture_task(&task.id, &[2, 3, 4])?;
            {
                let mut sa = session(&a)?;
                let mut sb = session(&b)?;
                for prefix in [&[2, 3, 4][..], &[2, 3, 4, 7][..], &[2, 3, 4, 7, 8][..]] {
                    let before = profile::snapshot();
                    let actual = sa.logits_for_ids(prefix, |_| {})?;
                    let observed = profile::counter_delta(&before, &profile::snapshot());
                    let before = profile::snapshot();
                    let expected = sb.logits_for_ids(prefix, |_| {})?;
                    let reference = profile::counter_delta(&before, &profile::snapshot());
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    let mut extra = observed;
                    let is_generated_input = prefix.len() > 3;
                    extra.d2h_copies -= if is_generated_input {
                        2 * selected.len() as u64
                    } else {
                        0
                    };
                    extra.d2h_bytes -= if is_generated_input {
                        selected.len() as u64 * (64 + 128) * 4
                    } else {
                        0
                    };
                    extra.stream_syncs -= if is_generated_input {
                        selected.len() as u64
                    } else {
                        0
                    };
                    assert_eq!(
                        extra, reference,
                        "only the documented read-only copies/drains may differ"
                    );
                    assert_eq!(observed.device_allocs, 0);
                    assert_eq!(observed.device_frees, 0);
                }
            }
            a.finish_ffn_capture_task(&[7, 8, 9])?;
            compare_kv(&a, &b, 5)?;
        }
        let artifact = a.finish_ffn_capture()?;
        assert!(a.ffn_capture.is_none());
        assert_eq!(
            artifact
                .layers
                .iter()
                .map(|layer| layer.source.layer_id)
                .collect::<Vec<_>>(),
            selected
        );
        assert!(artifact.layers.iter().all(|layer| layer.rows.len() == 12));
        verify_replay(&a, &artifact)?;
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Invalid setup publishes no capture/allocation; incomplete export retains
/// state. Unsupported batch, packed, graph, sequence and compression paths must
/// fail before launches or session scratch allocation rather than miss rows.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_ffn_capture_rejects_unsupported_paths_before_device_work() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let _dense = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let mut model = model()?;
        let before = CudaContext::total_device_bytes();
        assert!(model.begin_ffn_capture(plan()?, 1).is_err());
        assert!(model.ffn_capture.is_none());
        assert_eq!(CudaContext::total_device_bytes(), before);
        let mut wrong = plan()?;
        wrong.d_model = 128;
        assert!(model.begin_ffn_capture(wrong, 1_000_000).is_err());
        {
            let _prefill = EnvRestore::set("M40LLM_Q8_DECODE_STREAM", "prefill");
            assert!(model.begin_ffn_capture(plan()?, 1_000_000).is_err());
        }
        model.begin_ffn_capture(plan()?, 1_000_000)?;
        assert!(model.finish_ffn_capture().is_err());
        assert!(model.ffn_capture.is_some());
        assert!(model.set_diagnostic_layer_mask(&[]).is_err());
        let activation_plan = ActivationCalibrationPlan::new(
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            &corpus()?,
            2,
        )?;
        assert!(model
            .begin_activation_calibration(activation_plan, 1_000_000)
            .is_err());
        assert!(model
            .begin_ffn_capture_task(&corpus()?.held_out.tasks[0].id, &[2, 3, 4])
            .is_err());
        model.allocate_kv_cache_for_layer_sequences(16, 2)?;
        let input = DeviceBuffer::new_tagged(&model.cuda, 256, "ffn-reject:input")?;
        let output = DeviceBuffer::new_tagged(&model.cuda, 256, "ffn-reject:output")?;
        let params = DeviceBuffer::new_tagged(&model.cuda, 4, "ffn-reject:params")?;
        let before = profile::snapshot();
        assert!(DecodeSession::new_for_sequence_with_kv_config(
            &model,
            1,
            KvCompressionConfig::dense_reference(),
            64,
            true,
            "reject",
            "reject-in",
            "reject-out"
        )
        .is_err());
        assert!(DecodeSession::new_for_sequence_with_kv_config(
            &model,
            0,
            KvCompressionConfig::for_mode(KvCompressMode::BlockSelectExact),
            64,
            true,
            "reject",
            "reject-in",
            "reject-out"
        )
        .is_err());
        {
            let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "1");
            assert!(session(&model).is_err());
        }
        unsafe {
            assert!(model
                .forward_one_token_all_layers_batched_for_sequences(&[ForwardBatchItem {
                    sequence_id: 0,
                    seq_len: 1,
                    d_x_f32: input.as_mut_ptr(),
                    d_out_f32: output.as_mut_ptr()
                }])
                .is_err());
            assert!(model
                .forward_prefill_all_layers_varlen_for_sequences(&[ForwardPrefillSequence {
                    sequence_id: 0,
                    token_ids: &[2, 3],
                    d_out_f32: output.as_mut_ptr()
                }])
                .is_err());
            assert!(model
                .forward_one_token_all_layers_for_sequence_graph_params(
                    input.as_ptr(),
                    0,
                    params.as_ptr().cast(),
                    params.as_ptr().cast(),
                    output.as_mut_ptr()
                )
                .is_err());
            assert!(model
                .forward_one_token_with_layer_graph_params(
                    input.as_ptr(),
                    0,
                    0,
                    params.as_ptr().cast(),
                    params.as_ptr().cast(),
                    output.as_mut_ptr()
                )
                .is_err());
            assert!(model
                .forward_one_token_all_layers_for_sequence(
                    input.as_ptr(),
                    1,
                    1,
                    output.as_mut_ptr()
                )
                .is_err());
            assert!(model
                .forward_one_token_with_layer(input.as_ptr(), 0, 1, 1, output.as_mut_ptr())
                .is_err());
        }
        let delta = profile::counter_delta(&before, &profile::snapshot());
        assert_eq!(delta.launches, 0);
        assert_eq!(delta.device_allocs, 0);
        assert_eq!(delta.h2d_copies, 0);
        assert!(model.begin_ffn_capture(plan()?, 1_000_000).is_err());
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

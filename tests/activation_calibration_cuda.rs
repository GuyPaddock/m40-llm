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
    cuda::CudaContext,
    decode_session::DecodeSession,
    evaluation::{corpus::Corpus, teacher_forced::compare_loaded_models},
    infer::{
        activation_sparsity::{ActivationCalibration, ActivationCalibrationPlan},
        execution_weights::Q8ExecutionConfig,
        LoadedModel,
    },
    kv_compression::{KvCompressionConfig, ScopedRuntimeConfig},
    profile,
};

fn model() -> Result<LoadedModel> {
    model_with_layers(2)
}

fn model_with_layers(layer_count: u32) -> Result<LoadedModel> {
    let (mut gguf, mut bytes) = fixture::build(0);
    let original = bytes.clone();
    let tensors = gguf.tensors.clone();
    for layer in 1..layer_count {
        for tensor in &tensors {
            if !tensor.name.starts_with("layers.0.") {
                continue;
            }
            let mut copied = tensor.clone();
            copied.name = copied
                .name
                .replacen("layers.0.", &format!("layers.{layer}."), 1);
            bytes.resize(bytes.len().div_ceil(32) * 32, 0);
            copied.offset = bytes.len() as u64;
            let size = tensor.shape.iter().product::<u64>() as usize / 32 * 34;
            let mut payload =
                original[tensor.offset as usize..tensor.offset as usize + size].to_vec();
            for block in payload.chunks_exact_mut(34) {
                for value in &mut block[2..] {
                    *value = value.wrapping_add((layer as u8).wrapping_mul(7));
                }
            }
            bytes.extend_from_slice(&payload);
            gguf.tensors.push(copied);
        }
    }
    gguf.metadata.insert(
        "llama.block_count".to_owned(),
        m40_llm::gguf::GgufValue::Scalar(m40_llm::gguf::GgufScalar::U32(layer_count)),
    );
    LoadedModel::from_gguf_with_q8_execution(gguf, bytes, -1, Q8ExecutionConfig::disabled())
}

fn configure() -> Vec<EnvRestore> {
    [
        ("M40LLM_ENABLE_NVCC", "1"),
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_PROJECTION_BACKEND", "large-model"),
        ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
        ("M40LLM_DECODE_GRAPH", "0"),
        ("M40LLM_FUSED_Q8_QKV", "1"),
        ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
    ]
    .into_iter()
    .map(|(key, value)| EnvRestore::set(key, value))
    .collect()
}

fn session(model: &LoadedModel) -> Result<DecodeSession> {
    DecodeSession::new_for_sequence_with_kv_config(
        model,
        0,
        KvCompressionConfig::dense_reference(),
        64,
        true,
        "activation-test",
        "activation-test:input",
        "activation-test:output",
    )
}

fn capture() -> Result<ActivationCalibration> {
    capture_layers(2)
}

fn capture_layers(layer_count: u32) -> Result<ActivationCalibration> {
    let corpus = Corpus::parse(include_str!("../data/research_quality/v1.json"))?;
    let plan =
        ActivationCalibrationPlan::new("1".repeat(64), "2".repeat(64), "3".repeat(64), &corpus, 2)?;
    let mut observed = model_with_layers(layer_count)?;
    let mut reference = model_with_layers(layer_count)?;
    observed.begin_activation_calibration(plan, 1_000_000)?;
    assert!(observed.set_diagnostic_layer_mask(&[]).is_err());
    observed.allocate_kv_cache_for_layers(16)?;
    reference.allocate_kv_cache_for_layers(16)?;
    for task in &corpus.calibration.tasks {
        observed.reset_kv_cache()?;
        reference.reset_kv_cache()?;
        observed.begin_activation_calibration_task(&task.id, 3)?;
        let mut a = session(&observed)?;
        let mut b = session(&reference)?;
        // Controlled generated-prefix simulation isolates boundary accounting;
        // real-model calibration uses actual greedy generation in the runner.
        for prefix in [&[2, 3, 4][..], &[2, 3, 4, 7][..], &[2, 3, 4, 7, 8][..]] {
            let actual = a.logits_for_ids(prefix, |_| {})?;
            let expected = b.logits_for_ids(prefix, |_| {})?;
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        drop(a);
        drop(b);
        observed.finish_activation_calibration_task(3)?;
    }
    let artifact = observed.finish_activation_calibration()?;
    assert!(observed.activation_research.is_none());
    assert_eq!(artifact.layers.len(), layer_count as usize);
    assert!(artifact
        .layers
        .iter()
        .all(|layer| layer.observed_rows == 12));
    assert!(artifact
        .tasks
        .iter()
        .all(|task| task.captured_absolute_positions == [3, 4]));
    Ok(artifact)
}

/// Capture observes completed SwiGLU only on generated input positions and
/// cannot change native model logits. Zero-threshold application preserves
/// full-model logits; nonzero thresholds produce measured omission with reusable
/// scratch and independently reset counters, not per-token readbacks.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_capture_and_applied_thresholds_preserve_model_contracts() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let _dense = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    let artifact = capture()?;
    for target in [0.0, 0.5] {
        let mut reference = model()?;
        let mut candidate = model()?;
        candidate.apply_activation_calibration(&artifact, target, 1_000_000)?;
        let prepared = candidate.activation_research_stats()?;
        if target == 0.0 {
            assert_eq!(prepared.extra_weight_bytes, 0);
            assert_eq!(prepared.selection_workspace_bytes, 0);
            assert_eq!(prepared.counter_bytes, 0);
            assert_eq!(prepared.native_bypass_layer_ids, [0, 1]);
        } else {
            assert!(prepared.extra_weight_bytes > 0);
            assert_eq!(prepared.selection_workspace_bytes, 128 * 8 + 4);
            assert_eq!(prepared.counter_bytes, 32);
        }
        assert!(candidate.finish_activation_calibration().is_err());
        assert!(candidate.activation_research.is_some());
        reference.allocate_kv_cache_for_layers(16)?;
        candidate.allocate_kv_cache_for_layers(16)?;
        candidate.reset_activation_statistics()?;
        let before = profile::snapshot();
        let report = compare_loaded_models(&mut reference, &mut candidate, &[2, 3, 4, 7, 8, 9])?;
        let delta = profile::counter_delta(&before, &profile::snapshot());
        let stats = candidate.activation_research_stats()?;
        if target != 0.0 {
            assert_eq!(stats.layers[0].input_values, 5 * 128);
            assert_eq!(stats.layers[1].input_values, 5 * 128);
        }
        assert!(stats.layers[0].retained_values <= stats.layers[0].input_values);
        assert!(report.candidate.nll_sum.is_finite());
        if target == 0.0 {
            assert_eq!(report.max_abs_logit_diff, 0.0);
        } else {
            assert!(stats.layers[0].achieved_sparsity.unwrap() > 0.0);
        }
        assert!(delta.launches > 0);
        let input = m40_llm::cuda::DeviceBuffer::new_tagged(
            &candidate.cuda,
            64 * 4,
            "activation:test-input",
        )?;
        let output = m40_llm::cuda::DeviceBuffer::new_tagged(
            &candidate.cuda,
            64 * 4,
            "activation:test-output",
        )?;
        let before = profile::snapshot();
        let row = m40_llm::infer::ForwardBatchItem {
            sequence_id: 0,
            seq_len: 1,
            d_x_f32: input.as_mut_ptr(),
            d_out_f32: output.as_mut_ptr(),
        };
        assert!(
            unsafe { candidate.forward_one_token_all_layers_batched_for_sequences(&[row]) }
                .is_err()
        );
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()).launches,
            0
        );
        let prefill = m40_llm::infer::ForwardPrefillSequence {
            token_ids: &[2, 3],
            sequence_id: 0,
            d_out_f32: output.as_mut_ptr(),
        };
        assert!(
            unsafe { candidate.forward_prefill_all_layers_varlen_for_sequences(&[prefill]) }
                .is_err()
        );
        let parameter =
            m40_llm::cuda::DeviceBuffer::new_tagged(&candidate.cuda, 4, "activation:test-param")?;
        let before = profile::snapshot();
        assert!(unsafe {
            candidate.forward_one_token_with_layer_graph_params(
                input.as_ptr(),
                0,
                0,
                parameter.as_ptr().cast(),
                parameter.as_ptr().cast(),
                output.as_mut_ptr(),
            )
        }
        .is_err());
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()).launches,
            0
        );
        candidate.reset_activation_statistics()?;
        assert_eq!(
            candidate.activation_research_stats()?.layers[0].input_values,
            0
        );
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Stale source identities, insufficient preparation budgets and reconfiguration
/// after KV allocation fail without publishing a partial model cache or leaking
/// buffers. Compressed sessions cannot silently enter the dense-only experiment.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_activation_setup_rejects_mismatched_artifacts_and_unsupported_sessions() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let _dense = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    let _hardware = CudaContext::new_m40()?;
    let artifact = capture()?;
    let mut candidate = model()?;
    let initial = CudaContext::total_device_bytes();
    let mut stale = artifact.clone();
    stale.source_weights_sha256 = "9".repeat(64);
    assert!(candidate
        .apply_activation_calibration(&stale, 0.5, 1_000_000)
        .is_err());
    assert!(candidate
        .apply_activation_calibration(&artifact, 0.5, 1)
        .is_err());
    assert!(candidate.activation_research.is_none());
    assert_eq!(CudaContext::total_device_bytes(), initial);
    candidate.apply_activation_calibration(&artifact, 0.5, 1_000_000)?;
    candidate.allocate_kv_cache_for_layers(16)?;
    assert!(DecodeSession::new_for_sequence_with_kv_config(
        &candidate,
        0,
        KvCompressionConfig::default(),
        64,
        true,
        "test",
        "test:in",
        "test:out"
    )
    .is_err());
    assert!(candidate
        .apply_activation_calibration(&artifact, 0.5, 1_000_000)
        .is_err());
    Ok(())
}

/// Frozen thresholds retain original source IDs with an interior compute mask.
/// The zero control matches the masked native model exactly; positive thresholds
/// count only active layers. Capture and post-application mask changes must fail
/// before publishing state or allocating additional storage.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_masked_activation_application_preserves_original_layer_routing() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let _dense = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    let artifact = capture_layers(3)?;
    {
        let mut rejected = model_with_layers(3)?;
        rejected.set_diagnostic_layer_mask(&[1])?;
        let before = CudaContext::total_device_bytes();
        assert!(rejected
            .begin_activation_calibration(artifact.plan.clone(), 1_000_000)
            .is_err());
        assert!(rejected.activation_research.is_none());
        assert_eq!(CudaContext::total_device_bytes(), before);
    }
    for target in [0.0, 0.5] {
        let mut reference = model_with_layers(3)?;
        let mut candidate = model_with_layers(3)?;
        reference.set_diagnostic_layer_mask(&[1])?;
        candidate.set_diagnostic_layer_mask(&[1])?;
        if target > 0.0 {
            let before = CudaContext::total_device_bytes();
            assert!(candidate
                .apply_activation_calibration(&artifact, target, 1)
                .is_err());
            assert!(candidate.activation_research.is_none());
            assert_eq!(CudaContext::total_device_bytes(), before);
        }
        candidate.apply_activation_calibration(
            &artifact,
            target,
            if target == 0.0 { 0 } else { 1_000_000 },
        )?;
        let prepared = candidate.activation_research_stats()?;
        assert_eq!(prepared.skipped_layer_ids, [1]);
        assert_eq!(prepared.thresholds.len(), 3);
        let before = CudaContext::total_device_bytes();
        assert!(candidate.set_diagnostic_layer_mask(&[]).is_err());
        assert_eq!(candidate.layer_mask.skipped_layers(), [1]);
        assert_eq!(CudaContext::total_device_bytes(), before);
        reference.allocate_kv_cache_for_layers(16)?;
        candidate.allocate_kv_cache_for_layers(16)?;
        let compared = compare_loaded_models(&mut reference, &mut candidate, &[2, 3, 4, 7, 8, 9])?;
        let stats = candidate.activation_research_stats()?;
        assert_eq!(stats.layers.len(), 3);
        assert_eq!(stats.layers[1].input_values, 0);
        assert_eq!(stats.layers[1].retained_values, 0);
        assert_eq!(stats.layers[1].achieved_sparsity, None);
        if target == 0.0 {
            assert_eq!(compared.max_abs_logit_diff, 0.0);
            assert_eq!(prepared.extra_weight_bytes, 0);
            assert_eq!(prepared.counter_bytes, 0);
        } else {
            assert!(prepared.extra_weight_bytes > 0);
            assert_eq!(prepared.counter_bytes, 3 * 16);
            for layer in [0, 2] {
                assert_eq!(stats.layers[layer].layer_id, layer);
                assert_eq!(stats.layers[layer].input_values, 5 * 128);
                assert!(stats.layers[layer].achieved_sparsity.unwrap() > 0.0);
            }
            assert!(compared.candidate.nll_sum.is_finite());
        }
        candidate.reset_activation_statistics()?;
        assert!(candidate
            .activation_research_stats()?
            .layers
            .iter()
            .all(|layer| layer.input_values == 0 && layer.retained_values == 0));
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

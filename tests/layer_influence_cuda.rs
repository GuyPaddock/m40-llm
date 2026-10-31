#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/env_restore.rs"]
mod env_restore;
#[path = "common/tiny_gguf.rs"]
mod fixture;

use anyhow::Result;
use env_restore::EnvRestore;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    decode_session::DecodeSession,
    evaluation::{
        corpus::Corpus,
        layer_ranking::{measure_block_influence, LayerRankingPassage, LayerRankingPlan},
    },
    infer::{ForwardBatchItem, ForwardPrefillSequence, LoadedModel},
    kv_compression::{KvCompressMode, KvCompressionConfig, ScopedRuntimeConfig},
    profile,
};

fn model() -> Result<LoadedModel> {
    let (gguf, bytes) = fixture::make_qwen2_attention_bias_tiny_gguf(fixture::TinyGgufConfig {
        vocab: 256,
        d_model: 64,
        hidden: 128,
        head_count: 1,
        block_count: 6,
        context_length: 16,
    });
    LoadedModel::from_gguf(gguf, bytes, -1)
}

fn plan() -> Result<LayerRankingPlan> {
    let corpus = Corpus::parse(include_str!("../data/research_quality/v1.json"))?;
    let passages = corpus
        .calibration
        .passages
        .iter()
        .enumerate()
        .map(|(index, passage)| LayerRankingPassage {
            id: passage.id.clone(),
            token_ids: if index == 0 {
                vec![2, 3, 4]
            } else {
                vec![5, 6, 7, 8]
            },
        })
        .collect();
    LayerRankingPlan::new(
        "1".repeat(64),
        "2".repeat(64),
        "3".repeat(64),
        6,
        64,
        &corpus,
        passages,
    )
}

fn configure() -> Vec<EnvRestore> {
    [
        ("M40LLM_ENABLE_NVCC", "1"),
        ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
        ("M40LLM_DECODE_GRAPH", "0"),
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
        "influence-test",
        "influence:input",
        "influence:output",
    )
}

fn drain(model: &LoadedModel) -> Result<()> {
    model.cuda.synchronize_stream(CudaStream::Decode)?;
    model.cuda.synchronize_stream(CudaStream::Prefill)
}

fn hidden(model: &LoadedModel, buffer: &DeviceBuffer) -> Result<Vec<f32>> {
    drain(model)?;
    let mut values = vec![0.0; 64];
    unsafe {
        model
            .cuda
            .memcpy_d2h(values.as_mut_ptr().cast(), buffer.as_ptr(), 256)?;
    }
    Ok(values)
}

fn manual_token(model: &LoadedModel, token: u32, position: u32) -> Result<Vec<f64>> {
    let a = DeviceBuffer::new_tagged(&model.cuda, 256, "influence:oracle-a")?;
    let b = DeviceBuffer::new_tagged(&model.cuda, 256, "influence:oracle-b")?;
    unsafe {
        model.load_token_embedding_to_f32(token as u64, a.as_mut_ptr())?;
    }
    let mut values = Vec::new();
    for layer in 0..6 {
        let (input, output) = if layer % 2 == 0 { (&a, &b) } else { (&b, &a) };
        let before = hidden(model, input)?;
        unsafe {
            model.forward_one_token_with_layer(
                input.as_ptr(),
                layer,
                0,
                position + 1,
                output.as_mut_ptr(),
            )?;
        }
        let after = hidden(model, output)?;
        values.push(measure_block_influence(&before, &after)?);
    }
    Ok(values)
}

/// The read-only observer must preserve full-model logits/launches and report
/// the same original-layer hidden transitions as independent explicit calls.
/// Every declared absolute position is observed; no final target is forwarded.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_layer_influence_matches_manual_transitions_without_changing_logits() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let _dense = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let plan = plan()?;
        let mut observed = model()?;
        let mut reference = model()?;
        let mut manual = model()?;
        let bytes_before = CudaContext::total_device_bytes();
        observed.begin_layer_influence_capture(plan.clone(), 1_000_000)?;
        assert_eq!(CudaContext::total_device_bytes(), bytes_before);
        for model in [&mut observed, &mut reference, &mut manual] {
            model.allocate_kv_cache_for_layers(16)?;
        }
        let mut expected = vec![Vec::new(); 6];
        for passage in &plan.passages {
            for model in [&observed, &reference, &manual] {
                model.reset_kv_cache()?;
            }
            observed.begin_layer_influence_passage(&passage.id)?;
            {
                let mut a = session(&observed)?;
                let mut b = session(&reference)?;
                for length in 1..passage.token_ids.len() {
                    let before = profile::snapshot();
                    let actual = a.logits_for_ids(&passage.token_ids[..length], |_| {})?;
                    let observed_counts = profile::counter_delta(&before, &profile::snapshot());
                    let before = profile::snapshot();
                    let ordinary = b.logits_for_ids(&passage.token_ids[..length], |_| {})?;
                    let reference_counts = profile::counter_delta(&before, &profile::snapshot());
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        ordinary.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    assert_eq!(observed_counts.launches, reference_counts.launches);
                    assert_eq!(observed_counts.cublas_calls, reference_counts.cublas_calls);
                    let measured =
                        manual_token(&manual, passage.token_ids[length - 1], length as u32 - 1)?;
                    for (layer, value) in measured.into_iter().enumerate() {
                        expected[layer].push(value);
                    }
                }
            }
            observed.finish_layer_influence_passage()?;
        }
        let artifact = observed.finish_layer_influence_capture()?;
        assert!(observed.layer_influence.is_none());
        artifact.validate()?;
        for (layer, expected) in artifact.layers.iter().zip(&expected) {
            assert_eq!(layer.block_influence.len(), 5);
            for (&actual, &expected) in layer.block_influence.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "layer {}: {actual} vs {expected}",
                    layer.layer_id
                );
            }
        }
        assert!(artifact.layers.iter().any(|layer| layer.mean() > 1e-12));
        assert!(artifact.ranked_layers()?.iter().all(|&id| id > 0 && id < 5));
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Invalid setup must publish no observer or device allocation. Unsupported
/// execution paths fail before GPU launches; incomplete finish preserves capture
/// so callers cannot silently publish a partial calibration artifact.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_layer_influence_rejects_unsupported_paths_and_incomplete_capture() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let _dense = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let mut model = model()?;
        assert!(model.begin_layer_influence_capture(plan()?, 1).is_err());
        assert!(model.layer_influence.is_none());
        let mut wrong = plan()?;
        wrong.hidden_dim = 128;
        assert!(model
            .begin_layer_influence_capture(wrong, 1_000_000)
            .is_err());
        model.begin_layer_influence_capture(plan()?, 1_000_000)?;
        assert!(model.finish_layer_influence_capture().is_err());
        assert!(model.layer_influence.is_some());
        assert!(model.set_diagnostic_layer_mask(&[1]).is_err());
        assert!(model
            .begin_layer_influence_passage("held-passage-market")
            .is_err());
        model.allocate_kv_cache_for_layer_sequences(16, 2)?;
        assert!(DecodeSession::new_for_sequence_with_kv_config(
            &model,
            0,
            KvCompressionConfig::for_mode(KvCompressMode::BlockSelectExact),
            64,
            true,
            "influence-invalid",
            "influence:bad-input",
            "influence:bad-output"
        )
        .is_err());
        {
            let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "1");
            assert!(session(&model).is_err());
        }
        let input = DeviceBuffer::new_tagged(&model.cuda, 256, "influence:reject-input")?;
        let output = DeviceBuffer::new_tagged(&model.cuda, 256, "influence:reject-output")?;
        let params = DeviceBuffer::new_tagged(&model.cuda, 4, "influence:reject-params")?;
        let before = profile::snapshot();
        unsafe {
            assert!(model
                .forward_one_token_all_layers_batched_for_sequences(&[ForwardBatchItem {
                    d_x_f32: input.as_mut_ptr(),
                    sequence_id: 0,
                    seq_len: 1,
                    d_out_f32: output.as_mut_ptr(),
                }])
                .is_err());
            assert!(model
                .forward_prefill_all_layers_varlen_for_sequences(&[ForwardPrefillSequence {
                    token_ids: &[2, 3],
                    sequence_id: 0,
                    d_out_f32: output.as_mut_ptr(),
                }])
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
        }
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()).launches,
            0
        );
        assert!(model
            .begin_layer_influence_capture(plan()?, 1_000_000)
            .is_err());
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

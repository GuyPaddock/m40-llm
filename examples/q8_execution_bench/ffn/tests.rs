use super::{
    fixture::FfnInputs,
    pipeline::{DownProjection, FfnKernel, FfnPipeline, GuardedF32, Stage},
    validation,
};
use anyhow::Result;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer, Q8FfnFp32Variant, Q8FfnQ81Launch},
    evaluation::{
        corpus::Corpus,
        ffn_capture::{
            FfnCaptureArtifact, FfnCaptureCollector, FfnCapturePlan, FfnLayerSamples,
            FfnLayerSource, FfnRowPosition, FfnTensorSource,
        },
        identity::sha256,
    },
    gguf::GgmlDType,
    infer::{execution_weights::Q8ExecutionConfig, LoadedModel},
};
use serde_json::{json, Value};

#[path = "../../../tests/common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "../../../tests/common/env_restore.rs"]
mod env_restore;
#[path = "../../../tests/common/q8_model.rs"]
#[allow(dead_code)]
mod model_fixture;

fn corpus() -> Result<Corpus> {
    Corpus::parse(include_str!("../../../data/research_quality/v1.json"))
}

fn tensor(name: &str, index: usize) -> FfnTensorSource {
    FfnTensorSource {
        name: format!("blk.0.ffn_{name}.weight"),
        byte_offset: (index * 2176) as u64,
        dtype: GgmlDType::Q8_0,
        shape: if name == "down" { [64, 32] } else { [32, 64] },
        source_sha256: "4".repeat(64),
    }
}

fn evidence() -> Result<(FfnCaptureArtifact, Vec<Value>, Corpus, String)> {
    let corpus = corpus()?;
    let corpus_sha = sha256(&serde_json::to_vec(&corpus)?)?;
    let configuration = json!({"environment":{},"kv_mode":"off","prefill_mode":"sequential"});
    let plan = FfnCapturePlan::new(
        "1".repeat(64),
        corpus_sha.clone(),
        sha256(&serde_json::to_vec(&configuration)?)?,
        &corpus,
        1,
        vec![0],
        32,
        64,
        4,
    )?;
    let source = FfnLayerSource {
        layer_id: 0,
        gate: tensor("gate", 0),
        up: tensor("up", 1),
        down: tensor("down", 2),
    };
    let mut collector =
        FfnCaptureCollector::new(plan.clone(), "5".repeat(64), vec![source], 16384)?;
    for (index, task) in corpus.calibration.tasks.iter().enumerate() {
        collector.begin_task(&task.id, &[1, 2])?;
        collector.observe(
            0,
            2,
            &vec![(index + 1) as f32; 32],
            &vec![-((index + 1) as f32); 64],
        )?;
        collector.finish_task(&[3, 4])?;
    }
    let artifact = collector.finish()?;
    let mut records = vec![
        json!({"kind":"ffn_capture_manifest","schema_version":1,"build_profile":"release","worktree":"",
        "model":{"sha256":plan.model_sha256},"corpus_sha256":corpus_sha,"execution_sha256":plan.execution_sha256,
        "execution_configuration":configuration}),
    ];
    for task in &artifact.tasks {
        let generation = json!({"generated_token_ids":task.generated_token_ids,"output":"answer","metrics":{"stop_reason":"eos"}});
        records.push(
            json!({"kind":"ffn_capture_task","id":task.id,"prompt_token_ids":task.prompt_token_ids,
            "has_identical_generation":true,"baseline":generation,"captured":generation}),
        );
    }
    records.push(json!({"kind":"ffn_capture_complete","is_complete":true,"artifact_sha256":sha256(&serde_json::to_vec(&artifact)?)?}));
    Ok((artifact, records, corpus, corpus_sha))
}

fn parse(
    artifact: &FfnCaptureArtifact,
    records: &[Value],
    corpus: &Corpus,
    corpus_sha: &str,
) -> Result<FfnInputs> {
    let text = serde_json::to_string(artifact)?;
    let report = records
        .iter()
        .map(serde_json::to_string)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .join("\n");
    FfnInputs::parse(&text, &report, corpus, corpus_sha, &"1".repeat(64))
}

/// Benchmark inputs bind actual completed task histories, model/corpus hashes
/// and execution configuration to distinct captured rows, not a model label.
#[test]
fn test_ffn_evidence_binds_complete_histories_and_distinct_rows() -> Result<()> {
    let (artifact, records, corpus, hash) = evidence()?;
    let parsed = parse(&artifact, &records, &corpus, &hash)?;
    assert_eq!(parsed.artifact.layers[0].rows.len(), 6);
    assert_eq!(parsed.artifact.layers[0].normalized_inputs[0], 1.0);
    assert_eq!(parsed.artifact.layers[0].normalized_inputs[32], 2.0);
    for (record, path, replacement) in [
        (0, "/worktree", json!("modified")),
        (0, "/build_profile", json!("dev")),
        (0, "/execution_sha256", json!("0".repeat(64))),
        (0, "/model/sha256", json!("0".repeat(64))),
        (0, "/corpus_sha256", json!("0".repeat(64))),
        (
            0,
            "/execution_configuration/kv_mode",
            json!("block-select-exact"),
        ),
        (1, "/baseline/generated_token_ids", json!([8, 9])),
        (1, "/captured/generated_token_ids", json!([3, 5])),
        (1, "/prompt_token_ids", json!([2, 1])),
        (1, "/captured/output", json!("different")),
        (1, "/captured/metrics/stop_reason", json!("length")),
        (1, "/has_identical_generation", json!(false)),
        (7, "/is_complete", json!(false)),
        (7, "/artifact_sha256", json!("0".repeat(64))),
    ] {
        let mut altered = records.clone();
        *altered[record].pointer_mut(path).unwrap() = replacement;
        assert!(
            parse(&artifact, &altered, &corpus, &hash).is_err(),
            "record={record} path={path}"
        );
    }
    assert!(parse(&artifact, &records[..records.len() - 1], &corpus, &hash).is_err());
    let mut duplicated = artifact.clone();
    for row in 1..4 {
        duplicated.layers[0].normalized_inputs[row * 32..(row + 1) * 32].fill(1.0);
    }
    let mut records = records;
    records.last_mut().unwrap()["artifact_sha256"] =
        json!(sha256(&serde_json::to_vec(&duplicated)?)?);
    assert!(parse(&duplicated, &records, &corpus, &hash).is_err());
    Ok(())
}

/// A nominally faster kernel cannot pass validation with nonfinite output,
/// incomplete vectors or numerical drift hidden outside a top-token check.
#[test]
fn test_ffn_numerical_gate_rejects_full_output_drift() -> Result<()> {
    assert!(validation::compare(&[1.0, 2.0], &[1.0, 2.0])?.bitwise_equal);
    assert!(validation::compare(&[1.0, 2.1], &[1.0, 2.0]).is_err());
    assert!(validation::compare(&[1.0, f32::NAN], &[1.0, 2.0]).is_err());
    assert!(validation::compare(&[1.0], &[1.0, 2.0]).is_err());
    assert!(validation::compare(&[], &[]).is_err());
    assert!((validation::measure(&[1.0, 2.1], &[1.0, 2.0])?.max_abs_diff - 0.1).abs() < 1e-6);
    assert!(validation::measure(&[f32::NAN], &[1.0]).is_err());
    Ok(())
}

/// Q8_1 candidates must reserve reusable guarded scratch before selection,
/// include one quantizer in every gate/complete step, and keep strict numerical
/// correctness separate from intentionally reported FP32-activation drift.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_q81_pipeline_accounts_for_quantization_and_keeps_down() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let (gguf, bytes) = model_fixture::build(0);
        let model = LoadedModel::from_gguf_with_q8_execution(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
        )?;
        let layer = sample_layer(&model)?;
        for rows in [1, 4] {
            let mut pipeline = FfnPipeline::new(&model, &layer, rows)?;
            let capacity = pipeline.scratch_bytes();
            let before = m40_llm::profile::snapshot();
            assert!(pipeline
                .select_kernel(FfnKernel::Q81(Q8FfnQ81Launch::Threads128))
                .is_err());
            assert!(pipeline.time(Stage::Quantize, 1).is_err());
            assert_eq!(
                m40_llm::profile::counter_delta(&before, &m40_llm::profile::snapshot()),
                Default::default()
            );
            pipeline.reserve_quantized_scratch()?;
            assert!(pipeline.reserve_quantized_scratch().is_err());
            assert_eq!(
                pipeline.scratch_bytes(),
                capacity + rows * 64 / 32 * 36 + 32
            );
            let resident = CudaContext::total_device_bytes();
            for launch in Q8FfnQ81Launch::ALL {
                pipeline.select_kernel(FfnKernel::Q81(launch))?;
                let checks = validation::validate(&pipeline, &layer)?;
                assert_eq!(checks["quantization"]["encoded_bytes_match_cpu"], true);
                assert_eq!(checks["quantization"]["payload_bytes"], rows * 64 / 32 * 36);
                for stage in [
                    Stage::Quantize,
                    Stage::GateUpSwiglu,
                    Stage::IsolatedDown,
                    Stage::Complete,
                    Stage::Complete,
                ] {
                    let timing = pipeline.time(stage, 8)?;
                    let counters = &timing["enqueue_counters"];
                    let launches = match stage {
                        Stage::Quantize | Stage::IsolatedDown => 1,
                        Stage::GateUpSwiglu => 2,
                        Stage::Complete => 3,
                    };
                    let waits = if rows == 1 {
                        0
                    } else if matches!(stage, Stage::Complete) {
                        2
                    } else {
                        1
                    };
                    assert_eq!(counters["launches"], 8 * launches);
                    assert_eq!(counters["stream_waits"], 8 * waits);
                    for key in [
                        "device_allocs",
                        "device_frees",
                        "h2d_copies",
                        "d2h_copies",
                        "stream_syncs",
                        "cublas_calls",
                    ] {
                        assert_eq!(counters[key], 0, "{key}");
                    }
                }
                let after = validation::validate(&pipeline, &layer)?;
                assert_eq!(after["guards_and_immutable_inputs"], true);
                assert_eq!(CudaContext::total_device_bytes(), resident);
                pipeline.select_kernel(FfnKernel::Native)?;
                assert!(validation::validate(&pipeline, &layer)?["quantization"].is_null());
            }
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// The prepared-down screens change only the selected primitive: both genuine
/// row counts retain split gate/up bits, complete output, owned cache bytes
/// and allocation-free timing across the existing cross-stream schedule.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_prepared_down_candidates_keep_complete_ffn_bits_and_cache() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let (gguf, bytes) = model_fixture::build(0);
        let model = LoadedModel::from_gguf_with_q8_execution(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
        )?;
        let layer = sample_layer(&model)?;
        let cache = serde_json::to_value(model.q8_execution_stats())?;
        assert!(model
            .q8_execution
            .get_separate_weight(std::ptr::null())
            .is_none());
        for rows in [1, 4] {
            let mut p = FfnPipeline::new(&model, &layer, rows)?;
            p.prepare_split_weights(&layer)?;
            p.select_kernel(FfnKernel::SplitF32x4)?;
            validation::validate(&p, &layer)?;
            let hidden = p.hidden.read(&model.cuda)?;
            let output = p.output.read(&model.cuda)?;
            let resident = CudaContext::total_device_bytes();
            for projection in [DownProjection::WarpFinish, DownProjection::PackedActivation] {
                p.select_down_projection(projection)?;
                validation::validate(&p, &layer)?;
                assert_eq!(p.hidden.read(&model.cuda)?, hidden);
                assert!(p
                    .output
                    .read(&model.cuda)?
                    .iter()
                    .zip(&output)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
                let timing = p.time(Stage::Complete, 7)?;
                assert_eq!(timing["enqueue_counters"]["launches"], 14);
                assert_eq!(timing["enqueue_counters"]["device_allocs"], 0);
                assert_eq!(CudaContext::total_device_bytes(), resident);
                p.verify_guards_and_inputs(&layer)?;
            }
        }
        assert_eq!(serde_json::to_value(model.q8_execution_stats())?, cache);
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Protect exact-control variants' bits and the eight-value variant's unchanged
/// numerical bounds, then verify actual complete-FFN dispatch and ownership.
fn compare_split_pipeline(kernel: FfnKernel) -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let (gguf, bytes) = model_fixture::build(0);
        let model = LoadedModel::from_gguf_with_q8_execution(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
        )?;
        let layer = sample_layer(&model)?;
        let cache = serde_json::to_value(model.q8_execution_stats())?;
        for rows in [1, 4] {
            let mut pipeline = FfnPipeline::new(&model, &layer, rows)?;
            pipeline.prepare_split_weights(&layer)?;
            pipeline.select_kernel(FfnKernel::SplitF32x4)?;
            validation::validate(&pipeline, &layer)?;
            let read = || -> Result<Vec<u32>> {
                let mut bits = Vec::new();
                for buffer in [
                    &pipeline.gate,
                    &pipeline.up,
                    &pipeline.hidden,
                    &pipeline.output,
                ] {
                    bits.extend(buffer.read(&model.cuda)?.into_iter().map(f32::to_bits));
                }
                Ok(bits)
            };
            let expected = read()?;
            let capacity = pipeline.scratch_bytes();
            let resident = CudaContext::total_device_bytes();
            pipeline.select_kernel(kernel)?;
            validation::validate(&pipeline, &layer)?;
            let mut actual = Vec::new();
            for buffer in [
                &pipeline.gate,
                &pipeline.up,
                &pipeline.hidden,
                &pipeline.output,
            ] {
                actual.extend(buffer.read(&model.cuda)?.into_iter().map(f32::to_bits));
            }
            if kernel == FfnKernel::SplitF32x8 {
                validation::compare(
                    &actual
                        .iter()
                        .copied()
                        .map(f32::from_bits)
                        .collect::<Vec<_>>(),
                    &expected
                        .iter()
                        .copied()
                        .map(|v| f64::from(f32::from_bits(v)))
                        .collect::<Vec<_>>(),
                )?;
            } else {
                assert_eq!(actual, expected);
            }
            let timing = pipeline.time(Stage::Complete, 7)?;
            if kernel == FfnKernel::SplitF32x4Unfused {
                assert_eq!(timing["enqueue_counters"]["launches"], 28);
                assert_eq!(timing["unfused_dispatch"]["projection_launches"], 14);
                assert_eq!(timing["unfused_dispatch"]["swiglu_launches"], 7);
            } else {
                assert_eq!(timing["enqueue_counters"]["launches"], 14);
                let dispatch = if kernel == FfnKernel::SplitF32x8 {
                    "eight_value_dispatch"
                } else if kernel == FfnKernel::SplitF32x4Cohorts {
                    "cohorts_dispatch"
                } else if kernel == FfnKernel::SplitF32x4PersistentTiles {
                    "persistent_tiles_dispatch"
                } else if kernel == FfnKernel::SplitF32x4VirtualWarp {
                    "virtual_warp_dispatch"
                } else if kernel == FfnKernel::SplitF32x4WeightCg {
                    "weight_cg_dispatch"
                } else {
                    "byte_permute_dispatch"
                };
                assert_eq!(timing[dispatch]["fused_launches"], 7);
                if kernel == FfnKernel::SplitF32x8 {
                    assert_eq!(timing[dispatch]["values_per_task"], 8);
                    assert_eq!(timing[dispatch]["payload_load_bytes"], 8);
                    assert_eq!(timing[dispatch]["threads"], 128);
                    assert_eq!(timing[dispatch]["blocks_per_row"], pipeline.hidden_dim);
                    assert_eq!(timing[dispatch]["shared_bytes"], 1024);
                    assert_eq!(timing[dispatch]["has_activation_quantization"], false);
                }
                if kernel == FfnKernel::SplitF32x4Cohorts {
                    assert_eq!(timing[dispatch]["threads_per_output"], 128);
                    assert_eq!(timing[dispatch]["cohorts_per_block"], 2);
                    assert_eq!(timing[dispatch]["threads"], 256);
                    assert_eq!(
                        timing[dispatch]["blocks_per_row"],
                        pipeline.hidden_dim.div_ceil(2)
                    );
                    assert_eq!(timing[dispatch]["shared_bytes"], 2048);
                }
                if kernel == FfnKernel::SplitF32x4PersistentTiles {
                    assert_eq!(timing[dispatch]["block_cap_per_row"], 192);
                    assert_eq!(
                        timing[dispatch]["blocks_per_row"],
                        pipeline.hidden_dim.min(192)
                    );
                    assert_eq!(timing[dispatch]["has_device_wide_barrier"], false);
                }
            }
            for counter in [
                "device_allocs",
                "device_frees",
                "h2d_copies",
                "d2h_copies",
                "stream_syncs",
            ] {
                assert_eq!(timing["enqueue_counters"][counter], 0);
            }
            assert_eq!(pipeline.scratch_bytes(), capacity);
            assert_eq!(CudaContext::total_device_bytes(), resident);
            pipeline.verify_guards_and_inputs(&layer)?;
        }
        assert_eq!(serde_json::to_value(model.q8_execution_stats())?, cache);
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// The unfused experiment must retain all complete-FFN bits with existing
/// scratch and immutable down, counting three gate/up launches plus down.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_unfused_ffn_preserves_complete_bits_and_reuses_scratch() -> Result<()> {
    compare_split_pipeline(FfnKernel::SplitF32x4Unfused)
}

/// Changing only weight-cache policy must retain gate/up, hidden and complete
/// output bits with the same scratch/cache. Seven iterations enqueue fourteen
/// kernels without allocation, transfer or synchronization inside enqueue.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_weight_cg_ffn_preserves_complete_bits_and_reuses_scratch() -> Result<()> {
    compare_split_pipeline(FfnKernel::SplitF32x4WeightCg)
}

/// Exact byte-permute conversion must preserve all captured projection, hidden
/// and completed down bits with unchanged storage and actual fused dispatch.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_byte_permute_ffn_preserves_complete_bits_and_reuses_scratch() -> Result<()> {
    compare_split_pipeline(FfnKernel::SplitF32x4BytePermute)
}

/// Virtual-thread ownership must preserve captured projections, hidden and
/// completed down bits across distinct rows without changing scratch/cache.
/// Actual timed dispatch must be the new primitive, not a silent control run.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_virtual_warp_ffn_preserves_complete_bits_and_reuses_scratch() -> Result<()> {
    compare_split_pipeline(FfnKernel::SplitF32x4VirtualWarp)
}

/// CTA reuse must retain captured gate/up, hidden and complete down bits,
/// unchanged scratch/cache ownership and actual allocation-free dispatch.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_persistent_tiles_ffn_preserves_complete_bits_and_reuses_scratch() -> Result<()> {
    compare_split_pipeline(FfnKernel::SplitF32x4PersistentTiles)
}

/// Two physical output cohorts preserve the captured projections, hidden and
/// completed down bits with actual two-cohort dispatch and unchanged scratch.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_physical_cohorts_ffn_preserves_complete_bits_and_reuses_scratch() -> Result<()> {
    compare_split_pipeline(FfnKernel::SplitF32x4Cohorts)
}

/// Eight-value ownership keeps existing CPU/control tolerances, no quantized
/// activations or new scratch, and dispatches the real wider-task kernel.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_eight_value_ffn_preserves_cpu_bounds_and_reuses_scratch() -> Result<()> {
    compare_split_pipeline(FfnKernel::SplitF32x8)
}

/// Complete-FFN screening needs a reproducible positive paired interval and
/// at least 5% gain. Bad, tied or regressing timing samples cannot qualify.
#[test]
fn test_ffn_screen_requires_reproducible_complete_gain() -> Result<()> {
    let pairs: Vec<_> = (1..=5).map(|i| (i as f64, i as f64 / 1.1)).collect();
    let first = super::ablation::summarize(&pairs)?;
    assert_eq!(first, super::ablation::summarize(&pairs)?);
    assert_eq!(first["passes_micro_screen"], true);
    assert!(first["bootstrap_gain_ci95"][0].as_f64().unwrap() > 0.0);
    for bound in first["bootstrap_gain_ci95"].as_array().unwrap() {
        assert!((bound.as_f64().unwrap() - 0.1).abs() < 1e-12);
    }
    for pairs in [
        vec![(1.0, 1.0); 5],
        vec![(1.0, 1.1); 5],
        vec![(1.0, 0.97); 5],
    ] {
        assert_eq!(
            super::ablation::summarize(&pairs)?["passes_micro_screen"],
            false
        );
    }
    for pairs in [
        vec![(1.0, 0.9); 4],
        vec![(1.0, 0.0); 5],
        vec![(f64::NAN, 1.0); 5],
    ] {
        assert!(super::ablation::summarize(&pairs).is_err());
    }
    Ok(())
}

/// Even a self-consistent capture cannot silently select another projection
/// backend, materialization, stream or fusion when timing the native control.
#[test]
fn test_ffn_timing_requires_the_qualified_control_environment() -> Result<()> {
    let manifest: Value = serde_json::from_str(
        include_str!("../../../docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl")
            .lines()
            .next()
            .unwrap(),
    )?;
    let environment = serde_json::from_value::<std::collections::BTreeMap<String, String>>(
        manifest["execution_configuration"]["environment"].clone(),
    )?;
    super::fixture::validate_environment_values(&environment, &environment)?;
    let mut altered = environment.clone();
    altered.insert("M40LLM_MATERIALIZE_F32_WEIGHTS".to_owned(), "1".to_owned());
    assert!(super::fixture::validate_environment_values(&environment, &altered).is_err());
    assert!(super::fixture::validate_environment_values(&altered, &altered).is_err());
    let mut altered = environment.clone();
    altered.insert("M40LLM_FUSED_Q8_MLP_SWIGLU".to_owned(), "0".to_owned());
    assert!(super::fixture::validate_environment_values(&altered, &altered).is_err());
    Ok(())
}

fn configure() -> Vec<env_restore::EnvRestore> {
    [
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_Q8_DECODE_COL_TILE", "4"),
        ("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4"),
        ("M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK", "4"),
        ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
        ("M40LLM_PROJECTION_BACKEND", "large-model"),
        ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
        ("M40LLM_Q8_DECODE_EVENT_LOG", "0"),
    ]
    .into_iter()
    .map(|(key, value)| env_restore::EnvRestore::set(key, value))
    .collect()
}

fn sample_layer(model: &LoadedModel) -> Result<FfnLayerSamples> {
    let mapped = model.map_standard_layer(0)?;
    let source = |view: &m40_llm::infer::DeviceTensorView| -> Result<FfnTensorSource> {
        let tensor = model
            .gguf
            .tensors
            .iter()
            .find(|tensor| tensor.offset == view.byte_offset)
            .unwrap();
        let mut source = FfnTensorSource {
            name: tensor.name.clone(),
            byte_offset: tensor.offset,
            dtype: tensor.dtype,
            shape: [tensor.shape[0] as usize, tensor.shape[1] as usize],
            source_sha256: String::new(),
        };
        let start = source.byte_offset as usize;
        source.source_sha256 = sha256(&model.host_weights[start..start + source.byte_len()?])?;
        Ok(source)
    };
    let input = model_fixture::projection::make_input(4, 64);
    let a = DeviceBuffer::new_tagged(&model.cuda, 64 * 4, "ffn-test:input")?;
    let h = DeviceBuffer::new_tagged(&model.cuda, 128 * 4, "ffn-test:hidden")?;
    let mut hidden = vec![0.0f32; 4 * 128];
    for row in 0..4 {
        unsafe {
            model.cuda.memcpy_h2d(
                a.as_mut_ptr(),
                input[row * 64..(row + 1) * 64].as_ptr().cast(),
                256,
            )?;
            assert!(model.try_mlp_gate_up_swiglu_f32xf16_gguf_decode_async(
                a.as_ptr(),
                1,
                64,
                mapped.w_gate.dptr,
                mapped.w_up.dptr,
                128,
                h.as_mut_ptr(),
                None,
            )?);
        }
        model.cuda.synchronize_stream(CudaStream::Decode)?;
        unsafe {
            model.cuda.memcpy_d2h(
                hidden[row * 128..(row + 1) * 128].as_mut_ptr().cast(),
                h.as_ptr(),
                512,
            )?;
        }
    }
    Ok(FfnLayerSamples {
        source: FfnLayerSource {
            layer_id: 0,
            gate: source(&mapped.w_gate)?,
            up: source(&mapped.w_up)?,
            down: source(&mapped.w_down)?,
        },
        rows: (0..4)
            .map(|index| FfnRowPosition {
                task_id: "synthetic-matrix-rows".to_owned(),
                absolute_position: 5 + index,
            })
            .collect(),
        normalized_inputs: input,
        swiglu_inputs: hidden,
    })
}

/// Genuine one/four-row pipelines use the production wrappers and prepared
/// down, preserve guards/inputs across repeated asynchronous stages, and match
/// scalar CPU references without per-iteration allocation/copy/synchronization.
/// Warmup records actual work and fails at its declared finite batch cap.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_ffn_pipeline_matches_cpu_and_reuses_cross_stream_scratch() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let (gguf, bytes) = model_fixture::build(0);
        let model = LoadedModel::from_gguf_with_q8_execution(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
        )?;
        let layer = sample_layer(&model)?;
        for rows in [1, 4] {
            let pipeline = FfnPipeline::new(&model, &layer, rows)?;
            let result = validation::validate(&pipeline, &layer)?;
            assert_eq!(result["guards_and_immutable_inputs"], true);
            assert_eq!(result["gate_vs_cpu"]["compared_values"], rows * 128);
            if rows == 1 {
                assert_eq!(result["swiglu_vs_frozen_capture"]["bitwise_equal"], true);
            }
            let capacity = pipeline.scratch_bytes();
            let resident = CudaContext::total_device_bytes();
            for stage in [
                Stage::Complete,
                Stage::GateUpSwiglu,
                Stage::IsolatedDown,
                Stage::Complete,
            ] {
                let timing = pipeline.time(stage, 7)?;
                let counters = &timing["enqueue_counters"];
                let launches = match (rows, stage) {
                    (1, Stage::GateUpSwiglu) => 7,
                    (1, Stage::Complete) => 14,
                    (_, Stage::IsolatedDown) => 7,
                    (_, Stage::GateUpSwiglu) => 21,
                    (_, Stage::Complete) => 28,
                    (_, Stage::Quantize) => unreachable!("native stage list excludes quantization"),
                };
                assert_eq!(counters["launches"], launches);
                for key in [
                    "device_allocs",
                    "device_frees",
                    "h2d_copies",
                    "d2h_copies",
                    "stream_syncs",
                    "cublas_calls",
                ] {
                    assert_eq!(counters[key], 0, "{key}");
                }
                assert!(timing["gpu_ms"].as_f64().unwrap() >= 0.0);
                assert_eq!(pipeline.scratch_bytes(), capacity);
                assert_eq!(CudaContext::total_device_bytes(), resident);
            }
            pipeline.verify_guards_and_inputs(&layer)?;
            let warmup = pipeline.warm_up(0.25, 128)?;
            assert!(warmup["elapsed_gpu_ms"].as_f64().unwrap() >= 0.25);
            assert!(warmup["batches"].as_u64().unwrap() <= 128);
            assert!(pipeline.warm_up(f64::MAX, 1).is_err());
            assert!(pipeline.warm_up(f64::NAN, 1).is_err());
            assert!(pipeline.warm_up(0.0, 1).is_err());
            assert!(pipeline.warm_up(1.0, 0).is_err());
            assert_eq!(CudaContext::total_device_bytes(), resident);
            // Queue a changing producer on decode before each full pipeline.
            // Four-row gate/up consumes on prefill and must wait for this write.
            let mut expected_input = layer.normalized_inputs[..rows * 64].to_vec();
            for _ in 0..3 {
                for value in &mut expected_input {
                    *value += *value;
                }
                unsafe {
                    model.cuda.residual_add_f32_async(
                        pipeline.input.pointer(),
                        pipeline.input.pointer(),
                        pipeline.input.pointer(),
                        rows * 64,
                    )?;
                }
                pipeline.enqueue(Stage::Complete)?;
                pipeline.drain()?;
                assert_eq!(pipeline.input.read(&model.cuda)?, expected_input);
                let gate_cpu = validation::project(
                    &expected_input,
                    super::fixture::source_bytes(&model, &layer.source.gate)?,
                    &layer.source.gate,
                    rows,
                )?;
                let up_cpu = validation::project(
                    &expected_input,
                    super::fixture::source_bytes(&model, &layer.source.up)?,
                    &layer.source.up,
                    rows,
                )?;
                let expected_hidden: Vec<_> = gate_cpu
                    .iter()
                    .zip(up_cpu)
                    .map(|(&gate, up)| gate / (1.0 + (-gate).exp()) * up)
                    .collect();
                let hidden = pipeline.hidden.read(&model.cuda)?;
                validation::compare(&hidden, &expected_hidden)?;
                let expected_down = validation::project(
                    &hidden,
                    super::fixture::source_bytes(&model, &layer.source.down)?,
                    &layer.source.down,
                    rows,
                )?;
                validation::compare(&pipeline.output.read(&model.cuda)?, &expected_down)?;
            }
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Original and follow-up variants, including prepared split planes, reuse
/// one allocation set across kernel switches, including the two-column
/// split and joint-record consumers with identical one-column results.
/// Repeated row-batched complete work waits before hidden reuse and
/// returns CPU-matching outputs without timed allocation, copying or sync.
/// The K=2048 consumer rejects this K=64 fixture without changing allocations
/// or silently switching to the dynamic kernel.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_ffn_ablations_reuse_scratch_and_preserve_complete_outputs() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let (gguf, bytes) = model_fixture::build(0);
        let model = LoadedModel::from_gguf_with_q8_execution(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
        )?;
        let layer = sample_layer(&model)?;
        for rows in [1, 4] {
            let mut pipeline = FfnPipeline::new(&model, &layer, rows)?;
            assert!(pipeline.select_kernel(FfnKernel::SplitF32x4).is_err());
            assert!(pipeline
                .select_kernel(FfnKernel::SplitF32x4TwoColumns)
                .is_err());
            assert!(pipeline.select_kernel(FfnKernel::JointF32x4).is_err());
            let qualified = serde_json::to_value(model.q8_execution_stats())?;
            pipeline.prepare_split_weights(&layer)?;
            assert!(pipeline.prepare_split_weights(&layer).is_err());
            assert!(pipeline.prepared.as_ref().unwrap().bytes > 0);
            pipeline.joint = Some(super::joint::JointFfnWeights::new(
                &model,
                &layer,
                pipeline.prepared.as_ref().unwrap().bytes,
            )?);
            assert!(pipeline.joint.as_ref().unwrap().bytes > 0);
            let before = m40_llm::profile::snapshot();
            assert!(pipeline
                .select_kernel(FfnKernel::SplitF32x4K2048)
                .unwrap_err()
                .to_string()
                .contains("requires K=2048"));
            assert_eq!(
                m40_llm::profile::counter_delta(&before, &m40_llm::profile::snapshot()),
                m40_llm::profile::counter_delta(&before, &before)
            );
            let capacity = pipeline.scratch_bytes();
            let resident = CudaContext::total_device_bytes();
            validation::validate(&pipeline, &layer)?;
            let native_hidden = pipeline.hidden.read(&model.cuda)?;
            let mut split_hidden: Option<Vec<f32>> = None;
            let mut split_output: Option<Vec<f32>> = None;
            for kernel in Q8FfnFp32Variant::ALL
                .into_iter()
                .chain([
                    Q8FfnFp32Variant::PackedWarp128FactoredScales,
                    Q8FfnFp32Variant::PackedWarp128CoalescedPairs,
                ])
                .map(FfnKernel::Fp32)
                .chain([
                    FfnKernel::SplitF32x4,
                    FfnKernel::SplitF32x4TwoColumns,
                    FfnKernel::JointF32x4,
                ])
            {
                pipeline.select_kernel(kernel)?;
                let checks = validation::validate(&pipeline, &layer)?;
                assert_eq!(checks["guards_and_immutable_inputs"], true);
                if kernel.has_intermediate_capture() {
                    assert_eq!(checks["intermediate_capture_matches_normal_kernel"], true);
                }
                if kernel == FfnKernel::SplitF32x4 {
                    split_hidden = Some(pipeline.hidden.read(&model.cuda)?);
                    split_output = Some(pipeline.output.read(&model.cuda)?);
                } else if matches!(
                    kernel,
                    FfnKernel::SplitF32x4TwoColumns | FfnKernel::JointF32x4
                ) {
                    assert_eq!(
                        pipeline.hidden.read(&model.cuda)?,
                        *split_hidden.as_ref().unwrap()
                    );
                    assert_eq!(
                        pipeline.output.read(&model.cuda)?,
                        *split_output.as_ref().unwrap()
                    );
                }
                if rows == 1 && kernel.threads() == 256 {
                    assert_eq!(pipeline.hidden.read(&model.cuda)?, native_hidden);
                }
                for stage in [
                    Stage::Complete,
                    Stage::GateUpSwiglu,
                    Stage::IsolatedDown,
                    Stage::Complete,
                ] {
                    let timing = pipeline.time(stage, 32)?;
                    let counters = &timing["enqueue_counters"];
                    assert_eq!(
                        counters["launches"],
                        if matches!(stage, Stage::Complete) {
                            64
                        } else {
                            32
                        }
                    );
                    for key in [
                        "device_allocs",
                        "device_frees",
                        "h2d_copies",
                        "d2h_copies",
                        "stream_syncs",
                        "cublas_calls",
                    ] {
                        assert_eq!(counters[key], 0, "{key}");
                    }
                }
                let hidden = pipeline.hidden.read(&model.cuda)?;
                let expected = validation::project(
                    &hidden,
                    super::fixture::source_bytes(&model, &layer.source.down)?,
                    &layer.source.down,
                    rows,
                )?;
                validation::compare(&pipeline.output.read(&model.cuda)?, &expected)?;
                pipeline.verify_guards_and_inputs(&layer)?;
                assert_eq!(pipeline.scratch_bytes(), capacity);
                assert_eq!(CudaContext::total_device_bytes(), resident);
                assert_eq!(serde_json::to_value(model.q8_execution_stats())?, qualified);
                pipeline.select_kernel(FfnKernel::Native)?;
                pipeline.enqueue(Stage::Complete)?;
                pipeline.drain()?;
                assert_eq!(pipeline.hidden.read(&model.cuda)?, native_hidden);
            }
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Split-plane and joint dispatch have distinct report identities while legacy native
/// and packed variants retain their existing JSON values.
#[test]
fn test_split_ffn_reports_do_not_alias_native_or_packed_controls() -> Result<()> {
    assert_eq!(serde_json::to_value(FfnKernel::Native)?, Value::Null);
    assert_eq!(
        serde_json::to_value(FfnKernel::Fp32(Q8FfnFp32Variant::PackedWarp128))?,
        "packed-warp128"
    );
    assert_eq!(
        serde_json::to_value(FfnKernel::SplitF32x4)?,
        "split-f32x4-128"
    );
    assert_eq!(
        serde_json::to_value(FfnKernel::SplitF32x4TwoColumns)?,
        "split-f32x4-cols2-128"
    );
    assert_eq!(
        serde_json::to_value(FfnKernel::JointF32x4)?,
        "joint-f32x4-128"
    );
    assert_eq!(
        serde_json::to_value(FfnKernel::SplitF32x4K2048)?,
        "split-f32x4-k2048-128"
    );
    Ok(())
}

/// Guard verification checks real buffer contents and catches an out-of-bounds
/// write while leaving ordinary initialized payload values readable.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_ffn_guards_detect_boundary_corruption() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let buffer = GuardedF32::new(&ctx, &[1.0, 2.0, 3.0], "ffn-test:guards")?;
        assert_eq!(buffer.read(&ctx)?, [1.0, 2.0, 3.0]);
        let value = 0.0f32;
        unsafe {
            ctx.memcpy_h2d(
                buffer.pointer().cast::<f32>().sub(1).cast(),
                (&value as *const f32).cast(),
                4,
            )?;
        }
        assert!(buffer.read(&ctx).is_err());
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

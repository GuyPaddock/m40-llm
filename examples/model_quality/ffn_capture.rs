use super::{
    records::{self, Report},
    research, run, tasks, Args, SplitChoice,
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaContext,
    evaluation::{
        corpus::{Corpus, Split},
        ffn_capture::FfnCapturePlan,
    },
    infer::execution_weights::{Q8ExecutionConfig, Q8SeparateLaunch},
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const ROW_CAP: usize = 4;
const HOST_BUDGET: usize = 32 * 1024 * 1024;

fn validate_request(args: &Args) -> Result<()> {
    ensure!(
        matches!(args.split, SplitChoice::Calibration)
            && args.context_tokens == 1024
            && args.skip_layers.is_empty()
            && args.step_timing_trials == 0,
        "FFN capture requires --split calibration, context 1024, no transforms or timing trials"
    );
    let destination = args
        .capture_ffn_inputs
        .as_ref()
        .context("provide --capture-ffn-inputs")?;
    ensure!(
        !destination.exists(),
        "refuse to overwrite FFN input artifact"
    );
    let canonical_destination = |path: &Path| -> Result<_> {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        Ok(fs::canonicalize(parent)?.join(path.file_name().context("expected file destination")?))
    };
    ensure!(
        canonical_destination(destination)? != canonical_destination(&args.output)?,
        "FFN artifact and report need distinct destinations"
    );
    Ok(())
}

fn execution_settings(environment: &BTreeMap<String, String>) -> Result<Value> {
    research::validate_execution(environment)?;
    for (key, value) in [
        ("M40LLM_ENABLE_CUBLAS", "1"),
        ("M40LLM_Q8_DECODE_COL_TILE", "4"),
        ("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4"),
        ("M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK", "4"),
        ("M40LLM_FUSED_Q8_QKV", "1"),
        ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
        ("M40LLM_CUDA_GREEDY_ARGMAX", "1"),
        ("M40LLM_Q8_LM_HEAD_ARGMAX", "1"),
        ("M40LLM_Q8_LM_HEAD_ARGMAX_SHARED_A", "1"),
        ("M40LLM_ATTN_HEAD128_KERNEL", "parallel"),
        ("M40LLM_ATTN_HEAD128_PARTITIONS", "8"),
        ("M40LLM_ATTN_HEAD128_THREADS", "256"),
    ] {
        ensure!(
            environment.get(key).map(String::as_str) == Some(value),
            "FFN capture requires qualified control {key}={value}"
        );
    }
    Ok(
        json!({"environment":environment,"allocated_context_tokens":1024,
        "prepared_weights":{"mode":"split","budget_mib":6144,"families":["down","vocabulary"],
            "single_row_down_columns":4,"single_row_down_threads":256},
        "kv_mode":"off","prefill_mode":"sequential","sampling":"greedy top_k=1 seed=0; actual natural stopping",
        "task_template":"canonical Qwen system+user ChatML", "source_layer_ids":[0,17,35],
        "max_generated_input_rows_per_task":ROW_CAP,"capture_host_payload_budget_bytes":HOST_BUDGET}),
    )
}

/// Capture is deliberately untimed. Each task first runs with the configured
/// observer inactive, then again active; every actual output ID must match.
pub(super) fn capture(args: &Args) -> Result<()> {
    validate_request(args)?;
    let environment = run::environment()?;
    let settings = execution_settings(&environment)?;
    let execution_sha256 = records::sha256(&serde_json::to_vec(&settings)?)?;
    let corpus_bytes = fs::read(&args.corpus)?;
    let corpus = Corpus::parse(std::str::from_utf8(&corpus_bytes)?)?;
    let corpus_sha256 = records::sha256(&corpus_bytes)?;
    let model_sha256 = records::file_sha256(&args.model)?;
    ensure!(
        records::file_sha256(&args.reference_model)? == model_sha256,
        "FFN capture requires the same original reference/model weights"
    );
    let mut report = Report::create(&args.output)?;
    let initial_device_bytes = CudaContext::total_device_bytes();
    let started = Instant::now();
    let (mut model, tokenizer) = run::load_model_with_setup(
        &args.model,
        args.context_tokens,
        Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
        |model| {
            model.set_diagnostic_layer_mask(&[])?;
            let config = &model.model_config;
            ensure!(
                config.architecture == "qwen2"
                    && config.block_count == 36
                    && config.embedding_length == 2048
                    && config.feed_forward_length == 11008
                    && config.attention_key_length == 128,
                "M02 FFN capture requires the Qwen2.5-3B Q8_0 model shape"
            );
            let plan = FfnCapturePlan::new(
                model_sha256.clone(),
                corpus_sha256.clone(),
                execution_sha256.clone(),
                &corpus,
                36,
                vec![0, 17, 35],
                2048,
                11008,
                ROW_CAP,
            )?;
            model.begin_ffn_capture(plan, HOST_BUDGET)
        },
    )?;
    let prepared = model.q8_execution_stats();
    ensure!(
        prepared.has_prepared_vocabulary
            && prepared.single_row_down_split_launch == Some(Q8SeparateLaunch::new(4, 256)?),
        "FFN capture did not install the qualified prepared down/vocabulary control"
    );
    run::validate_inputs(
        &corpus,
        &[Split::Calibration],
        &tokenizer,
        &tokenizer,
        args.context_tokens,
        model.model_config.vocab_size,
    )?;
    report.append(json!({"kind":"ffn_capture_manifest","schema_version":1,"variant_id":args.variant_id,
        "created_unix_s":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "model":run::model_identity(&args.model,&model,&tokenizer)?,"corpus_path":args.corpus,"corpus_sha256":corpus_sha256,
        "execution_configuration":settings,"execution_sha256":execution_sha256,"prepared_weights":prepared,
        "revision":records::command_text("git",&["rev-parse","HEAD"])?,"worktree":records::command_text("git",&["status","--short"])?,
        "executable_sha256":records::file_sha256(&std::env::current_exe()?)?,"build_profile":"release",
        "rustc":records::command_text("rustc",&["--version"])?,"nvcc":records::command_text("nvcc",&["--version"])?,
        "host_compiler":records::command_text("g++",&["--version"])?,"cuda_flags":research::CUDA_FLAGS,
        "device":format!("{:?}",model.cuda.current_device_props()?),
        "setup_through_identity_ms":started.elapsed().as_secs_f64()*1000.0,"is_performance_measurement":false,
        "scope":"calibration-only read-only paired inputs; active capture adds diagnostic D2H and stream drains; no held-out/browser observations"}))?;
    tasks::warmup(&model, &tokenizer)?;
    for task in &corpus.calibration.tasks {
        let (prompt, prompt_ids) = tasks::encode_task(&tokenizer, &task.prompt)?;
        let baseline_started = Instant::now();
        let baseline = tasks::generate(&model, &prompt, task.max_tokens)?;
        let baseline_ms = baseline_started.elapsed().as_secs_f64() * 1000.0;
        model.begin_ffn_capture_task(&task.id, &prompt_ids)?;
        let capture_started = Instant::now();
        let captured = tasks::generate(&model, &prompt, task.max_tokens)?;
        let capture_ms = capture_started.elapsed().as_secs_f64() * 1000.0;
        ensure!(
            baseline.token_ids[..baseline.prompt_token_len] == prompt_ids
                && captured.token_ids[..captured.prompt_token_len] == prompt_ids
                && baseline.token_ids == captured.token_ids
                && baseline.output == captured.output
                && baseline.metrics.stop_reason == captured.metrics.stop_reason,
            "FFN capture changed actual task generation: {}",
            task.id
        );
        let generated_ids = &captured.token_ids[captured.prompt_token_len..];
        ensure!(
            captured.metrics.generated_tokens == generated_ids.len(),
            "FFN capture token accounting mismatch"
        );
        model.finish_ffn_capture_task(generated_ids)?;
        report.append(json!({"kind":"ffn_capture_task","id":task.id,"family":task.family,"rendered_prompt":prompt,
            "prompt_token_ids":prompt_ids,"max_tokens":task.max_tokens,"scoring_rule":task.scoring,
            "baseline":tasks::generation_record(&baseline,baseline_ms,&model)?,"captured":tasks::generation_record(&captured,capture_ms,&model)?,
            "score":task.scoring.score(&captured.output)?,"has_identical_generation":true,
            "captured_rows_per_layer":generated_ids.len().saturating_sub(1).min(ROW_CAP),"is_performance_measurement":false}))?;
        eprintln!(
            "FFN capture {}: {} identical generated IDs, {} input rows/layer",
            task.id,
            generated_ids.len(),
            generated_ids.len().saturating_sub(1).min(ROW_CAP)
        );
    }
    let artifact = model.finish_ffn_capture()?;
    artifact.plan.validate_corpus(&corpus, &corpus_sha256)?;
    ensure!(
        artifact.layers.iter().all(|layer| layer.rows.len() >= 4),
        "FFN capture lacks four genuine rows for benchmark"
    );
    let destination = args
        .capture_ffn_inputs
        .as_ref()
        .context("FFN artifact destination")?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    serde_json::to_writer_pretty(&mut file, &artifact)?;
    drop(file);
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial_device_bytes,
        "FFN capture leaked tracked device allocations"
    );
    report.append(json!({"kind":"ffn_capture_complete","is_complete":true,"artifact":destination,
        "artifact_sha256":records::file_sha256(destination)?,"artifact_file_bytes":fs::metadata(destination)?.len(),
        "rows_per_layer":artifact.layers.iter().map(|layer| json!({"layer_id":layer.source.layer_id,"rows":layer.rows.len()})).collect::<Vec<_>>(),
        "observation_payload_bytes":artifact.observation_payload_bytes,"observation_capacity_bytes":artifact.observation_capacity_bytes,
        "read_scratch_bytes":4*(artifact.plan.d_model+artifact.plan.hidden_dim),
        "host_payload_budget_bytes":HOST_BUDGET,"is_performance_measurement":false,
        "device_bytes_before":initial_device_bytes,"device_bytes_after":CudaContext::total_device_bytes()}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Capture never touches a model for an invalid split, context, reused
    /// artifact or aliased report destination. Rejection preserves prior data.
    #[test]
    fn test_capture_request_rejects_leakage_and_destination_conflicts() -> Result<()> {
        let artifact = crate::test_files::TestReportPath::new();
        let report = crate::test_files::TestReportPath::new();
        let parse = |split: &str, context: &str| -> Result<Args> {
            Ok(Args::try_parse_from([
                "model_quality",
                "--model",
                "source.gguf",
                "--reference-model",
                "source.gguf",
                "--variant-id",
                "paired-inputs",
                "--capture-ffn-inputs",
                artifact.path().to_str().unwrap(),
                "--output",
                report.path().to_str().unwrap(),
                "--split",
                split,
                "--context-tokens",
                context,
            ])?)
        };
        validate_request(&parse("calibration", "1024")?)?;
        for (split, context) in [
            ("held-out", "1024"),
            ("all", "1024"),
            ("calibration", "512"),
        ] {
            assert!(validate_request(&parse(split, context)?).is_err());
        }
        let mut alias = parse("calibration", "1024")?;
        alias.output = artifact
            .path()
            .parent()
            .unwrap()
            .join(".")
            .join(artifact.path().file_name().unwrap());
        assert!(validate_request(&alias).is_err());
        fs::write(artifact.path(), b"preserve earlier evidence")?;
        assert!(validate_request(&parse("calibration", "1024")?).is_err());
        assert_eq!(fs::read(artifact.path())?, b"preserve earlier evidence");
        assert!(!report.path().exists());
        Ok(())
    }

    /// The control is explicit and immutable; prepared down/vocabulary must not
    /// be accidentally replaced by ambient preparation, changed fusion or prefill.
    #[test]
    fn test_capture_requires_qualified_control_configuration() -> Result<()> {
        let mut environment = BTreeMap::from(
            [
                ("M40LLM_Q8_DECODE_STREAM", "decode"),
                ("M40LLM_ENABLE_CUBLAS", "1"),
                ("M40LLM_Q8_DECODE_COL_TILE", "4"),
                ("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4"),
                ("M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK", "4"),
                ("M40LLM_FUSED_Q8_QKV", "1"),
                ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
                ("M40LLM_CUDA_GREEDY_ARGMAX", "1"),
                ("M40LLM_Q8_LM_HEAD_ARGMAX", "1"),
                ("M40LLM_Q8_LM_HEAD_ARGMAX_SHARED_A", "1"),
                ("M40LLM_ATTN_HEAD128_KERNEL", "parallel"),
                ("M40LLM_ATTN_HEAD128_PARTITIONS", "8"),
                ("M40LLM_ATTN_HEAD128_THREADS", "256"),
            ]
            .map(|(key, value)| (key.to_owned(), value.to_owned())),
        );
        let settings = execution_settings(&environment)?;
        assert_eq!(
            settings["prepared_weights"]["families"],
            json!(["down", "vocabulary"])
        );
        assert_eq!(settings["max_generated_input_rows_per_task"], 4);
        for key in [
            "M40LLM_ENABLE_CUBLAS",
            "M40LLM_Q8_DECODE_COL_TILE",
            "M40LLM_FUSED_Q8_MLP_SWIGLU",
        ] {
            let original = environment.insert(key.to_owned(), "0".to_owned()).unwrap();
            assert!(execution_settings(&environment).is_err());
            environment.insert(key.to_owned(), original);
        }
        environment.insert("M40LLM_PREFILL_CHUNK_SIZE".to_owned(), "64".to_owned());
        assert!(execution_settings(&environment).is_err());
        Ok(())
    }
}

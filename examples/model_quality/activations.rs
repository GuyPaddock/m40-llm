use super::{
    records::{self, Report},
    run, tasks, Args, SplitChoice,
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaContext,
    evaluation::corpus::{Corpus, Split},
    infer::{
        activation_sparsity::{ActivationCalibration, ActivationCalibrationPlan},
        LoadedModel,
    },
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub(super) struct AppliedCalibration {
    pub artifact: ActivationCalibration,
    path: PathBuf,
    hash: String,
    target: f64,
    budget_bytes: usize,
}

impl AppliedCalibration {
    pub fn describe(&self) -> Value {
        json!({"kind":"calibrated_activation_sparsity","artifact":self.path,
            "artifact_sha256":self.hash,"target_sparsity":self.target,"extra_payload_budget_bytes":self.budget_bytes,
            "layout":"Q8 input-tiled width16 threads128; zero thresholds retain native execution",
            "fit_split":"calibration","is_model_semantics_change":self.target != 0.0})
    }
}

pub(super) enum ActivationSetup<'a> {
    Disabled,
    Apply(&'a AppliedCalibration),
    Capture {
        plan: ActivationCalibrationPlan,
        budget_bytes: usize,
    },
}

impl ActivationSetup<'_> {
    pub fn configure(self, model: &mut LoadedModel) -> Result<()> {
        match self {
            Self::Disabled => Ok(()),
            Self::Apply(value) => model.apply_activation_calibration(
                &value.artifact,
                value.target,
                value.budget_bytes,
            ),
            Self::Capture { plan, budget_bytes } => {
                model.begin_activation_calibration(plan, budget_bytes)
            }
        }
    }
}

fn execution_hash(environment: &BTreeMap<String, String>, context: u32) -> Result<String> {
    records::sha256(&serde_json::to_vec(
        &json!({"environment":environment,"context_tokens":context,
        "reference":"native full-layer Q8, explicit dense sequential prefill/decode, greedy top_k=1 seed=0",
        "task_template":"canonical Qwen system+user ChatML; Llama3 auto chat"}),
    )?)
}

fn validate_execution(environment: &BTreeMap<String, String>) -> Result<()> {
    ensure!(
        environment
            .get("M40LLM_Q8_DECODE_STREAM")
            .map(String::as_str)
            == Some("decode"),
        "activation study requires M40LLM_Q8_DECODE_STREAM=decode"
    );
    for key in [
        "M40LLM_PREFILL_CHUNK_SIZE",
        "M40LLM_KV_COMPRESSED_PREFILL_CHUNK_SIZE",
    ] {
        ensure!(
            environment.get(key).is_none_or(|value| value == "0"),
            "activation study requires sequential prefill; unset {key} or set 0"
        );
    }
    Ok(())
}

pub(super) fn load_variant(
    args: &Args,
    environment: &BTreeMap<String, String>,
    corpus: &Corpus,
    corpus_bytes: &[u8],
) -> Result<Option<AppliedCalibration>> {
    let Some(path) = &args.activation_calibration else {
        return Ok(None);
    };
    validate_execution(environment)?;
    let text = fs::read_to_string(path)?;
    let artifact = ActivationCalibration::parse(&text)?;
    let model_hash = records::file_sha256(&args.model)?;
    ensure!(
        artifact.plan.model_sha256 == model_hash
            && records::file_sha256(&args.reference_model)? == model_hash,
        "activation artifact must match both original reference/candidate GGUF files"
    );
    ensure!(
        artifact.plan.corpus_sha256 == records::sha256(corpus_bytes)?
            && artifact.plan.execution_sha256 == execution_hash(environment, args.context_tokens)?
            && artifact.plan.calibration_task_ids
                == corpus
                    .calibration
                    .tasks
                    .iter()
                    .map(|task| task.id.clone())
                    .collect::<Vec<_>>(),
        "activation artifact corpus/execution/calibration-task identity mismatch"
    );
    let target = args
        .activation_target
        .context("provide --activation-target explicitly")?;
    artifact.thresholds(target)?;
    Ok(Some(AppliedCalibration {
        artifact,
        path: path.clone(),
        hash: records::sha256(text.as_bytes())?,
        target,
        budget_bytes: args
            .activation_budget_mb
            .checked_mul(1024 * 1024)
            .context("activation budget overflow")?,
    }))
}

pub(super) fn capture(args: &Args) -> Result<()> {
    ensure!(
        matches!(args.split, SplitChoice::Calibration)
            && args.skip_layers.is_empty()
            && args.activation_calibration.is_none()
            && args.activation_target.is_none()
            && args.step_timing_trials == 0,
        "capture requires --split calibration with no layer/activation variants or timing trials"
    );
    let environment = run::environment()?;
    validate_execution(&environment)?;
    let corpus_bytes = fs::read(&args.corpus)?;
    let corpus = Corpus::parse(std::str::from_utf8(&corpus_bytes)?)?;
    let model_hash = records::file_sha256(&args.model)?;
    ensure!(
        records::file_sha256(&args.reference_model)? == model_hash,
        "capture reference/model must name the same original weights"
    );
    let plan = ActivationCalibrationPlan::new(
        model_hash,
        records::sha256(&corpus_bytes)?,
        execution_hash(&environment, args.context_tokens)?,
        &corpus,
        args.activation_capture_rows,
    )?;
    let destination = args
        .capture_activations
        .as_ref()
        .context("capture artifact destination")?;
    ensure!(
        !destination.exists(),
        "refuse to overwrite calibration artifact"
    );
    let mut report = Report::create(&args.output)?;
    let initial_device_bytes = CudaContext::total_device_bytes();
    let load_started = Instant::now();
    let (mut model, tokenizer) = run::load_model(
        &args.model,
        args.context_tokens,
        &[],
        ActivationSetup::Capture {
            plan: plan.clone(),
            budget_bytes: args
                .activation_observation_budget_mb
                .checked_mul(1024 * 1024)
                .context("observation budget overflow")?,
        },
    )?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    run::validate_inputs(
        &corpus,
        &[Split::Calibration],
        &tokenizer,
        &tokenizer,
        args.context_tokens,
        model.model_config.vocab_size,
    )?;
    report.append(json!({"kind":"activation_calibration_manifest","schema_version":1,
        "created_unix_s":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "plan":plan,"model":run::model_identity(&args.model,&model,&tokenizer)?,
        "environment":environment,"revision":records::command_text("git",&["rev-parse","HEAD"])?,
        "worktree":records::command_text("git",&["status","--short"])?,
        "executable_sha256":records::file_sha256(&std::env::current_exe()?)?,
        "rustc":records::command_text("rustc",&["--version"])?,"nvcc":records::command_text("nvcc",&["--version"])?,
        "cold_load_and_capture_setup_ms":load_ms,"is_timed":false,
        "scope":"calibration-only generated-input capture; D2H barriers are diagnostic, not generation speed"}))?;
    tasks::warmup(&model, &tokenizer)?;
    for task in &corpus.calibration.tasks {
        let (prompt, expected_ids) = tasks::encode_task(&tokenizer, &task.prompt)?;
        model.begin_activation_calibration_task(&task.id, expected_ids.len())?;
        let started = Instant::now();
        let generated = tasks::generate(&model, &prompt, task.max_tokens)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        ensure!(
            generated.token_ids[..generated.prompt_token_len] == expected_ids
                && generated.metrics.generated_tokens
                    == generated.token_ids.len() - generated.prompt_token_len,
            "capture generation token accounting mismatch"
        );
        model.finish_activation_calibration_task(generated.metrics.generated_tokens)?;
        report.append(
            json!({"kind":"activation_calibration_task","id":task.id,"family":task.family,
            "prompt_token_ids":expected_ids,"rendered_prompt":prompt,"max_tokens":task.max_tokens,
            "generated":tasks::generation_record(&generated,elapsed_ms,&model)?,
            "score":task.scoring.score(&generated.output)?,"is_timed":false}),
        )?;
        eprintln!(
            "calibration {}: {} generated IDs, at most {} observed input rows",
            task.id, generated.metrics.generated_tokens, args.activation_capture_rows
        );
    }
    let fit_started = Instant::now();
    let artifact = model.finish_activation_calibration()?;
    let fit_ms = fit_started.elapsed().as_secs_f64() * 1000.0;
    let mut artifact_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    serde_json::to_writer_pretty(&mut artifact_file, &artifact)?;
    drop(artifact_file);
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial_device_bytes,
        "activation capture allocation leak"
    );
    report.append(
        json!({"kind":"activation_calibration_complete","artifact":destination,
        "artifact_sha256":records::file_sha256(destination)?,"fit_elapsed_ms":fit_ms,
        "observation_payload_bytes":artifact.observation_payload_bytes,
        "observation_capacity_bytes":artifact.observation_capacity_bytes,
        "device_bytes_after":CudaContext::total_device_bytes(),"is_complete":true}),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Base execution identity is deterministic and binds both kernel flags and
    /// context capacity; sequential decode is required before model loading.
    #[test]
    fn test_activation_execution_identity_and_sequential_preflight() -> Result<()> {
        let mut environment =
            BTreeMap::from([("M40LLM_Q8_DECODE_STREAM".to_owned(), "decode".to_owned())]);
        validate_execution(&environment)?;
        let original = execution_hash(&environment, 1024)?;
        assert_eq!(original, execution_hash(&environment, 1024)?);
        assert_ne!(original, execution_hash(&environment, 2048)?);
        environment.insert("M40LLM_Q8_DECODE_COL_TILE".to_owned(), "4".to_owned());
        assert_ne!(original, execution_hash(&environment, 1024)?);
        environment.insert("M40LLM_PREFILL_CHUNK_SIZE".to_owned(), "64".to_owned());
        assert!(validate_execution(&environment).is_err());
        environment.remove("M40LLM_PREFILL_CHUNK_SIZE");
        environment.insert("M40LLM_Q8_DECODE_STREAM".to_owned(), "prefill".to_owned());
        assert!(validate_execution(&environment).is_err());
        Ok(())
    }
}

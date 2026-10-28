//! One bounded full-target experiment. No drafting or generation integration.
mod correctness;
mod measurement;
use crate::run::verification::protocol;

use crate::run::{command_output, inspect_gpu, write_record, Args};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaContext,
    evaluation::identity::{file_sha256, sha256},
    gguf,
    infer::{
        execution_weights::Q8ExecutionConfig, q8_prefill::Q8PrefillBackend, LoadedModel,
        Q8AttentionProjectionBackend, Q8FfnConfig,
    },
    tokenizer::Tokenizer,
    verification::VerificationSession,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    path::Path,
};

const REFERENCE: &str = "docs/experiments/2026-10-27-compound-reference.json";
const QUALITY: &str = "docs/experiments/2026-10-27-maxwell-compound-runtime-quality.jsonl";

struct Trace {
    id: String,
    prompt: Vec<u32>,
    tokens: Vec<u32>,
    stop_reason: String,
    output: String,
}
struct Frozen {
    browser: Trace,
    tasks: Vec<Trace>,
    stop_ids: Vec<u32>,
    reference: Value,
}

fn read_frozen() -> Result<Frozen> {
    let reference: Value = serde_json::from_slice(&fs::read(REFERENCE)?)?;
    ensure!(
        file_sha256(Path::new(QUALITY))? == reference["artifact_sha256"][QUALITY],
        "frozen quality changed"
    );
    let rows = fs::read_to_string(QUALITY)?
        .lines()
        .map(serde_json::from_str)
        .collect::<serde_json::Result<Vec<Value>>>()?;
    let browser = rows
        .iter()
        .find(|r| r["kind"] == "q8_runtime_browser_generation" && r["label"] == "candidate")
        .context("missing frozen browser")?;
    let trace = |id: String, prompt: &Value, result: &Value| -> Result<Trace> {
        Ok(Trace {
            id,
            prompt: serde_json::from_value(prompt.clone())?,
            tokens: serde_json::from_value(result["generated_token_ids"].clone())?,
            stop_reason: result["metrics"]["stop_reason"]
                .as_str()
                .context("missing stop")?
                .into(),
            output: result["output"].as_str().context("missing output")?.into(),
        })
    };
    let browser = trace(
        "browser".into(),
        &browser["prompt_token_ids"],
        &browser["generation"],
    )?;
    let tasks = rows
        .iter()
        .filter(|r| r["kind"] == "model_quality_task")
        .map(|r| {
            trace(
                r["id"].as_str().unwrap().into(),
                &r["prompt_token_ids"],
                &r["candidate"],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        browser.prompt.len() == 65
            && browser.tokens.len() == 512
            && browser.stop_reason == "length"
            && tasks.len() == 12,
        "wrong frozen workload"
    );
    Ok(Frozen {
        browser,
        tasks,
        stop_ids: serde_json::from_value(rows[0]["candidate"]["stop_ids"].clone())?,
        reference,
    })
}

fn new_session<'m>(
    model: &'m mut LoadedModel,
    stops: &[u32],
    is_direct: bool,
) -> Result<VerificationSession<'m>> {
    if is_direct {
        VerificationSession::new_direct_four_rows(model, stops)
    } else {
        VerificationSession::new(model, stops)
    }
}

fn check_gpu() -> Result<Value> {
    let state = inspect_gpu()?;
    let fields: Vec<_> = state["device"]
        .as_str()
        .context("missing GPU state")?
        .split(',')
        .map(str::trim)
        .collect();
    ensure!(
        fields.len() == 9
            && fields[0] == "GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852"
            && fields[3] == "P0"
            && fields[5].trim_end_matches(" MHz") == "1113"
            && fields[6].trim_end_matches(" MHz") == "3004",
        "GPU clocks/identity changed: {state}"
    );
    Ok(state)
}

fn target_dispatch(
    before: &m40_llm::profile::ProfileSnapshot,
    after: &m40_llm::profile::ProfileSnapshot,
) -> BTreeMap<&'static str, u64> {
    after
        .by_op
        .iter()
        .filter_map(|(name, c)| {
            let old = before.by_op.get(name).copied().unwrap_or_default();
            (c.launches > old.launches).then_some((*name, c.launches - old.launches))
        })
        .collect()
}

fn require_target_dispatch(dispatch: &BTreeMap<&str, u64>, is_direct: bool) -> Result<()> {
    for (name, count) in [
        (
            "ffn_q8_joint_four_rows_128",
            if is_direct { 127 * 36 } else { 0 },
        ),
        (
            "ffn_q8_joint_f32x4_shared_scales_128",
            if is_direct { 3 * 36 } else { 511 * 36 },
        ),
        ("qkv_f32xq8_0_gguf_packed_warp", 511 * 36),
        ("gemm_f32xq8_0_gguf_coltile4_packed_warp", 511 * 36),
        ("gemm_f32xq8_separate_f32", 511 * 37),
    ] {
        ensure!(
            dispatch.get(name).copied().unwrap_or(0) == count,
            "wrong qualified target dispatch {name}"
        );
    }
    Ok(())
}

fn validate_correctness(rows: &[Value], frozen: &Frozen, executable: &str) -> Result<()> {
    ensure!(rows.len() == 36, "incomplete correctness records");
    let manifest = &rows[0];
    ensure!(
        manifest["kind"] == "direct_verifier_manifest"
            && manifest["executable_sha256"] == executable
            && manifest["correctness_only"] == true
            && manifest["quality_sha256"] == frozen.reference["artifact_sha256"][QUALITY]
            && manifest["model_sha256"] == frozen.reference["model"]["sha256"]
            && rows
                .last()
                .is_some_and(|r| r["kind"] == "direct_verifier_complete"
                    && r["is_correctness_passed"] == true
                    && r["device_bytes_after"] == 0),
        "correctness binding/completion differs"
    );
    let browser = rows
        .iter()
        .filter(|r| r["kind"] == "direct_verifier_browser_correctness")
        .collect::<Vec<_>>();
    ensure!(browser.len() == 1, "missing browser correctness");
    let b = browser[0];
    ensure!(
        b["actual_token_ids"] == json!(frozen.browser.tokens)
            && b["has_exact_valid_kv"] == true
            && b["valid_kv_words"] == 36 * 576 * 512
            && b["target_counters"]["cublas_calls"] == 0,
        "browser actual IDs/KV/target path differ"
    );
    let logits = b["full_logit_comparisons"]
        .as_array()
        .context("missing full logits")?;
    ensure!(logits.len() == 511, "missing full-logit rows");
    for (index, l) in logits.iter().enumerate() {
        ensure!(
            l["input_row"] == index
                && l["actual_next"] == frozen.browser.tokens[index + 1]
                && l["expected_next"] == l["actual_next"]
                && l["comparison"]["values"] == 151936
                && l["comparison"]["has_tolerance_passed"] == true,
            "full-logit alignment differs"
        );
    }
    for partial in [false, true] {
        for accepted in 0..4 {
            ensure!(
                rows.iter()
                    .filter(|r| r["kind"] == "direct_verifier_transaction"
                        && r["accepted"] == accepted
                        && r["is_forced_partial"] == partial
                        && r["has_exact_retained_kv"] == true
                        && r["has_cleared_suffix_before_correction"] == true
                        && r["has_exact_correction"] == true)
                    .count()
                    == 1,
                "missing transaction case"
            );
        }
    }
    for task in &frozen.tasks {
        for direct in [false, true] {
            ensure!(
                rows.iter()
                    .filter(|r| r["kind"] == "direct_verifier_quality"
                        && r["id"] == task.id
                        && r["is_direct"] == direct
                        && r["actual_token_ids"] == json!(task.tokens)
                        && r["output"] == task.output
                        && r["stop_reason"] == task.stop_reason
                        && r["has_identical_generation"] == true)
                    .count()
                    == 1,
                "missing frozen task correctness"
            );
        }
    }
    ensure!(
        rows.iter()
            .filter(|r| r["kind"] == "direct_verifier_invalid_cases"
                && r["has_rejected_alias"] == true
                && r["has_drop_rollback"] == true
                && r["has_rejected_unverified_commit"] == true
                && r["has_rejected_invalid_width_and_id"] == true
                && r["has_stop_cap"] == true
                && r["has_rejected_capacity"] == true)
            .count()
            == 1,
        "missing ownership/invalid cases"
    );
    Ok(())
}

pub(super) fn benchmark(args: &Args) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions)
            && cfg!(have_cublas)
            && args.rows == [4]
            && args.iterations == 10
            && args.trials == 5,
        "direct verifier requires release/cuBLAS, rows4, iterations10, initial trials5; statistical extensions are automatic"
    );
    let frozen = read_frozen()?;
    let environment: BTreeMap<_, _> = std::env::vars()
        .filter(|(k, _)| k.starts_with("M40LLM_") || k == "CUDA_VISIBLE_DEVICES")
        .collect();
    ensure!(
        serde_json::to_value(&environment)? == frozen.reference["configuration"]["environment"],
        "restore exact qualified environment"
    );
    ensure!(
        file_sha256(&args.model)? == frozen.reference["model"]["sha256"],
        "model hash differs"
    );
    ensure!(
        command_output("git", &["status", "--short"])?.is_empty(),
        "research requires clean committed source"
    );
    let executable_sha256 = file_sha256(&std::env::current_exe()?)?;
    if !args.direct_verifier_correctness_only {
        let path = args
            .direct_verifier_correctness_report
            .as_ref()
            .context("provide passing --direct-verifier-correctness-report")?;
        let rows = fs::read_to_string(path)?
            .lines()
            .map(serde_json::from_str)
            .collect::<serde_json::Result<Vec<Value>>>()?;
        validate_correctness(&rows, &frozen, &executable_sha256)?;
    }
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    let initial = CudaContext::total_device_bytes();
    let gguf = gguf::load_gguf(&args.model)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata)?;
    let mut model = LoadedModel::from_gguf_with_q8_runtime(
        gguf,
        fs::read(&args.model)?,
        -1,
        Q8ExecutionConfig::parse("split", Some("6144"), Some("gate-up,down,vocabulary"))?,
        Q8FfnConfig::joint_shared_scales(),
        Q8PrefillBackend::DequantSgemm,
        Q8AttentionProjectionBackend::PackedWarp256,
    )?;
    model.allocate_kv_cache_for_layers(1024)?;
    let prepared = serde_json::to_value(model.q8_execution_stats())?;
    let joint = serde_json::to_value(model.q8_ffn_pair_stats())?;
    let joint_hashes = model.q8_execution.debug_joint_payload_hashes()?;
    write_record(
        &mut out,
        json!({"kind":"direct_verifier_manifest","schema_version":1,
        "revision":command_output("git",&["rev-parse","HEAD"])?,"worktree":"","executable_sha256":executable_sha256,
        "model_sha256":frozen.reference["model"]["sha256"],"quality_sha256":file_sha256(Path::new(QUALITY))?,
        "environment":environment,"prepared":prepared,"joint":joint,"joint_payload_hashes":joint_hashes,
        "logical_sources":model.q8_execution.logical_sources(),"source_bytes":model.weights_len,
        "dense_kv_bytes":model.kv_cache.as_ref().unwrap().actual_bytes(),"correctness_only":args.direct_verifier_correctness_only,
        "trials":args.trials,"rows":4,"build_profile":"release","nvcc":command_output("nvcc",&["--version"])?,
        "cuda_flags":["-O3","sm_52","compute_52","no fast math"],"gpu":inspect_gpu()?,
        "scope":"fully direct target research; shared prompt SGEMM permitted, target SGEMM prohibited; no HTTP claim"}),
    )?;
    if args.direct_verifier_correctness_only {
        correctness::run(&mut model, &frozen, &tokenizer, &mut out)?;
    } else {
        measurement::run(&mut model, &frozen, &mut out)?;
    }
    ensure!(
        serde_json::to_value(model.q8_execution_stats())? == prepared
            && serde_json::to_value(model.q8_ffn_pair_stats())? == joint
            && model.q8_execution.debug_joint_payload_hashes()? == joint_hashes,
        "prepared ownership/payloads changed"
    );
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "direct verifier leaked"
    );
    write_record(
        &mut out,
        json!({"kind":"direct_verifier_complete","is_complete":true,
        "is_correctness_passed":args.direct_verifier_correctness_only,"device_bytes_after":CudaContext::total_device_bytes(),
        "is_runtime_claim":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// A research target cannot accidentally select the rejected verifier or FFN experiment.
    #[test]
    fn test_direct_cli_is_exclusive() {
        let args = [
            "b",
            "--model",
            "m",
            "--output",
            "o",
            "--direct-four-row-verifier",
        ];
        assert!(Args::try_parse_from(args).is_ok());
        for extra in [["--verification-trace", "t"], ["--ffn-inputs", "f"]] {
            assert!(Args::try_parse_from(args.into_iter().chain(extra)).is_err());
        }
        assert!(Args::try_parse_from(args.into_iter().chain(["--direct-q8-feasibility"])).is_err());
    }

    /// Completed headers alone cannot admit missing tasks, transaction cases,
    /// vocabulary rows, wrong targets or failed numerical evidence to timing.
    #[test]
    fn test_correctness_admission_requires_complete_behavior() -> Result<()> {
        let frozen = read_frozen()?;
        let mut rows = vec![
            json!({"kind":"direct_verifier_manifest","executable_sha256":"unit",
            "correctness_only":true,"quality_sha256":frozen.reference["artifact_sha256"][QUALITY],
            "model_sha256":frozen.reference["model"]["sha256"]}),
            json!({"kind":"direct_verifier_browser_correctness","actual_token_ids":frozen.browser.tokens,
                "has_exact_valid_kv":true,"valid_kv_words":36*576*512,"target_counters":{"cublas_calls":0},
                "full_logit_comparisons":(0..511).map(|i|json!({"input_row":i,"actual_next":frozen.browser.tokens[i+1],
                    "expected_next":frozen.browser.tokens[i+1],"comparison":{"values":151936,"has_tolerance_passed":true}})).collect::<Vec<_>>()}),
        ];
        for partial in [false, true] {
            for accepted in 0..4 {
                rows.push(json!({"kind":"direct_verifier_transaction","accepted":accepted,
                "is_forced_partial":partial,"has_exact_retained_kv":true,"has_cleared_suffix_before_correction":true,
                "has_exact_correction":true}));
            }
        }
        rows.push(json!({"kind":"direct_verifier_invalid_cases","has_rejected_alias":true,"has_drop_rollback":true,
            "has_rejected_unverified_commit":true,"has_rejected_invalid_width_and_id":true,"has_stop_cap":true,"has_rejected_capacity":true}));
        for task in &frozen.tasks {
            for direct in [false, true] {
                rows.push(json!({"kind":"direct_verifier_quality","id":task.id,"is_direct":direct,
                "actual_token_ids":task.tokens,"output":task.output,"stop_reason":task.stop_reason,"has_identical_generation":true}));
            }
        }
        rows.push(json!({"kind":"direct_verifier_complete","is_correctness_passed":true,"device_bytes_after":0}));
        validate_correctness(&rows, &frozen, "unit")?;
        for remove in [1, 2, 10, 11] {
            let mut changed = rows.clone();
            changed.remove(remove);
            assert!(validate_correctness(&changed, &frozen, "unit").is_err());
        }
        let mut changed = rows.clone();
        changed[1]["full_logit_comparisons"][0]["comparison"]["has_tolerance_passed"] =
            json!(false);
        assert!(validate_correctness(&changed, &frozen, "unit").is_err());
        let mut changed = rows.clone();
        changed[1]["actual_token_ids"][0] = json!(0);
        assert!(validate_correctness(&changed, &frozen, "unit").is_err());
        assert!(validate_correctness(&rows, &frozen, "wrong-executable").is_err());
        Ok(())
    }
}

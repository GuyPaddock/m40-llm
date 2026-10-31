use super::simulate::Baseline;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

pub const SELECTED_ENV: &[(&str, &str)] = &[
    ("M40LLM_Q8_EXECUTION_WEIGHTS", "off"),
    ("M40LLM_Q8_DECODE_STREAM", "decode"),
    ("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4"),
    ("M40LLM_Q8_DECODE_COL_TILE", "4"),
    ("M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK", "4"),
    ("M40LLM_FUSED_Q8_QKV", "1"),
    ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
    ("M40LLM_CUDA_GREEDY_ARGMAX", "1"),
    ("M40LLM_Q8_LM_HEAD_ARGMAX", "1"),
    ("M40LLM_Q8_LM_HEAD_ARGMAX_SHARED_A", "1"),
    ("M40LLM_PROJECTION_BACKEND", "large-model"),
    ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
    ("M40LLM_DECODE_GRAPH", "0"),
    ("M40LLM_ATTN_HEAD128_KERNEL", "parallel"),
    ("M40LLM_ATTN_HEAD128_PARTITIONS", "8"),
    ("M40LLM_ATTN_HEAD128_THREADS", "256"),
];

// Frozen before measuring acceptance. The first three are existing C07 smokes.
pub const REGRESSION_PROMPTS: &[(&str, &str, usize, Option<&str>)] = &[
    ("arithmetic", "What is 2+2? Answer with one digit.", 16, Some("4")),
    ("exact_ok", "Reply with exactly the word OK.", 16, Some("OK")),
    ("config_lookup", "Configuration note: the active value is M40_BATCH_LIMIT=37. What is M40_BATCH_LIMIT? Answer with the number only.", 16, Some("37")),
    ("explanation", "Explain why binary search requires sorted data and describe its time complexity.", 128, None),
    ("java_code", "Write a small Java method that checks whether a string is a palindrome. Include a short explanation.", 128, None),
];

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenTrace {
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub output: String,
}

pub struct Case {
    pub label: String,
    pub trace: TokenTrace,
    pub stop_ids: Vec<u32>,
    pub baseline: Option<Baseline>,
}

pub struct Browser {
    pub case: Case,
    pub model_sha256: String,
    pub configuration: Value,
    pub source: Value,
}

pub fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line)
                .with_context(|| format!("{} line {}", path.display(), index + 1))
        })
        .collect()
}

pub fn command_text(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    ensure!(
        output.status.success(),
        "{program}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

pub fn sha256(bytes: &[u8]) -> Result<String> {
    let mut process = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    process
        .stdin
        .take()
        .context("hash stdin")?
        .write_all(bytes)?;
    let output = process.wait_with_output()?;
    ensure!(output.status.success(), "sha256sum failed");
    Ok(String::from_utf8(output.stdout)?
        .split_whitespace()
        .next()
        .context("missing hash")?
        .to_owned())
}

pub fn configuration(environment: &Value) -> Result<Value> {
    for &(key, expected) in SELECTED_ENV {
        ensure!(
            environment[key].as_str() == Some(expected),
            "selected C11 configuration requires {key}={expected}"
        );
    }
    for key in environment
        .as_object()
        .context("environment object")?
        .keys()
        .filter(|key| key.starts_with("M40LLM_"))
    {
        ensure!(
            SELECTED_ENV.iter().any(|(name, _)| key == name)
                || [
                    "M40LLM_ENABLE_NVCC",
                    "M40LLM_ENABLE_CUBLAS",
                    "M40LLM_SYSROOT"
                ]
                .contains(&key.as_str()),
            "unset extra runtime override {key} for matched study input"
        );
    }
    Ok(
        json!({"environment":SELECTED_ENV.iter().map(|(k,v)|((*k).to_owned(),json!(v))).collect::<serde_json::Map<_,_>>(),
        "build_profile":"release", "kv_mode":"off", "context_tokens":1024,
        "sampling":"greedy top_k=1, seed=0, no penalties", "prompt_format":"raw"}),
    )
}

fn median(values: impl Iterator<Item = f64>) -> Result<f64> {
    let mut values: Vec<_> = values.collect();
    ensure!(
        !values.is_empty() && values.iter().all(|v| v.is_finite() && *v >= 0.0),
        "invalid/missing timings"
    );
    values.sort_by(f64::total_cmp);
    Ok(if values.len().is_multiple_of(2) {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
    } else {
        values[values.len() / 2]
    })
}

/// Require actual IDs, stable measured output, and the selected configuration;
/// do not silently substitute an arbitrary prepared/early-EOS continuation.
pub fn read_browser(path: &Path) -> Result<Browser> {
    let records = read_jsonl(path)?;
    let manifests: Vec<_> = records.iter().filter(|r| r["kind"] == "manifest").collect();
    ensure!(manifests.len() == 1, "need exactly one benchmark manifest");
    let manifest = manifests[0];
    ensure!(
        manifest["build_profile"] == "release"
            && manifest["kv_mode"] == "off"
            && manifest["context_tokens"] == 1024,
        "browser source must be release dense ctx1024"
    );
    let configuration = configuration(&manifest["environment"])?;
    let traces: Vec<_> = records
        .iter()
        .filter(|r| r["kind"] == "token_trace" && r["runtime"] == "m40-llm")
        .collect();
    ensure!(
        traces.len() == 1 && traces[0]["matches_measured_output"] == true,
        "need one verified actual m40 token trace"
    );
    let trace: TokenTrace = serde_json::from_value(traces[0].clone())?;
    let expected_prompt: Vec<u32> = serde_json::from_value(manifest["prompt_token_ids"].clone())?;
    ensure!(
        trace.prompt_token_ids == expected_prompt
            && expected_prompt.len() == 65
            && trace.generated_token_ids.len() == 512,
        "browser token counts/IDs do not match the fixed workload"
    );
    let trials: Vec<_> = records
        .iter()
        .filter(|r| r["kind"] == "trial" && r["runtime"] == "m40-llm" && r["phase"] == "screen")
        .collect();
    ensure!(
        trials.len() >= 5
            && trials.iter().all(|r| r["is_fixed_workload"] == true
                && r["metrics"]["generated_tokens"] == 512
                && r["metrics"]["prompt_tokens"] == 65
                && r["metrics"]["stop_reason"] == "length"
                && r["output"] == trace.output
                && r["cache"]["reused_prompt_tokens"] == 0
                && r["cache"]["evaluated_prompt_tokens"] == 65),
        "source trials are not stable, fixed-workload, fresh-KV browser runs"
    );
    let timing = |field: &str, nested: bool| -> Result<f64> {
        let values: Result<Vec<_>> = trials
            .iter()
            .map(|r| {
                if nested {
                    &r["metrics"][field]
                } else {
                    &r[field]
                }
            })
            .map(|v| v.as_f64().context("missing numeric timing"))
            .collect();
        median(values?.into_iter())
    };
    let baseline = Baseline {
        http_elapsed_ms: timing("client_elapsed_ms", false)?,
        prefill_elapsed_ms: timing("prefill_elapsed_ms", true)?,
        decode_elapsed_ms: timing("decode_elapsed_ms", true)?,
        generated_tokens: 512,
    };
    let stop_ids = serde_json::from_value(manifest["m40_stop_token_ids"].clone())?;
    let model_sha256 = manifest["model_sha256"]
        .as_str()
        .context("missing model hash")?
        .to_owned();
    Ok(Browser {
        case: Case {
            label: "browser".into(),
            trace,
            stop_ids,
            baseline: Some(baseline),
        },
        model_sha256,
        configuration,
        source: json!({"path":path,"sha256":sha256(&fs::read(path)?)?,"manifest":manifest}),
    })
}

pub fn read_regressions(path: &Path, browser: &Browser) -> Result<Vec<Case>> {
    let records = read_jsonl(path)?;
    let manifest = records.first().context("empty regression artifact")?;
    ensure!(
        manifest["kind"] == "ngram_regression_manifest"
            && manifest["model_sha256"] == browser.model_sha256
            && manifest["configuration"] == browser.configuration,
        "regression model/configuration differs from C11 browser reference"
    );
    let mut cases = Vec::new();
    for &(label, prompt, _, expected) in REGRESSION_PROMPTS {
        let matches: Vec<_> = records
            .iter()
            .filter(|r| r["kind"] == "ngram_regression_trace" && r["case"] == label)
            .collect();
        ensure!(
            matches.len() == 1,
            "missing/duplicate regression case {label}"
        );
        let row = matches[0];
        ensure!(
            row["user_prompt"] == prompt,
            "regression prompt changed: {label}"
        );
        ensure!(
            row["is_stable_two_runs"] == true,
            "unstable/unverified regression trace: {label}"
        );
        let trace: TokenTrace = serde_json::from_value(row.clone())?;
        ensure!(
            !trace.prompt_token_ids.is_empty() && !trace.generated_token_ids.is_empty(),
            "empty regression IDs: {label}"
        );
        if let Some(expected) = expected {
            ensure!(
                trace.output.trim() == expected,
                "reference quality smoke failed: {label}"
            );
        }
        cases.push(Case {
            label: label.into(),
            trace,
            stop_ids: serde_json::from_value(row["stop_ids"].clone())?,
            baseline: None,
        });
    }
    Ok(cases)
}

#[cfg(test)]
#[path = "records_tests.rs"]
mod tests;

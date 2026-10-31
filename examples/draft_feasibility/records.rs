use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    io::Write,
    path::Path,
    process::Command,
};

pub const PROMPTS: &str = "docs/experiments/2026-10-28-draft-heldout-prompts.json";
pub const TARGET: &str = "docs/experiments/2026-10-28-direct-verifier-measurement.jsonl";
pub const REFERENCE: &str = "docs/experiments/2026-10-27-compound-reference.json";
const QUALITY: &str = "docs/experiments/2026-10-27-maxwell-compound-runtime-quality.jsonl";

#[derive(Clone, Serialize, Deserialize)]
pub struct Trace {
    pub id: String,
    pub class: String,
    pub split: String,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub stop_ids: Vec<u32>,
    pub stop_reason: String,
    pub output: String,
    pub serial_fixed_ms: Option<f64>,
    pub max_tokens: usize,
}
impl Trace {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.prompt_token_ids.is_empty() && !self.generated_token_ids.is_empty(),
            "empty trace"
        );
        let n = self.generated_token_ids.len();
        ensure!(n <= self.max_tokens, "output exceeds configured budget");
        ensure!(
            self.stop_reason != "length" || n == self.max_tokens,
            "length stop before budget"
        );
        ensure!(
            !self.generated_token_ids[..n - 1]
                .iter()
                .any(|t| self.stop_ids.contains(t)),
            "tokens after stop"
        );
        ensure!(
            matches!(self.stop_reason.as_str(), "stop" | "length"),
            "unknown stop"
        );
        ensure!(
            (self.stop_reason == "stop")
                == self.stop_ids.contains(&self.generated_token_ids[n - 1]),
            "stop accounting"
        );
        Ok(())
    }
}
pub fn rows(path: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .map(|l| Ok(serde_json::from_str(l)?))
        .collect()
}
pub fn command(program: &str, args: &[&str]) -> Result<String> {
    let result = Command::new(program).args(args).output()?;
    ensure!(
        result.status.success(),
        "{program}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(String::from_utf8(result.stdout)?.trim().to_owned())
}
pub fn hash_file(path: &Path) -> Result<String> {
    Ok(
        command("sha256sum", &[path.to_str().context("non-UTF8 path")?])?
            .split_whitespace()
            .next()
            .context("missing digest")?
            .into(),
    )
}
pub fn write(out: &mut File, row: Value) -> Result<()> {
    writeln!(out, "{row}")?;
    out.flush()?;
    Ok(())
}

pub fn read_traces(heldout: Option<&Path>) -> Result<Vec<Trace>> {
    let reference: Value = serde_json::from_slice(&fs::read(REFERENCE)?)?;
    ensure!(
        hash_file(QUALITY.as_ref())? == reference["artifact_sha256"][QUALITY],
        "frozen quality differs"
    );
    let data = rows(QUALITY.as_ref())?;
    let stops = data[0]["candidate"]["stop_ids"].clone();
    let mut traces = Vec::new();
    for r in data {
        let (id, class, prompt, generation, budget) =
            if r["kind"] == "q8_runtime_browser_generation" && r["label"] == "candidate" {
                (
                    "browser".into(),
                    "prose".into(),
                    r["prompt_token_ids"].clone(),
                    r["generation"].clone(),
                    json!(512),
                )
            } else if r["kind"] == "model_quality_task" {
                (
                    r["id"].clone(),
                    r["family"].clone(),
                    r["prompt_token_ids"].clone(),
                    r["candidate"].clone(),
                    r["max_tokens"].clone(),
                )
            } else {
                continue;
            };
        traces.push(serde_json::from_value(json!({"id":id,"class":class,"split":"development",
            "prompt_token_ids":prompt,"generated_token_ids":generation["generated_token_ids"],"stop_ids":stops,
            "stop_reason":generation["metrics"]["stop_reason"],"output":generation["output"],"serial_fixed_ms":null,"max_tokens":budget}))?);
    }
    ensure!(traces.len() == 13, "missing development traces");
    if let Some(path) = heldout {
        let data = rows(path)?;
        ensure!(
            data[0]["kind"] == "draft_capture_manifest"
                && data
                    .last()
                    .is_some_and(|r| r["is_complete"] == true && r["device_bytes_after"] == 0)
                && data[0]["prompt_definitions_sha256"] == hash_file(PROMPTS.as_ref())?
                && data[0]["model_sha256"] == reference["model"]["sha256"],
            "incomplete or changed heldout capture"
        );
        let held: Vec<Trace> = data
            .into_iter()
            .filter(|r| r["kind"] == "draft_target_trace")
            .map(|r| {
                ensure!(r["is_stable_two_runs"] == true, "unstable target trace");
                ensure!(
                    r["repeat_generated_token_ids"] == r["trace"]["generated_token_ids"]
                        && r["repeat_stop_reason"] == r["trace"]["stop_reason"],
                    "repeat evidence differs"
                );
                Ok(serde_json::from_value(r["trace"].clone())?)
            })
            .collect::<Result<_>>()?;
        ensure!(
            held.len() == 8 && held.iter().all(|t| t.split == "heldout"),
            "eight heldout cases required"
        );
        let definitions: Vec<Value> = serde_json::from_slice(&fs::read(PROMPTS)?)?;
        for definition in definitions {
            ensure!(
                held.iter()
                    .filter(|t| json!(t.id) == definition["id"]
                        && json!(t.class) == definition["class"])
                    .count()
                    == 1,
                "missing or duplicate held-out identity"
            );
        }
        traces.extend(held);
    }
    Ok(traces)
}

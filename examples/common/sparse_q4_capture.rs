//! Validate the report and the native Q4 observations before consuming either split.
use anyhow::{ensure, Context, Result};
use m40_llm::evaluation::{
    corpus::{Corpus, Split},
    ffn_capture::FfnCaptureArtifact,
    identity::sha256,
};
use serde_json::{json, Value};
use std::{fs, path::Path};

pub fn load(
    artifact_path: &Path,
    report_path: &Path,
    corpus_path: &Path,
    split: Split,
) -> Result<(FfnCaptureArtifact, String, String)> {
    ensure!(
        fs::metadata(artifact_path)?.len() <= 64 * 1024 * 1024,
        "capture exceeds size budget"
    );
    let bytes = fs::read(artifact_path)?;
    let artifact: FfnCaptureArtifact = serde_json::from_slice(&bytes)?;
    ensure!(
        artifact.plan.is_sparse_q4 && artifact.plan.split == split,
        "expected a sparse Q4 capture for the requested split"
    );
    artifact.validate()?;
    let corpus = fs::read_to_string(corpus_path)?;
    artifact
        .plan
        .validate_corpus(&Corpus::parse(&corpus)?, &sha256(corpus.as_bytes())?)?;
    let digest = sha256(&bytes)?;
    let report = fs::read_to_string(report_path)?;
    let records = report
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()?;
    let first = records.first().context("empty observation report")?;
    let last = records.last().context("missing completion")?;
    ensure!(
        records.len() == artifact.tasks.len() + 2
            && first["kind"] == "ffn_capture_manifest"
            && first["execution_sha256"] == artifact.plan.execution_sha256
            && first["model"]["sha256"] == artifact.plan.model_sha256
            && first["corpus_sha256"] == artifact.plan.corpus_sha256
            && first["split"] == json!(split)
            && last["kind"] == "ffn_capture_complete"
            && last["is_complete"] == true
            && last["artifact_sha256"] == digest
            && last["device_bytes_before"].as_u64().is_some()
            && last["device_bytes_before"] == last["device_bytes_after"],
        "sparse Q4 capture provenance mismatch"
    );
    for (task, row) in artifact.tasks.iter().zip(&records[1..records.len() - 1]) {
        ensure!(
            row["kind"] == "ffn_capture_task"
                && row["id"] == task.id
                && row["prompt_token_ids"] == json!(task.prompt_token_ids)
                && row["has_identical_generation"] == true
                && row["baseline_observer_uninstalled"] == true
                && row["baseline"]["generated_token_ids"] == json!(task.generated_token_ids)
                && row["captured"]["generated_token_ids"] == row["baseline"]["generated_token_ids"]
                && row["captured"]["output"] == row["baseline"]["output"]
                && row["captured"]["metrics"]["stop_reason"]
                    == row["baseline"]["metrics"]["stop_reason"],
            "unpaired Q4 task {}",
            task.id
        );
    }
    Ok((artifact, digest, sha256(report.as_bytes())?))
}

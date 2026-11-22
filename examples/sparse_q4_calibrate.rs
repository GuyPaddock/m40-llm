//! Freeze three layer-local cutoffs before opening the held-out observations.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use m40_llm::evaluation::{
    corpus::{Corpus, Split},
    ffn_capture::FfnCaptureArtifact,
    identity::sha256,
};
use serde_json::{json, Value};
use std::{fs, io::Write, path::PathBuf};

#[derive(Parser)]
struct Args {
    artifact: PathBuf,
    report: PathBuf,
    output: PathBuf,
    #[arg(long, default_value = "data/research_quality/v1.json")]
    corpus: PathBuf,
}

fn cutoffs(values: &[f32]) -> Result<[f32; 3]> {
    ensure!(
        !values.is_empty() && values.iter().all(|v| v.is_finite()),
        "invalid calibration vector"
    );
    let mut magnitudes: Vec<_> = values.iter().map(|v| v.abs()).collect();
    magnitudes.sort_unstable_by(f32::total_cmp);
    let quantile = |percent: usize| magnitudes[(magnitudes.len() * percent).div_ceil(100) - 1];
    Ok([0.0, quantile(50), quantile(75)])
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        fs::metadata(&args.artifact)?.len() <= 64 * 1024 * 1024,
        "capture exceeds size budget"
    );
    let bytes = fs::read(args.artifact)?;
    let artifact: FfnCaptureArtifact = serde_json::from_slice(&bytes)?;
    ensure!(
        artifact.plan.is_sparse_q4 && artifact.plan.split == Split::Calibration,
        "cutoffs require a calibration-only sparse Q4 capture"
    );
    artifact.validate()?;
    let corpus = fs::read_to_string(args.corpus)?;
    artifact
        .plan
        .validate_corpus(&Corpus::parse(&corpus)?, &sha256(corpus.as_bytes())?)?;
    let digest = sha256(&bytes)?;
    let report = fs::read_to_string(args.report)?;
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
            && first["split"] == "calibration"
            && last["kind"] == "ffn_capture_complete"
            && last["is_complete"] == true
            && last["artifact_sha256"] == digest
            && last["device_bytes_before"].as_u64().is_some()
            && last["device_bytes_before"] == last["device_bytes_after"],
        "calibration capture provenance mismatch"
    );
    for (task, row) in artifact.tasks.iter().zip(&records[1..records.len() - 1]) {
        ensure!(
            row["kind"] == "ffn_capture_task"
                && row["id"] == task.id
                && row["prompt_token_ids"] == json!(task.prompt_token_ids)
                && row["has_identical_generation"] == true
                && row["baseline_observer_uninstalled"] == true
                && row["baseline"]["generated_token_ids"] == json!(task.generated_token_ids)
                && row["captured"]["generated_token_ids"] == row["baseline"]["generated_token_ids"],
            "unpaired calibration task {}",
            task.id
        );
    }
    let mut layers = Vec::new();
    for layer in &artifact.layers {
        let thresholds = cutoffs(&layer.swiglu_inputs)?;
        let retained = thresholds.map(|cutoff| {
            layer
                .swiglu_inputs
                .iter()
                .filter(|v| v.abs() > cutoff)
                .count()
        });
        layers.push(
            json!({"layer":layer.source.layer_id,"down":layer.source.down,
            "rows":layer.rows.len(),"values":layer.swiglu_inputs.len(),"cutoffs":thresholds,
            "retained_values":retained}),
        );
    }
    let frozen = json!({"schema_version":1,"model_sha256":artifact.plan.model_sha256,
        "corpus_sha256":artifact.plan.corpus_sha256,"source_weights_sha256":artifact.source_weights_sha256,
        "calibration_artifact_sha256":digest,"calibration_report_sha256":sha256(report.as_bytes())?,
        "quantiles":["zero","nearest-rank 50%","nearest-rank 75%"],"retain":"abs(value) > cutoff","layers":layers});
    let mut output = fs::File::create_new(args.output)?;
    serde_json::to_writer_pretty(&mut output, &frozen)?;
    writeln!(output)?;
    output.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Rank ties stay on the discarded side of the strict GPU comparison.
    #[test]
    fn nearest_rank_resolves_ties_and_preserves_zero_control() -> Result<()> {
        assert_eq!(cutoffs(&[-8., 0., -2., 2.])?, [0., 2., 2.]);
        assert_eq!(cutoffs(&[0.])?, [0.; 3]);
        assert!(cutoffs(&[]).is_err());
        assert!(cutoffs(&[f32::NAN]).is_err());
        Ok(())
    }
}

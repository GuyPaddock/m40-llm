//! Freeze three layer-local cutoffs before opening the held-out observations.
#[path = "common/sparse_q4_capture.rs"]
mod capture;
use anyhow::{ensure, Result};
use clap::Parser;
use m40_llm::evaluation::corpus::Split;
use serde_json::json;
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
    let (artifact, digest, report_digest) = capture::load(
        &args.artifact,
        &args.report,
        &args.corpus,
        Split::Calibration,
    )?;
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
        "calibration_artifact_sha256":digest,"calibration_report_sha256":report_digest,
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

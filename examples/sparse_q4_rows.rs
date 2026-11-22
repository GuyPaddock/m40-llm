//! Bind the held-out real SwiGLU rows to the frozen cutoffs and weight planes.
#[path = "common/sparse_q4_capture.rs"]
mod capture;
use anyhow::{ensure, Context, Result};
use clap::Parser;
use m40_llm::evaluation::{
    corpus::Split,
    identity::{file_sha256, sha256},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
};

#[derive(Parser)]
struct Args {
    artifact: PathBuf,
    report: PathBuf,
    thresholds: PathBuf,
    planes: PathBuf,
    output: PathBuf,
    #[arg(long, default_value = "data/research_quality/v1.json")]
    corpus: PathBuf,
}

#[derive(Deserialize, Serialize)]
struct LayerThresholds {
    layer: usize,
    down: m40_llm::evaluation::ffn_capture::FfnTensorSource,
    cutoffs: [f32; 3],
}

#[derive(Deserialize)]
struct Thresholds {
    schema_version: u32,
    model_sha256: String,
    corpus_sha256: String,
    source_weights_sha256: String,
    layers: Vec<LayerThresholds>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (artifact, artifact_sha, report_sha) =
        capture::load(&args.artifact, &args.report, &args.corpus, Split::HeldOut)?;
    let frozen = fs::read(&args.thresholds)?;
    let fit: Thresholds = serde_json::from_slice(&frozen)?;
    ensure!(
        fit.schema_version == 1
            && fit.model_sha256 == artifact.plan.model_sha256
            && fit.corpus_sha256 == artifact.plan.corpus_sha256
            && fit.source_weights_sha256 == artifact.source_weights_sha256
            && fit.layers.len() == artifact.layers.len(),
        "calibration and held-out identities differ"
    );
    fs::create_dir(&args.output)?;
    let mut layers = Vec::new();
    for (layer, fitted) in artifact.layers.iter().zip(fit.layers) {
        let id = layer.source.layer_id;
        ensure!(
            id == fitted.layer
                && layer.source.down == fitted.down
                && fitted.cutoffs[0].to_bits() == 0
                && fitted.cutoffs.iter().all(|x| x.is_finite())
                && fitted.cutoffs.windows(2).all(|x| x[0] <= x[1]),
            "invalid layer cutoff identity"
        );
        let weights = args.planes.join(id.to_string()).join("native");
        ensure!(
            file_sha256(&weights)? == layer.source.down.source_sha256,
            "down plane differs from observed layer"
        );
        let path = args.output.join(format!("{id}.f32le"));
        let mut writer = BufWriter::new(fs::File::create_new(&path)?);
        for value in &layer.swiglu_inputs {
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.flush()?;
        let mut rows = Vec::new();
        for row in &layer.rows {
            let task = artifact
                .tasks
                .iter()
                .find(|task| task.id == row.task_id)
                .context("missing task")?;
            let offset = (row.absolute_position as usize)
                .checked_sub(task.prompt_token_ids.len())
                .context("row in prompt")?;
            rows.push(json!({"task":task.id,"position":row.absolute_position,"token":task.generated_token_ids[offset]}));
        }
        layers.push(json!({"layer":id,"cutoffs":fitted.cutoffs,"rows":rows,"vector_sha256":file_sha256(&path)?,
            "down_sha256":layer.source.down.source_sha256}));
    }
    let manifest = json!({"schema_version":1,"model_sha256":fit.model_sha256,"cutoffs_sha256":sha256(&frozen)?,
        "held_out_artifact_sha256":artifact_sha,"held_out_report_sha256":report_sha,"layers":layers});
    serde_json::to_writer_pretty(
        fs::File::create_new(args.output.join("manifest.json"))?,
        &manifest,
    )?;
    Ok(())
}

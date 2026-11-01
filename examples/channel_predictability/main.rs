//! Fixed offline diagnostics. Never dispatches a CUDA kernel or fits a model.
mod evidence;
mod metrics;
mod predictor;
#[cfg(test)]
mod tests;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use evidence::Evidence;
use metrics::{measure, remaining_budget, Summary};
use predictor::{sketch, History, Prediction, Signal};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::PathBuf,
    process::Command,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    inputs: PathBuf,
    #[arg(long)]
    capture_report: PathBuf,
    #[arg(long, default_value = "data/research_quality/v1.json")]
    corpus: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

struct Report {
    output: BufWriter<std::fs::File>,
    summaries: BTreeMap<(usize, usize, String, String), Summary>,
}

impl Report {
    fn append(&mut self, value: Value) -> Result<()> {
        serde_json::to_writer(&mut self.output, &value)?;
        self.output.write_all(b"\n")?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        layer: usize,
        percent: usize,
        task: &str,
        row: usize,
        position: u32,
        method: &str,
        is_dense: bool,
        prediction: Prediction,
        hidden: &[f32],
        norms: &[f64],
        oracle: &[usize],
    ) -> Result<()> {
        let metrics = measure(hidden, norms, oracle, &prediction.channels)?;
        for scope in [task, "all_tasks"] {
            self.summaries
                .entry((layer, percent, method.to_owned(), scope.to_owned()))
                .or_default()
                .observe(&metrics, is_dense);
        }
        self.append(
            json!({"kind":"selector_row", "layer":layer, "retained_percent":percent,
            "task":task, "row":row, "absolute_position":position, "method":method,
            "is_dense_refresh":is_dense, "source_rows":prediction.source_rows,
            "selected_channels":prediction.channels, "metrics":metrics}),
        )
    }
}

/// Fixed historical medians from c16559e, 2026-10-31-q4-stronger-baselines.md.
/// These are estimates for triage, not measurements of this analyzer or policy.
fn historical_times(layer: usize, percent: usize) -> Result<(f64, f64)> {
    match (layer, percent) {
        (0, 10) => Ok((0.324390, 0.156682)),
        (17, 10) => Ok((0.324803, 0.155686)),
        (35, 10) => Ok((0.324778, 0.158432)),
        (0, 20) => Ok((0.324598, 0.250483)),
        (17, 20) => Ok((0.324554, 0.249312)),
        (35, 20) => Ok((0.324627, 0.252275)),
        _ => anyhow::bail!("no frozen timing estimate for case"),
    }
}

fn analyze(evidence: &Evidence, report: &mut Report) -> Result<()> {
    let artifact = &evidence.artifact;
    let width = artifact.plan.hidden_dim;
    let input_width = artifact.plan.d_model;
    for layer_id in [0, 17, 35] {
        let layer = artifact
            .layers
            .iter()
            .find(|layer| layer.source.layer_id == layer_id)
            .context("missing target layer")?;
        let predecessor = artifact
            .layers
            .iter()
            .find(|layer| layer.source.layer_id + 1 == layer_id);
        let norms = evidence
            .down_norms
            .get(&layer_id)
            .context("missing down norms")?;
        for percent in [10, 20] {
            let count = (width * percent).div_ceil(100);
            let mut offset = 0;
            for (task_index, task) in artifact.tasks.iter().enumerate() {
                let rows = task.generated_token_ids.len().saturating_sub(1);
                let mut histories = [
                    History::new(width, count, 8)?,
                    History::new(width, count, 16)?,
                ];
                let mut first = Vec::new();
                let mut previous = Vec::new();
                let mut names = vec![
                    "current_oracle".to_owned(),
                    "previous_oracle".to_owned(),
                    "random".to_owned(),
                    "first_token".to_owned(),
                ];
                for interval in [8, 16] {
                    for name in [
                        "refresh_latest",
                        "refresh_frequency",
                        "current_sketch",
                        "previous_layer_sketch",
                    ] {
                        if name == "previous_layer_sketch" && predecessor.is_none() {
                            continue;
                        }
                        names.push(format!("{name}_{interval}"));
                    }
                }
                for name in names {
                    for scope in [task.id.as_str(), "all_tasks"] {
                        report
                            .summaries
                            .entry((layer_id, percent, name.clone(), scope.to_owned()))
                            .or_default();
                    }
                }
                for row in 0..rows {
                    let global_row = offset + row;
                    let hidden = &layer.swiglu_inputs[global_row * width..(global_row + 1) * width];
                    let oracle = predictor::select_top_k(
                        &hidden
                            .iter()
                            .map(|value| f64::from(value.abs()))
                            .collect::<Vec<_>>(),
                        count,
                    )?;
                    let current = sketch(
                        &layer.normalized_inputs
                            [global_row * input_width..(global_row + 1) * input_width],
                    )?;
                    let preceding = predecessor
                        .map(|previous_layer| {
                            sketch(
                                &previous_layer.normalized_inputs
                                    [global_row * input_width..(global_row + 1) * input_width],
                            )
                        })
                        .transpose()?;
                    let position = layer.rows[global_row].absolute_position;
                    if row == 0 {
                        first = oracle.clone();
                    }
                    for (method, is_dense, channels, source_rows) in [
                        ("current_oracle", false, oracle.clone(), vec![row]),
                        (
                            "previous_oracle",
                            row == 0,
                            if row == 0 {
                                oracle.clone()
                            } else {
                                previous.clone()
                            },
                            vec![row.saturating_sub(1)],
                        ),
                        (
                            "random",
                            false,
                            predictor::random_selection(
                                width,
                                count,
                                ((layer_id as u64) << 48)
                                    | ((task_index as u64) << 32)
                                    | row as u64,
                            )?,
                            vec![],
                        ),
                        ("first_token", row == 0, first.clone(), vec![0]),
                    ] {
                        report.record(
                            layer_id,
                            percent,
                            &task.id,
                            row,
                            position,
                            method,
                            is_dense,
                            Prediction {
                                channels,
                                source_rows,
                            },
                            hidden,
                            norms,
                            &oracle,
                        )?;
                    }
                    for (history, interval) in histories.iter_mut().zip([8, 16]) {
                        let is_dense = history.should_refresh(row);
                        for (name, signal, query) in [
                            ("refresh_latest", Signal::Latest, None),
                            ("refresh_frequency", Signal::Frequency, None),
                            ("current_sketch", Signal::CurrentSketch, Some(current)),
                            ("previous_layer_sketch", Signal::PreviousSketch, preceding),
                        ] {
                            if matches!(signal, Signal::PreviousSketch) && predecessor.is_none() {
                                continue;
                            }
                            let prediction = if is_dense {
                                Prediction {
                                    channels: oracle.clone(),
                                    source_rows: vec![row],
                                }
                            } else {
                                history.predict(row, signal, query)?
                            };
                            report.record(
                                layer_id,
                                percent,
                                &task.id,
                                row,
                                position,
                                &format!("{name}_{interval}"),
                                is_dense,
                                prediction,
                                hidden,
                                norms,
                                &oracle,
                            )?;
                        }
                        // Only the scheduled dense branch ever exposes labels.
                        if is_dense {
                            history.observe_refresh(row, &oracle, current, preceding)?;
                        }
                    }
                    previous = oracle;
                }
                offset += rows;
            }
        }
    }
    let summaries = std::mem::take(&mut report.summaries);
    let mut decisions: BTreeMap<(usize, String), Vec<(usize, bool)>> = BTreeMap::new();
    for ((layer, percent, method, scope), summary) in summaries {
        let is_causal =
            method == "first_token" || method.starts_with("refresh_") || method.contains("sketch_");
        let tasks = if scope == "all_tasks" {
            artifact.tasks.len()
        } else {
            1
        };
        let (dense_ms, sparse_ms) = historical_times(layer, percent)?;
        if is_causal && scope == "all_tasks" {
            decisions
                .entry((percent, method.clone()))
                .or_default()
                .push((layer, summary.passes_screen()));
        }
        report.append(json!({"kind":"selector_summary", "layer":layer, "retained_percent":percent,
            "method":method, "scope":scope, "is_causal_label_policy":is_causal,
            "bootstrap_dense_steps":tasks, "summary":summary,
            "passes_fixed_screen":summary.passes_screen(),
            "optimistic_remaining_selector_us_per_sparse_step":if is_causal { remaining_budget(dense_ms, sparse_ms, &summary, tasks) } else { None },
            "cost_scope":"historical per-layer medians with dense bootstrap/refresh; no new selector timing or recursive sparse quality"}))?;
    }
    let decisions: Vec<_> = decisions
        .into_iter()
        .map(|((percent, method), layers)| {
            json!({
        "retained_percent":percent,"method":method,"layer_results":layers,
        "passes_all_applicable_layers":layers.iter().all(|(_, pass)| *pass),
        "covers_all_three_target_layers":layers.len()==3})
        })
        .collect();
    report.append(json!({"kind":"selector_complete","is_complete":true,"decisions":decisions,
        "scope":"fixed six-prompt dense-trajectory diagnostics; promising overlap still requires masked output, measured cost and recursive generation gates"}))?;
    report.output.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "refuse to overwrite analysis");
    let evidence = Evidence::load(
        &args.model,
        &args.inputs,
        &args.capture_report,
        &args.corpus,
    )?;
    let revision = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    let state = Command::new("git")
        .args(["status", "--porcelain"])
        .output()?;
    ensure!(
        revision.status.success() && state.status.success() && state.stdout.is_empty(),
        "analyze from a clean source checkpoint"
    );
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    let mut report = Report {
        output: BufWriter::new(output),
        summaries: BTreeMap::new(),
    };
    report.append(json!({"kind":"selector_manifest", "revision":String::from_utf8(revision.stdout)?.trim(),
        "inputs_sha256":evidence.artifact_sha256,"capture_report_sha256":evidence.report_sha256,
        "capture_revision":evidence.manifest["revision"],
        "executable_sha256":m40_llm::evaluation::identity::file_sha256(&std::env::current_exe()?)?,
        "model_sha256":evidence::MODEL_SHA256,"corpus_sha256":evidence::CORPUS_SHA256,
        "sketch_seed":format!("{:016x}",predictor::HASH_SEED),"sketch_width":predictor::SKETCH_WIDTH,
        "bank_capacity":predictor::BANK_CAPACITY,"refresh_intervals":[8,16],
        "uses_cuda":false,"is_performance_measurement":false,"is_trained_selector":false}))?;
    analyze(&evidence, &mut report)
}

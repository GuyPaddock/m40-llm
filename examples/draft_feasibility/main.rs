#[cfg(all(feature = "cuda", nvcc))]
mod capture;
mod cost;
mod records;
mod replay;
mod source;
#[cfg(test)]
mod tests;

use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use serde_json::json;
use std::{fs::OpenOptions, path::PathBuf};

#[derive(Parser)]
#[command(about = "Causal CPU draft costs and measured-target projections, not HTTP benchmarks")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Capture {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Study {
        #[arg(long)]
        heldout: Option<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Capture { model, output } => {
            #[cfg(all(feature = "cuda", nvcc))]
            return capture::capture(&model, &output);
            #[cfg(not(all(feature = "cuda", nvcc)))]
            {
                let _ = (model, output);
                anyhow::bail!("capture requires release CUDA/NVCC; study is CPU-only");
            }
        }
        Command::Study { heldout, output } => study(heldout, output),
    }
}

fn study(heldout: Option<PathBuf>, output: PathBuf) -> Result<()> {
    ensure!(!cfg!(debug_assertions), "CPU cost study requires --release");
    ensure!(
        records::command("git", &["status", "--short"])?.is_empty(),
        "commit source first"
    );
    let traces = records::read_traces(heldout.as_deref())?;
    let model = cost::CostModel::load()?;
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    records::write(
        &mut out,
        json!({"kind":"draft_study_manifest","schema_version":1,
        "revision":records::command("git", &["rev-parse","HEAD"])?,"worktree":"",
        "executable_sha256":records::hash_file(&std::env::current_exe()?)?,
        "build_profile":"release","rustc":records::command("rustc",&["--version"])?,
        "heldout_source":heldout.as_ref().map(|p|records::hash_file(p)).transpose()?,
        "prompt_definitions_sha256":records::hash_file(records::PROMPTS.as_ref())?,
        "target_measurement_sha256":records::hash_file(records::TARGET.as_ref())?,
        "historical_ngram_report":"docs/experiments/2026-10-25-ngram-feasibility.md",
        "protocol":"three fixed sources, known target seed plus three guesses, five CPU replays; no fitting or future access",
        "is_measured_http":false,"runtime_defaults_changed":false}),
    )?;
    for trace in traces {
        trace.validate()?;
        for kind in source::SourceKind::ALL {
            let _warmup = replay::run(&trace, kind)?;
            let trials = (0..5)
                .map(|_| replay::run(&trace, kind))
                .collect::<Result<Vec<_>>>()?;
            let first = &trials[0];
            for trial in &trials {
                ensure!(
                    trial.signature() == first.signature(),
                    "nondeterministic causal decisions"
                );
            }
            let positions = replay::all_positions(&trace, kind)?;
            let costs: Vec<_> = trials.iter().map(|r| r.cpu_elapsed_ns).collect();
            let projections = model.project(&trace, &trials)?;
            records::write(
                &mut out,
                json!({"kind":"causal_draft_case","id":trace.id,
                "class":trace.class,"split":trace.split,"source":kind,"prompt_tokens":trace.prompt_token_ids.len(),
                "generated_tokens":trace.generated_token_ids.len(),"stop_reason":trace.stop_reason,
                "actual_output":trace.output,"position_stats":positions,"visited_stats":first.statistics(),
                "trials":trials,"all_in_cpu_ns":costs,"projections":projections,
                "has_causal_reference_replay":true}),
            )?;
            eprintln!(
                "{} {kind:?}: {} eligible / {} visited decisions",
                trace.id,
                first.rounds.iter().filter(|r| r.guesses.is_some()).count(),
                first.rounds.len()
            );
        }
    }
    records::write(
        &mut out,
        json!({"kind":"draft_study_complete","is_complete":true,
        "is_measured_gpu_speedup":false,"is_measured_http":false,"has_live_drafter":false}),
    )
}

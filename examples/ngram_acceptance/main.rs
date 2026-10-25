#[cfg(all(feature = "cuda", nvcc))]
mod capture;
mod records;
mod simulate;

use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use serde_json::json;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
};

#[derive(Parser)]
#[command(
    about = "History-only ngram feasibility; verifier costs are assumptions, not measured GPU speed"
)]
struct Args {
    #[command(subcommand)]
    command: StudyCommand,
}

#[derive(Subcommand)]
enum StudyCommand {
    Study {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        regression_traces: Option<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    Capture {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    match Args::parse().command {
        StudyCommand::Study {
            input,
            regression_traces,
            output,
        } => run_study(input, regression_traces, output),
        StudyCommand::Capture { model, output } => {
            #[cfg(all(feature = "cuda", nvcc))]
            {
                capture::capture(model, output)
            }
            #[cfg(not(all(feature = "cuda", nvcc)))]
            {
                let _ = (model, output);
                anyhow::bail!(
                    "capture requires --release --features cuda and NVCC; study itself is CPU-only"
                )
            }
        }
    }
}

fn run_study(input: PathBuf, regression_traces: Option<PathBuf>, output: PathBuf) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "use --release for CPU drafting timings"
    );
    let browser = records::read_browser(&input)?;
    let mut cases = Vec::new();
    if let Some(path) = regression_traces.as_ref() {
        cases = records::read_regressions(path, &browser)?;
    }
    let has_regression_checkpoint = !cases.is_empty();
    let configuration_sha256 = records::sha256(&serde_json::to_vec(&browser.configuration)?)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)?;
    writeln!(
        file,
        "{}",
        json!({"kind":"ngram_study_manifest","source":browser.source,
        "model_sha256":browser.model_sha256,"configuration":browser.configuration,"configuration_sha256":configuration_sha256,
        "analysis_revision":records::command_text("git",&["rev-parse","HEAD"])?,"analysis_worktree":records::command_text("git",&["status","--short"])?,
        "analysis_executable_sha256":records::sha256(&fs::read(std::env::current_exe()?)?)?,
        "analysis_rustc":records::command_text("rustc",&["--version"])?,
        "build_profile":"release","regression_source":regression_traces.as_ref().map(|path|->Result<_>{Ok(json!({"path":path,"sha256":records::sha256(&fs::read(path)?)?}))}).transpose()?,
        "lookup":"longest suffix 3-8, most recent usable occurrence, known history only, draft capped at first stop or remaining budget",
        "cost_model":"fixed HTTP minus decode residual, weighted visited rounds less one prefilling prediction, zero drafting/commit/rollback/copy/correction overhead; hypothetical, not measured verification speed",
        "screening_threshold":0.10,"has_regression_checkpoint":has_regression_checkpoint})
    )?;
    cases.insert(0, browser.case);
    let mut best_gain = f64::NEG_INFINITY;
    for case in cases {
        for draft_len in [2, 4, 8, 16] {
            let study = simulate::study(
                &case.trace.prompt_token_ids,
                &case.trace.generated_token_ids,
                &case.stop_ids,
                draft_len,
            )?;
            ensure!(
                study.replayed_tokens == case.trace.generated_token_ids,
                "replay changed committed tokens"
            );
            let estimates = case
                .baseline
                .as_ref()
                .map(|baseline| {
                    [1.0, 1.5, 2.0, 4.0]
                        .into_iter()
                        .map(|cost| simulate::estimate_cost(baseline, &study.visited, cost))
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?;
            if let Some(estimates) = &estimates {
                best_gain = best_gain.max(estimates[0].estimated_throughput_gain);
            }
            writeln!(
                file,
                "{}",
                json!({"kind":"ngram_acceptance","case":case.label,"candidate_tokens":draft_len,
                "prompt_tokens":case.trace.prompt_token_ids.len(),"generated_tokens":case.trace.generated_token_ids.len(),
                "positions":study.positions.report(),"visited_rounds":study.visited.report(),"rounds":study.rounds,
                "baseline":case.baseline,"hypothetical_costs":estimates,"replay_matches_reference":true})
            )?;
            eprintln!("{} draft={draft_len}: eligible={:.1}% positions, accepted/offer={:.3}, visited_rounds={}",
                case.label,100.0*study.positions.eligible_steps as f64/study.positions.steps as f64,
                study.visited.accepted_tokens as f64/study.visited.eligible_steps.max(1) as f64,study.visited.steps);
        }
    }
    let passes_browser_screen = best_gain >= 0.10;
    writeln!(
        file,
        "{}",
        json!({"kind":"ngram_feasibility_decision","best_browser_optimistic_throughput_gain":best_gain,
        "passes_browser_screen":passes_browser_screen,"has_regression_checkpoint":has_regression_checkpoint,
        "may_proceed_to_verifier_feasibility":passes_browser_screen && has_regression_checkpoint,
        "verifier_implemented":false,"speedup_measured":false,"runtime_defaults_changed":false})
    )?;
    eprintln!("Best optimistic browser gain={:.3}%, screen={}, regressions_present={has_regression_checkpoint}",best_gain*100.0,passes_browser_screen);
    Ok(())
}

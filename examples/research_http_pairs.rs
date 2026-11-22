//! Apply the controlled benchmark's paired statistics to separate server processes.
#[allow(dead_code)]
#[path = "compare_ollama/protocol.rs"]
mod protocol;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use protocol::{compare_pairs, Pair};
use serde::Deserialize;
use serde_json::json;
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    timings: PathBuf,
    /// Independent confirmation runs use the label prefix "confirm".
    #[arg(long)]
    confirmation: bool,
}

#[derive(Deserialize)]
struct Observation {
    label: String,
    arm: usize,
    http_seconds: f64,
    metrics: Metrics,
}

#[derive(Deserialize)]
struct Metrics {
    generated_tokens: usize,
    prompt_tokens: usize,
    stop_reason: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let prefix = if args.confirmation {
        "confirm-"
    } else {
        "pair-"
    };
    let mut trials: BTreeMap<usize, [Option<f64>; 2]> = BTreeMap::new();
    let mut workload = None;
    for line in fs::read_to_string(args.timings)?.lines() {
        let sample: Observation = serde_json::from_str(line)?;
        let Some(suffix) = sample.label.strip_prefix(prefix) else {
            continue;
        };
        let (index, arm) = suffix.split_once('-').context("invalid trial label")?;
        let arm = arm.parse::<usize>()?;
        ensure!(arm < 2 && arm == sample.arm, "invalid paired arm");
        let metrics = sample.metrics;
        let signature = (
            metrics.prompt_tokens,
            metrics.generated_tokens,
            metrics.stop_reason,
        );
        ensure!(
            workload.get_or_insert(signature.clone()) == &signature,
            "prompt, generation or stop differs"
        );
        ensure!(
            sample.http_seconds.is_finite()
                && sample.http_seconds > 0.
                && metrics.generated_tokens > 0,
            "invalid timing"
        );
        let tps = metrics.generated_tokens as f64 / sample.http_seconds;
        ensure!(
            trials.entry(index.parse()?).or_default()[arm]
                .replace(tps)
                .is_none(),
            "duplicate observation"
        );
    }
    let mut pairs = Vec::new();
    for (key, rates) in trials {
        ensure!(key == pairs.len(), "missing pair index");
        pairs.push(Pair {
            control_tps: rates[0].context("missing control")?,
            candidate_tps: rates[1].context("missing candidate")?,
        });
    }
    let summary = compare_pairs(&pairs)?;
    let unstable = summary.needs_variation_extension;
    ensure!(
        if args.confirmation {
            pairs.len() == 20
        } else {
            [5, 10].contains(&pairs.len())
        },
        "unexpected series size"
    );
    let threshold = 0.05_f64.max(summary.promotion_threshold);
    let near_threshold = (summary.gain - threshold).abs() <= 0.01;
    let next = if unstable && pairs.len() == 5 {
        "extend"
    } else if unstable {
        "inconclusive"
    } else if (summary.needs_fresh_confirmation || near_threshold) && !args.confirmation {
        "confirm"
    } else {
        "complete"
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "pairs":pairs.len(),"control_tps":summary.control,"candidate_tps":summary.candidate,
            "gain":summary.gain,"promotion_threshold":threshold,"bootstrap_gain_ci95":summary.bootstrap_gain_ci95,
            "bootstrap_resamples":summary.bootstrap_resamples,"bootstrap_seed":summary.bootstrap_seed,
            "next":next,"passes_november_speed_gate":next=="complete" && summary.gain >= threshold && summary.bootstrap_gain_ci95[0]>0.,
            "workload":workload
        }))?
    );
    Ok(())
}

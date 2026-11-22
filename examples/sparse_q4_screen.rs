//! Apply project paired-trial rules to charged Q4 operator latency.
#[allow(dead_code)]
#[path = "compare_ollama/protocol.rs"]
mod protocol;
use anyhow::{ensure, Context, Result};
use clap::Parser;
use protocol::{compare_pairs, Pair};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    #[arg(long)]
    confirmation: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut samples = BTreeMap::new();
    for path in args.inputs {
        let mut complete = false;
        for line in fs::read_to_string(path)?.lines() {
            ensure!(!complete, "data after completion");
            let row: Value = serde_json::from_str(line)?;
            complete = row["kind"] == "complete";
            if row["kind"] != "trial" || row["round"] == -1 {
                continue;
            }
            let key = (
                row["layer"].as_u64().context("layer")?,
                row["count"].as_u64().context("count")?,
                row["arm"].as_str().context("arm")?.to_owned(),
                row["round"].as_u64().context("round")?,
            );
            let us = row["wall_us"].as_f64().context("wall")?;
            ensure!(
                us.is_finite() && us > 0. && samples.insert(key, us).is_none(),
                "invalid or duplicate latency"
            );
        }
        ensure!(complete, "unfinished operator cohort");
    }
    let rounds = samples.keys().map(|k| k.3).max().context("no trials")? + 1;
    ensure!(
        if args.confirmation {
            rounds == 20
        } else {
            [5, 10].contains(&rounds)
        },
        "unexpected trial count"
    );
    ensure!(
        samples.len() == 3 * 4 * 8 * rounds as usize,
        "incorrect grid size"
    );
    let mut comparisons = Vec::new();
    let mut extend = false;
    let mut confirm = false;
    let mut passing = BTreeMap::new();
    for layer in [0, 17, 35] {
        for count in [1376, 2752, 5504, 11008] {
            for arm in [
                "dense_zigzag",
                "native_oracle",
                "packed_oracle",
                "zigzag_oracle",
                "native_charged",
                "packed_charged",
                "zigzag_charged",
            ] {
                let mut pairs = Vec::new();
                for round in 0..rounds {
                    let lookup = |name: &str| {
                        samples
                            .get(&(layer, count, name.to_owned(), round))
                            .copied()
                            .context("missing arm")
                    };
                    pairs.push(Pair {
                        control_tps: 1e6 / lookup("dense_native")?,
                        candidate_tps: 1e6 / lookup(arm)?,
                    });
                }
                let stats = compare_pairs(&pairs)?;
                // Ten percent lower latency is 1/0.90-1 higher service rate.
                let threshold = (1.0 / 0.90_f64 - 1.0).max(stats.promotion_threshold);
                let primary = arm.ends_with("_charged") && [2752, 5504].contains(&count);
                let unstable = stats.needs_variation_extension;
                let borderline = (stats.gain - threshold).abs() <= 0.01;
                if primary {
                    extend |= unstable && rounds == 5;
                    confirm |= !unstable
                        && (borderline
                            || (stats.gain >= threshold && stats.bootstrap_gain_ci95[0] <= 0.0));
                }
                let passes = primary
                    && !unstable
                    && stats.gain >= threshold
                    && stats.bootstrap_gain_ci95[0] > 0.;
                if passes {
                    *passing.entry((arm, count)).or_insert(0) += 1;
                }
                comparisons.push(json!({"layer":layer,"count":count,"arm":arm,"control_calls_s":stats.control,
                    "candidate_calls_s":stats.candidate,"gain":stats.gain,"ci95":stats.bootstrap_gain_ci95,
                    "threshold":threshold,"unstable":unstable,"primary":primary,"passes":passes}));
            }
        }
    }
    let next = if extend {
        "extend"
    } else if confirm && !args.confirmation {
        "confirm"
    } else {
        "complete"
    };
    let admitted: Vec<_> = passing.into_iter().filter(|(_,layers)| *layers>=2)
        .map(|((arm,count),layers)|json!({"arm":arm,"active_inputs":count,"passing_layers":layers})).collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"rounds":rounds,"next":next,
        "admitted_if_complete":admitted,"comparisons":comparisons}))?
    );
    Ok(())
}

#[cfg(any(all(feature = "cuda", nvcc), test))]
mod config;
#[cfg(all(feature = "cuda", nvcc))]
mod evidence;
#[cfg(all(feature = "cuda", nvcc))]
mod execution;
mod fixture;
#[cfg(all(test, feature = "cuda", nvcc))]
mod hardware_tests;
#[cfg(all(feature = "cuda", nvcc))]
mod history;
mod history_inputs;
#[cfg(any(all(feature = "cuda", nvcc), test))]
mod metrics;
mod november;
#[cfg(all(feature = "cuda", nvcc))]
mod prefix;
#[cfg(all(feature = "cuda", nvcc))]
mod residency;
#[cfg(all(feature = "cuda", nvcc))]
mod run;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Frozen real-model KV quality research; no runtime default changes")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Prepare the 2K equivalence and >64-block quality inputs.
    PrepareHistory {
        model: PathBuf,
        november_inputs: PathBuf,
        output: PathBuf,
    },
    /// Evaluate existing mixed consumers on a frozen history boundary.
    History {
        model: PathBuf,
        november_inputs: PathBuf,
        inputs: PathBuf,
        output: PathBuf,
    },
    /// Measure one cache in a fresh process with the quality model's weight owners.
    Residency {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_parser=["dense","top64","top8"])]
        arm: String,
        #[arg(long)]
        capacity: u32,
        #[arg(long)]
        output: PathBuf,
    },
    /// Freeze the separate November expansion without invoking inference.
    PrepareNovember {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Run all three cache arms and matched-prefix observations on frozen inputs.
    Run {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "data/research_quality/v1.json")]
        corpus: PathBuf,
        #[arg(long)]
        inputs: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Prepare and identify all inputs without loading a GPU model or running inference.
    Prepare {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "data/research_quality/v1.json")]
        corpus: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::PrepareHistory {
            model,
            november_inputs,
            output,
        } => history_inputs::prepare(&model, &november_inputs, &output),
        Command::History {
            model,
            november_inputs,
            inputs,
            output,
        } => {
            #[cfg(all(feature = "cuda", nvcc))]
            {
                history::run(&model, &november_inputs, &inputs, &output)
            }
            #[cfg(not(all(feature = "cuda", nvcc)))]
            {
                let _ = (model, november_inputs, inputs, output);
                anyhow::bail!("history evaluation requires CUDA/NVCC")
            }
        }
        Command::Residency {
            model,
            arm,
            capacity,
            output,
        } => {
            #[cfg(all(feature = "cuda", nvcc))]
            {
                residency::measure(&model, &arm, capacity, &output)
            }
            #[cfg(not(all(feature = "cuda", nvcc)))]
            {
                let _ = (model, arm, capacity, output);
                anyhow::bail!("residency requires CUDA/NVCC")
            }
        }
        Command::PrepareNovember { model, output } => november::prepare(&model, &output),
        Command::Run {
            model,
            corpus,
            inputs,
            output,
        } => {
            #[cfg(all(feature = "cuda", nvcc))]
            {
                run::run(&model, &corpus, &inputs, &output)
            }
            #[cfg(not(all(feature = "cuda", nvcc)))]
            {
                let _ = (model, corpus, inputs, output);
                anyhow::bail!("KV quality inference requires CUDA/NVCC; preparation is CPU-only")
            }
        }
        Command::Prepare {
            model,
            corpus,
            output,
        } => fixture::prepare(&model, &corpus, &output),
    }
}

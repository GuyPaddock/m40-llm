#[cfg(any(all(feature = "cuda", nvcc), test))]
mod config;
#[cfg(all(feature = "cuda", nvcc))]
mod evidence;
#[cfg(all(feature = "cuda", nvcc))]
mod execution;
mod fixture;
#[cfg(all(test, feature = "cuda", nvcc))]
mod hardware_tests;
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
    /// Measure one cache in a fresh process with the quality model's weight owners.
    Residency {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_parser=["dense","top64","top8"])]
        arm: String,
        #[arg(long)]
        capacity: u32,
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=3))]
        cycles: u8,
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
        Command::Residency {
            model,
            arm,
            capacity,
            cycles,
            output,
        } => {
            #[cfg(all(feature = "cuda", nvcc))]
            {
                residency::measure(&model, &arm, capacity, cycles, &output)
            }
            #[cfg(not(all(feature = "cuda", nvcc)))]
            {
                let _ = (model, arm, capacity, cycles, output);
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

mod fixture;

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
        Command::Prepare {
            model,
            corpus,
            output,
        } => fixture::prepare(&model, &corpus, &output),
    }
}

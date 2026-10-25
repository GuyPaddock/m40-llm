#[cfg(any(all(feature = "cuda", nvcc), test))]
mod records;
#[cfg(all(feature = "cuda", nvcc))]
mod run;
#[cfg(all(feature = "cuda", nvcc))]
mod tasks;
#[cfg(all(test, feature = "cuda", nvcc))]
mod test_files;

use anyhow::{ensure, Result};
use clap::{Parser, ValueEnum};
use m40_llm::evaluation::corpus::Split;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SplitChoice {
    Calibration,
    HeldOut,
    All,
}

impl SplitChoice {
    fn selected(self) -> Vec<Split> {
        match self {
            Self::Calibration => vec![Split::Calibration],
            Self::HeldOut => vec![Split::HeldOut],
            Self::All => vec![Split::Calibration, Split::HeldOut],
        }
    }
}

#[derive(Parser)]
#[command(
    about = "Explicit model-variant quality, loss, and resident generation timing; not HTTP throughput"
)]
struct Args {
    #[arg(long)]
    reference_model: PathBuf,
    #[arg(long)]
    model: PathBuf,
    /// Descriptive label; actual model hashes determine source identity.
    #[arg(long)]
    variant_id: String,
    #[arg(long, default_value = "data/research_quality/v1.json")]
    corpus: PathBuf,
    #[arg(long, value_enum, default_value = "all")]
    split: SplitChoice,
    #[arg(long, default_value_t = 1024)]
    context_tokens: u32,
    /// Must not exist. Flush each completed row; never overwrite prior evidence.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!cfg!(debug_assertions), "use --release for evaluation");
    ensure!(
        !args.variant_id.trim().is_empty() && args.context_tokens > 1,
        "provide a nonempty variant ID and at least two context slots"
    );
    #[cfg(all(feature = "cuda", nvcc))]
    {
        run::evaluate(args)
    }
    #[cfg(not(all(feature = "cuda", nvcc)))]
    {
        // Keep the CLI/error available in non-CUDA builds, not a successful skip.
        let _ = args.split.selected();
        anyhow::bail!("evaluation requires --features cuda and an NVCC build")
    }
}

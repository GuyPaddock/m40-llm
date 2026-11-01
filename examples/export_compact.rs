//! CPU-only, quality-bound compact GGUF export. This does not promote a model.

use anyhow::{Context, Result};
use clap::Parser;
use m40_llm::gguf::compact::{export_compact_gguf, CompactExportEvidence};
use serde_json::json;
use std::{fs::OpenOptions, io::Write, path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(about = "Export a separate compact GGUF from a frozen, quality-evaluated layer mask")]
struct Args {
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    layer_ranking: PathBuf,
    #[arg(long)]
    quality_report: PathBuf,
    #[arg(long, default_value = "data/research_quality/v1.json")]
    corpus: PathBuf,
    #[arg(long, value_parser=clap::value_parser!(u32).range(1..=3))]
    remove_layer_count: u32,
    /// New JSON record; existing files are never overwritten.
    #[arg(long)]
    report: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let started = Instant::now();
    let evidence = CompactExportEvidence::load(
        &args.layer_ranking,
        &args.quality_report,
        &args.corpus,
        args.remove_layer_count as usize,
    )?;
    let evidence_elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut report = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.report)
        .context("create new compact-export report")?;
    let result = export_compact_gguf(&args.source, &args.output, &evidence)?;
    let record = json!({"kind":"compact_export_complete","is_complete":true,
        "build_profile":if cfg!(debug_assertions) { "debug" } else { "release" },
        "is_exact_engine_improvement":false,
        "timing_scope":"evidence validation and export publication; excludes process startup and report serialization/sync",
        "evidence_elapsed_ms":evidence_elapsed_ms,
        "total_preparation_elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"export":result});
    serde_json::to_writer(&mut report, &record)?;
    writeln!(report)?;
    report.sync_all()?;
    println!("{}", serde_json::to_string(&record)?);
    Ok(())
}

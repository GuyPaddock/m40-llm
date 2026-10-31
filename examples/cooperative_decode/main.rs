//! Post-prefill scheduling simulation. No model generation or hardware timing.

mod input;
mod simulation;
#[cfg(test)]
mod tests;

use anyhow::{ensure, Result};
use clap::Parser;
use input::{Evidence, Trace};
use m40_llm::evaluation::identity::file_sha256;
use serde_json::json;
use simulation::{simulate, Mode};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
};

#[derive(Parser)]
struct Args {
    /// Explicit post-prefill arrivals and per-token stage costs, schema 1.
    #[arg(long)]
    trace: PathBuf,
    /// New JSONL report; existing results are never overwritten.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        fs::metadata(&args.trace)?.len() <= 8 * 1024 * 1024,
        "trace exceeds 8 MiB"
    );
    let trace: Trace = serde_json::from_slice(&fs::read(&args.trace)?)?;
    trace.validate()?;
    for evidence in trace.evidence.values() {
        if let Evidence::Artifact { path, sha256, .. } = evidence {
            let path = args.trace.parent().unwrap().join(path);
            ensure!(
                file_sha256(&path)?.eq_ignore_ascii_case(sha256),
                "cost artifact hash differs"
            );
        }
    }
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.output)?;
    let mut record = |value: serde_json::Value| -> Result<()> {
        serde_json::to_writer(&mut output, &value)?;
        writeln!(output)?;
        output.flush()?;
        Ok(())
    };
    record(
        json!({"kind":"cooperative_simulation_manifest","schema_version":1,
        "trace_sha256":file_sha256(&args.trace)?, "trace":trace,
        "executable_sha256":file_sha256(&std::env::current_exe()?)?,
        "is_simulation":true,"is_serving_measurement":false,"generated_token_count":0,
        "has_assumed_costs":trace.evidence.values().any(|value| matches!(value, Evidence::Assumption { .. })),
        "is_model_correctness_evaluated":false,
        "timing_scope":"decode-ready through last simulated token; prefill and initial KV migration excluded",
        "cost_scope":"declared component costs; artifact integrity checked, values not automatically extracted",
        "quantiles":"nearest-rank, pooled within-request intervals; excludes first-token latency"}),
    )?;
    let mut admitted = 0;
    for mode in [Mode::GpuOnly, Mode::Independent, Mode::Assisted] {
        match simulate(&trace, mode) {
            Ok(result) => {
                admitted += 1;
                record(
                    json!({"kind":"cooperative_simulation","is_admitted":true,"result":result}),
                )?;
            }
            Err(error) => record(json!({"kind":"cooperative_simulation","mode":mode,
                "is_admitted":false,"error":format!("{error:#}")}))?,
        }
    }
    record(
        json!({"kind":"cooperative_simulation_complete","is_complete":true,
        "admitted_scenarios":admitted,"is_performance_gate_evaluated":false}),
    )
}

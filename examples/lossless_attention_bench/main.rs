//! Full-context lossless FP16 attention and append-cost research.

mod capture;
mod fixture;

use anyhow::{ensure, Result};
use clap::Parser;
use m40_llm::evaluation::identity::file_sha256;
use serde_json::json;
use std::{fs::OpenOptions, io::Write, path::PathBuf};

#[derive(Parser)]
struct Args {
    /// Complete, hash-verified eight-sequence capture report.
    #[arg(long)]
    capture_report: PathBuf,
    /// Original raw FP16 artifact belonging to the report.
    #[arg(long)]
    kv_inputs: PathBuf,
    /// New JSONL report; default mode inspects CPU fixtures only.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "output must be new");
    let mut capture = capture::Capture::open(&args.capture_report, &args.kv_inputs)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    let mut record = |value: serde_json::Value| -> Result<()> {
        serde_json::to_writer(&mut output, &value)?;
        writeln!(output)?;
        output.flush()?;
        Ok(())
    };
    record(
        json!({"kind":"lossless_attention_fixture_manifest","schema_version":1,
        "capture":capture.manifest(),"artifact_sha256":capture.artifact_sha256(),
        "capture_report_sha256":file_sha256(&args.capture_report)?,
        "query_source":"deterministic synthetic FP32 vectors; not captured model queries",
        "is_gpu_measured":false,"generated_token_count":0}),
    )?;
    for layer in capture::LAYERS {
        for length in capture::CONTEXTS {
            let sample = capture.read_layer(layer, length, fixture::DIMENSION)?;
            let fixture = fixture::Fixture::from_sample(&sample, length)?;
            let encoded = fixture.encode()?;
            record(
                json!({"kind":"lossless_attention_fixture","layer":layer,"length":length,
                "input_sha256":sample.sha256()?,"host_fp32_source_bytes":sample.bytes(),
                "encoded":encoded.describe(),"has_bit_exact_reconstruction":true}),
            )?;
        }
    }
    record(
        json!({"kind":"lossless_attention_fixture_complete","is_complete":true,"cases":12,
        "generated_token_count":0,"stop_reason":"fixed_fixture_grid_complete"}),
    )
}

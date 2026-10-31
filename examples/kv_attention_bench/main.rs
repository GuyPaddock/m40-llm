//! Quantized-attention fixture admission and paired operator research.

mod capture;

use anyhow::{ensure, Result};
use clap::Parser;
use m40_llm::evaluation::identity::file_sha256;
use serde_json::json;
use std::{fs::OpenOptions, io::Write, path::PathBuf};

#[derive(Parser)]
struct Args {
    /// Complete lossless-KV capture JSONL; partial captures are rejected.
    #[arg(long)]
    capture_report: PathBuf,
    /// Raw FP16 words belonging to the capture report.
    #[arg(long)]
    kv_inputs: PathBuf,
    /// New fixture-inspection report. This mode does not access a GPU.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "output must be new");
    let mut capture = capture::Capture::open(&args.capture_report, &args.kv_inputs)?;
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
        json!({"kind":"kv_attention_fixture_manifest","schema_version":1,
        "capture_report_sha256":file_sha256(&args.capture_report)?,
        "artifact_sha256":capture.artifact_sha256(),"capture":capture.manifest(),
        "query_source":"deterministic synthetic FP32 query vectors, not captured model queries",
        "head64_scope":"first 64 channels of each captured head; diagnostic fixture, not Qwen execution",
        "is_gpu_measured":false,"is_quality_measured":false}),
    )?;
    for layer in capture::LAYERS {
        for length in capture::CONTEXTS {
            for head_dim in [64, 128] {
                let sample = capture.read_layer(layer, length, head_dim)?;
                record(json!({"kind":"kv_attention_fixture","layer":layer,
                    "context_tokens":length,"head_dim":head_dim,"sequences":capture::SEQUENCES,
                    "kv_heads":capture::KV_HEADS,"query_heads":capture::QUERY_HEADS,
                    "input_sha256":sample.sha256()?,"fp32_host_payload_bytes":sample.bytes(),
                    "is_finite":true}))?;
            }
        }
    }
    record(
        json!({"kind":"kv_attention_fixture_complete","is_complete":true,
        "cases":24,"generated_token_count":0,"stop_reason":"fixed_fixture_grid_complete"}),
    )
}

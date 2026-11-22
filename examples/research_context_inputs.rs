//! Freeze the controlled Qwen browser prompt at three real tokenizer lengths.
#[path = "common/qwen_workload.rs"]
mod workload;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use m40_llm::{
    evaluation::identity::{file_sha256, sha256},
    gguf::load_gguf,
    tokenizer::Tokenizer,
};
use serde_json::{json, Value};
use std::{fmt::Write as _, fs, io::Write as _, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    /// Completed compare_ollama report with the historical tokenizer oracle.
    #[arg(long)]
    oracle_report: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let model_hash = file_sha256(&args.model)?;
    ensure!(
        model_hash == "4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d",
        "wrong GGUF"
    );
    let metadata = load_gguf(&args.model)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&metadata.metadata)?;
    let records = fs::read_to_string(&args.oracle_report)?;
    let oracle: Value = records
        .lines()
        .map(serde_json::from_str)
        .collect::<serde_json::Result<Vec<Value>>>()?
        .into_iter()
        .find(|row| row["kind"] == "tokenizer_oracle" && row["is_equal"] == true)
        .context("missing verified tokenizer oracle")?;
    let mut archive = String::new();
    let mut lines = 0;
    let mut cases = Vec::new();
    for (name, target, max_tokens) in [
        ("short", 65, workload::MAX_TOKENS),
        ("medium", 2048, 128),
        ("long", 8192, 128),
    ] {
        let (prompt, ids) = loop {
            let prompt =
                workload::render_user_prompt(&format!("{archive}{}", workload::USER_PROMPT));
            let ids = tokenizer.encode_with_specials(&prompt, false, false)?;
            if ids.len() >= target {
                break (prompt, ids);
            }
            ensure!(lines < 1000, "archive exceeded bound");
            lines += 1;
            writeln!(
                archive,
                "Archive {lines:03}: stone, wool, copper; label {:04}.",
                (lines * 73 + 19) % 10000
            )?;
        };
        ensure!(
            tokenizer.decode(&ids)? == prompt,
            "prompt round trip failed"
        );
        if name == "short" {
            ensure!(
                ids.len() == 65 && json!(ids) == oracle["prompt_token_ids"],
                "historical token IDs differ"
            );
        }
        let context = if name == "short" {
            workload::CONTEXT_TOKENS as usize
        } else {
            (ids.len() + max_tokens).next_multiple_of(256)
        };
        cases.push(json!({"name":name,"minimum_prompt_tokens":target,"prompt_token_count":ids.len(),
            "prompt_ids":ids,"prompt_sha256":sha256(prompt.as_bytes())?,"archive_lines":lines,"context":context,
            "request":{"prompt":prompt,"prompt_format":"raw","max_tokens":max_tokens,"seed":0,
            "top_k":1,"stream":false,"include_metrics":true}}));
    }
    let manifest = json!({"model_sha256":model_hash,"oracle_report_sha256":file_sha256(&args.oracle_report)?,"cases":cases});
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(args.output)?;
    serde_json::to_writer(&mut output, &manifest)?;
    writeln!(output)?;
    Ok(())
}

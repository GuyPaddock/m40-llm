//! CPU-only runtime tokenization for source eligibility and normalized span audits.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use m40_llm::{
    generate::{prepare_prompt, PromptFormat},
    gguf::load_gguf,
    tokenizer::Tokenizer,
};
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::BTreeSet,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::Command,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Deserialize)]
struct Record {
    id: String,
    text: String,
    comparison_text: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !cfg!(feature = "cuda"),
        "CPU-only audit requires --no-default-features"
    );
    ensure!(
        !args.output.exists(),
        "refusing to overwrite token evidence"
    );
    let digest = Command::new("sha256sum").arg(&args.model).output()?;
    ensure!(digest.status.success(), "model hash command failed");
    ensure!(
        String::from_utf8(digest.stdout)?.split_whitespace().next()
            == Some("4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"),
        "target model hash differs"
    );
    let gguf = load_gguf(&args.model)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata)?;
    ensure!(tokenizer.is_qwen2(), "original Qwen tokenizer required");
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.output)?;
    let mut seen = BTreeSet::new();
    for line in BufReader::new(File::open(&args.input)?).lines() {
        let record: Record = serde_json::from_str(&line?)?;
        ensure!(seen.insert(record.id.clone()), "duplicate source ID");
        let (prompt, bos) = prepare_prompt(&tokenizer, &record.text, PromptFormat::QwenChat);
        ensure!(!bos, "unexpected BOS");
        let ids = tokenizer.encode_with_specials(&prompt, false, false)?;
        ensure!(
            tokenizer.decode(&ids)? == prompt,
            "prompt roundtrip differs at {}",
            record.id
        );
        let comparison = tokenizer
            .encode(&record.comparison_text)
            .context("comparison tokenization")?;
        serde_json::to_writer(
            &mut output,
            &json!({"id": record.id,
            "prompt_tokens": ids.len(), "prompt_ids": ids, "comparison_ids": comparison}),
        )?;
        writeln!(output)?;
    }
    eprintln!(
        "CPU-tokenized {} source records; no CUDA context created",
        seen.len()
    );
    Ok(())
}

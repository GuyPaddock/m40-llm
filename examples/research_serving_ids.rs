//! Resolve a stored HTTP request using Phase 4's serving prompt and tokenizer.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use m40_llm::{
    generate::{prepare_prompt, PromptFormat},
    gguf::load_gguf,
    tokenizer::Tokenizer,
};
use serde_json::{json, Value};
use std::{fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    model: PathBuf,
    request: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let gguf = load_gguf(&args.model)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata)?;
    let body: Value = serde_json::from_slice(&fs::read(&args.request)?)?;
    ensure!(body["prompt_format"] == "raw", "requires frozen raw prompt");
    let prompt = body["prompt"].as_str().context("prompt missing")?;
    let (prepared, add_bos) = prepare_prompt(&tokenizer, prompt, PromptFormat::Raw);
    ensure!(prepared == prompt, "raw prompt changed");
    let ids = tokenizer.encode_with_specials(&prepared, add_bos, false)?;
    println!(
        "{}",
        json!({"ids":ids,"length":ids.len(),"add_bos":add_bos,"stop_ids":tokenizer.stop_ids()})
    );
    Ok(())
}

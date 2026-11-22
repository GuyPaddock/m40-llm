//! Exercise the pinned main loader and raw serving prompt preparation on real GGUFs.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use m40_llm::{
    generate::{prepare_prompt, PromptFormat},
    gguf::load_gguf,
    infer::LoadedModel,
    tokenizer::Tokenizer,
};
use serde_json::{json, Value};
use std::{fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    model: PathBuf,
    manifest: PathBuf,
    #[arg(long)]
    tiny: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let gguf = load_gguf(&args.model)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata)?;
    let fixture: Value = serde_json::from_slice(&fs::read(args.manifest)?)?;
    let mut prompts = Vec::new();
    if args.tiny {
        prompts.push(("tiny", "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nExplain how a web browser loads a page, from typing the URL to displaying the result.</s>\n<|assistant|>\n"));
    } else if tokenizer.is_llama3() {
        prompts.push(("llama3","<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nExplain how a web browser loads a page.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"));
    } else {
        for case in fixture["cases"].as_array().context("missing cases")? {
            prompts.push((
                case["name"].as_str().context("case name")?,
                case["request"]["prompt"].as_str().context("case prompt")?,
            ));
        }
    }
    let mut inputs = Vec::new();
    for (name, prompt) in prompts {
        let (raw, bos) = prepare_prompt(&tokenizer, prompt, PromptFormat::Raw);
        ensure!(raw == prompt, "raw prompt bytes changed");
        let ids = tokenizer.encode_with_specials(&raw, bos, false)?;
        inputs.push(json!({"name":name,"add_bos":bos,"length":ids.len(),"ids":ids,"prompt":raw,"stop_ids":tokenizer.stop_ids()}));
    }
    let model = LoadedModel::from_gguf(gguf, fs::read(&args.model)?, -1)?;
    let layers = model.validate_standard_layers();
    let head = model.map_lm_head();
    println!(
        "{}",
        json!({"path":args.model,"architecture":model.model_config.architecture,
        "layers":model.model_config.block_count,"context":model.model_config.context_length,
        "mapped_all":layers.is_ok(),"layer_dimensions":layers.as_ref().ok(),
        "layer_error":layers.as_ref().err().map(|e|format!("{e:#}")),
        "mapped_head":head.is_ok(),"head_error":head.as_ref().err().map(|e|format!("{e:#}")),
        "head":head.as_ref().ok().map(|(name,tensor,dim,vocab,tied)|json!({"name":name,"type":tensor.dtype,"width":dim,"vocab":vocab,"tied":tied})),
        "inputs":inputs})
    );
    Ok(())
}

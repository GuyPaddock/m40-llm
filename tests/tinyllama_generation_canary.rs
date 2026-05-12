#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use m40_llm::generate::{generate_text, GenerateOptions};
use m40_llm::gguf;
use m40_llm::infer::LoadedModel;
use std::path::PathBuf;

const DEFAULT_TINYLLAMA: &str =
    "/mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf";
const EXPECTED_GENERATED_TOKENS: &[u32] = &[
    13, 13, 29896, 29889, 22402, 385, 1409, 310, 10961, 29879, 411, 1009, 5829, 29892, 1024, 29892,
    322, 8666, 472, 278,
];

struct EnvRestore {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvRestore {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        std::env::set_var(key, value);
        Self { key, previous }
    }
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => std::env::set_var(self.key, value),
            None => std::env::remove_var(self.key),
        }
    }
}

fn tinyllama_canary_model_path() -> Option<PathBuf> {
    std::env::var_os("M40LLM_TINYLLAMA_CANARY_MODEL")
        .map(PathBuf::from)
        .or_else(|| {
            let path = PathBuf::from(DEFAULT_TINYLLAMA);
            path.exists().then_some(path)
        })
}

#[test]
fn tinyllama_stock_quotes_prompt_matches_expected_tokens() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{e}");
        return Ok(());
    }
    drop(ctx);

    let Some(path) = tinyllama_canary_model_path() else {
        eprintln!("skipping: set M40LLM_TINYLLAMA_CANARY_MODEL to run TinyLlama canary");
        return Ok(());
    };

    // Keep this canary on the normal async decode path. Graph mode is covered by
    // separate fixed-shape smoke tests until full-token graph capture is ready.
    let _graph_env = EnvRestore::set("M40LLM_DECODE_GRAPH", "0");

    let gguf_bytes = std::fs::read(&path)?;
    let gguf_model = gguf::load_gguf(&path)?;
    let mut model = LoadedModel::from_gguf(gguf_model, gguf_bytes, -1)?;
    model.allocate_kv_cache_for_layers(model.model_config.context_length)?;

    let generated = generate_text(
        &model,
        GenerateOptions {
            prompt: "Please write a small Java program to check for stock quotes: ".to_string(),
            max_tokens: Some(EXPECTED_GENERATED_TOKENS.len()),
            top_k: Some(10),
            log_prefix: "tinyllama_canary",
            ..Default::default()
        },
    )?;

    let generated_start = generated
        .token_ids
        .len()
        .checked_sub(EXPECTED_GENERATED_TOKENS.len())
        .expect("generated ids should include prompt plus generated tokens");
    let generated_tokens = &generated.token_ids[generated_start..];

    assert_eq!(
        generated_tokens, EXPECTED_GENERATED_TOKENS,
        "TinyLlama generation canary regressed; output was {:?}",
        generated.output
    );
    assert_eq!(
        generated.output,
        "\n\n1. Define an array of stocks with their symbol, name, and price at the"
    );
    Ok(())
}

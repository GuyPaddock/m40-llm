#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::Result;
use m40_llm::generate::{generate_text, GenerateOptions};
use m40_llm::gguf;
use m40_llm::infer::LoadedModel;
use std::path::PathBuf;

const DEFAULT_TINYLLAMA: &str =
    "/mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf";
const EXPECTED_GENERATED_TOKEN_COUNT: usize = 19;
const STOCK_PROMPT: &str = "Please write a small Java program to check for stock quotes: ";

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

fn load_tinyllama_canary_model() -> Result<Option<LoadedModel>> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(None),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{e}");
        return Ok(None);
    }
    drop(ctx);

    let Some(path) = tinyllama_canary_model_path() else {
        eprintln!("skipping: set M40LLM_TINYLLAMA_CANARY_MODEL to run TinyLlama canary");
        return Ok(None);
    };

    let gguf_bytes = std::fs::read(&path)?;
    let gguf_model = gguf::load_gguf(&path)?;
    let mut model = LoadedModel::from_gguf(gguf_model, gguf_bytes, -1)?;
    model.allocate_kv_cache_for_layers(model.model_config.context_length)?;
    Ok(Some(model))
}

fn generate_stock_quotes(model: &LoadedModel, seed: u64) -> Result<(Vec<u32>, String)> {
    let generated = generate_text(
        model,
        GenerateOptions {
            prompt: STOCK_PROMPT.to_string(),
            max_tokens: Some(EXPECTED_GENERATED_TOKEN_COUNT),
            top_k: Some(10),
            temperature: Some(0.7),
            seed: Some(seed),
            log_prefix: "tinyllama_canary",
            ..Default::default()
        },
    )?;

    let generated_start = generated
        .token_ids
        .len()
        .checked_sub(EXPECTED_GENERATED_TOKEN_COUNT)
        .expect("generated ids should include prompt plus generated tokens");
    let generated_tokens = generated.token_ids[generated_start..].to_vec();
    Ok((generated_tokens, generated.output))
}

fn assert_stock_quotes_output_looks_reasonable(output: &str) {
    let lower = output.to_lowercase();
    let has_keyword = ["stock", "quote", "price", "java", "array", "import", "code"]
        .iter()
        .any(|kw| lower.contains(kw));
    assert!(
        has_keyword,
        "canary output should contain stock- or Java-related terms; got: {output}"
    );
    assert!(
        output.chars().any(|ch| ch.is_alphabetic()),
        "canary output should contain alphabetic text: {output}"
    );
    assert!(
        output.len() >= 24,
        "canary output should not be too short: {output}"
    );
    assert!(
        !output.contains("Helloijk"),
        "canary output should not look like prior broken token path: {output}"
    );
}

#[test]
fn tinyllama_stock_quotes_prompt_seeded_anchor() -> Result<()> {
    let Some(model) = load_tinyllama_canary_model()? else {
        return Ok(());
    };
    let _graph_env = EnvRestore::set("M40LLM_DECODE_GRAPH", "0");

    let (generated_tokens, output) = generate_stock_quotes(&model, 0)?;
    assert_eq!(
        generated_tokens.len(),
        EXPECTED_GENERATED_TOKEN_COUNT,
        "TinyLlama generation canary token count changed; output was {output:?}"
    );
    assert_stock_quotes_output_looks_reasonable(&output);
    Ok(())
}

#[test]
fn tinyllama_stock_quotes_prompt_multi_seed_sanity() -> Result<()> {
    let Some(model) = load_tinyllama_canary_model()? else {
        return Ok(());
    };
    let _graph_env = EnvRestore::set("M40LLM_DECODE_GRAPH", "0");

    for seed in [0u64, 7u64, 42u64] {
        let (generated_tokens, output) = generate_stock_quotes(&model, seed)?;
        assert_eq!(
            generated_tokens.len(),
            EXPECTED_GENERATED_TOKEN_COUNT,
            "TinyLlama canary should produce expected token count for seed {seed}"
        );
        assert_stock_quotes_output_looks_reasonable(&output);
    }
    Ok(())
}

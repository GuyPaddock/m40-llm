#![cfg(feature = "cuda")]

use anyhow::{Context, Result};
use m40_llm::generate::{generate_text, GenerateOptions};
use m40_llm::gguf;
use m40_llm::infer::LoadedModel;
use m40_llm::kv_compression::{KvCompressMode, KvCompressionConfig};
use m40_llm::tokenizer::Tokenizer;

const NEEDLE: &str = "ZXQ-NEEDLE-41729";

fn load_model_from_env() -> Result<Option<LoadedModel>> {
    let Some(path) = std::env::var_os("M40LLM_LONG_CONTEXT_RETRIEVAL_MODEL") else {
        eprintln!("M40LLM_LONG_CONTEXT_RETRIEVAL_MODEL not set; skipping long-context KV quality");
        return Ok(None);
    };
    let path = std::path::PathBuf::from(path);
    if !path.exists() {
        eprintln!(
            "M40LLM_LONG_CONTEXT_RETRIEVAL_MODEL does not exist: {}",
            path.display()
        );
        return Ok(None);
    }
    let gguf_bytes = std::fs::read(&path)?;
    let gguf_model = gguf::load_gguf(&path)?;
    let mut model = LoadedModel::from_gguf(gguf_model, gguf_bytes, -1)?;
    model.allocate_kv_cache_for_layers(model.model_config.context_length)?;
    Ok(Some(model))
}

fn retrieval_prompt(tokenizer: &Tokenizer, target_tokens: usize, needle_position: &str) -> String {
    let mut prompt = String::new();
    prompt.push_str("You are doing an exact retrieval task. ");
    prompt.push_str("When asked, answer with only the secret code.\n");
    if needle_position == "old" {
        prompt.push_str("Important secret code: ");
        prompt.push_str(NEEDLE);
        prompt.push('\n');
    }
    while tokenizer
        .encode_with_specials(&prompt, true, false)
        .map(|ids| ids.len())
        .unwrap_or(usize::MAX)
        < target_tokens.saturating_sub(64)
    {
        prompt
            .push_str("Filler sentence about CUDA kernels, memory bandwidth, and cache locality. ");
    }
    if needle_position == "recent" {
        prompt.push_str("\nImportant secret code: ");
        prompt.push_str(NEEDLE);
        prompt.push('\n');
    }
    prompt.push_str("\nQuestion: What is the secret code? Answer with only the code.");
    prompt
}

fn run_retrieval_case(
    model: &LoadedModel,
    tokenizer: &Tokenizer,
    target_tokens: usize,
    needle_position: &str,
    mode: KvCompressMode,
) -> Result<String> {
    let prompt = retrieval_prompt(tokenizer, target_tokens, needle_position);
    let prompt_tokens = tokenizer
        .encode_with_specials(&prompt, true, false)
        .context("encode retrieval prompt")?
        .len();
    if prompt_tokens >= model.model_config.context_length as usize {
        anyhow::bail!(
            "retrieval prompt has {prompt_tokens} tokens, model context is {}",
            model.model_config.context_length
        );
    }
    let generated = generate_text(
        model,
        GenerateOptions {
            prompt,
            max_tokens: Some(24),
            top_k: Some(1),
            log_prefix: "kv_retrieval",
            kv_compression: KvCompressionConfig {
                mode,
                recent_window: 1024,
                block_size: 32,
                top_blocks: 16,
                representatives: 2,
            },
            ..Default::default()
        },
    )?;
    Ok(generated.output)
}

#[test]
fn long_context_needle_retrieval_quality_smoke() -> Result<()> {
    let Some(model) = load_model_from_env()? else {
        return Ok(());
    };
    let tokenizer = Tokenizer::from_gguf_metadata(&model.gguf.metadata)
        .unwrap_or_else(|_| Tokenizer::byte_level());
    let contexts = [4096usize, 8192, 16384, 32768];
    let modes = [
        KvCompressMode::Off,
        KvCompressMode::BlockSelectExact,
        KvCompressMode::BlockSummary,
        KvCompressMode::BlockSelectLossy,
    ];

    for target_tokens in contexts {
        if target_tokens + 128 >= model.model_config.context_length as usize {
            eprintln!(
                "skipping target_tokens={target_tokens}; model context={}",
                model.model_config.context_length
            );
            continue;
        }
        for needle_position in ["old", "recent"] {
            for mode in modes {
                let output =
                    run_retrieval_case(&model, &tokenizer, target_tokens, needle_position, mode)?;
                assert!(
                    output.contains(NEEDLE),
                    "mode={mode:?} target_tokens={target_tokens} needle_position={needle_position} failed retrieval; output={output:?}"
                );
            }
        }
    }
    Ok(())
}

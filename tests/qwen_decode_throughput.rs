#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::{Context, Result};
use m40_llm::generate::{generate_text, GenerateOptions, PromptFormat};
use m40_llm::gguf;
use m40_llm::infer::LoadedModel;
use m40_llm::kv_compression::KvCompressionConfig;
use std::path::{Path, PathBuf};

const DEFAULT_QWEN_F16: &str =
    "/mnt/array-fastest/home/guyep/.cache/m40-llm/models/Qwen2.5-3B-Instruct-f16.gguf";
const DEFAULT_QWEN_Q8: &str = "/mnt/array-fastest/home/guyep/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d";
const DEFAULT_PROMPT: &str =
    "Repeat the word BLUE over and over, separated by spaces. Continue until stopped.";

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

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str) -> Option<f64> {
    std::env::var(key).ok().and_then(|value| value.parse().ok())
}

fn model_path_from_env() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("M40LLM_QWEN_THROUGHPUT_MODEL").map(PathBuf::from) {
        return Some(path);
    }
    for default in [DEFAULT_QWEN_F16, DEFAULT_QWEN_Q8] {
        let path = PathBuf::from(default);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn load_model(path: &Path, context_bound: u32, kv: &KvCompressionConfig) -> Result<LoadedModel> {
    let gguf_bytes = std::fs::read(path)
        .with_context(|| format!("read model weights from {}", path.display()))?;
    let gguf_model =
        gguf::load_gguf(path).with_context(|| format!("parse GGUF at {}", path.display()))?;
    let mut model = LoadedModel::from_gguf(gguf_model, gguf_bytes, -1)
        .with_context(|| format!("load model from {}", path.display()))?;
    let context_tokens = context_bound.min(model.model_config.context_length);
    if kv.mode.is_enabled() {
        model
            .allocate_compressed_kv_cache_for_layers(context_tokens, kv)
            .with_context(|| {
                format!("allocate compressed KV cache with {context_tokens} tokens")
            })?;
    } else {
        model
            .allocate_kv_cache_for_layers(context_tokens)
            .with_context(|| format!("allocate dense KV cache with {context_tokens} tokens"))?;
    }
    Ok(model)
}

fn throughput_kv_config() -> KvCompressionConfig {
    match std::env::var("M40LLM_QWEN_THROUGHPUT_KV").ok().as_deref() {
        Some("default") | Some("compressed") => KvCompressionConfig::default(),
        Some("off") | Some("dense") | None => KvCompressionConfig::dense_reference(),
        Some(other) => panic!(
            "unsupported M40LLM_QWEN_THROUGHPUT_KV={other}; use off/dense or default/compressed"
        ),
    }
}

#[test]
fn qwen_decode_throughput_goal_probe() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("skipping qwen throughput probe: {e}");
        return Ok(());
    }

    let Some(model_path) = model_path_from_env() else {
        eprintln!(
            "skipping qwen throughput probe: set M40LLM_QWEN_THROUGHPUT_MODEL or place a supported Qwen2.5 GGUF at the default cache path"
        );
        return Ok(());
    };

    let backend =
        std::env::var("M40LLM_QWEN_THROUGHPUT_BACKEND").unwrap_or_else(|_| "fast-fits".to_string());
    let _backend_env = EnvRestore::set("M40LLM_PROJECTION_BACKEND", &backend);
    let _materialize_env = EnvRestore::set(
        "M40LLM_MATERIALIZE_F32_WEIGHTS",
        if backend == "fast-fits" { "1" } else { "0" },
    );
    let kv = throughput_kv_config();
    let context_bound = env_u32("M40LLM_QWEN_THROUGHPUT_CONTEXT", 2048);
    let model = load_model(&model_path, context_bound, &kv)?;

    let prompt =
        std::env::var("M40LLM_QWEN_THROUGHPUT_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.into());
    let max_tokens = env_usize("M40LLM_QWEN_THROUGHPUT_MAX_TOKENS", 96);
    let min_generated = env_usize("M40LLM_QWEN_THROUGHPUT_MIN_GENERATED", 32);

    let generated = generate_text(
        &model,
        GenerateOptions {
            prompt,
            max_tokens: Some(max_tokens),
            top_k: Some(1),
            temperature: None,
            seed: Some(0),
            log_prefix: "qwen_decode_throughput",
            kv_compression: kv.clone(),
            prompt_format: PromptFormat::Auto,
            ..Default::default()
        },
    )?;

    let generated_tokens = generated
        .token_ids
        .len()
        .saturating_sub(generated.prompt_token_len);
    let decode_tps = if generated.generated_decode_elapsed_ms > 0 {
        generated_tokens as f64 * 1000.0 / generated.generated_decode_elapsed_ms as f64
    } else {
        0.0
    };
    let total_tps = if generated.total_elapsed_ms > 0 {
        generated_tokens as f64 * 1000.0 / generated.total_elapsed_ms as f64
    } else {
        0.0
    };

    eprintln!(
        "[qwen-throughput] model={} backend={} kv_mode={:?} top_blocks={} prompt_tokens={} generated_tokens={} prefill_ms={} decode_ms={} total_ms={} decode_tps={:.3} total_tps={:.3} prefill_mode={} final_kv_bytes={:?} dense_equiv_kv_bytes={:?} materialized_bytes={:?} output={:?}",
        model_path.display(),
        backend,
        kv.mode,
        kv.top_blocks,
        generated.prompt_token_len,
        generated_tokens,
        generated.prompt_prefill_elapsed_ms,
        generated.generated_decode_elapsed_ms,
        generated.total_elapsed_ms,
        decode_tps,
        total_tps,
        generated.prefill_mode,
        generated.final_kv_allocated_bytes,
        generated.dense_equivalent_kv_bytes,
        generated.materialized_f32_cache_bytes,
        generated.output,
    );

    assert!(
        generated_tokens >= min_generated,
        "throughput probe generated too few tokens: {generated_tokens} < {min_generated}; output={:?}",
        generated.output
    );
    assert!(
        !generated.output.trim().is_empty(),
        "throughput probe should produce non-empty output"
    );
    assert!(
        !generated.output.contains('\0') && !generated.output.contains('\u{fffd}'),
        "throughput probe output should not contain invalid replacement/control data: {:?}",
        generated.output
    );
    if let Some(min_tps) = env_f64("M40LLM_QWEN_THROUGHPUT_MIN_TPS") {
        assert!(
            decode_tps >= min_tps,
            "Qwen decode throughput {decode_tps:.3} tok/s below required {min_tps:.3} tok/s"
        );
    }

    Ok(())
}

#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::{Context, Result};
use m40_llm::generate::{generate_text, GenerateOptions, GeneratedText, PromptFormat};
use m40_llm::gguf;
use m40_llm::infer::LoadedModel;
use m40_llm::kv_compression::KvCompressionConfig;
use m40_llm::profile::{self, ProfileSnapshot};
use std::path::{Path, PathBuf};

const DEFAULT_CONTEXT_TOKENS: u32 = 256;
const DEFAULT_COMPRESSED_CONTEXT_TOKENS: u32 = 2048;
const DEFAULT_QWEN_F16: &str =
    "/mnt/array-fastest/home/guyep/.cache/m40-llm/models/Qwen2.5-3B-Instruct-f16.gguf";
const DEFAULT_QWEN_Q8: &str = "/mnt/array-fastest/home/guyep/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d";

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

    fn unset(key: &'static str) -> Self {
        let previous = std::env::var_os(key);
        std::env::remove_var(key);
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

#[derive(Debug, Clone, Copy)]
struct PromptCase {
    name: &'static str,
    prompt: &'static str,
    expected_substring: Option<&'static str>,
}

const PROMPTS: [PromptCase; 3] = [
    PromptCase {
        name: "arithmetic",
        prompt: "What is 2+2? Answer with one digit.",
        expected_substring: Some("4"),
    },
    PromptCase {
        name: "exact_ok",
        prompt: "Reply with exactly the word OK.",
        expected_substring: Some("OK"),
    },
    PromptCase {
        name: "config_lookup",
        prompt: "Configuration note: the active value is M40_BATCH_LIMIT=37. What is M40_BATCH_LIMIT? Answer with the number only.",
        expected_substring: Some("37"),
    },
];

const Q8_PREFILL_CHUNK_SIZES: [Option<usize>; 4] = [None, Some(32), Some(64), Some(128)];
const Q8_COMPRESSED_PROMPTS: [PromptCase; 2] = [PROMPTS[0], PROMPTS[2]];

fn env_flag(key: &str) -> bool {
    matches!(
        std::env::var(key).ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn optional_model_path(key: &str, default: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from).or_else(|| {
        let path = PathBuf::from(default);
        path.exists().then_some(path)
    })
}

fn load_model(path: &Path, context_bound: u32) -> Result<LoadedModel> {
    let gguf_bytes = std::fs::read(path)
        .with_context(|| format!("read model weights from {}", path.display()))?;
    let gguf_model =
        gguf::load_gguf(path).with_context(|| format!("parse GGUF at {}", path.display()))?;
    let mut model = LoadedModel::from_gguf(gguf_model, gguf_bytes, -1)
        .with_context(|| format!("load model from {}", path.display()))?;
    let context_tokens = context_bound.min(model.model_config.context_length);
    model
        .allocate_kv_cache_for_layers(context_tokens)
        .with_context(|| format!("allocate dense KV cache with {context_tokens} tokens"))?;
    Ok(model)
}

fn load_compressed_model(
    path: &Path,
    context_bound: u32,
    kv_compression: &KvCompressionConfig,
) -> Result<LoadedModel> {
    let gguf_bytes = std::fs::read(path)
        .with_context(|| format!("read model weights from {}", path.display()))?;
    let gguf_model =
        gguf::load_gguf(path).with_context(|| format!("parse GGUF at {}", path.display()))?;
    let mut model = LoadedModel::from_gguf(gguf_model, gguf_bytes, -1)
        .with_context(|| format!("load model from {}", path.display()))?;
    let context_tokens = context_bound.min(model.model_config.context_length);
    model
        .allocate_compressed_kv_cache_for_layers(context_tokens, kv_compression)
        .with_context(|| format!("allocate compressed KV cache with {context_tokens} tokens"))?;
    Ok(model)
}

fn q8_projection_launches(snapshot: &ProfileSnapshot) -> u64 {
    snapshot
        .by_op
        .iter()
        .filter(|(op, _)| op.contains("q8_0"))
        .map(|(_, counters)| counters.launches)
        .sum()
}

fn cublas_calls(snapshot: &ProfileSnapshot) -> u64 {
    snapshot
        .by_op
        .values()
        .map(|counters| counters.cublas_calls)
        .sum()
}

fn output_quality_pass(generated: &GeneratedText, expected_substring: Option<&str>) -> bool {
    let trimmed = generated.output.trim();
    !trimmed.is_empty()
        && !generated.output.contains('\0')
        && !generated.output.contains('\u{fffd}')
        && expected_substring
            .map(|expected| generated.output.contains(expected))
            .unwrap_or(true)
}

fn run_case(
    model: &LoadedModel,
    backend: &str,
    prompt: PromptCase,
    kv_compression: KvCompressionConfig,
    prefill_chunk_size: Option<usize>,
) -> Result<()> {
    let _prefill_chunk = match prefill_chunk_size {
        Some(size) => EnvRestore::set("M40LLM_PREFILL_CHUNK_SIZE", &size.to_string()),
        None => EnvRestore::unset("M40LLM_PREFILL_CHUNK_SIZE"),
    };
    profile::reset();
    let generated = generate_text(
        model,
        GenerateOptions {
            prompt: prompt.prompt.to_string(),
            max_tokens: Some(env_u32("M40LLM_QWEN_BACKEND_COMPARE_MAX_TOKENS", 16) as usize),
            top_k: Some(1),
            temperature: None,
            seed: Some(0),
            log_prefix: "qwen_backend_compare",
            kv_compression,
            prompt_format: PromptFormat::Auto,
            ..Default::default()
        },
    )?;
    let snapshot = profile::snapshot();
    let quality_pass = output_quality_pass(&generated, prompt.expected_substring);
    let prompt_tps = if generated.prompt_prefill_elapsed_ms > 0 {
        Some(
            generated.prompt_token_len as f64 * 1000.0 / generated.prompt_prefill_elapsed_ms as f64,
        )
    } else {
        None
    };
    let generated_tokens = generated
        .token_ids
        .len()
        .saturating_sub(generated.prompt_token_len);
    let decode_tps = if generated.generated_decode_elapsed_ms > 0 {
        Some(generated_tokens as f64 * 1000.0 / generated.generated_decode_elapsed_ms as f64)
    } else {
        None
    };

    eprintln!(
        "[qwen-backend-compare] backend={} case={} pass={} prompt_tokens={} generated_tokens={} prefill_ms={} decode_ms={} total_ms={} prefill_tps={} decode_tps={} prefill_mode={} prefill_chunk_size={:?} final_kv_bytes={:?} dense_equiv_kv_bytes={:?} materialized_entries_before={:?} materialized_entries_after_prompt={:?} materialized_entries={:?} materialized_bytes_before={:?} materialized_bytes_after_prompt={:?} materialized_bytes={:?} materialized_bytes_added_prompt={:?} materialized_bytes_added_total={:?} q8_projection_launches={} cublas_calls={} output={:?}",
        backend,
        prompt.name,
        quality_pass,
        generated.prompt_token_len,
        generated_tokens,
        generated.prompt_prefill_elapsed_ms,
        generated.generated_decode_elapsed_ms,
        generated.total_elapsed_ms,
        prompt_tps
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
        decode_tps
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
        generated.prefill_mode,
        generated.prefill_chunk_size,
        generated.final_kv_allocated_bytes,
        generated.dense_equivalent_kv_bytes,
        generated.materialized_f32_cache_entries_before,
        generated.materialized_f32_cache_entries_after_prompt,
        generated.materialized_f32_cache_entries,
        generated.materialized_f32_cache_bytes_before,
        generated.materialized_f32_cache_bytes_after_prompt,
        generated.materialized_f32_cache_bytes,
        generated.materialized_f32_cache_bytes_added_prompt,
        generated.materialized_f32_cache_bytes_added_total,
        q8_projection_launches(&snapshot),
        cublas_calls(&snapshot),
        generated.output,
    );

    assert!(
        quality_pass,
        "backend {backend} case {} should pass lightweight output check; output={:?}",
        prompt.name, generated.output
    );
    Ok(())
}

#[test]
fn qwen_f16_fast_fits_vs_q8_large_model_dense_kv() -> Result<()> {
    if !env_flag("M40LLM_QWEN_BACKEND_COMPARE") {
        eprintln!("skipping: set M40LLM_QWEN_BACKEND_COMPARE=1 to compare Qwen F16 fast-fits against Qwen Q8_0 large-model");
        return Ok(());
    }

    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{e}");
        return Ok(());
    }
    drop(ctx);

    let Some(f16_path) = optional_model_path("M40LLM_QWEN_F16_MODEL", DEFAULT_QWEN_F16) else {
        eprintln!("skipping: set M40LLM_QWEN_F16_MODEL=/path/to/Qwen2.5-3B-Instruct-f16.gguf");
        return Ok(());
    };
    let Some(q8_path) = optional_model_path("M40LLM_QWEN_Q8_MODEL", DEFAULT_QWEN_Q8) else {
        eprintln!("skipping: set M40LLM_QWEN_Q8_MODEL=/path/to/Qwen2.5-3B-Instruct-q8_0.gguf");
        return Ok(());
    };

    let context_bound = env_u32(
        "M40LLM_QWEN_BACKEND_COMPARE_CONTEXT_TOKENS",
        DEFAULT_CONTEXT_TOKENS,
    );

    {
        let _backend = EnvRestore::set("M40LLM_PROJECTION_BACKEND", "fast-fits");
        let _materialize = EnvRestore::unset("M40LLM_MATERIALIZE_F32_WEIGHTS");
        let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "0");
        let model = load_model(&f16_path, context_bound)?;
        eprintln!(
            "[qwen-backend-compare] backend=f16-fast-fits model={} context_bound={}",
            f16_path.display(),
            context_bound.min(model.model_config.context_length)
        );
        for prompt in PROMPTS {
            run_case(
                &model,
                "f16-fast-fits",
                prompt,
                KvCompressionConfig::dense_reference(),
                None,
            )?;
        }
    }

    {
        let _backend = EnvRestore::set("M40LLM_PROJECTION_BACKEND", "large-model");
        let _materialize = EnvRestore::set("M40LLM_MATERIALIZE_F32_WEIGHTS", "0");
        let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "0");
        let model = load_model(&q8_path, context_bound)?;
        eprintln!(
            "[qwen-backend-compare] backend=q8-large-model model={} context_bound={}",
            q8_path.display(),
            context_bound.min(model.model_config.context_length)
        );
        for prefill_chunk_size in Q8_PREFILL_CHUNK_SIZES {
            for prompt in PROMPTS {
                run_case(
                    &model,
                    match prefill_chunk_size {
                        Some(_) => "q8-large-model-packed-prefix",
                        None => "q8-large-model-sequential-prefill",
                    },
                    prompt,
                    KvCompressionConfig::dense_reference(),
                    prefill_chunk_size,
                )?;
            }
        }
    }

    {
        let _backend = EnvRestore::set("M40LLM_PROJECTION_BACKEND", "large-model");
        let _materialize = EnvRestore::set("M40LLM_MATERIALIZE_F32_WEIGHTS", "0");
        let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "0");
        let compressed_context_bound = env_u32(
            "M40LLM_QWEN_BACKEND_COMPARE_COMPRESSED_CONTEXT_TOKENS",
            DEFAULT_COMPRESSED_CONTEXT_TOKENS,
        );
        let kv_compression = KvCompressionConfig::default();
        let model = load_compressed_model(&q8_path, compressed_context_bound, &kv_compression)?;
        eprintln!(
            "[qwen-backend-compare] backend=q8-large-model-compressed-top8 model={} context_bound={} kv_mode={:?} top_blocks={} exact_old_backing={} exact_old_attention={}",
            q8_path.display(),
            compressed_context_bound.min(model.model_config.context_length),
            kv_compression.mode,
            kv_compression.top_blocks,
            kv_compression.effective_exact_old_backing().as_str(),
            kv_compression.effective_exact_old_attention().as_str()
        );
        for prompt in Q8_COMPRESSED_PROMPTS {
            run_case(
                &model,
                "q8-large-model-compressed-top8",
                prompt,
                kv_compression.clone(),
                None,
            )?;
        }
    }

    Ok(())
}

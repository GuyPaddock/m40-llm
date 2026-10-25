#![cfg(all(feature = "cuda", nvcc))]

mod cuda_env;

use anyhow::{Context, Result};
use m40_llm::generate::{generate_text, GenerateOptions, PromptFormat};
use m40_llm::gguf::{self, GgmlDType, GgufModel};
use m40_llm::infer::{LoadedModel, ModelConfig};
use m40_llm::kv_compression::KvCompressionConfig;
use m40_llm::profile;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const DEFAULT_MAX_CONTEXT_TOKENS: u32 = 256;
const DEFAULT_MAX_WEIGHT_MB: u64 = 8 * 1024;
const PROJECTION_NAMES: [&str; 7] = ["wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"];

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

#[derive(Debug)]
struct Q8Probe {
    architecture: String,
    block_count: u32,
    context_length: u32,
    d_model: u32,
    head_dim: u32,
    q_heads: u32,
    kv_heads: u32,
    weights_bytes: u64,
    dense_kv_bytes_at_bound: u64,
    q8_projection_tensors: usize,
    expected_projection_tensors: usize,
    dtype_counts: BTreeMap<String, usize>,
    unsupported_reason: Option<String>,
}

impl Q8Probe {
    fn supported(&self) -> bool {
        self.unsupported_reason.is_none()
    }

    fn log(&self, path: &Path, context_bound: u32) {
        eprintln!(
            "[q8-canary] model={} arch={} layers={} d_model={} q_heads={} kv_heads={} head_dim={} context={} context_bound={} weights_bytes={} dense_kv_bytes_at_bound={} q8_projection_tensors={}/{} dtype_counts={:?} supported={}{}",
            path.display(),
            self.architecture,
            self.block_count,
            self.d_model,
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            self.context_length,
            context_bound,
            self.weights_bytes,
            self.dense_kv_bytes_at_bound,
            self.q8_projection_tensors,
            self.expected_projection_tensors,
            self.dtype_counts,
            self.supported(),
            self.unsupported_reason
                .as_ref()
                .map(|reason| format!(" reason={reason}"))
                .unwrap_or_default()
        );
    }
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn candidate_projection_names(layer: u32, logical: &str) -> [String; 2] {
    match logical {
        "wq" => [
            format!("layers.{layer}.attention.wq.weight"),
            format!("blk.{layer}.attn_q.weight"),
        ],
        "wk" => [
            format!("layers.{layer}.attention.wk.weight"),
            format!("blk.{layer}.attn_k.weight"),
        ],
        "wv" => [
            format!("layers.{layer}.attention.wv.weight"),
            format!("blk.{layer}.attn_v.weight"),
        ],
        "wo" => [
            format!("layers.{layer}.attention.wo.weight"),
            format!("blk.{layer}.attn_output.weight"),
        ],
        "w_gate" => [
            format!("layers.{layer}.feed_forward.w3.weight"),
            format!("blk.{layer}.ffn_gate.weight"),
        ],
        "w_up" => [
            format!("layers.{layer}.feed_forward.w1.weight"),
            format!("blk.{layer}.ffn_up.weight"),
        ],
        "w_down" => [
            format!("layers.{layer}.feed_forward.w2.weight"),
            format!("blk.{layer}.ffn_down.weight"),
        ],
        _ => unreachable!("unknown projection name"),
    }
}

fn probe_q8_model(gguf_model: &GgufModel, weights_bytes: u64, context_bound: u32) -> Q8Probe {
    let dtype_counts = gguf_model
        .tensors
        .iter()
        .fold(BTreeMap::new(), |mut counts, tensor| {
            *counts.entry(format!("{:?}", tensor.dtype)).or_insert(0) += 1;
            counts
        });
    let config_result = ModelConfig::from_metadata(&gguf_model.metadata, &gguf_model.tensors);
    let Ok(config) = config_result else {
        return Q8Probe {
            architecture: gguf_model
                .metadata
                .get("general.architecture")
                .and_then(|value| value.as_str())
                .unwrap_or("<missing>")
                .to_string(),
            block_count: 0,
            context_length: 0,
            d_model: 0,
            head_dim: 0,
            q_heads: 0,
            kv_heads: 0,
            weights_bytes,
            dense_kv_bytes_at_bound: 0,
            q8_projection_tensors: 0,
            expected_projection_tensors: 0,
            dtype_counts,
            unsupported_reason: Some(format!(
                "model config parse failed: {}",
                config_result.unwrap_err()
            )),
        };
    };

    let mut q8_projection_tensors = 0usize;
    let mut missing_projection_tensors = Vec::new();
    for layer in 0..config.block_count {
        for logical in PROJECTION_NAMES {
            let candidates = candidate_projection_names(layer, logical);
            match candidates
                .iter()
                .filter_map(|name| {
                    gguf_model
                        .tensors
                        .iter()
                        .find(|tensor| tensor.name == *name)
                })
                .next()
            {
                Some(tensor) if tensor.dtype == GgmlDType::Q8_0 => q8_projection_tensors += 1,
                Some(_) => {}
                None => missing_projection_tensors.push(format!("layer {layer} {logical}")),
            }
        }
    }
    let expected_projection_tensors = config.block_count as usize * PROJECTION_NAMES.len();
    let dense_kv_bytes_at_bound = u64::from(config.block_count)
        * u64::from(context_bound)
        * u64::from(config.attention_head_count_kv)
        * u64::from(config.attention_key_length)
        * 2
        * 2;
    let max_weight_bytes = env_u64("M40LLM_Q8_CANARY_MAX_WEIGHT_MB", DEFAULT_MAX_WEIGHT_MB)
        .saturating_mul(1024 * 1024);
    let architecture_supported =
        config.architecture.starts_with("llama") || config.architecture.starts_with("qwen");
    let unsupported_reason = if !architecture_supported {
        Some(format!(
            "unsupported architecture {}; canary currently expects llama/qwen-style tensor names and tokenizer handling",
            config.architecture
        ))
    } else if !missing_projection_tensors.is_empty() {
        Some(format!(
            "missing standard projection tensors: {}",
            missing_projection_tensors.join(", ")
        ))
    } else if q8_projection_tensors == 0 {
        Some("no Q8_0 standard projection tensors found".to_string())
    } else if weights_bytes > max_weight_bytes {
        Some(format!(
            "weights_bytes {weights_bytes} exceeds M40LLM_Q8_CANARY_MAX_WEIGHT_MB budget {max_weight_bytes}"
        ))
    } else {
        None
    };

    Q8Probe {
        architecture: config.architecture,
        block_count: config.block_count,
        context_length: config.context_length,
        d_model: config.embedding_length,
        head_dim: config.attention_key_length,
        q_heads: config.attention_head_count,
        kv_heads: config.attention_head_count_kv,
        weights_bytes,
        dense_kv_bytes_at_bound,
        q8_projection_tensors,
        expected_projection_tensors,
        dtype_counts,
        unsupported_reason,
    }
}

fn q8_canary_model_path() -> Option<PathBuf> {
    std::env::var_os("M40LLM_Q8_GENERATION_MODEL").map(PathBuf::from)
}

fn q8_projection_launches() -> u64 {
    profile::snapshot()
        .by_op
        .iter()
        .filter(|(op, _)| op.contains("q8_0"))
        .map(|(_, counters)| counters.launches)
        .sum()
}

#[test]
fn q8_0_real_model_generation_canary() -> Result<()> {
    let ctx = match cuda_env::ctx_m40_or_skip() {
        Some(ctx) => ctx,
        None => return Ok(()),
    };
    if let Err(e) = cuda_env::require_sm52(&ctx) {
        eprintln!("{e}");
        return Ok(());
    }
    drop(ctx);

    let Some(path) = q8_canary_model_path() else {
        eprintln!("skipping: set M40LLM_Q8_GENERATION_MODEL=/path/to/q8_0.gguf to run the Q8_0 generation canary");
        return Ok(());
    };

    let gguf_model = gguf::load_gguf(&path).with_context(|| {
        format!(
            "parse M40LLM_Q8_GENERATION_MODEL GGUF at {}",
            path.display()
        )
    })?;
    let file_len = std::fs::metadata(&path)?.len();
    let weights_bytes = file_len.saturating_sub(gguf_model.data_offset);
    let context_bound = env_u32(
        "M40LLM_Q8_CANARY_CONTEXT_TOKENS",
        DEFAULT_MAX_CONTEXT_TOKENS,
    )
    .min(
        ModelConfig::from_metadata(&gguf_model.metadata, &gguf_model.tensors)
            .map(|config| config.context_length)
            .unwrap_or(DEFAULT_MAX_CONTEXT_TOKENS),
    );
    let probe = probe_q8_model(&gguf_model, weights_bytes, context_bound);
    probe.log(&path, context_bound);
    if let Some(reason) = &probe.unsupported_reason {
        eprintln!("[q8-canary] skipping unsupported model: {reason}");
        return Ok(());
    }

    let _backend = EnvRestore::set("M40LLM_PROJECTION_BACKEND", "large-model");
    let _materialize = EnvRestore::set("M40LLM_MATERIALIZE_F32_WEIGHTS", "0");
    let _graph = EnvRestore::set("M40LLM_DECODE_GRAPH", "0");
    let _logit_compare = EnvRestore::set("M40LLM_KV_LOGIT_COMPARE", "1");

    let gguf_bytes = std::fs::read(&path)?;
    let mut model = LoadedModel::from_gguf(gguf_model, gguf_bytes, -1)?;
    model.allocate_kv_cache_for_layers(context_bound)?;

    profile::reset();
    let generated = generate_text(
        &model,
        GenerateOptions {
            prompt: "What is 2+2? Answer with one digit.".to_string(),
            max_tokens: Some(env_u32("M40LLM_Q8_CANARY_MAX_TOKENS", 8) as usize),
            top_k: Some(1),
            temperature: None,
            seed: Some(0),
            log_prefix: "q8_generation_canary",
            kv_compression: KvCompressionConfig::dense_reference(),
            prompt_format: PromptFormat::Auto,
            ..Default::default()
        },
    )?;

    eprintln!(
        "[q8-canary] output={:?} prompt_tokens={} total_tokens={} prefill_ms={} decode_ms={} total_ms={} q8_projection_launches={}",
        generated.output,
        generated.prompt_token_len,
        generated.token_ids.len(),
        generated.prompt_prefill_elapsed_ms,
        generated.generated_decode_elapsed_ms,
        generated.total_elapsed_ms,
        q8_projection_launches()
    );
    let trimmed = generated.output.trim();
    assert!(
        !trimmed.is_empty(),
        "Q8_0 canary should produce non-empty decoded text"
    );
    assert!(
        trimmed.contains('4'),
        "Q8_0 canary should answer a simple arithmetic prompt with 4; got: {:?}",
        generated.output
    );
    assert!(
        !generated.output.contains('\0') && !generated.output.contains('\u{fffd}'),
        "Q8_0 canary output should not contain NUL or replacement characters: {:?}",
        generated.output
    );
    let prompt_logits = generated
        .prompt_logits
        .as_ref()
        .context("prompt logits should be captured")?;
    assert!(
        prompt_logits.iter().all(|value| value.is_finite()),
        "Q8_0 canary prompt logits should be finite"
    );
    assert!(
        q8_projection_launches() > 0,
        "Q8_0 canary should exercise fused Q8_0 projection launches"
    );

    Ok(())
}

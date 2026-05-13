#![cfg(feature = "cuda")]

use anyhow::{Context, Result};
use m40_llm::generate::{generate_text, GenerateOptions, GeneratedText};
use m40_llm::gguf::{self, GgmlDType, GgufModel};
use m40_llm::infer::{LoadedModel, ModelConfig};
use m40_llm::kv_compression::{KvCompressMode, KvCompressionConfig};
use m40_llm::tokenizer::Tokenizer;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const NEEDLE: &str = "ZXQ-NEEDLE-41729";

#[derive(Debug, Clone)]
struct Candidate {
    path: PathBuf,
    resolved_path: PathBuf,
    size: u64,
}

#[derive(Debug, Clone)]
struct CandidateProbe {
    candidate: Candidate,
    config: Option<ModelConfig>,
    dtype_summary: BTreeMap<String, usize>,
    skip_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CaseStatus {
    Pass,
    Fail,
    Inconclusive,
    Error,
}

#[derive(Debug, Clone, Serialize)]
struct CaseRecord {
    model_path: String,
    resolved_model_path: String,
    target_tokens: usize,
    prompt_tokens: usize,
    generated_tokens: usize,
    needle_position: String,
    mode: String,
    status: CaseStatus,
    prompt_prefill_elapsed_ms: u128,
    generated_decode_elapsed_ms: u128,
    total_elapsed_ms: u128,
    attention_compression_elapsed_ms: Option<u128>,
    prefill_tokens_per_sec: Option<f64>,
    decode_tokens_per_sec: Option<f64>,
    prefill_chunk_size: Option<usize>,
    compressed_prefill_chunk_size: Option<usize>,
    prefill_mode: String,
    output: String,
    error: Option<String>,
}

fn resolved_size(path: &Path) -> Result<(PathBuf, u64)> {
    let resolved = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let size = fs::metadata(path)
        .with_context(|| format!("stat GGUF candidate {}", path.display()))?
        .len();
    Ok((resolved, size))
}

fn discover_candidates() -> Vec<Candidate> {
    let Some(path) = std::env::var_os("M40LLM_LONG_CONTEXT_RETRIEVAL_MODEL") else {
        eprintln!(
            "set M40LLM_LONG_CONTEXT_RETRIEVAL_MODEL=/path/to/model.gguf to run KV retrieval quality"
        );
        return Vec::new();
    };
    let paths = [PathBuf::from(path)];

    let mut candidates = Vec::new();
    for path in paths {
        if !path.exists() {
            eprintln!("GGUF candidate does not exist: {}", path.display());
            continue;
        }
        match resolved_size(&path) {
            Ok((resolved_path, size)) => candidates.push(Candidate {
                path,
                resolved_path,
                size,
            }),
            Err(err) => eprintln!("skipping GGUF candidate {}: {err}", path.display()),
        }
    }
    candidates
}

fn dtype_summary(model: &GgufModel) -> BTreeMap<String, usize> {
    let mut summary = BTreeMap::new();
    for tensor in &model.tensors {
        *summary.entry(format!("{:?}", tensor.dtype)).or_insert(0) += 1;
    }
    summary
}

fn tensor_dtype_supported(model: &GgufModel) -> Result<()> {
    for tensor in &model.tensors {
        let is_embedding = matches!(
            tensor.name.as_str(),
            "tok_embeddings.weight"
                | "token_embd.weight"
                | "token_embd"
                | "token_embeddings.weight"
        );
        let supported = match tensor.dtype {
            GgmlDType::F16 | GgmlDType::F32 => true,
            GgmlDType::Q8_0 => is_embedding,
            _ => false,
        };
        if !supported {
            anyhow::bail!(
                "tensor '{}' has unsupported dtype {:?}",
                tensor.name,
                tensor.dtype
            );
        }
    }
    Ok(())
}

fn probe_candidate(candidate: Candidate) -> CandidateProbe {
    let mut config = None;
    let mut dtype_counts = BTreeMap::new();
    let mut skip_reason = None;

    match gguf::load_gguf(&candidate.path) {
        Ok(model) => {
            dtype_counts = dtype_summary(&model);
            match ModelConfig::from_metadata(&model.metadata, &model.tensors) {
                Ok(cfg) => {
                    if let Err(err) = tensor_dtype_supported(&model) {
                        skip_reason = Some(err.to_string());
                    } else {
                        config = Some(cfg);
                    }
                }
                Err(err) => skip_reason = Some(format!("unsupported metadata: {err}")),
            }
        }
        Err(err) => skip_reason = Some(format!("failed to parse GGUF metadata: {err}")),
    }

    CandidateProbe {
        candidate,
        config,
        dtype_summary: dtype_counts,
        skip_reason,
    }
}

fn select_candidate() -> Result<Option<CandidateProbe>> {
    let candidates = discover_candidates();
    if candidates.is_empty() {
        eprintln!("no GGUF candidates found under configured cache roots");
        return Ok(None);
    }

    let probes: Vec<CandidateProbe> = candidates.into_iter().map(probe_candidate).collect();

    eprintln!("KV compression quality candidate table:");
    for probe in &probes {
        let context = probe
            .config
            .as_ref()
            .map(|cfg| cfg.context_length.to_string())
            .unwrap_or_else(|| "-".to_string());
        let arch = probe
            .config
            .as_ref()
            .map(|cfg| cfg.architecture.as_str())
            .unwrap_or("-");
        let d_model = probe
            .config
            .as_ref()
            .map(|cfg| cfg.embedding_length.to_string())
            .unwrap_or_else(|| "-".to_string());
        let blocks = probe
            .config
            .as_ref()
            .map(|cfg| cfg.block_count.to_string())
            .unwrap_or_else(|| "-".to_string());
        eprintln!(
            "  size={:.2}GiB ctx={context} arch={arch} d_model={d_model} layers={blocks} dtypes={:?} status={} path={}",
            probe.candidate.size as f64 / 1024.0 / 1024.0 / 1024.0,
            probe.dtype_summary,
            probe.skip_reason.as_deref().unwrap_or("candidate"),
            probe.candidate.path.display()
        );
    }

    Ok(probes.into_iter().find(|probe| probe.skip_reason.is_none()))
}

fn load_selected_model(probe: &CandidateProbe) -> Result<LoadedModel> {
    let path = &probe.candidate.path;
    let gguf_bytes =
        fs::read(path).with_context(|| format!("read selected GGUF {}", path.display()))?;
    let gguf_model = gguf::load_gguf(path)?;
    LoadedModel::from_gguf(gguf_model, gguf_bytes, -1)
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

fn parse_target_list(value: &str) -> Result<Vec<usize>> {
    let mut targets = Vec::new();
    for part in value.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let target = trimmed
            .parse::<usize>()
            .with_context(|| format!("invalid M40LLM_KV_QUALITY_TARGETS entry '{trimmed}'"))?;
        if target == 0 {
            anyhow::bail!("M40LLM_KV_QUALITY_TARGETS entries must be positive");
        }
        targets.push(target);
    }
    targets.sort_unstable();
    targets.dedup();
    if targets.is_empty() {
        anyhow::bail!("M40LLM_KV_QUALITY_TARGETS did not contain any target sizes");
    }
    Ok(targets)
}

fn retrieval_max_tokens() -> usize {
    std::env::var("M40LLM_KV_QUALITY_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16)
}

fn target_contexts(model_context: usize, full_quality: bool) -> Result<Vec<usize>> {
    let limit = model_context.saturating_sub(128);
    if let Some(targets) = std::env::var("M40LLM_KV_QUALITY_TARGETS")
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        return Ok(parse_target_list(&targets)?
            .into_iter()
            .filter(|ctx| *ctx < limit)
            .collect());
    }

    if let Some(smoke) = std::env::var("M40LLM_KV_QUALITY_SMOKE_TOKENS")
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        let smoke = smoke
            .parse::<usize>()
            .context("invalid M40LLM_KV_QUALITY_SMOKE_TOKENS")?;
        let smoke = smoke.min(limit.saturating_sub(1)).max(32);
        return Ok(if smoke < limit {
            vec![smoke]
        } else {
            Vec::new()
        });
    }

    if !full_quality {
        return Ok([64, 512].into_iter().filter(|ctx| *ctx < limit).collect());
    }

    let mut contexts: Vec<usize> = [64, 512, 1024, 2048, 4096]
        .into_iter()
        .filter(|ctx| *ctx < limit)
        .collect();
    if contexts.is_empty() && limit >= 192 {
        contexts.push(limit.saturating_sub(64).max(128));
    }
    contexts.sort_unstable();
    contexts.dedup();
    Ok(contexts)
}

fn prepare_kv_cache(model: &mut LoadedModel, mode: KvCompressMode) -> Result<()> {
    let max_len = model.model_config.context_length;
    let config = KvCompressionConfig {
        mode,
        recent_window: 1024,
        block_size: 32,
        top_blocks: 16,
        representatives: 2,
    };
    if matches!(
        mode,
        KvCompressMode::BlockSummary | KvCompressMode::BlockSelectLossy
    ) {
        model.allocate_compressed_kv_cache_for_layers(max_len, &config)
    } else {
        model.allocate_kv_cache_for_layers(max_len)
    }
}

fn run_retrieval_case(
    model: &mut LoadedModel,
    tokenizer: &Tokenizer,
    target_tokens: usize,
    needle_position: &str,
    mode: KvCompressMode,
) -> Result<(GeneratedText, usize)> {
    prepare_kv_cache(model, mode)?;
    let prompt = retrieval_prompt(tokenizer, target_tokens, needle_position);
    let preformatted_prompt_tokens = tokenizer
        .encode_with_specials(&prompt, true, false)
        .context("encode retrieval prompt")?
        .len();
    if preformatted_prompt_tokens >= model.model_config.context_length as usize {
        anyhow::bail!(
            "retrieval prompt has {preformatted_prompt_tokens} tokens before prompt formatting, model context is {}",
            model.model_config.context_length
        );
    }
    let options = GenerateOptions {
        prompt,
        max_tokens: Some(retrieval_max_tokens()),
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
    };
    let generated = generate_text(model, options.clone())?;
    if !generated.output.contains(NEEDLE) && generated.prefill_mode.starts_with("packed") {
        eprintln!(
            "[kv_retrieval] packed prefill missed needle for mode={} target={} needle={}; retrying sequential",
            mode_name(mode),
            target_tokens,
            needle_position
        );
        prepare_kv_cache(model, mode)?;
        let previous_prefill_mode = generated.prefill_mode.clone();
        let _guard = EnvVarGuard::unset("M40LLM_PREFILL_CHUNK_SIZE");
        let mut sequential = generate_text(
            model,
            GenerateOptions {
                prompt: options.prompt.clone(),
                max_tokens: options.max_tokens,
                top_k: options.top_k,
                log_prefix: options.log_prefix,
                kv_compression: options.kv_compression.clone(),
                ..Default::default()
            },
        )?;
        sequential.prefill_chunk_size = generated.prefill_chunk_size;
        sequential.prefill_mode =
            format!("sequential-quality-fallback-after-{previous_prefill_mode}");
        return Ok((sequential, preformatted_prompt_tokens));
    }
    Ok((generated, preformatted_prompt_tokens))
}

struct EnvVarGuard {
    name: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn unset(name: &'static str) -> Self {
        let previous = std::env::var_os(name);
        std::env::remove_var(name);
        Self { name, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => std::env::set_var(self.name, value),
            None => std::env::remove_var(self.name),
        }
    }
}

fn mode_name(mode: KvCompressMode) -> &'static str {
    match mode {
        KvCompressMode::Off => "off",
        KvCompressMode::BlockSelectExact => "block-select-exact",
        KvCompressMode::BlockSummary => "block-summary",
        KvCompressMode::BlockSelectLossy => "block-select-lossy",
    }
}

fn append_report(records: &[CaseRecord]) -> Result<()> {
    let Some(path) = std::env::var_os("M40LLM_KV_QUALITY_REPORT") else {
        return Ok(());
    };
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("open quality report {}", PathBuf::from(&path).display()))?;
    for record in records {
        serde_json::to_writer(&mut file, record)?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

fn print_records(records: &[CaseRecord]) {
    eprintln!("KV compression retrieval quality results:");
    for record in records {
        eprintln!(
            "  ctx={} prompt={} generated={} needle={} mode={} status={:?} prefill={}ms decode={}ms total={}ms prefill_mode={} compressed_chunk={:?} output={:?} error={}",
            record.target_tokens,
            record.prompt_tokens,
            record.generated_tokens,
            record.needle_position,
            record.mode,
            record.status,
            record.prompt_prefill_elapsed_ms,
            record.generated_decode_elapsed_ms,
            record.total_elapsed_ms,
            record.prefill_mode,
            record.compressed_prefill_chunk_size,
            record.output,
            record.error.as_deref().unwrap_or("-")
        );
    }
}

#[test]
fn long_context_needle_retrieval_quality_smoke() -> Result<()> {
    let Some(probe) = select_candidate()? else {
        return Ok(());
    };
    let config = probe
        .config
        .as_ref()
        .expect("selected candidate must have config");
    let full_quality = std::env::var("M40LLM_KV_QUALITY_FULL").ok().as_deref() == Some("1");
    let contexts = target_contexts(config.context_length as usize, full_quality)?;
    if contexts.is_empty() {
        eprintln!(
            "selected model context={} is too short for retrieval smoke",
            config.context_length
        );
        return Ok(());
    }

    eprintln!(
        "selected KV quality model: path={} resolved={} size={:.2}GiB ctx={} arch={} d_model={} layers={} heads={} kv_heads={} head_dim={}",
        probe.candidate.path.display(),
        probe.candidate.resolved_path.display(),
        probe.candidate.size as f64 / 1024.0 / 1024.0 / 1024.0,
        config.context_length,
        config.architecture,
        config.embedding_length,
        config.block_count,
        config.attention_head_count,
        config.attention_head_count_kv,
        config.attention_key_length
    );

    let mut model = load_selected_model(&probe)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&model.gguf.metadata)
        .unwrap_or_else(|_| Tokenizer::byte_level());
    let modes = [
        KvCompressMode::Off,
        KvCompressMode::BlockSelectExact,
        KvCompressMode::BlockSummary,
        KvCompressMode::BlockSelectLossy,
    ];

    let mut records = Vec::new();
    for target_tokens in contexts {
        for needle_position in ["old", "recent"] {
            let mut dense_passed = None;
            for mode in modes {
                let mut prompt_tokens = 0usize;
                let mut generated_tokens = 0usize;
                let mut prompt_prefill_elapsed_ms = 0u128;
                let mut generated_decode_elapsed_ms = 0u128;
                let mut total_elapsed_ms = 0u128;
                let mut attention_compression_elapsed_ms = None;
                let mut prefill_chunk_size = None;
                let mut compressed_prefill_chunk_size = None;
                let mut prefill_mode = "error".to_string();
                let (mut status, output, error) = match run_retrieval_case(
                    &mut model,
                    &tokenizer,
                    target_tokens,
                    needle_position,
                    mode,
                ) {
                    Ok((generated, _preformatted_prompt_tokens)) => {
                        prompt_tokens = generated.prompt_token_len;
                        generated_tokens = generated
                            .token_ids
                            .len()
                            .saturating_sub(generated.prompt_token_len);
                        prompt_prefill_elapsed_ms = generated.prompt_prefill_elapsed_ms;
                        generated_decode_elapsed_ms = generated.generated_decode_elapsed_ms;
                        total_elapsed_ms = generated.total_elapsed_ms;
                        attention_compression_elapsed_ms =
                            generated.attention_compression_elapsed_ms;
                        prefill_chunk_size = generated.prefill_chunk_size;
                        compressed_prefill_chunk_size = generated.compressed_prefill_chunk_size;
                        prefill_mode = generated.prefill_mode.clone();
                        let passed = generated.output.contains(NEEDLE);
                        if mode == KvCompressMode::Off {
                            dense_passed = Some(passed);
                        }
                        let status = if passed {
                            CaseStatus::Pass
                        } else {
                            CaseStatus::Fail
                        };
                        (status, generated.output, None)
                    }
                    Err(err) => {
                        if mode == KvCompressMode::Off {
                            dense_passed = Some(false);
                        }
                        (CaseStatus::Error, String::new(), Some(err.to_string()))
                    }
                };
                if mode != KvCompressMode::Off && dense_passed == Some(false) {
                    status = CaseStatus::Inconclusive;
                }
                records.push(CaseRecord {
                    model_path: probe.candidate.path.display().to_string(),
                    resolved_model_path: probe.candidate.resolved_path.display().to_string(),
                    target_tokens,
                    prompt_tokens,
                    generated_tokens,
                    needle_position: needle_position.to_string(),
                    mode: mode_name(mode).to_string(),
                    status,
                    prompt_prefill_elapsed_ms,
                    generated_decode_elapsed_ms,
                    total_elapsed_ms,
                    attention_compression_elapsed_ms,
                    prefill_tokens_per_sec: tokens_per_sec(
                        prompt_tokens,
                        prompt_prefill_elapsed_ms,
                    ),
                    decode_tokens_per_sec: tokens_per_sec(
                        generated_tokens,
                        generated_decode_elapsed_ms,
                    ),
                    prefill_chunk_size,
                    compressed_prefill_chunk_size,
                    prefill_mode,
                    output,
                    error,
                });
            }
        }
    }

    print_records(&records);
    append_report(&records)?;
    Ok(())
}

fn tokens_per_sec(tokens: usize, elapsed_ms: u128) -> Option<f64> {
    if tokens == 0 || elapsed_ms == 0 {
        None
    } else {
        Some(tokens as f64 / (elapsed_ms as f64 / 1000.0))
    }
}

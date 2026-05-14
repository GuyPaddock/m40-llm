#![cfg(feature = "cuda")]

use anyhow::{Context, Result};
use m40_llm::generate::{
    generate_text, prepare_prompt, GenerateOptions, GeneratedText, PromptFormat,
};
use m40_llm::gguf::{self, GgmlDType, GgufModel};
use m40_llm::infer::{LoadedModel, ModelConfig};
use m40_llm::kv_compression::{KvCompressMode, KvCompressionConfig, KvRepresentativePolicy};
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
    representatives: u32,
    representative_policy: String,
    status: CaseStatus,
    prompt_prefill_elapsed_ms: u128,
    generated_decode_elapsed_ms: u128,
    total_elapsed_ms: u128,
    attention_compression_elapsed_ms: Option<u128>,
    prefill_tokens_per_sec: Option<f64>,
    decode_tokens_per_sec: Option<f64>,
    prefill_chunk_size: Option<usize>,
    compressed_prefill_chunk_size: Option<usize>,
    temporary_dense_kv_bytes: Option<usize>,
    final_kv_allocated_bytes: Option<usize>,
    dense_equivalent_kv_bytes: Option<usize>,
    compression_ratio: Option<f64>,
    prefill_mode: String,
    top_blocks: Option<u32>,
    needle_block_index: Option<u32>,
    selected_block_indices: Option<Vec<u32>>,
    needle_block_selected: Option<bool>,
    needle_block_rank: Option<u32>,
    total_old_blocks: Option<u32>,
    recent_ring_absolute_start: Option<usize>,
    recent_ring_absolute_end: Option<usize>,
    needle_token_absolute_positions: Vec<usize>,
    question_token_absolute_positions: Vec<usize>,
    needle_tokens_in_recent_ring: bool,
    question_tokens_in_recent_ring: bool,
    expected_first_answer_token_id: Option<u32>,
    expected_first_answer_token_rank_dense: Option<usize>,
    expected_first_answer_token_rank_compressed: Option<usize>,
    expected_first_answer_token_logit_dense: Option<f32>,
    expected_first_answer_token_logit_compressed: Option<f32>,
    prompt_logit_max_abs_diff: Option<f32>,
    prompt_logit_mean_abs_diff: Option<f32>,
    prompt_logit_top10_overlap: Option<usize>,
    prompt_dense_top_token_id: Option<u32>,
    prompt_compressed_top_token_id: Option<u32>,
    first_decode_logit_max_abs_diff: Option<f32>,
    first_decode_logit_mean_abs_diff: Option<f32>,
    first_decode_logit_top10_overlap: Option<usize>,
    first_decode_dense_top_token_id: Option<u32>,
    first_decode_compressed_top_token_id: Option<u32>,
    attention_recent_mass: Option<f32>,
    attention_selected_old_exact_mass: Option<f32>,
    attention_summary_mass: Option<f32>,
    attention_representative_mass: Option<f32>,
    attention_other_mass: Option<f32>,
    attention_needle_block_mass: Option<f32>,
    attention_recent_logit_max: Option<f32>,
    attention_recent_logit_mean: Option<f32>,
    attention_summary_logit_max: Option<f32>,
    attention_summary_logit_mean: Option<f32>,
    attention_representative_logit_max: Option<f32>,
    attention_representative_logit_mean: Option<f32>,
    attention_top_entries: Option<Vec<String>>,
    attention_records: Option<Vec<String>>,
    output: String,
    error: Option<String>,
}

#[derive(Debug, Clone)]
struct PromptPositionMeta {
    recent_ring_absolute_start: Option<usize>,
    recent_ring_absolute_end: Option<usize>,
    needle_token_absolute_positions: Vec<usize>,
    question_token_absolute_positions: Vec<usize>,
    needle_tokens_in_recent_ring: bool,
    question_tokens_in_recent_ring: bool,
    expected_first_answer_token_id: Option<u32>,
}

#[derive(Debug, Clone)]
struct LogitDiffSummary {
    max_abs_diff: f32,
    mean_abs_diff: f32,
    top10_overlap: usize,
    dense_top_token_id: Option<u32>,
    compressed_top_token_id: Option<u32>,
}

#[derive(Debug, Clone)]
struct ExpectedTokenLogitSummary {
    rank_dense: Option<usize>,
    rank_compressed: Option<usize>,
    logit_dense: Option<f32>,
    logit_compressed: Option<f32>,
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

fn token_positions_for_byte_span(
    tokenizer: &Tokenizer,
    prompt: &str,
    add_bos: bool,
    start_byte: usize,
    end_byte: usize,
) -> Result<Vec<usize>> {
    let start = tokenizer
        .encode_with_specials(&prompt[..start_byte], add_bos, false)
        .context("encode prompt prefix for token positions")?
        .len();
    let end = tokenizer
        .encode_with_specials(&prompt[..end_byte], add_bos, false)
        .context("encode prompt span end for token positions")?
        .len();
    Ok((start..end).collect())
}

fn prompt_position_meta(
    tokenizer: &Tokenizer,
    prompt: &str,
    add_bos: bool,
    prompt_tokens: usize,
    recent_window: usize,
) -> Result<PromptPositionMeta> {
    let needle_token_absolute_positions = if let Some(start) = prompt.find(NEEDLE) {
        token_positions_for_byte_span(tokenizer, prompt, add_bos, start, start + NEEDLE.len())?
    } else {
        Vec::new()
    };
    let question = "Question: What is the secret code?";
    let question_token_absolute_positions = if let Some(start) = prompt.find(question) {
        token_positions_for_byte_span(tokenizer, prompt, add_bos, start, start + question.len())?
    } else {
        Vec::new()
    };
    let recent_start = prompt_tokens.saturating_sub(recent_window);
    let recent_end = prompt_tokens;
    let in_recent = |positions: &[usize]| {
        !positions.is_empty()
            && positions
                .iter()
                .all(|pos| *pos >= recent_start && *pos < recent_end)
    };
    let expected_first_answer_token_id = tokenizer
        .encode_with_specials(NEEDLE, false, false)
        .ok()
        .and_then(|ids| ids.first().copied());
    Ok(PromptPositionMeta {
        recent_ring_absolute_start: Some(recent_start),
        recent_ring_absolute_end: Some(recent_end),
        needle_tokens_in_recent_ring: in_recent(&needle_token_absolute_positions),
        question_tokens_in_recent_ring: in_recent(&question_token_absolute_positions),
        needle_token_absolute_positions,
        question_token_absolute_positions,
        expected_first_answer_token_id,
    })
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
        if lossy_packed_sweep_enabled() {
            return Ok([1024, 2048, 4096]
                .into_iter()
                .filter(|ctx| *ctx < limit)
                .collect());
        }
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

fn lossy_packed_sweep_enabled() -> bool {
    std::env::var("M40LLM_KV_QUALITY_LOSSY_PACKED_SWEEP")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn exact_selection_sweep_enabled() -> bool {
    std::env::var("M40LLM_KV_QUALITY_EXACT_SELECTION_SWEEP")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn logit_compare_enabled() -> bool {
    std::env::var("M40LLM_KV_LOGIT_COMPARE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn parse_u32_list(value: &str, var_name: &str) -> Result<Vec<u32>> {
    let mut values = Vec::new();
    for part in value.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parsed = trimmed
            .parse::<u32>()
            .with_context(|| format!("invalid {var_name} entry '{trimmed}'"))?;
        if parsed == 0 {
            anyhow::bail!("{var_name} entries must be positive");
        }
        values.push(parsed);
    }
    values.sort_unstable();
    values.dedup();
    if values.is_empty() {
        anyhow::bail!("{var_name} did not contain any values");
    }
    Ok(values)
}

fn top_block_cases(exact_selection_sweep: bool, mode: KvCompressMode) -> Result<Vec<Option<u32>>> {
    if mode == KvCompressMode::Off {
        return Ok(vec![None]);
    }
    if matches!(
        mode,
        KvCompressMode::RecentOnly | KvCompressMode::BlockSummary
    ) {
        return Ok(vec![Some(0)]);
    }
    let values = std::env::var("M40LLM_KV_QUALITY_TOP_BLOCKS")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| parse_u32_list(&value, "M40LLM_KV_QUALITY_TOP_BLOCKS"))
        .transpose()?
        .unwrap_or_else(|| {
            if exact_selection_sweep {
                vec![4, 8, 16, 32, 64]
            } else {
                vec![16]
            }
        });
    Ok(values.into_iter().map(Some).collect())
}

#[derive(Debug, Clone, Copy)]
struct RepresentativeCase {
    representatives: u32,
    policy: KvRepresentativePolicy,
}

fn representative_cases(lossy_packed_sweep: bool) -> Vec<RepresentativeCase> {
    if !lossy_packed_sweep {
        return vec![RepresentativeCase {
            representatives: 0,
            policy: KvRepresentativePolicy::Last,
        }];
    }
    let reps = std::env::var("M40LLM_KV_QUALITY_REPRESENTATIVES")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<u32>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| vec![0, 1, 2, 4]);
    let policies = std::env::var("M40LLM_KV_QUALITY_REP_POLICIES")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| match part.trim() {
                    "last" => Some(KvRepresentativePolicy::Last),
                    "stride" => Some(KvRepresentativePolicy::Stride),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| vec![KvRepresentativePolicy::Last]);
    let mut cases = Vec::new();
    for representatives in reps {
        if representatives == 0 {
            cases.push(RepresentativeCase {
                representatives,
                policy: KvRepresentativePolicy::Last,
            });
        } else {
            for &policy in &policies {
                cases.push(RepresentativeCase {
                    representatives,
                    policy,
                });
            }
        }
    }
    cases
}

fn quality_modes(
    lossy_packed_sweep: bool,
    exact_selection_sweep: bool,
) -> &'static [KvCompressMode] {
    if exact_selection_sweep {
        &[
            KvCompressMode::Off,
            KvCompressMode::BlockSelectExact,
            KvCompressMode::RecentOnly,
            KvCompressMode::BlockSummary,
            KvCompressMode::BlockSelectLossy,
        ]
    } else if lossy_packed_sweep {
        &[
            KvCompressMode::Off,
            KvCompressMode::BlockSummary,
            KvCompressMode::BlockSelectLossy,
        ]
    } else {
        &[
            KvCompressMode::Off,
            KvCompressMode::BlockSelectExact,
            KvCompressMode::RecentOnly,
            KvCompressMode::BlockSummary,
            KvCompressMode::BlockSelectLossy,
        ]
    }
}

fn prepare_kv_cache(
    model: &mut LoadedModel,
    mode: KvCompressMode,
    rep_case: RepresentativeCase,
    top_blocks: u32,
) -> Result<()> {
    let max_len = model.model_config.context_length;
    let config = KvCompressionConfig {
        mode,
        recent_window: 1024,
        block_size: 32,
        top_blocks,
        representatives: rep_case.representatives,
        representative_policy: rep_case.policy,
    };
    if matches!(
        mode,
        KvCompressMode::RecentOnly
            | KvCompressMode::BlockSummary
            | KvCompressMode::BlockSelectLossy
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
    rep_case: RepresentativeCase,
    top_blocks: u32,
) -> Result<(GeneratedText, usize, Option<u32>)> {
    let exact_selection_sweep = exact_selection_sweep_enabled();
    let _prefill_guard = if (lossy_packed_sweep_enabled()
        || (exact_selection_sweep
            && matches!(mode, KvCompressMode::Off | KvCompressMode::BlockSelectExact)))
        && matches!(mode, KvCompressMode::Off | KvCompressMode::BlockSelectExact)
    {
        Some(EnvVarGuard::set(
            "M40LLM_PREFILL_CHUNK_SIZE",
            target_tokens.to_string(),
        ))
    } else {
        None
    };
    let _packed_then_compress_guard = if (lossy_packed_sweep_enabled() || exact_selection_sweep)
        && matches!(
            mode,
            KvCompressMode::RecentOnly
                | KvCompressMode::BlockSummary
                | KvCompressMode::BlockSelectLossy
        ) {
        Some(EnvVarGuard::set(
            "M40LLM_KV_PACKED_THEN_COMPRESS_PREFILL",
            "1",
        ))
    } else {
        None
    };
    let _selection_telemetry_guard = if exact_selection_sweep
        && matches!(
            mode,
            KvCompressMode::BlockSelectExact
                | KvCompressMode::RecentOnly
                | KvCompressMode::BlockSummary
                | KvCompressMode::BlockSelectLossy
        ) {
        Some(EnvVarGuard::set("M40LLM_KV_SELECTION_TELEMETRY", "1"))
    } else {
        None
    };

    prepare_kv_cache(model, mode, rep_case, top_blocks)?;
    let prompt = retrieval_prompt(tokenizer, target_tokens, needle_position);
    let (formatted_prompt, add_bos) = prepare_prompt(tokenizer, &prompt, PromptFormat::Auto);
    let preformatted_prompt_tokens = tokenizer
        .encode_with_specials(&formatted_prompt, add_bos, false)
        .context("encode retrieval prompt")?
        .len();
    let needle_block_index = needle_block_index(
        tokenizer,
        &formatted_prompt,
        add_bos,
        preformatted_prompt_tokens,
        1024,
        32,
    )?;
    let _needle_block_guard = if let Some(block) = needle_block_index {
        Some(EnvVarGuard::set(
            "M40LLM_KV_TELEMETRY_NEEDLE_BLOCK",
            block.to_string(),
        ))
    } else {
        Some(EnvVarGuard::unset("M40LLM_KV_TELEMETRY_NEEDLE_BLOCK"))
    };
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
            top_blocks,
            representatives: rep_case.representatives,
            representative_policy: rep_case.policy,
        },
        ..Default::default()
    };
    let generated = generate_text(model, options.clone())?;
    if !lossy_packed_sweep_enabled()
        && !exact_selection_sweep
        && !generated.output.contains(NEEDLE)
        && generated.prefill_mode.starts_with("packed")
    {
        eprintln!(
            "[kv_retrieval] packed prefill missed needle for mode={} target={} needle={}; retrying sequential",
            mode_name(mode),
            target_tokens,
            needle_position
        );
        prepare_kv_cache(model, mode, rep_case, top_blocks)?;
        let previous_prefill_mode = generated.prefill_mode.clone();
        let _packed_guard = EnvVarGuard::unset("M40LLM_PREFILL_CHUNK_SIZE");
        let _packed_then_compress_guard =
            EnvVarGuard::unset("M40LLM_KV_PACKED_THEN_COMPRESS_PREFILL");
        let _compressed_chunk_guard = EnvVarGuard::unset("M40LLM_KV_COMPRESSED_PREFILL_CHUNK_SIZE");
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
        sequential.temporary_dense_kv_bytes = generated.temporary_dense_kv_bytes;
        sequential.prefill_mode =
            format!("sequential-quality-fallback-after-{previous_prefill_mode}");
        return Ok((sequential, preformatted_prompt_tokens, needle_block_index));
    }
    Ok((generated, preformatted_prompt_tokens, needle_block_index))
}

fn needle_block_index(
    tokenizer: &Tokenizer,
    prompt: &str,
    add_bos: bool,
    prompt_tokens: usize,
    recent_window: usize,
    block_size: usize,
) -> Result<Option<u32>> {
    let Some(needle_byte) = prompt.find(NEEDLE) else {
        return Ok(None);
    };
    let needle_token = tokenizer
        .encode_with_specials(&prompt[..needle_byte], add_bos, false)
        .context("encode prompt prefix before needle")?
        .len();
    let old_len = prompt_tokens.saturating_sub(recent_window);
    if needle_token >= old_len {
        return Ok(None);
    }
    Ok(Some((needle_token / block_size) as u32))
}

struct EnvVarGuard {
    name: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set(name: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
        let previous = std::env::var_os(name);
        std::env::set_var(name, value);
        Self { name, previous }
    }

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
        KvCompressMode::RecentOnly => "recent-only",
        KvCompressMode::BlockSummary => "block-summary",
        KvCompressMode::BlockSelectLossy => "block-select-lossy",
    }
}

fn representative_policy_name(policy: KvRepresentativePolicy) -> &'static str {
    match policy {
        KvRepresentativePolicy::Last => "last",
        KvRepresentativePolicy::Stride => "stride",
    }
}

fn append_report_record(record: &CaseRecord) -> Result<()> {
    let Some(path) = std::env::var_os("M40LLM_KV_QUALITY_REPORT") else {
        return Ok(());
    };
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("open quality report {}", PathBuf::from(&path).display()))?;
    serde_json::to_writer(&mut file, record)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn attention_top_entry_strings(
    attention: &m40_llm::kv_selection::KvAttentionTelemetrySummary,
) -> Vec<String> {
    attention
        .top_entries
        .iter()
        .map(|entry| {
            format!(
                "{}:block={:?}:token={:?}:score={:.4}:p={:.6}",
                entry.group,
                entry.block_index,
                entry.token_position,
                entry.score,
                entry.probability
            )
        })
        .collect()
}

fn attention_record_strings(summary: &m40_llm::kv_selection::KvSelectionSummary) -> Vec<String> {
    summary
        .attentions
        .iter()
        .map(|record| {
            format!(
                "layer={:?}:token={:?}:mass(recent={:.6},old_exact={:.6},summary={:.6},rep={:.6},other={:.6})",
                record.layer,
                record.token,
                record.attention.recent.prob_mass,
                record.attention.selected_old_exact.prob_mass,
                record.attention.summary.prob_mass,
                record.attention.representatives.prob_mass,
                record.attention.other.prob_mass
            )
        })
        .collect()
}

fn top_token_id(logits: &[f32]) -> Option<u32> {
    logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| f32::total_cmp(&a.1, &b.1))
        .map(|(idx, _)| idx as u32)
}

fn top_token_ids(logits: &[f32], k: usize) -> Vec<u32> {
    let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    top.sort_by(|a, b| f32::total_cmp(&b.1, &a.1));
    top.truncate(k);
    top.into_iter().map(|(idx, _)| idx as u32).collect()
}

fn logit_diff_summary(dense: &[f32], compressed: &[f32]) -> Option<LogitDiffSummary> {
    if dense.len() != compressed.len() || dense.is_empty() {
        return None;
    }
    let mut max_abs_diff = 0.0f32;
    let mut sum_abs_diff = 0.0f64;
    for (a, b) in dense.iter().zip(compressed.iter()) {
        let diff = (*a - *b).abs();
        max_abs_diff = max_abs_diff.max(diff);
        sum_abs_diff += diff as f64;
    }
    let dense_top10 = top_token_ids(dense, 10);
    let compressed_top10 = top_token_ids(compressed, 10);
    let top10_overlap = dense_top10
        .iter()
        .filter(|token| compressed_top10.contains(token))
        .count();
    Some(LogitDiffSummary {
        max_abs_diff,
        mean_abs_diff: (sum_abs_diff / dense.len() as f64) as f32,
        top10_overlap,
        dense_top_token_id: top_token_id(dense),
        compressed_top_token_id: top_token_id(compressed),
    })
}

fn expected_token_summary(
    dense: Option<&[f32]>,
    compressed: Option<&[f32]>,
    token_id: Option<u32>,
) -> ExpectedTokenLogitSummary {
    let Some(token_id) = token_id.map(|id| id as usize) else {
        return ExpectedTokenLogitSummary {
            rank_dense: None,
            rank_compressed: None,
            logit_dense: None,
            logit_compressed: None,
        };
    };
    let rank = |logits: &[f32]| -> Option<usize> {
        let target = *logits.get(token_id)?;
        Some(1 + logits.iter().filter(|logit| **logit > target).count())
    };
    ExpectedTokenLogitSummary {
        rank_dense: dense.and_then(rank),
        rank_compressed: compressed.and_then(rank),
        logit_dense: dense.and_then(|logits| logits.get(token_id).copied()),
        logit_compressed: compressed.and_then(|logits| logits.get(token_id).copied()),
    }
}

fn print_records(records: &[CaseRecord]) {
    eprintln!("KV compression retrieval quality results:");
    for record in records {
        eprintln!(
            "  ctx={} prompt={} generated={} needle={} mode={} reps={} rep_policy={} top_blocks={:?} status={:?} ring={:?}..{:?} needle_pos={:?} question_pos={:?} needle_recent={} question_recent={} expected_id={:?} expected_rank(dense={:?},mode={:?}) expected_logit(dense={:?},mode={:?}) prompt_diff(max={:?},mean={:?},top10={:?}) first_decode_diff(max={:?},mean={:?},top10={:?}) needle_block={:?} selected={:?} needle_selected={:?} needle_rank={:?} old_blocks={:?} mass(recent={:?},old_exact={:?},summary={:?},rep={:?},needle={:?}) logits(recent_max={:?},recent_mean={:?},summary_max={:?},summary_mean={:?}) top_attn={:?} attn_records={:?} prefill={}ms decode={}ms total={}ms prefill_tps={:?} decode_tps={:?} final_kv_bytes={:?} dense_equiv_kv_bytes={:?} temp_dense_kv_bytes={:?} compression_ratio={:?} prefill_mode={} output={:?} error={}",
            record.target_tokens,
            record.prompt_tokens,
            record.generated_tokens,
            record.needle_position,
            record.mode,
            record.representatives,
            record.representative_policy,
            record.top_blocks,
            record.status,
            record.recent_ring_absolute_start,
            record.recent_ring_absolute_end,
            record.needle_token_absolute_positions,
            record.question_token_absolute_positions,
            record.needle_tokens_in_recent_ring,
            record.question_tokens_in_recent_ring,
            record.expected_first_answer_token_id,
            record.expected_first_answer_token_rank_dense,
            record.expected_first_answer_token_rank_compressed,
            record.expected_first_answer_token_logit_dense,
            record.expected_first_answer_token_logit_compressed,
            record.prompt_logit_max_abs_diff,
            record.prompt_logit_mean_abs_diff,
            record.prompt_logit_top10_overlap,
            record.first_decode_logit_max_abs_diff,
            record.first_decode_logit_mean_abs_diff,
            record.first_decode_logit_top10_overlap,
            record.needle_block_index,
            record.selected_block_indices,
            record.needle_block_selected,
            record.needle_block_rank,
            record.total_old_blocks,
            record.attention_recent_mass,
            record.attention_selected_old_exact_mass,
            record.attention_summary_mass,
            record.attention_representative_mass,
            record.attention_needle_block_mass,
            record.attention_recent_logit_max,
            record.attention_recent_logit_mean,
            record.attention_summary_logit_max,
            record.attention_summary_logit_mean,
            record.attention_top_entries,
            record.attention_records,
            record.prompt_prefill_elapsed_ms,
            record.generated_decode_elapsed_ms,
            record.total_elapsed_ms,
            record.prefill_tokens_per_sec,
            record.decode_tokens_per_sec,
            record.final_kv_allocated_bytes,
            record.dense_equivalent_kv_bytes,
            record.temporary_dense_kv_bytes,
            record.compression_ratio,
            record.prefill_mode,
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
    let lossy_packed_sweep = lossy_packed_sweep_enabled();
    let exact_selection_sweep = exact_selection_sweep_enabled();
    let modes = quality_modes(lossy_packed_sweep, exact_selection_sweep);
    let rep_cases = representative_cases(lossy_packed_sweep);

    let mut records = Vec::new();
    for target_tokens in contexts {
        for needle_position in ["old", "recent"] {
            let mut dense_passed = None;
            let mut dense_prompt_logits: Option<Vec<f32>> = None;
            let mut dense_first_decode_logits: Option<Vec<f32>> = None;
            let prompt_for_meta = retrieval_prompt(&tokenizer, target_tokens, needle_position);
            let (formatted_prompt_for_meta, add_bos_for_meta) =
                prepare_prompt(&tokenizer, &prompt_for_meta, PromptFormat::Auto);
            let prompt_tokens_for_meta = tokenizer
                .encode_with_specials(&formatted_prompt_for_meta, add_bos_for_meta, false)
                .context("encode retrieval prompt for position metadata")?
                .len();
            let position_meta = prompt_position_meta(
                &tokenizer,
                &formatted_prompt_for_meta,
                add_bos_for_meta,
                prompt_tokens_for_meta,
                1024,
            )?;
            for &mode in modes {
                let mode_rep_cases: Vec<RepresentativeCase> = if mode == KvCompressMode::Off {
                    vec![RepresentativeCase {
                        representatives: 0,
                        policy: KvRepresentativePolicy::Last,
                    }]
                } else {
                    rep_cases.clone()
                };
                for rep_case in mode_rep_cases {
                    for top_blocks_case in top_block_cases(exact_selection_sweep, mode)? {
                        let top_blocks = top_blocks_case.unwrap_or(16);
                        let mut prompt_tokens = 0usize;
                        let mut generated_tokens = 0usize;
                        let mut prompt_prefill_elapsed_ms = 0u128;
                        let mut generated_decode_elapsed_ms = 0u128;
                        let mut total_elapsed_ms = 0u128;
                        let mut attention_compression_elapsed_ms = None;
                        let mut prefill_chunk_size = None;
                        let mut compressed_prefill_chunk_size = None;
                        let mut temporary_dense_kv_bytes = None;
                        let mut final_kv_allocated_bytes = None;
                        let mut dense_equivalent_kv_bytes = None;
                        let mut prefill_mode = "error".to_string();
                        let mut selection_summary = None;
                        let mut needle_block_index = None;
                        let mut prompt_logits = None;
                        let mut first_decode_logits = None;
                        let (mut status, output, error) = match run_retrieval_case(
                            &mut model,
                            &tokenizer,
                            target_tokens,
                            needle_position,
                            mode,
                            rep_case,
                            top_blocks,
                        ) {
                            Ok((generated, _preformatted_prompt_tokens, needle_block)) => {
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
                                compressed_prefill_chunk_size =
                                    generated.compressed_prefill_chunk_size;
                                temporary_dense_kv_bytes = generated.temporary_dense_kv_bytes;
                                final_kv_allocated_bytes = generated.final_kv_allocated_bytes;
                                dense_equivalent_kv_bytes = generated.dense_equivalent_kv_bytes;
                                prefill_mode = generated.prefill_mode.clone();
                                selection_summary = generated.kv_selection.clone();
                                needle_block_index = needle_block;
                                prompt_logits = generated.prompt_logits.clone();
                                first_decode_logits = generated.first_decode_logits.clone();
                                let passed = generated.output.contains(NEEDLE);
                                if mode == KvCompressMode::Off {
                                    dense_passed = Some(passed);
                                    dense_prompt_logits = generated.prompt_logits.clone();
                                    dense_first_decode_logits =
                                        generated.first_decode_logits.clone();
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
                        let selected_block_indices = selection_summary
                            .as_ref()
                            .map(|summary| summary.selected_block_indices.clone());
                        let needle_block_selected = needle_block_index.map(|needle| {
                            selected_block_indices
                                .as_ref()
                                .is_some_and(|selected| selected.contains(&needle))
                        });
                        let needle_block_rank = needle_block_index.and_then(|needle| {
                            selected_block_indices.as_ref().and_then(|selected| {
                                selected
                                    .iter()
                                    .position(|block| *block == needle)
                                    .map(|rank| rank as u32)
                            })
                        });
                        let total_old_blocks = selection_summary
                            .as_ref()
                            .map(|summary| summary.total_old_blocks);
                        let attention = selection_summary
                            .as_ref()
                            .and_then(|summary| summary.attention.as_ref());
                        let prompt_logit_diff = if logit_compare_enabled() {
                            dense_prompt_logits
                                .as_deref()
                                .zip(prompt_logits.as_deref())
                                .and_then(|(dense, compressed)| {
                                    logit_diff_summary(dense, compressed)
                                })
                        } else {
                            None
                        };
                        let first_decode_logit_diff = if logit_compare_enabled() {
                            dense_first_decode_logits
                                .as_deref()
                                .zip(first_decode_logits.as_deref())
                                .and_then(|(dense, compressed)| {
                                    logit_diff_summary(dense, compressed)
                                })
                        } else {
                            None
                        };
                        let expected_token_logits = if logit_compare_enabled() {
                            expected_token_summary(
                                dense_prompt_logits.as_deref(),
                                prompt_logits.as_deref(),
                                position_meta.expected_first_answer_token_id,
                            )
                        } else {
                            ExpectedTokenLogitSummary {
                                rank_dense: None,
                                rank_compressed: None,
                                logit_dense: None,
                                logit_compressed: None,
                            }
                        };
                        let record = CaseRecord {
                            model_path: probe.candidate.path.display().to_string(),
                            resolved_model_path: probe
                                .candidate
                                .resolved_path
                                .display()
                                .to_string(),
                            target_tokens,
                            prompt_tokens,
                            generated_tokens,
                            needle_position: needle_position.to_string(),
                            mode: mode_name(mode).to_string(),
                            representatives: rep_case.representatives,
                            representative_policy: representative_policy_name(rep_case.policy)
                                .to_string(),
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
                            temporary_dense_kv_bytes,
                            final_kv_allocated_bytes,
                            dense_equivalent_kv_bytes,
                            compression_ratio: match (
                                dense_equivalent_kv_bytes,
                                final_kv_allocated_bytes,
                            ) {
                                (Some(dense), Some(actual)) if actual > 0 => {
                                    Some(dense as f64 / actual as f64)
                                }
                                _ => None,
                            },
                            prefill_mode,
                            top_blocks: top_blocks_case,
                            needle_block_index,
                            selected_block_indices,
                            needle_block_selected,
                            needle_block_rank,
                            total_old_blocks,
                            recent_ring_absolute_start: position_meta.recent_ring_absolute_start,
                            recent_ring_absolute_end: position_meta.recent_ring_absolute_end,
                            needle_token_absolute_positions: position_meta
                                .needle_token_absolute_positions
                                .clone(),
                            question_token_absolute_positions: position_meta
                                .question_token_absolute_positions
                                .clone(),
                            needle_tokens_in_recent_ring: position_meta
                                .needle_tokens_in_recent_ring,
                            question_tokens_in_recent_ring: position_meta
                                .question_tokens_in_recent_ring,
                            expected_first_answer_token_id: position_meta
                                .expected_first_answer_token_id,
                            expected_first_answer_token_rank_dense: expected_token_logits
                                .rank_dense,
                            expected_first_answer_token_rank_compressed: expected_token_logits
                                .rank_compressed,
                            expected_first_answer_token_logit_dense: expected_token_logits
                                .logit_dense,
                            expected_first_answer_token_logit_compressed: expected_token_logits
                                .logit_compressed,
                            prompt_logit_max_abs_diff: prompt_logit_diff
                                .as_ref()
                                .map(|summary| summary.max_abs_diff),
                            prompt_logit_mean_abs_diff: prompt_logit_diff
                                .as_ref()
                                .map(|summary| summary.mean_abs_diff),
                            prompt_logit_top10_overlap: prompt_logit_diff
                                .as_ref()
                                .map(|summary| summary.top10_overlap),
                            prompt_dense_top_token_id: prompt_logit_diff
                                .as_ref()
                                .and_then(|summary| summary.dense_top_token_id),
                            prompt_compressed_top_token_id: prompt_logit_diff
                                .as_ref()
                                .and_then(|summary| summary.compressed_top_token_id),
                            first_decode_logit_max_abs_diff: first_decode_logit_diff
                                .as_ref()
                                .map(|summary| summary.max_abs_diff),
                            first_decode_logit_mean_abs_diff: first_decode_logit_diff
                                .as_ref()
                                .map(|summary| summary.mean_abs_diff),
                            first_decode_logit_top10_overlap: first_decode_logit_diff
                                .as_ref()
                                .map(|summary| summary.top10_overlap),
                            first_decode_dense_top_token_id: first_decode_logit_diff
                                .as_ref()
                                .and_then(|summary| summary.dense_top_token_id),
                            first_decode_compressed_top_token_id: first_decode_logit_diff
                                .as_ref()
                                .and_then(|summary| summary.compressed_top_token_id),
                            attention_recent_mass: attention.map(|a| a.recent.prob_mass),
                            attention_selected_old_exact_mass: attention
                                .map(|a| a.selected_old_exact.prob_mass),
                            attention_summary_mass: attention.map(|a| a.summary.prob_mass),
                            attention_representative_mass: attention
                                .map(|a| a.representatives.prob_mass),
                            attention_other_mass: attention.map(|a| a.other.prob_mass),
                            attention_needle_block_mass: attention
                                .and_then(|a| a.needle_block_mass),
                            attention_recent_logit_max: attention.map(|a| a.recent.logit_max),
                            attention_recent_logit_mean: attention.map(|a| a.recent.logit_mean),
                            attention_summary_logit_max: attention.map(|a| a.summary.logit_max),
                            attention_summary_logit_mean: attention.map(|a| a.summary.logit_mean),
                            attention_representative_logit_max: attention
                                .map(|a| a.representatives.logit_max),
                            attention_representative_logit_mean: attention
                                .map(|a| a.representatives.logit_mean),
                            attention_top_entries: attention.map(attention_top_entry_strings),
                            attention_records: selection_summary
                                .as_ref()
                                .map(attention_record_strings),
                            output,
                            error,
                        };
                        append_report_record(&record)?;
                        records.push(record);
                    }
                }
            }
        }
    }

    print_records(&records);
    Ok(())
}

fn tokens_per_sec(tokens: usize, elapsed_ms: u128) -> Option<f64> {
    if tokens == 0 || elapsed_ms == 0 {
        None
    } else {
        Some(tokens as f64 / (elapsed_ms as f64 / 1000.0))
    }
}

#![cfg(feature = "cuda")]

use anyhow::{Context, Result};
use m40_llm::generate::{
    generate_text, prepare_prompt, GenerateOptions, GeneratedLogitTraceStep, GeneratedText,
    PromptFormat,
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
    exact_block_staging_enabled: bool,
    staged_workspace_reused: bool,
    staged_workspace_bytes: Option<usize>,
    staged_workspace_capacity_tokens: Option<u32>,
    staged_workspace_allocations: usize,
    staged_kv_tokens: Option<usize>,
    staged_kv_bytes: Option<usize>,
    staged_old_tokens: Option<usize>,
    staged_recent_tokens: Option<usize>,
    staged_position_min: Option<usize>,
    staged_position_max: Option<usize>,
    prefill_tokens_per_sec: Option<f64>,
    decode_tokens_per_sec: Option<f64>,
    prefill_chunk_size: Option<usize>,
    compressed_prefill_chunk_size: Option<usize>,
    temporary_dense_kv_bytes: Option<usize>,
    final_kv_allocated_bytes: Option<usize>,
    dense_equivalent_kv_bytes: Option<usize>,
    exact_old_backing: Option<String>,
    exact_old_attention_backend: Option<String>,
    exact_block_backend_variant: Option<String>,
    q8_old_backing_bytes: Option<usize>,
    q8_old_backing_scale_bytes: Option<usize>,
    old_k_fp16_bytes: Option<usize>,
    q4_old_v_payload_bytes: Option<usize>,
    q4_old_v_scale_bytes: Option<usize>,
    recent_fp16_bytes: Option<usize>,
    summary_index_bytes: Option<usize>,
    compression_ratio: Option<f64>,
    prefill_mode: String,
    top_blocks: Option<u32>,
    needle_block_index: Option<u32>,
    selected_block_indices: Option<Vec<u32>>,
    selected_block_score_entries: Option<Vec<String>>,
    selection_records: Option<Vec<String>>,
    selected_blocks_score_order_is_chronological: Option<bool>,
    selected_block_order: String,
    selected_blocks_attention_order_is_chronological: Option<bool>,
    needle_block_selected: Option<bool>,
    needle_block_rank: Option<u32>,
    total_old_blocks: Option<u32>,
    active_attended_kv_tokens: Option<usize>,
    active_attended_kv_bytes: Option<usize>,
    active_attended_kv_bytes_all_layers: Option<usize>,
    active_attended_old_block_tokens: Option<usize>,
    active_attended_recent_tokens: Option<usize>,
    recent_ring_absolute_start: Option<usize>,
    recent_ring_absolute_end: Option<usize>,
    dense_window_candidate_positions: Option<Vec<usize>>,
    compressed_recent_candidate_positions: Option<Vec<usize>>,
    compressed_recent_ring_slots: Option<Vec<usize>>,
    attention_candidate_position_min: Option<u32>,
    attention_candidate_position_max: Option<u32>,
    attention_candidate_order_is_chronological: Option<bool>,
    needle_token_absolute_positions: Vec<usize>,
    question_token_absolute_positions: Vec<usize>,
    needle_tokens_in_recent_ring: bool,
    question_tokens_in_recent_ring: bool,
    expected_first_answer_token_id: Option<u32>,
    expected_first_answer_token_rank_dense: Option<usize>,
    expected_first_answer_token_rank_dense_window: Option<usize>,
    expected_first_answer_token_rank_compressed: Option<usize>,
    expected_first_answer_token_logit_dense: Option<f32>,
    expected_first_answer_token_logit_dense_window: Option<f32>,
    expected_first_answer_token_logit_compressed: Option<f32>,
    prompt_logit_max_abs_diff: Option<f32>,
    prompt_logit_mean_abs_diff: Option<f32>,
    prompt_logit_top10_overlap: Option<usize>,
    prompt_dense_top_token_id: Option<u32>,
    prompt_compressed_top_token_id: Option<u32>,
    prompt_dense_window_max_abs_diff: Option<f32>,
    prompt_dense_window_mean_abs_diff: Option<f32>,
    prompt_dense_window_top10_overlap: Option<usize>,
    prompt_dense_window_top_token_id: Option<u32>,
    first_decode_logit_max_abs_diff: Option<f32>,
    first_decode_logit_mean_abs_diff: Option<f32>,
    first_decode_logit_top10_overlap: Option<usize>,
    first_decode_dense_top_token_id: Option<u32>,
    first_decode_compressed_top_token_id: Option<u32>,
    first_decode_dense_window_max_abs_diff: Option<f32>,
    first_decode_dense_window_mean_abs_diff: Option<f32>,
    first_decode_dense_window_top10_overlap: Option<usize>,
    first_decode_dense_window_top_token_id: Option<u32>,
    attention_recent_mass: Option<f32>,
    attention_selected_old_exact_mass: Option<f32>,
    attention_summary_mass: Option<f32>,
    attention_representative_mass: Option<f32>,
    attention_other_mass: Option<f32>,
    attention_needle_block_mass: Option<f32>,
    attention_selected_block_masses: Option<Vec<String>>,
    attention_recent_logit_max: Option<f32>,
    attention_recent_logit_mean: Option<f32>,
    attention_summary_logit_max: Option<f32>,
    attention_summary_logit_mean: Option<f32>,
    attention_representative_logit_max: Option<f32>,
    attention_representative_logit_mean: Option<f32>,
    attention_top_entries: Option<Vec<String>>,
    attention_records: Option<Vec<String>>,
    generated_logit_trace: Option<Vec<GeneratedLogitTraceRecord>>,
    output: String,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct GeneratedLogitTraceRecord {
    step: usize,
    dense_generated_token: Option<u32>,
    compressed_generated_token: Option<u32>,
    dense_reference_token: Option<u32>,
    dense_reference_token_rank_dense: Option<usize>,
    dense_reference_token_rank_compressed: Option<usize>,
    dense_reference_token_logit_dense: Option<f32>,
    dense_reference_token_logit_compressed: Option<f32>,
    expected_answer_token: Option<u32>,
    expected_token_rank_dense: Option<usize>,
    expected_token_rank_compressed: Option<usize>,
    expected_token_logit_dense: Option<f32>,
    expected_token_logit_compressed: Option<f32>,
    eot_token: u32,
    eot_token_rank_dense: Option<usize>,
    eot_token_rank_compressed: Option<usize>,
    eot_token_logit_dense: Option<f32>,
    eot_token_logit_compressed: Option<f32>,
    dense_top_token: Option<u32>,
    compressed_top_token: Option<u32>,
    dense_top_logit: Option<f32>,
    compressed_top_logit: Option<f32>,
    max_logit_diff: Option<f32>,
    mean_logit_diff: Option<f32>,
    top10_overlap: Option<usize>,
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
        if q4_v_diagnostic_enabled() || mixed_q4_v_direct_sweep_enabled() {
            return Ok([2048, 4096]
                .into_iter()
                .filter(|ctx| *ctx < limit)
                .collect());
        }
        if exact_q8_diagnostic_enabled() {
            return Ok(if 4096 < limit { vec![4096] } else { Vec::new() });
        }
        if exact_block_retrieval_sweep_enabled() {
            return Ok(if 2048 < limit { vec![2048] } else { Vec::new() });
        }
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

fn needle_positions() -> Result<Vec<&'static str>> {
    let Some(value) = std::env::var("M40LLM_KV_QUALITY_NEEDLES")
        .ok()
        .filter(|value| !value.trim().is_empty())
    else {
        return Ok(vec!["old", "recent"]);
    };
    let mut needles = Vec::new();
    for part in value.split(',') {
        match part.trim() {
            "old" => needles.push("old"),
            "recent" => needles.push("recent"),
            "" => {}
            other => anyhow::bail!(
                "invalid M40LLM_KV_QUALITY_NEEDLES entry '{other}', expected old or recent"
            ),
        }
    }
    needles.sort_unstable();
    needles.dedup();
    if needles.is_empty() {
        anyhow::bail!("M40LLM_KV_QUALITY_NEEDLES did not contain any values");
    }
    Ok(needles)
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

fn exact_block_retrieval_sweep_enabled() -> bool {
    std::env::var("M40LLM_KV_EXACT_BLOCK_RETRIEVAL_SWEEP")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn q8_drift_diagnostic_enabled() -> bool {
    std::env::var("M40LLM_KV_Q8_DRIFT_DIAG")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn q8_precision_split_diagnostic_enabled() -> bool {
    std::env::var("M40LLM_KV_Q8_PRECISION_SPLIT_DIAG")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn q4_v_diagnostic_enabled() -> bool {
    std::env::var("M40LLM_KV_Q4_V_DIAG")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn mixed_q4_v_backing_sweep_enabled() -> bool {
    std::env::var("M40LLM_KV_MIXED_Q4_V_SWEEP")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn mixed_q4_v_direct_sweep_enabled() -> bool {
    std::env::var("M40LLM_KV_MIXED_Q4_V_DIRECT_SWEEP")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn top_block_robustness_diagnostic_enabled() -> bool {
    std::env::var("M40LLM_KV_TOP_BLOCK_ROBUSTNESS_DIAG")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn exact_q8_diagnostic_enabled() -> bool {
    q8_drift_diagnostic_enabled()
        || q8_precision_split_diagnostic_enabled()
        || q4_v_diagnostic_enabled()
        || mixed_q4_v_backing_sweep_enabled()
        || mixed_q4_v_direct_sweep_enabled()
        || top_block_robustness_diagnostic_enabled()
}

fn exact_block_staging_enabled() -> bool {
    std::env::var("M40LLM_KV_EXACT_BLOCK_STAGING")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn recent_equivalence_sequential_enabled() -> bool {
    std::env::var("M40LLM_KV_RECENT_EQUIV_SEQUENTIAL")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn logit_compare_enabled() -> bool {
    q4_v_diagnostic_enabled()
        || std::env::var("M40LLM_KV_LOGIT_COMPARE")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
}

fn selected_block_order() -> String {
    match std::env::var("M40LLM_KV_SELECTED_BLOCK_ORDER")
        .ok()
        .as_deref()
    {
        Some("chronological") => "chronological".to_string(),
        _ => "score".to_string(),
    }
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
        KvCompressMode::DenseRecentOnly | KvCompressMode::RecentOnly | KvCompressMode::BlockSummary
    ) {
        return Ok(vec![Some(0)]);
    }
    let values = std::env::var("M40LLM_KV_QUALITY_TOP_BLOCKS")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| parse_u32_list(&value, "M40LLM_KV_QUALITY_TOP_BLOCKS"))
        .transpose()?
        .unwrap_or_else(|| {
            if q4_v_diagnostic_enabled()
                || mixed_q4_v_backing_sweep_enabled()
                || mixed_q4_v_direct_sweep_enabled()
            {
                vec![2, 4, 8, 16]
            } else if q8_precision_split_diagnostic_enabled() {
                vec![4]
            } else if top_block_robustness_diagnostic_enabled() {
                vec![4, 8, 16]
            } else if q8_drift_diagnostic_enabled() {
                vec![4, 8]
            } else if exact_block_retrieval_sweep_enabled() {
                vec![1, 2, 4, 8, 16]
            } else if exact_selection_sweep {
                vec![4, 8, 16, 32, 64]
            } else {
                vec![16]
            }
        });
    Ok(values.into_iter().map(Some).collect())
}

#[derive(Debug, Clone, Copy)]
struct ExactBlockBackendCase {
    variant: Option<&'static str>,
    exact_old_backing: Option<&'static str>,
    exact_old_attention: Option<&'static str>,
    exact_old_precision: Option<&'static str>,
    q8_dense_shadow: bool,
    staging: Option<&'static str>,
}

fn exact_block_backend_cases(mode: KvCompressMode) -> Vec<ExactBlockBackendCase> {
    if q4_v_diagnostic_enabled() && mode == KvCompressMode::BlockSelectExact {
        vec![
            ExactBlockBackendCase {
                variant: Some("fp16-k-fp16-v"),
                exact_old_backing: None,
                exact_old_attention: None,
                exact_old_precision: None,
                q8_dense_shadow: false,
                staging: None,
            },
            ExactBlockBackendCase {
                variant: Some("fp16-k-q8-v"),
                exact_old_backing: Some("q8"),
                exact_old_attention: None,
                exact_old_precision: Some("fp16-k-q8-v"),
                q8_dense_shadow: true,
                staging: Some("1"),
            },
            ExactBlockBackendCase {
                variant: Some("fp16-k-q4-v"),
                exact_old_backing: Some("q8"),
                exact_old_attention: None,
                exact_old_precision: Some("fp16-k-q4-v"),
                q8_dense_shadow: true,
                staging: Some("1"),
            },
        ]
    } else if (mixed_q4_v_backing_sweep_enabled() || mixed_q4_v_direct_sweep_enabled())
        && mode == KvCompressMode::BlockSelectExact
    {
        let mut cases = vec![
            ExactBlockBackendCase {
                variant: Some("fp16-k-fp16-v"),
                exact_old_backing: None,
                exact_old_attention: None,
                exact_old_precision: None,
                q8_dense_shadow: false,
                staging: None,
            },
            ExactBlockBackendCase {
                variant: Some("fp16-k-q4-v"),
                exact_old_backing: Some("fp16-k-q4-v"),
                exact_old_attention: None,
                exact_old_precision: None,
                q8_dense_shadow: false,
                staging: Some("1"),
            },
        ];
        if mixed_q4_v_direct_sweep_enabled() {
            cases.push(ExactBlockBackendCase {
                variant: Some("fp16-k-q4-v-direct"),
                exact_old_backing: Some("fp16-k-q4-v"),
                exact_old_attention: Some("fp16-k-q4-v-direct"),
                exact_old_precision: None,
                q8_dense_shadow: false,
                staging: None,
            });
        }
        cases
    } else if q8_precision_split_diagnostic_enabled() && mode == KvCompressMode::BlockSelectExact {
        vec![
            ExactBlockBackendCase {
                variant: Some("fp16-k-fp16-v"),
                exact_old_backing: None,
                exact_old_attention: None,
                exact_old_precision: None,
                q8_dense_shadow: false,
                staging: None,
            },
            ExactBlockBackendCase {
                variant: Some("q8-k-q8-v"),
                exact_old_backing: Some("q8"),
                exact_old_attention: None,
                exact_old_precision: Some("q8-k-q8-v"),
                q8_dense_shadow: true,
                staging: Some("1"),
            },
            ExactBlockBackendCase {
                variant: Some("fp16-k-q8-v"),
                exact_old_backing: Some("q8"),
                exact_old_attention: None,
                exact_old_precision: Some("fp16-k-q8-v"),
                q8_dense_shadow: true,
                staging: Some("1"),
            },
            ExactBlockBackendCase {
                variant: Some("q8-k-fp16-v"),
                exact_old_backing: Some("q8"),
                exact_old_attention: None,
                exact_old_precision: Some("q8-k-fp16-v"),
                q8_dense_shadow: true,
                staging: Some("1"),
            },
        ]
    } else if (q8_drift_diagnostic_enabled() || top_block_robustness_diagnostic_enabled())
        && mode == KvCompressMode::BlockSelectExact
    {
        vec![
            ExactBlockBackendCase {
                variant: Some("fp16-exact"),
                exact_old_backing: None,
                exact_old_attention: None,
                exact_old_precision: None,
                q8_dense_shadow: false,
                staging: None,
            },
            ExactBlockBackendCase {
                variant: Some("staged-q8"),
                exact_old_backing: Some("q8"),
                exact_old_attention: None,
                exact_old_precision: Some("q8-k-q8-v"),
                q8_dense_shadow: top_block_robustness_diagnostic_enabled(),
                staging: Some("1"),
            },
            ExactBlockBackendCase {
                variant: Some("direct-q8"),
                exact_old_backing: Some("q8"),
                exact_old_attention: Some("q8-direct"),
                exact_old_precision: Some("q8-k-q8-v"),
                q8_dense_shadow: false,
                staging: Some("1"),
            },
        ]
    } else {
        vec![ExactBlockBackendCase {
            variant: None,
            exact_old_backing: None,
            exact_old_attention: None,
            exact_old_precision: None,
            q8_dense_shadow: false,
            staging: None,
        }]
    }
}

fn apply_exact_block_backend_case(case: ExactBlockBackendCase) -> Vec<EnvVarGuard> {
    let mut guards = Vec::new();
    if let Some(value) = case.exact_old_backing {
        guards.push(EnvVarGuard::set("M40LLM_KV_EXACT_OLD_BACKING", value));
    } else if exact_q8_diagnostic_enabled() {
        guards.push(EnvVarGuard::unset("M40LLM_KV_EXACT_OLD_BACKING"));
    }
    if let Some(value) = case.exact_old_attention {
        guards.push(EnvVarGuard::set("M40LLM_KV_EXACT_OLD_ATTENTION", value));
    } else if exact_q8_diagnostic_enabled() {
        guards.push(EnvVarGuard::unset("M40LLM_KV_EXACT_OLD_ATTENTION"));
    }
    if let Some(value) = case.exact_old_precision {
        guards.push(EnvVarGuard::set("M40LLM_KV_EXACT_OLD_PRECISION", value));
    } else if exact_q8_diagnostic_enabled() {
        guards.push(EnvVarGuard::unset("M40LLM_KV_EXACT_OLD_PRECISION"));
    }
    if case.q8_dense_shadow {
        guards.push(EnvVarGuard::set("M40LLM_KV_Q8_DENSE_SHADOW", "1"));
    } else if exact_q8_diagnostic_enabled() {
        guards.push(EnvVarGuard::unset("M40LLM_KV_Q8_DENSE_SHADOW"));
    }
    if matches!(case.exact_old_precision, Some("fp16-k-q4-v")) {
        guards.push(EnvVarGuard::set("M40LLM_KV_Q4_V_DIAG", "1"));
    }
    if let Some(value) = case.staging {
        guards.push(EnvVarGuard::set("M40LLM_KV_EXACT_BLOCK_STAGING", value));
    } else if exact_q8_diagnostic_enabled() {
        guards.push(EnvVarGuard::unset("M40LLM_KV_EXACT_BLOCK_STAGING"));
    }
    guards
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
    if recent_equivalence_sequential_enabled() {
        &[
            KvCompressMode::Off,
            KvCompressMode::DenseRecentOnly,
            KvCompressMode::RecentOnly,
            KvCompressMode::BlockSelectExact,
        ]
    } else if exact_q8_diagnostic_enabled() || exact_block_retrieval_sweep_enabled() {
        &[KvCompressMode::Off, KvCompressMode::BlockSelectExact]
    } else if exact_selection_sweep {
        &[
            KvCompressMode::Off,
            KvCompressMode::DenseRecentOnly,
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
            KvCompressMode::DenseRecentOnly,
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
    let use_compressed_kv = matches!(
        mode,
        KvCompressMode::RecentOnly
            | KvCompressMode::BlockSummary
            | KvCompressMode::BlockSelectLossy
    ) || (mode == KvCompressMode::BlockSelectExact
        && std::env::var("M40LLM_KV_EXACT_OLD_BACKING")
            .map(|value| matches!(value.as_str(), "q8" | "Q8" | "fp16-k-q4-v" | "FP16-K-Q4-V"))
            .unwrap_or(false));
    if use_compressed_kv {
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
    let exact_selection_sweep = exact_selection_sweep_enabled()
        || exact_block_retrieval_sweep_enabled()
        || exact_q8_diagnostic_enabled();
    let recent_equivalence_sequential = recent_equivalence_sequential_enabled();
    let _prefill_guard = if (lossy_packed_sweep_enabled()
        || (exact_selection_sweep
            && matches!(
                mode,
                KvCompressMode::Off
                    | KvCompressMode::DenseRecentOnly
                    | KvCompressMode::BlockSelectExact
            )))
        && matches!(
            mode,
            KvCompressMode::Off
                | KvCompressMode::DenseRecentOnly
                | KvCompressMode::BlockSelectExact
        ) {
        Some(EnvVarGuard::set(
            "M40LLM_PREFILL_CHUNK_SIZE",
            target_tokens.to_string(),
        ))
    } else {
        None
    };
    let _packed_then_compress_guard = if (lossy_packed_sweep_enabled()
        || (exact_selection_sweep && !recent_equivalence_sequential))
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
                | KvCompressMode::DenseRecentOnly
                | KvCompressMode::RecentOnly
                | KvCompressMode::BlockSummary
                | KvCompressMode::BlockSelectLossy
        ) {
        Some(EnvVarGuard::set("M40LLM_KV_SELECTION_TELEMETRY", "1"))
    } else {
        None
    };
    let _logit_compare_guard =
        q4_v_diagnostic_enabled().then(|| EnvVarGuard::set("M40LLM_KV_LOGIT_COMPARE", "1"));
    let _logit_trace_guard =
        q4_v_diagnostic_enabled().then(|| EnvVarGuard::set("M40LLM_KV_LOGIT_TRACE", "1"));

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
        KvCompressMode::DenseRecentOnly => "dense-recent-only",
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

fn selected_block_entry_strings(
    summary: &m40_llm::kv_selection::KvSelectionSummary,
) -> Vec<String> {
    summary
        .selection_records
        .iter()
        .flat_map(|record| {
            record.selected_blocks.iter().map(move |block| {
                format!(
                    "layer={:?}:token={:?}:rank={}:block={}:score={:.6}:range={}..{}",
                    record.layer,
                    record.token,
                    block.rank,
                    block.block_index,
                    block.score,
                    block.absolute_start,
                    block.absolute_end
                )
            })
        })
        .collect()
}

fn selection_record_strings(summary: &m40_llm::kv_selection::KvSelectionSummary) -> Vec<String> {
    summary
        .selection_records
        .iter()
        .map(|record| {
            let blocks = record
                .selected_blocks
                .iter()
                .map(|block| {
                    format!(
                        "{}@{:.4}[{}..{}]",
                        block.block_index, block.score, block.absolute_start, block.absolute_end
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "layer={:?}:token={:?}:top_blocks={}:total_old_blocks={}:blocks={}",
                record.layer, record.token, record.top_blocks, record.total_old_blocks, blocks
            )
        })
        .collect()
}

fn first_selection_is_chronological(
    summary: &m40_llm::kv_selection::KvSelectionSummary,
) -> Option<bool> {
    let first = summary.selection_records.first()?;
    Some(
        first
            .selected_blocks
            .windows(2)
            .all(|pair| pair[0].block_index <= pair[1].block_index),
    )
}

fn attention_block_mass_strings(
    attention: &m40_llm::kv_selection::KvAttentionTelemetrySummary,
) -> Vec<String> {
    attention
        .selected_block_masses
        .iter()
        .map(|mass| {
            format!(
                "block={}:mass={:.6}:logit_max={:.4}:logit_mean={:.4}:count={}",
                mass.block_index, mass.prob_mass, mass.logit_max, mass.logit_mean, mass.count
            )
        })
        .collect()
}

fn first_attention_candidate_bounds(
    summary: &m40_llm::kv_selection::KvSelectionSummary,
    recent_start: Option<usize>,
    recent_end: Option<usize>,
) -> (Option<u32>, Option<u32>, Option<bool>) {
    let Some(record) = summary.selection_records.first() else {
        return (None, None, None);
    };
    let mut min_pos: Option<u32> = None;
    let mut max_pos: Option<u32> = None;
    for block in &record.selected_blocks {
        min_pos = Some(min_pos.map_or(block.absolute_start, |pos| pos.min(block.absolute_start)));
        max_pos = Some(max_pos.map_or(block.absolute_end, |pos| pos.max(block.absolute_end)));
    }
    if let (Some(start), Some(end)) = (recent_start, recent_end) {
        min_pos = Some(min_pos.map_or(start as u32, |pos| pos.min(start as u32)));
        max_pos = Some(max_pos.map_or(end as u32, |pos| pos.max(end as u32)));
    }
    (min_pos, max_pos, first_selection_is_chronological(summary))
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

fn recent_candidate_positions(start: Option<usize>, end: Option<usize>) -> Option<Vec<usize>> {
    Some((start?..end?).collect())
}

fn recent_ring_slots(
    start: Option<usize>,
    end: Option<usize>,
    recent_window: usize,
) -> Option<Vec<usize>> {
    Some((start?..end?).map(|pos| pos % recent_window).collect())
}

#[derive(Debug, Clone, Copy)]
struct ActiveKvWorkingSet {
    tokens: usize,
    bytes_per_layer: usize,
    bytes_all_layers: usize,
    old_block_tokens: usize,
    recent_tokens: usize,
}

#[derive(Debug, Clone, Copy)]
struct ActiveKvAccounting {
    recent_window: usize,
    block_size: usize,
    kv_heads: usize,
    head_dim: usize,
    layer_count: usize,
}

fn active_kv_working_set(
    mode: KvCompressMode,
    prompt_tokens: usize,
    top_blocks: Option<u32>,
    accounting: ActiveKvAccounting,
) -> Option<ActiveKvWorkingSet> {
    if prompt_tokens == 0 || accounting.kv_heads == 0 || accounting.head_dim == 0 {
        return None;
    }
    let recent_tokens = prompt_tokens.min(accounting.recent_window);
    let old_len = prompt_tokens.saturating_sub(recent_tokens);
    let old_block_tokens = match mode {
        KvCompressMode::Off => old_len,
        KvCompressMode::DenseRecentOnly | KvCompressMode::RecentOnly => 0,
        KvCompressMode::BlockSelectExact => {
            let old_blocks = old_len.div_ceil(accounting.block_size);
            let selected_blocks = (top_blocks? as usize).min(old_blocks);
            old_len.min(selected_blocks * accounting.block_size)
        }
        KvCompressMode::BlockSummary | KvCompressMode::BlockSelectLossy => 0,
    };
    let tokens = recent_tokens + old_block_tokens;
    let bytes_per_token =
        accounting.kv_heads * accounting.head_dim * 2 * std::mem::size_of::<half::f16>();
    let bytes_per_layer = tokens * bytes_per_token;
    Some(ActiveKvWorkingSet {
        tokens,
        bytes_per_layer,
        bytes_all_layers: bytes_per_layer * accounting.layer_count,
        old_block_tokens,
        recent_tokens,
    })
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

fn top_token_with_logit(logits: &[f32]) -> Option<(u32, f32)> {
    logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| f32::total_cmp(&a.1, &b.1))
        .map(|(idx, logit)| (idx as u32, logit))
}

fn generated_trace_records(
    dense: Option<&[GeneratedLogitTraceStep]>,
    compressed: Option<&[GeneratedLogitTraceStep]>,
    expected_token_id: Option<u32>,
) -> Option<Vec<GeneratedLogitTraceRecord>> {
    let compressed = compressed?;
    let max_len = dense.map_or(compressed.len(), |dense| dense.len().max(compressed.len()));
    let mut records = Vec::with_capacity(max_len);
    for step in 0..max_len {
        let dense_step = dense.and_then(|trace| trace.get(step));
        let compressed_step = compressed.get(step);
        let diff = dense_step
            .zip(compressed_step)
            .and_then(|(dense, compressed)| logit_diff_summary(&dense.logits, &compressed.logits));
        let expected = expected_token_summary(
            dense_step.map(|step| step.logits.as_slice()),
            compressed_step.map(|step| step.logits.as_slice()),
            expected_token_id,
        );
        let dense_reference_token = dense_step.map(|step| step.sampled_token_id);
        let dense_reference = expected_token_summary(
            dense_step.map(|step| step.logits.as_slice()),
            compressed_step.map(|step| step.logits.as_slice()),
            dense_reference_token,
        );
        let eot_token = 128009u32;
        let eot = expected_token_summary(
            dense_step.map(|step| step.logits.as_slice()),
            compressed_step.map(|step| step.logits.as_slice()),
            Some(eot_token),
        );
        let dense_top = dense_step.and_then(|step| top_token_with_logit(&step.logits));
        let compressed_top = compressed_step.and_then(|step| top_token_with_logit(&step.logits));
        records.push(GeneratedLogitTraceRecord {
            step,
            dense_generated_token: dense_step.map(|step| step.sampled_token_id),
            compressed_generated_token: compressed_step.map(|step| step.sampled_token_id),
            dense_reference_token,
            dense_reference_token_rank_dense: dense_reference.rank_dense,
            dense_reference_token_rank_compressed: dense_reference.rank_compressed,
            dense_reference_token_logit_dense: dense_reference.logit_dense,
            dense_reference_token_logit_compressed: dense_reference.logit_compressed,
            expected_answer_token: expected_token_id,
            expected_token_rank_dense: expected.rank_dense,
            expected_token_rank_compressed: expected.rank_compressed,
            expected_token_logit_dense: expected.logit_dense,
            expected_token_logit_compressed: expected.logit_compressed,
            eot_token,
            eot_token_rank_dense: eot.rank_dense,
            eot_token_rank_compressed: eot.rank_compressed,
            eot_token_logit_dense: eot.logit_dense,
            eot_token_logit_compressed: eot.logit_compressed,
            dense_top_token: dense_top.map(|(token, _)| token),
            compressed_top_token: compressed_top.map(|(token, _)| token),
            dense_top_logit: dense_top.map(|(_, logit)| logit),
            compressed_top_logit: compressed_top.map(|(_, logit)| logit),
            max_logit_diff: diff.as_ref().map(|summary| summary.max_abs_diff),
            mean_logit_diff: diff.as_ref().map(|summary| summary.mean_abs_diff),
            top10_overlap: diff.as_ref().map(|summary| summary.top10_overlap),
        });
    }
    Some(records)
}

fn print_records(records: &[CaseRecord]) {
    eprintln!("KV compression retrieval quality results:");
    for record in records {
        let dense_candidate_summary =
            record
                .dense_window_candidate_positions
                .as_ref()
                .map(|positions| {
                    (
                        positions.first().copied(),
                        positions.last().copied(),
                        positions.len(),
                    )
                });
        let compressed_candidate_summary = record
            .compressed_recent_candidate_positions
            .as_ref()
            .map(|positions| {
                (
                    positions.first().copied(),
                    positions.last().copied(),
                    positions.len(),
                )
            });
        let compressed_slot_summary = record
            .compressed_recent_ring_slots
            .as_ref()
            .map(|slots| (slots.first().copied(), slots.last().copied(), slots.len()));
        eprintln!(
            "  ctx={} prompt={} generated={} needle={} mode={} reps={} rep_policy={} top_blocks={:?} status={:?} ring={:?}..{:?} dense_candidates={:?} compressed_candidates={:?} compressed_slots={:?} needle_pos={:?} question_pos={:?} needle_recent={} question_recent={} expected_id={:?} expected_rank(dense={:?},dense_window={:?},mode={:?}) expected_logit(dense={:?},dense_window={:?},mode={:?}) prompt_diff(max={:?},mean={:?},top10={:?},dense_top={:?},mode_top={:?}) prompt_window_diff(max={:?},mean={:?},top10={:?},window_top={:?}) first_decode_diff(max={:?},mean={:?},top10={:?},dense_top={:?},mode_top={:?}) first_decode_window_diff(max={:?},mean={:?},top10={:?},window_top={:?}) needle_block={:?} selected={:?} needle_selected={:?} needle_rank={:?} old_blocks={:?} exact_backend={:?} exact_old_backing={:?} exact_old_attention={:?} q8_old(bytes={:?},scale_bytes={:?}) old_k_fp16_bytes={:?} q4_v(payload_bytes={:?},scale_bytes={:?}) recent_fp16_bytes={:?} summary_index_bytes={:?} staging={} staging_workspace(reused={},bytes={:?},capacity_tokens={:?},allocations={}) staged(tokens={:?},bytes={:?},old={:?},recent={:?},pos={:?}..{:?}) active_kv(tokens={:?},bytes={:?},all_layers={:?},old={:?},recent={:?}) mass(recent={:?},old_exact={:?},summary={:?},rep={:?},needle={:?}) logits(recent_max={:?},recent_mean={:?},summary_max={:?},summary_mean={:?}) top_attn={:?} attn_records={:?} prefill={}ms decode={}ms total={}ms prefill_tps={:?} decode_tps={:?} final_kv_bytes={:?} dense_equiv_kv_bytes={:?} temp_dense_kv_bytes={:?} compression_ratio={:?} prefill_mode={} output={:?} error={}",
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
            dense_candidate_summary,
            compressed_candidate_summary,
            compressed_slot_summary,
            record.needle_token_absolute_positions,
            record.question_token_absolute_positions,
            record.needle_tokens_in_recent_ring,
            record.question_tokens_in_recent_ring,
            record.expected_first_answer_token_id,
            record.expected_first_answer_token_rank_dense,
            record.expected_first_answer_token_rank_dense_window,
            record.expected_first_answer_token_rank_compressed,
            record.expected_first_answer_token_logit_dense,
            record.expected_first_answer_token_logit_dense_window,
            record.expected_first_answer_token_logit_compressed,
            record.prompt_logit_max_abs_diff,
            record.prompt_logit_mean_abs_diff,
            record.prompt_logit_top10_overlap,
            record.prompt_dense_top_token_id,
            record.prompt_compressed_top_token_id,
            record.prompt_dense_window_max_abs_diff,
            record.prompt_dense_window_mean_abs_diff,
            record.prompt_dense_window_top10_overlap,
            record.prompt_dense_window_top_token_id,
            record.first_decode_logit_max_abs_diff,
            record.first_decode_logit_mean_abs_diff,
            record.first_decode_logit_top10_overlap,
            record.first_decode_dense_top_token_id,
            record.first_decode_compressed_top_token_id,
            record.first_decode_dense_window_max_abs_diff,
            record.first_decode_dense_window_mean_abs_diff,
            record.first_decode_dense_window_top10_overlap,
            record.first_decode_dense_window_top_token_id,
            record.needle_block_index,
            record.selected_block_indices,
            record.needle_block_selected,
            record.needle_block_rank,
            record.total_old_blocks,
            record.exact_block_backend_variant,
            record.exact_old_backing,
            record.exact_old_attention_backend,
            record.q8_old_backing_bytes,
            record.q8_old_backing_scale_bytes,
            record.old_k_fp16_bytes,
            record.q4_old_v_payload_bytes,
            record.q4_old_v_scale_bytes,
            record.recent_fp16_bytes,
            record.summary_index_bytes,
            record.exact_block_staging_enabled,
            record.staged_workspace_reused,
            record.staged_workspace_bytes,
            record.staged_workspace_capacity_tokens,
            record.staged_workspace_allocations,
            record.staged_kv_tokens,
            record.staged_kv_bytes,
            record.staged_old_tokens,
            record.staged_recent_tokens,
            record.staged_position_min,
            record.staged_position_max,
            record.active_attended_kv_tokens,
            record.active_attended_kv_bytes,
            record.active_attended_kv_bytes_all_layers,
            record.active_attended_old_block_tokens,
            record.active_attended_recent_tokens,
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
    let exact_selection_sweep = exact_selection_sweep_enabled() || exact_q8_diagnostic_enabled();
    let modes = quality_modes(lossy_packed_sweep, exact_selection_sweep);
    let rep_cases = representative_cases(lossy_packed_sweep);
    let needle_positions = needle_positions()?;

    let mut records = Vec::new();
    for target_tokens in contexts {
        for &needle_position in &needle_positions {
            let mut dense_passed = None;
            let mut dense_prompt_logits: Option<Vec<f32>> = None;
            let mut dense_first_decode_logits: Option<Vec<f32>> = None;
            let mut dense_generated_logit_trace: Option<Vec<GeneratedLogitTraceStep>> = None;
            let mut dense_window_prompt_logits: Option<Vec<f32>> = None;
            let mut dense_window_first_decode_logits: Option<Vec<f32>> = None;
            let mut exact_reference_prompt_logits: Option<Vec<f32>> = None;
            let mut exact_reference_first_decode_logits: Option<Vec<f32>> = None;
            let mut exact_reference_generated_logit_trace: Option<Vec<GeneratedLogitTraceStep>> =
                None;
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
                if q8_precision_split_diagnostic_enabled() && needle_position != "old" {
                    continue;
                }
                if top_block_robustness_diagnostic_enabled() && needle_position != "recent" {
                    continue;
                }
                let mode_rep_cases: Vec<RepresentativeCase> =
                    if matches!(mode, KvCompressMode::Off | KvCompressMode::DenseRecentOnly) {
                        vec![RepresentativeCase {
                            representatives: 0,
                            policy: KvRepresentativePolicy::Last,
                        }]
                    } else {
                        rep_cases.clone()
                    };
                for rep_case in mode_rep_cases {
                    for top_blocks_case in top_block_cases(exact_selection_sweep, mode)? {
                        if q8_drift_diagnostic_enabled()
                            && mode == KvCompressMode::BlockSelectExact
                            && !matches!(
                                (needle_position, top_blocks_case),
                                ("old", Some(4)) | ("recent", Some(8))
                            )
                        {
                            continue;
                        }
                        if q8_precision_split_diagnostic_enabled()
                            && mode == KvCompressMode::BlockSelectExact
                            && !matches!((needle_position, top_blocks_case), ("old", Some(4)))
                        {
                            continue;
                        }
                        if top_block_robustness_diagnostic_enabled()
                            && mode == KvCompressMode::BlockSelectExact
                            && needle_position != "recent"
                        {
                            continue;
                        }
                        if q4_v_diagnostic_enabled()
                            && mode == KvCompressMode::BlockSelectExact
                            && !matches!(
                                (target_tokens, needle_position, top_blocks_case),
                                (2048, "old", Some(2))
                                    | (2048, "recent", Some(2))
                                    | (4096, "old", Some(4))
                                    | (4096, "recent", Some(4))
                                    | (4096, "recent", Some(8))
                                    | (4096, "recent", Some(16))
                            )
                        {
                            continue;
                        }
                        if (mixed_q4_v_backing_sweep_enabled() || mixed_q4_v_direct_sweep_enabled())
                            && mode == KvCompressMode::BlockSelectExact
                            && !matches!(
                                (target_tokens, needle_position, top_blocks_case),
                                (2048, "old", Some(2))
                                    | (2048, "recent", Some(2))
                                    | (4096, "old", Some(4))
                                    | (4096, "recent", Some(4))
                                    | (4096, "recent", Some(8))
                                    | (4096, "recent", Some(16))
                            )
                        {
                            continue;
                        }
                        for backend_case in exact_block_backend_cases(mode) {
                            let _backend_guards = apply_exact_block_backend_case(backend_case);
                            let exact_block_backend_variant =
                                backend_case.variant.map(str::to_string);
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
                            let mut exact_old_backing = None;
                            let mut exact_old_attention_backend = None;
                            let mut q8_old_backing_bytes = None;
                            let mut q8_old_backing_scale_bytes = None;
                            let mut old_k_fp16_bytes = None;
                            let mut q4_old_v_payload_bytes = None;
                            let mut q4_old_v_scale_bytes = None;
                            let mut recent_fp16_bytes = None;
                            let mut summary_index_bytes = None;
                            let mut staged_workspace_reused = false;
                            let mut staged_workspace_bytes = None;
                            let mut staged_workspace_capacity_tokens = None;
                            let mut staged_workspace_allocations = 0usize;
                            let mut prefill_mode = "error".to_string();
                            let mut selection_summary = None;
                            let mut needle_block_index = None;
                            let mut prompt_logits = None;
                            let mut first_decode_logits = None;
                            let mut generated_logit_trace_raw: Option<
                                Vec<GeneratedLogitTraceStep>,
                            > = None;
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
                                    generated_decode_elapsed_ms =
                                        generated.generated_decode_elapsed_ms;
                                    total_elapsed_ms = generated.total_elapsed_ms;
                                    attention_compression_elapsed_ms =
                                        generated.attention_compression_elapsed_ms;
                                    prefill_chunk_size = generated.prefill_chunk_size;
                                    compressed_prefill_chunk_size =
                                        generated.compressed_prefill_chunk_size;
                                    temporary_dense_kv_bytes = generated.temporary_dense_kv_bytes;
                                    final_kv_allocated_bytes = generated.final_kv_allocated_bytes;
                                    dense_equivalent_kv_bytes = generated.dense_equivalent_kv_bytes;
                                    exact_old_backing = generated.exact_old_backing.clone();
                                    exact_old_attention_backend =
                                        generated.exact_old_attention_backend.clone();
                                    q8_old_backing_bytes = generated.q8_old_backing_bytes;
                                    q8_old_backing_scale_bytes =
                                        generated.q8_old_backing_scale_bytes;
                                    old_k_fp16_bytes = generated.old_k_fp16_bytes;
                                    q4_old_v_payload_bytes = generated.q4_old_v_payload_bytes;
                                    q4_old_v_scale_bytes = generated.q4_old_v_scale_bytes;
                                    recent_fp16_bytes = generated.recent_fp16_bytes;
                                    summary_index_bytes = generated.summary_index_bytes;
                                    staged_workspace_reused = generated.staged_workspace_reused;
                                    staged_workspace_bytes = generated.staged_workspace_bytes;
                                    staged_workspace_capacity_tokens =
                                        generated.staged_workspace_capacity_tokens;
                                    staged_workspace_allocations =
                                        generated.staged_workspace_allocations;
                                    prefill_mode = generated.prefill_mode.clone();
                                    selection_summary = generated.kv_selection.clone();
                                    needle_block_index = needle_block;
                                    prompt_logits = generated.prompt_logits.clone();
                                    first_decode_logits = generated.first_decode_logits.clone();
                                    generated_logit_trace_raw =
                                        generated.generated_logit_trace.clone();
                                    let passed = generated.output.contains(NEEDLE);
                                    if mode == KvCompressMode::Off {
                                        dense_passed = Some(passed);
                                        dense_prompt_logits = generated.prompt_logits.clone();
                                        dense_first_decode_logits =
                                            generated.first_decode_logits.clone();
                                        dense_generated_logit_trace =
                                            generated.generated_logit_trace.clone();
                                    } else if mode == KvCompressMode::DenseRecentOnly {
                                        dense_window_prompt_logits =
                                            generated.prompt_logits.clone();
                                        dense_window_first_decode_logits =
                                            generated.first_decode_logits.clone();
                                    }
                                    if q4_v_diagnostic_enabled()
                                        && mode == KvCompressMode::BlockSelectExact
                                        && exact_block_backend_variant.as_deref()
                                            == Some("fp16-k-fp16-v")
                                    {
                                        exact_reference_prompt_logits =
                                            generated.prompt_logits.clone();
                                        exact_reference_first_decode_logits =
                                            generated.first_decode_logits.clone();
                                        exact_reference_generated_logit_trace =
                                            generated.generated_logit_trace.clone();
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
                            let selected_block_score_entries =
                                selection_summary.as_ref().map(selected_block_entry_strings);
                            let selection_records =
                                selection_summary.as_ref().map(selection_record_strings);
                            let selected_blocks_score_order_is_chronological = selection_summary
                                .as_ref()
                                .and_then(first_selection_is_chronological);
                            let selected_block_order = selected_block_order();
                            let selected_blocks_attention_order_is_chronological =
                                if selected_block_order == "chronological" {
                                    Some(true)
                                } else {
                                    selected_blocks_score_order_is_chronological
                                };
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
                            let attention_selected_block_masses =
                                attention.map(attention_block_mass_strings);
                            let (
                                attention_candidate_position_min,
                                attention_candidate_position_max,
                                attention_candidate_order_is_chronological,
                            ) = selection_summary
                                .as_ref()
                                .map(|summary| {
                                    first_attention_candidate_bounds(
                                        summary,
                                        position_meta.recent_ring_absolute_start,
                                        position_meta.recent_ring_absolute_end,
                                    )
                                })
                                .unwrap_or((None, None, None));
                            let attention_candidate_order_is_chronological = if selected_block_order
                                == "chronological"
                                && selection_summary.is_some()
                            {
                                Some(true)
                            } else {
                                attention_candidate_order_is_chronological
                            };
                            let prompt_logit_diff = if logit_compare_enabled() {
                                let reference = if q4_v_diagnostic_enabled() {
                                    exact_reference_prompt_logits.as_deref()
                                } else {
                                    dense_prompt_logits.as_deref()
                                };
                                reference.zip(prompt_logits.as_deref()).and_then(
                                    |(dense, compressed)| logit_diff_summary(dense, compressed),
                                )
                            } else {
                                None
                            };
                            let first_decode_logit_diff = if logit_compare_enabled() {
                                let reference = if q4_v_diagnostic_enabled() {
                                    exact_reference_first_decode_logits.as_deref()
                                } else {
                                    dense_first_decode_logits.as_deref()
                                };
                                reference.zip(first_decode_logits.as_deref()).and_then(
                                    |(dense, compressed)| logit_diff_summary(dense, compressed),
                                )
                            } else {
                                None
                            };
                            let prompt_dense_window_diff = if logit_compare_enabled() {
                                dense_window_prompt_logits
                                    .as_deref()
                                    .zip(prompt_logits.as_deref())
                                    .and_then(|(window, compressed)| {
                                        logit_diff_summary(window, compressed)
                                    })
                            } else {
                                None
                            };
                            let first_decode_dense_window_diff = if logit_compare_enabled() {
                                dense_window_first_decode_logits
                                    .as_deref()
                                    .zip(first_decode_logits.as_deref())
                                    .and_then(|(window, compressed)| {
                                        logit_diff_summary(window, compressed)
                                    })
                            } else {
                                None
                            };
                            let expected_token_window_logits = if logit_compare_enabled() {
                                expected_token_summary(
                                    dense_prompt_logits.as_deref(),
                                    dense_window_prompt_logits.as_deref(),
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
                            let expected_token_logits = if logit_compare_enabled() {
                                let reference = if q4_v_diagnostic_enabled() {
                                    exact_reference_prompt_logits.as_deref()
                                } else {
                                    dense_prompt_logits.as_deref()
                                };
                                expected_token_summary(
                                    reference,
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
                            let generated_logit_trace = generated_trace_records(
                                if q4_v_diagnostic_enabled() {
                                    exact_reference_generated_logit_trace.as_deref()
                                } else {
                                    dense_generated_logit_trace.as_deref()
                                },
                                generated_logit_trace_raw.as_deref(),
                                position_meta.expected_first_answer_token_id,
                            );
                            let active_kv = active_kv_working_set(
                                mode,
                                prompt_tokens,
                                top_blocks_case,
                                ActiveKvAccounting {
                                    recent_window: 1024,
                                    block_size: 32,
                                    kv_heads: config.attention_head_count_kv as usize,
                                    head_dim: config.attention_key_length as usize,
                                    layer_count: config.block_count as usize,
                                },
                            );
                            let staging_enabled = exact_block_staging_enabled()
                                && mode == KvCompressMode::BlockSelectExact;
                            let staged_position_min = if staging_enabled {
                                selected_block_indices
                                    .as_ref()
                                    .and_then(|selected| selected.iter().min().copied())
                                    .map(|block| block as usize * 32)
                                    .or(position_meta.recent_ring_absolute_start)
                            } else {
                                None
                            };
                            let staged_position_max = if staging_enabled {
                                prompt_tokens.checked_sub(1)
                            } else {
                                None
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
                                exact_block_staging_enabled: staging_enabled,
                                staged_workspace_reused,
                                staged_workspace_bytes,
                                staged_workspace_capacity_tokens,
                                staged_workspace_allocations,
                                staged_kv_tokens: staging_enabled
                                    .then(|| active_kv.map(|working| working.tokens))
                                    .flatten(),
                                staged_kv_bytes: staging_enabled
                                    .then(|| active_kv.map(|working| working.bytes_per_layer))
                                    .flatten(),
                                staged_old_tokens: staging_enabled
                                    .then(|| active_kv.map(|working| working.old_block_tokens))
                                    .flatten(),
                                staged_recent_tokens: staging_enabled
                                    .then(|| active_kv.map(|working| working.recent_tokens))
                                    .flatten(),
                                staged_position_min,
                                staged_position_max,
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
                                exact_old_backing,
                                exact_old_attention_backend,
                                exact_block_backend_variant,
                                q8_old_backing_bytes,
                                q8_old_backing_scale_bytes,
                                old_k_fp16_bytes,
                                q4_old_v_payload_bytes,
                                q4_old_v_scale_bytes,
                                recent_fp16_bytes,
                                summary_index_bytes,
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
                                selected_block_score_entries,
                                selection_records,
                                selected_blocks_score_order_is_chronological,
                                selected_block_order,
                                selected_blocks_attention_order_is_chronological,
                                needle_block_selected,
                                needle_block_rank,
                                total_old_blocks,
                                active_attended_kv_tokens: active_kv.map(|working| working.tokens),
                                active_attended_kv_bytes: active_kv
                                    .map(|working| working.bytes_per_layer),
                                active_attended_kv_bytes_all_layers: active_kv
                                    .map(|working| working.bytes_all_layers),
                                active_attended_old_block_tokens: active_kv
                                    .map(|working| working.old_block_tokens),
                                active_attended_recent_tokens: active_kv
                                    .map(|working| working.recent_tokens),
                                recent_ring_absolute_start: position_meta
                                    .recent_ring_absolute_start,
                                recent_ring_absolute_end: position_meta.recent_ring_absolute_end,
                                dense_window_candidate_positions: logit_compare_enabled()
                                    .then(|| {
                                        recent_candidate_positions(
                                            position_meta.recent_ring_absolute_start,
                                            position_meta.recent_ring_absolute_end,
                                        )
                                    })
                                    .flatten(),
                                compressed_recent_candidate_positions: logit_compare_enabled()
                                    .then(|| {
                                        if matches!(
                                            mode,
                                            KvCompressMode::RecentOnly
                                                | KvCompressMode::BlockSummary
                                                | KvCompressMode::BlockSelectLossy
                                        ) {
                                            recent_candidate_positions(
                                                position_meta.recent_ring_absolute_start,
                                                position_meta.recent_ring_absolute_end,
                                            )
                                        } else {
                                            None
                                        }
                                    })
                                    .flatten(),
                                compressed_recent_ring_slots: logit_compare_enabled()
                                    .then(|| {
                                        if matches!(
                                            mode,
                                            KvCompressMode::RecentOnly
                                                | KvCompressMode::BlockSummary
                                                | KvCompressMode::BlockSelectLossy
                                        ) {
                                            recent_ring_slots(
                                                position_meta.recent_ring_absolute_start,
                                                position_meta.recent_ring_absolute_end,
                                                1024,
                                            )
                                        } else {
                                            None
                                        }
                                    })
                                    .flatten(),
                                attention_candidate_position_min,
                                attention_candidate_position_max,
                                attention_candidate_order_is_chronological,
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
                                expected_first_answer_token_rank_dense_window:
                                    expected_token_window_logits.rank_compressed,
                                expected_first_answer_token_rank_compressed: expected_token_logits
                                    .rank_compressed,
                                expected_first_answer_token_logit_dense: expected_token_logits
                                    .logit_dense,
                                expected_first_answer_token_logit_dense_window:
                                    expected_token_window_logits.logit_compressed,
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
                                prompt_dense_window_max_abs_diff: prompt_dense_window_diff
                                    .as_ref()
                                    .map(|summary| summary.max_abs_diff),
                                prompt_dense_window_mean_abs_diff: prompt_dense_window_diff
                                    .as_ref()
                                    .map(|summary| summary.mean_abs_diff),
                                prompt_dense_window_top10_overlap: prompt_dense_window_diff
                                    .as_ref()
                                    .map(|summary| summary.top10_overlap),
                                prompt_dense_window_top_token_id: dense_window_prompt_logits
                                    .as_deref()
                                    .and_then(top_token_id),
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
                                first_decode_dense_window_max_abs_diff:
                                    first_decode_dense_window_diff
                                        .as_ref()
                                        .map(|summary| summary.max_abs_diff),
                                first_decode_dense_window_mean_abs_diff:
                                    first_decode_dense_window_diff
                                        .as_ref()
                                        .map(|summary| summary.mean_abs_diff),
                                first_decode_dense_window_top10_overlap:
                                    first_decode_dense_window_diff
                                        .as_ref()
                                        .map(|summary| summary.top10_overlap),
                                first_decode_dense_window_top_token_id:
                                    dense_window_first_decode_logits
                                        .as_deref()
                                        .and_then(top_token_id),
                                attention_recent_mass: attention.map(|a| a.recent.prob_mass),
                                attention_selected_old_exact_mass: attention
                                    .map(|a| a.selected_old_exact.prob_mass),
                                attention_summary_mass: attention.map(|a| a.summary.prob_mass),
                                attention_representative_mass: attention
                                    .map(|a| a.representatives.prob_mass),
                                attention_other_mass: attention.map(|a| a.other.prob_mass),
                                attention_needle_block_mass: attention
                                    .and_then(|a| a.needle_block_mass),
                                attention_selected_block_masses,
                                attention_recent_logit_max: attention.map(|a| a.recent.logit_max),
                                attention_recent_logit_mean: attention.map(|a| a.recent.logit_mean),
                                attention_summary_logit_max: attention.map(|a| a.summary.logit_max),
                                attention_summary_logit_mean: attention
                                    .map(|a| a.summary.logit_mean),
                                attention_representative_logit_max: attention
                                    .map(|a| a.representatives.logit_max),
                                attention_representative_logit_mean: attention
                                    .map(|a| a.representatives.logit_mean),
                                attention_top_entries: attention.map(attention_top_entry_strings),
                                attention_records: selection_summary
                                    .as_ref()
                                    .map(attention_record_strings),
                                generated_logit_trace,
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

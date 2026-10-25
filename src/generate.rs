use anyhow::{Context, Result};

#[cfg(feature = "cuda")]
use crate::cuda::{CudaContext, CudaStream};
use crate::decode::StoppingCriteria;
#[cfg(not(feature = "cuda"))]
use crate::gguf::GgmlDType;
use crate::infer::LoadedModel;
#[cfg(feature = "cuda")]
use crate::kv_compression::KvCompressMode;
use crate::kv_compression::{
    runtime_config, KvCompressionConfig, KvExactOldAttention, ScopedRuntimeConfig,
};
use crate::kv_selection::{self, KvSelectionSummary};
use crate::sampling::{Sampler, SamplerConfig};
use crate::timing;
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PromptFormat {
    #[default]
    Auto,
    Raw,
    Llama3Chat,
    QwenChat,
}

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub log_prefix: &'static str,
    pub sequence_id: u32,
    pub reset_kv_cache: bool,
    pub kv_compression: KvCompressionConfig,
    pub prompt_format: PromptFormat,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: None,
            temperature: None,
            top_k: None,
            top_p: None,
            seed: None,
            log_prefix: "generate",
            sequence_id: 0,
            reset_kv_cache: true,
            kv_compression: KvCompressionConfig::default(),
            prompt_format: PromptFormat::Auto,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeneratedText {
    pub output: String,
    pub token_ids: Vec<u32>,
    pub prompt_token_len: usize,
    pub prompt_prefill_elapsed_ms: u128,
    pub generated_decode_elapsed_ms: u128,
    pub total_elapsed_ms: u128,
    pub attention_compression_elapsed_ms: Option<u128>,
    pub prefill_mode: String,
    pub prefill_chunk_size: Option<usize>,
    pub compressed_prefill_chunk_size: Option<usize>,
    pub temporary_dense_kv_bytes: Option<usize>,
    pub packed_prefill_sync_wall_ms: Option<u128>,
    pub packed_prefill_sync_decode_gpu_ms: Option<f32>,
    pub packed_prefill_sync_prefill_gpu_ms: Option<f32>,
    pub prompt_forward_sync_wall_ms: Option<u128>,
    pub prompt_forward_sync_decode_gpu_ms: Option<f32>,
    pub prompt_forward_sync_prefill_gpu_ms: Option<f32>,
    pub final_kv_allocated_bytes: Option<usize>,
    pub dense_equivalent_kv_bytes: Option<usize>,
    pub materialized_f32_cache_entries: Option<usize>,
    pub materialized_f32_cache_bytes: Option<usize>,
    pub materialized_f32_cache_entries_before: Option<usize>,
    pub materialized_f32_cache_bytes_before: Option<usize>,
    pub materialized_f32_cache_entries_after_prompt: Option<usize>,
    pub materialized_f32_cache_bytes_after_prompt: Option<usize>,
    pub materialized_f32_cache_entries_added_prompt: Option<usize>,
    pub materialized_f32_cache_bytes_added_prompt: Option<usize>,
    pub materialized_f32_cache_entries_added_total: Option<usize>,
    pub materialized_f32_cache_bytes_added_total: Option<usize>,
    pub materialized_f32_warm_row: Option<bool>,
    pub exact_old_backing: Option<String>,
    pub exact_old_attention_backend: Option<String>,
    pub q8_old_backing_bytes: Option<usize>,
    pub q8_old_backing_scale_bytes: Option<usize>,
    pub old_k_fp16_bytes: Option<usize>,
    pub q4_old_v_payload_bytes: Option<usize>,
    pub q4_old_v_scale_bytes: Option<usize>,
    pub recent_fp16_bytes: Option<usize>,
    pub summary_index_bytes: Option<usize>,
    pub staged_workspace_reused: bool,
    pub staged_workspace_bytes: Option<usize>,
    pub staged_workspace_capacity_tokens: Option<u32>,
    pub staged_workspace_allocations: usize,
    pub kv_selection: Option<KvSelectionSummary>,
    pub prompt_logits: Option<Vec<f32>>,
    pub first_decode_logits: Option<Vec<f32>>,
    pub generated_logit_trace: Option<Vec<GeneratedLogitTraceStep>>,
}

#[derive(Debug, Clone)]
pub struct GeneratedLogitTraceStep {
    pub step: usize,
    pub sampled_token_id: u32,
    pub logits: Vec<f32>,
}

#[cfg(feature = "cuda")]
fn decode_session_log_enabled() -> bool {
    std::env::var("M40LLM_DECODE_SESSION_LOG").ok().as_deref() == Some("1")
}

fn long_decode_log_interval() -> Option<usize> {
    let value = std::env::var("M40LLM_LONG_DECODE_LOG").ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("0")
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("no")
    {
        return None;
    }
    if trimmed.eq_ignore_ascii_case("1")
        || trimmed.eq_ignore_ascii_case("true")
        || trimmed.eq_ignore_ascii_case("yes")
    {
        return Some(64);
    }
    trimmed
        .parse::<usize>()
        .ok()
        .filter(|interval| *interval > 0)
}

#[cfg(feature = "cuda")]
fn prefill_chunk_size_setting_from_env() -> Result<Option<Option<usize>>> {
    let Some(value) = std::env::var("M40LLM_PREFILL_CHUNK_SIZE")
        .ok()
        .filter(|value| !value.trim().is_empty())
    else {
        return Ok(None);
    };
    let parsed = value
        .parse::<usize>()
        .with_context(|| format!("invalid M40LLM_PREFILL_CHUNK_SIZE value '{value}'"))?;
    Ok(Some((parsed > 0).then_some(parsed)))
}

#[cfg(feature = "cuda")]
fn compressed_prefill_chunk_size_from_env() -> Result<Option<usize>> {
    let Some(value) = std::env::var("M40LLM_KV_COMPRESSED_PREFILL_CHUNK_SIZE")
        .ok()
        .filter(|value| !value.trim().is_empty())
    else {
        return Ok(None);
    };
    let parsed = value.parse::<usize>().with_context(|| {
        format!("invalid M40LLM_KV_COMPRESSED_PREFILL_CHUNK_SIZE value '{value}'")
    })?;
    Ok((parsed > 0).then_some(parsed))
}

fn exact_old_attention_backend_name() -> Option<String> {
    match runtime_config().effective_exact_old_attention() {
        KvExactOldAttention::Staged => None,
        other => Some(other.as_str().to_string()),
    }
}

#[cfg(feature = "cuda")]
fn packed_then_compress_prefill_enabled() -> bool {
    std::env::var("M40LLM_KV_PACKED_THEN_COMPRESS_PREFILL")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn kv_logit_compare_enabled() -> bool {
    std::env::var("M40LLM_KV_LOGIT_COMPARE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn kv_logit_trace_enabled() -> bool {
    std::env::var("M40LLM_KV_LOGIT_TRACE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

pub fn sanitize_output(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut cleaned = Vec::with_capacity(bytes.len());
    let mut i = 0usize;

    while i < bytes.len() {
        if bytes[i] == b'\0' {
            i += 1;
            continue;
        }

        if bytes[i] == b'<'
            && i + 5 < bytes.len()
            && bytes[i + 1] == b'0'
            && (bytes[i + 2] == b'x' || bytes[i + 2] == b'X')
            && bytes[i + 3].is_ascii_hexdigit()
            && bytes[i + 4].is_ascii_hexdigit()
            && bytes[i + 5] == b'>'
        {
            i += 6;
            continue;
        }

        let b = bytes[i];
        if b.is_ascii_control() && b != b'\n' && b != b'\r' && b != b'\t' {
            i += 1;
            continue;
        }

        cleaned.push(b);
        i += 1;
    }

    String::from_utf8_lossy(&cleaned).into_owned()
}

pub fn decode_generated_text(
    tokenizer: &Tokenizer,
    ids: &[u32],
    prompt_token_len: usize,
) -> Result<String> {
    let generated = ids
        .get(prompt_token_len..)
        .ok_or_else(|| anyhow::anyhow!("prompt token length exceeds decoded ids"))?;
    tokenizer.decode_ignoring_specials(generated)
}

pub fn sampler_from_options(options: &GenerateOptions) -> Result<Sampler> {
    let mut cfg = SamplerConfig::default();
    if let Some(t) = options.temperature {
        if t <= 0.0 {
            anyhow::bail!("temperature must be > 0");
        }
        cfg.temperature = t;
    }
    if let Some(k) = options.top_k {
        if k == 0 {
            anyhow::bail!("top_k must be > 0 when provided");
        }
        cfg.top_k = Some(k);
    }
    if let Some(p) = options.top_p {
        if !(0.0 < p && p <= 1.0) {
            anyhow::bail!("top_p must be in (0, 1]");
        }
        cfg.top_p = Some(p);
    }
    if let Some(seed) = options.seed {
        cfg.seed = seed;
    }
    Ok(Sampler::new(cfg))
}

pub fn prepare_prompt(
    tokenizer: &Tokenizer,
    prompt: &str,
    prompt_format: PromptFormat,
) -> (String, bool) {
    fn llama3_prompt_is_preformatted(prompt: &str) -> bool {
        prompt.trim_start().starts_with("<|begin_of_text|>")
            || prompt.contains("<|start_header_id|>")
    }

    let selected = match prompt_format {
        PromptFormat::Auto => {
            if tokenizer.is_llama3()
                && tokenizer.has_chat_template()
                && !llama3_prompt_is_preformatted(prompt)
            {
                PromptFormat::Llama3Chat
            } else if tokenizer.is_qwen2()
                && tokenizer.has_chat_template()
                && !prompt.contains("<|im_start|>")
            {
                PromptFormat::QwenChat
            } else {
                PromptFormat::Raw
            }
        }
        other => other,
    };

    match selected {
        PromptFormat::Llama3Chat => (
            format!(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                prompt.trim()
            ),
            false,
        ),
        PromptFormat::QwenChat => (
            format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                prompt.trim()
            ),
            false,
        ),
        PromptFormat::Auto | PromptFormat::Raw => {
            let add_bos = !(tokenizer.is_llama3() && llama3_prompt_is_preformatted(prompt));
            (prompt.to_string(), add_bos)
        }
    }
}

#[cfg(feature = "cuda")]
fn log_top_logits(logits: &[f32], k: usize, log_prefix: &str) {
    if std::env::var("M40LLM_LOGITS_LOG").ok().as_deref() != Some("1") {
        return;
    }
    let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    top.sort_by(|a, b| f32::total_cmp(&b.1, &a.1));
    top.truncate(k);
    eprintln!("[{log_prefix}] top_logits={top:?}");
}

fn log_tokenizer_diag(
    tokenizer: &Tokenizer,
    prompt: &str,
    add_bos: bool,
    prompt_ids: &[u32],
    log_prefix: &str,
) {
    if std::env::var("M40LLM_TOKENIZER_DIAG").ok().as_deref() != Some("1") {
        return;
    }
    let head: Vec<u32> = prompt_ids.iter().copied().take(32).collect();
    let mut tail: Vec<u32> = prompt_ids.iter().rev().copied().take(16).collect();
    tail.reverse();
    eprintln!(
        "[{log_prefix}] tokenizer kind={:?} bos={:?} eos={:?} pad={:?} unk={:?} stop_ids={:?} has_chat_template={} add_bos={} prompt_chars={} prompt_tokens={} prompt_head={:?} prompt_tail={:?}",
        tokenizer.kind(),
        tokenizer.bos_id(),
        tokenizer.eos_id(),
        tokenizer.pad_id(),
        tokenizer.unk_id(),
        tokenizer.stop_ids(),
        tokenizer.has_chat_template(),
        add_bos,
        prompt.chars().count(),
        prompt_ids.len(),
        head,
        tail,
    );
}

pub fn generate_text(model: &LoadedModel, options: GenerateOptions) -> Result<GeneratedText> {
    let total_start = std::time::Instant::now();
    #[cfg(feature = "cuda")]
    let (materialized_f32_cache_entries_before, materialized_f32_cache_bytes_before) =
        model.materialized_f32_cache_stats();
    #[cfg(not(feature = "cuda"))]
    let (materialized_f32_cache_entries_before, materialized_f32_cache_bytes_before) =
        (0usize, 0usize);
    options.kv_compression.validate()?;
    let _kv_runtime_guard = ScopedRuntimeConfig::new(options.kv_compression.clone());
    kv_selection::reset();
    #[cfg(not(feature = "cuda"))]
    if options.kv_compression.mode.is_enabled() {
        anyhow::bail!("compressed KV cache modes require the cuda feature");
    }
    #[cfg(feature = "cuda")]
    if decode_session_log_enabled() {
        eprintln!(
            "[mem] (start) pid={} device_id={} TOTAL_DEVICE_BYTES={}",
            std::process::id(),
            model.cuda.device_id(),
            CudaContext::total_device_bytes()
        );
    }

    let tokenizer = Tokenizer::from_gguf_metadata(&model.gguf.metadata)
        .unwrap_or_else(|_| Tokenizer::byte_level());
    let (prompt, add_bos) = prepare_prompt(&tokenizer, &options.prompt, options.prompt_format);
    let encode_start = std::time::Instant::now();
    let prompt_ids = tokenizer
        .encode_with_specials(&prompt, add_bos, false)
        .context("encode prompt")?;
    log_tokenizer_diag(
        &tokenizer,
        &prompt,
        add_bos,
        &prompt_ids,
        options.log_prefix,
    );
    let prompt_token_len = prompt_ids.len();
    timing::timing_log!(
        encode_start.elapsed(),
        "{}.prompt_encode",
        options.log_prefix
    );
    let context_len = model.model_config.context_length as usize;
    if prompt_token_len >= context_len {
        anyhow::bail!(
            "prompt has {prompt_token_len} tokens, but model context_length is {context_len}; shorten the prompt"
        );
    }
    #[cfg(feature = "cuda")]
    if options.kv_compression.mode.is_enabled() {
        options
            .kv_compression
            .validate_runtime_support(
                model.model_config.context_length,
                model.model_config.attention_key_length,
            )
            .map_err(|err| {
                anyhow::anyhow!(
                    "{err}; use --kv-compress-mode off for dense reference/compatibility mode"
                )
            })?;
    }
    let max_tokens = match options.max_tokens {
        Some(max_tokens) => {
            let remaining = context_len - prompt_token_len;
            if max_tokens > remaining {
                anyhow::bail!(
                    "requested max_tokens={} but only {} token slots remain in context_length={} after {} prompt tokens",
                    max_tokens,
                    remaining,
                    context_len,
                    prompt_token_len
                );
            }
            max_tokens
        }
        None => context_len - prompt_token_len,
    };
    let stopping = StoppingCriteria::with_stop_ids(Some(max_tokens), tokenizer.stop_ids());
    let mut sampler = sampler_from_options(&options)?;

    if options.reset_kv_cache && model.kv_cache.is_some() {
        let reset_start = std::time::Instant::now();
        model.reset_kv_cache()?;
        timing::timing_log!(reset_start.elapsed(), "{}.kv_reset", options.log_prefix);
    }

    #[cfg(feature = "cuda")]
    let (full_decode_d_model, full_decode_hidden_dim) = {
        let validate_start = std::time::Instant::now();
        let result = model
            .validate_full_layer_decode()
            .context("CUDA full-layer decode is unavailable")?;
        timing::timing_log!(
            validate_start.elapsed(),
            "{}.validate_full_layer_decode",
            options.log_prefix
        );
        result
    };

    let log_prefix = options.log_prefix;
    #[cfg(feature = "cuda")]
    let prefill_chunk_size_setting = prefill_chunk_size_setting_from_env()?;
    #[cfg(feature = "cuda")]
    let prefill_chunk_size = prefill_chunk_size_setting.flatten();
    #[cfg(feature = "cuda")]
    let preferred_compressed_prefill_chunk_size =
        if options.kv_compression.is_preferred_batched_runtime() {
            prefill_chunk_size_setting.unwrap_or(Some(128))
        } else {
            prefill_chunk_size
        };
    #[cfg(not(feature = "cuda"))]
    let prefill_chunk_size = None;
    #[cfg(feature = "cuda")]
    let compressed_prefill_chunk_size = compressed_prefill_chunk_size_from_env()?;
    #[cfg(not(feature = "cuda"))]
    let compressed_prefill_chunk_size = None;
    #[cfg(feature = "cuda")]
    let mut prefill_mode = "sequential".to_string();
    #[cfg(not(feature = "cuda"))]
    let prefill_mode = "sequential".to_string();
    #[cfg(feature = "cuda")]
    let mut temporary_dense_kv_bytes = None;
    #[cfg(not(feature = "cuda"))]
    let temporary_dense_kv_bytes = None;
    #[cfg(feature = "cuda")]
    let mut decode_session = {
        eprintln!(
            "[{log_prefix}] mapped {} standard layers d_model={} hidden_dim={}",
            model.model_config.block_count, full_decode_d_model, full_decode_hidden_dim
        );
        crate::decode_session::DecodeSession::new_for_sequence_with_kv_config(
            model,
            options.sequence_id,
            options.kv_compression.clone(),
            full_decode_d_model,
            true,
            log_prefix,
            "generate:d_x_embed_f32",
            "generate:d_out_hidden_f32",
        )?
    };
    let mut logits_call_count = 0usize;
    let capture_logits = kv_logit_compare_enabled();
    let capture_logit_trace = kv_logit_trace_enabled();
    let mut prompt_logits_snapshot = None;
    let mut first_decode_logits_snapshot = None;
    let mut generated_logit_trace = Vec::new();
    #[cfg(feature = "cuda")]
    let cuda_greedy_logits_enabled = options.top_k == Some(1)
        && options.temperature.is_none()
        && options.top_p.is_none()
        && !capture_logits
        && !capture_logit_trace
        && std::env::var("M40LLM_LOGITS_LOG").ok().as_deref() != Some("1")
        && std::env::var("M40LLM_CUDA_GREEDY_ARGMAX")
            .map(|value| value != "0")
            .unwrap_or(false);
    let mut prompt_prefill_elapsed = std::time::Duration::ZERO;
    let mut generated_decode_elapsed = std::time::Duration::ZERO;
    let mut materialized_f32_cache_after_prompt = None;
    #[cfg(feature = "cuda")]
    let mut prompt_forward_sync_diag = None;
    let mut logits_fn = {
        |ids: &[u32]| -> anyhow::Result<Vec<f32>> {
            let timed_logits_fn_start = std::time::Instant::now();
            let result = {
                let logits_fn_start = std::time::Instant::now();

                #[cfg(not(feature = "cuda"))]
                let embed_names = [
                    "tok_embeddings.weight",
                    "token_embd.weight",
                    "token_embd",
                    "token_embeddings.weight",
                ];
                #[cfg(not(feature = "cuda"))]
                let tok_opt = embed_names
                    .iter()
                    .find_map(|name| model.device_tensors.get(*name));
                #[cfg(not(feature = "cuda"))]
                if tok_opt.is_none() {
                    eprintln!(
                        "[{log_prefix}] warning: no known embedding tensor found; will try lm_head only"
                    );
                }
                #[cfg(not(feature = "cuda"))]
                let d_model_from_tok = tok_opt
                    .and_then(|tensor| tensor.shape.get(1).copied())
                    .unwrap_or(0) as usize;
                #[cfg(not(feature = "cuda"))]
                if let Some(tensor) = tok_opt {
                    eprintln!("[{log_prefix}] embeddings shape: {:?}", tensor.shape);
                }

                #[cfg(not(feature = "cuda"))]
                let (d, can_forward) = match model.validate_standard_layers() {
                    Ok((d_model, hidden_dim)) => {
                        let ok = model.kv_cache_can_address_layers();
                        if !ok {
                            eprintln!(
                                "[{log_prefix}] KV cache cannot address all layers; using embeddings-only logits fallback"
                            );
                        }
                        eprintln!(
                            "[{log_prefix}] mapped {} standard layers d_model={} hidden_dim={}",
                            model.model_config.block_count, d_model, hidden_dim
                        );
                        (d_model, ok)
                    }
                    Err(e) => {
                        eprintln!("[{log_prefix}] validate_standard_layers failed; falling back to embeddings/logits path: {e}");
                        let d_try = if d_model_from_tok > 0 {
                            d_model_from_tok
                        } else {
                            match model.map_lm_head() {
                                Ok((_name, _lm, d_m, _vocab, _tied)) => {
                                    eprintln!(
                                        "[{log_prefix}] derived d_model from lm_head: {}",
                                        d_m
                                    );
                                    d_m
                                }
                                Err(e2) => {
                                    eprintln!(
                                        "[{log_prefix}] could not derive d_model from lm_head: {e2}"
                                    );
                                    0
                                }
                            }
                        };
                        if d_try == 0 {
                            return Err(anyhow::anyhow!("could not determine d_model"));
                        }
                        (d_try, false)
                    }
                };
                #[cfg(not(feature = "cuda"))]
                {
                    let bytes = d * std::mem::size_of::<f32>();
                    let _ = (can_forward, bytes);
                }

                #[cfg(feature = "cuda")]
                {
                    let _ = logits_fn_start;
                    if cuda_greedy_logits_enabled {
                        let token = decode_session.greedy_token_for_ids(ids)?;
                        let mut logits = vec![f32::NEG_INFINITY; token as usize + 1];
                        logits[token as usize] = 0.0;
                        Ok(logits)
                    } else if logits_call_count == 0 {
                        if !matches!(options.kv_compression.mode, KvCompressMode::Off) {
                            if matches!(
                                options.kv_compression.mode,
                                KvCompressMode::BlockSelectExact
                            ) && preferred_compressed_prefill_chunk_size
                                .is_some_and(|chunk_size| ids.len() <= chunk_size)
                            {
                                match decode_session
                                    .logits_for_packed_prefix_then_ids(ids, |logits| {
                                        log_top_logits(logits, 8, log_prefix)
                                    }) {
                                    Ok(logits) => {
                                        prefill_mode = format!(
                                            "packed-prefix-{}",
                                            match options.kv_compression.mode {
                                                KvCompressMode::BlockSelectExact =>
                                                    "block-select-exact",
                                                _ => unreachable!(),
                                            }
                                        );
                                        Ok(logits)
                                    }
                                    Err(err) => {
                                        eprintln!(
                                            "[{log_prefix}] dense/exact diagnostic packed prefill fallback: {err:#}"
                                        );
                                        prefill_mode =
                                            "packed-prefix-diagnostic-fallback-sequential"
                                                .to_string();
                                        decode_session.logits_for_ids(ids, |logits| {
                                            log_top_logits(logits, 8, log_prefix)
                                        })
                                    }
                                }
                            } else if matches!(
                                options.kv_compression.mode,
                                KvCompressMode::DenseRecentOnly
                            ) {
                                prefill_mode = "sequential-dense-recent-only".to_string();
                                decode_session.logits_for_ids(ids, |logits| {
                                    log_top_logits(logits, 8, log_prefix)
                                })
                            } else if packed_then_compress_prefill_enabled()
                                && matches!(
                                    options.kv_compression.mode,
                                    KvCompressMode::RecentOnly
                                        | KvCompressMode::BlockSummary
                                        | KvCompressMode::BlockSelectLossy
                                )
                            {
                                match decode_session
                                    .logits_for_packed_then_compress_prefill_ids(ids, |logits| {
                                        log_top_logits(logits, 8, log_prefix)
                                    }) {
                                    Ok((logits, temp_bytes)) => {
                                        prefill_mode = "packed-then-compress".to_string();
                                        temporary_dense_kv_bytes = Some(temp_bytes);
                                        Ok(logits)
                                    }
                                    Err(err) => {
                                        eprintln!(
                                            "[{log_prefix}] packed-then-compress prefill fallback: {err:#}"
                                        );
                                        if let Some(chunk_size) = compressed_prefill_chunk_size {
                                            match decode_session
                                                .logits_for_compressed_chunked_prefill_ids(
                                                    ids,
                                                    chunk_size,
                                                    |logits| log_top_logits(logits, 8, log_prefix),
                                                ) {
                                                Ok(logits) => {
                                                    prefill_mode =
                                                        "packed-then-compress-fallback-chunked"
                                                            .to_string();
                                                    Ok(logits)
                                                }
                                                Err(chunk_err) => {
                                                    eprintln!(
                                                        "[{log_prefix}] compressed chunked prefill fallback: {chunk_err:#}"
                                                    );
                                                    prefill_mode =
                                                        "packed-then-compress-fallback-sequential"
                                                            .to_string();
                                                    decode_session.logits_for_ids(ids, |logits| {
                                                        log_top_logits(logits, 8, log_prefix)
                                                    })
                                                }
                                            }
                                        } else {
                                            prefill_mode =
                                                "packed-then-compress-fallback-sequential"
                                                    .to_string();
                                            decode_session.logits_for_ids(ids, |logits| {
                                                log_top_logits(logits, 8, log_prefix)
                                            })
                                        }
                                    }
                                }
                            } else if let Some(chunk_size) = compressed_prefill_chunk_size {
                                match decode_session.logits_for_compressed_chunked_prefill_ids(
                                    ids,
                                    chunk_size,
                                    |logits| log_top_logits(logits, 8, log_prefix),
                                ) {
                                    Ok(logits) => {
                                        prefill_mode = "chunked-kv-compressed".to_string();
                                        Ok(logits)
                                    }
                                    Err(err) => {
                                        eprintln!(
                                            "[{log_prefix}] compressed chunked prefill fallback: {err:#}"
                                        );
                                        prefill_mode =
                                            "sequential-kv-compressed-fallback".to_string();
                                        decode_session.logits_for_ids(ids, |logits| {
                                            log_top_logits(logits, 8, log_prefix)
                                        })
                                    }
                                }
                            } else {
                                prefill_mode = "sequential-kv-compressed".to_string();
                                decode_session.logits_for_ids(ids, |logits| {
                                    log_top_logits(logits, 8, log_prefix)
                                })
                            }
                        } else if let Some(chunk_size) = prefill_chunk_size {
                            if ids.len() <= chunk_size {
                                match decode_session
                                    .logits_for_packed_prefix_then_ids(ids, |logits| {
                                        log_top_logits(logits, 8, log_prefix)
                                    }) {
                                    Ok(logits) => {
                                        prefill_mode = "packed-prefix".to_string();
                                        Ok(logits)
                                    }
                                    Err(err) => {
                                        eprintln!(
                                            "[{log_prefix}] packed prefill fallback: {err:#}"
                                        );
                                        prefill_mode = "sequential-fallback".to_string();
                                        decode_session.logits_for_ids(ids, |logits| {
                                            log_top_logits(logits, 8, log_prefix)
                                        })
                                    }
                                }
                            } else {
                                prefill_mode = "sequential-over-bound".to_string();
                                decode_session.logits_for_ids(ids, |logits| {
                                    log_top_logits(logits, 8, log_prefix)
                                })
                            }
                        } else {
                            decode_session
                                .logits_for_ids(ids, |logits| log_top_logits(logits, 8, log_prefix))
                        }
                    } else {
                        decode_session
                            .logits_for_ids(ids, |logits| log_top_logits(logits, 8, log_prefix))
                    }
                }

                #[cfg(not(feature = "cuda"))]
                {
                    use half::f16;
                    use std::ffi::c_void;

                    let tok_id = *ids.last().ok_or_else(|| anyhow::anyhow!("empty ids"))? as usize;
                    let tok = embed_names
                        .iter()
                        .find_map(|name| model.device_tensors.get(*name))
                        .ok_or_else(|| {
                            anyhow::anyhow!("missing embeddings tensor (tried common names)")
                        })?;
                    if tok.shape.len() != 2 {
                        return Err(anyhow::anyhow!("embeddings must be [vocab, d_model]"));
                    }
                    let vocab = tok.shape[0] as usize;
                    if tok_id >= vocab {
                        return Err(anyhow::anyhow!(
                            "token id {} out of range (vocab={})",
                            tok_id,
                            vocab
                        ));
                    }
                    let d_model = tok.shape[1] as usize;
                    let mut hidden = vec![0f32; d_model];
                    match tok.dtype {
                        GgmlDType::F16 => {
                            let row_bytes = d_model * 2;
                            let off = tok.byte_offset as usize + tok_id * row_bytes;
                            let row = &model.host_weights[off..off + row_bytes];
                            for i in 0..d_model {
                                let lo = row[2 * i] as u16;
                                let hi = row[2 * i + 1] as u16;
                                hidden[i] = f16::from_bits(lo | (hi << 8)).to_f32();
                            }
                        }
                        GgmlDType::Q8_0 => {
                            const Q8_0_BLOCK: usize = 32;
                            const Q8_0_BLOCK_BYTES: usize = 34;
                            let blocks = d_model.div_ceil(Q8_0_BLOCK);
                            let row_bytes = blocks * Q8_0_BLOCK_BYTES;
                            let off = tok.byte_offset as usize + tok_id * row_bytes;
                            let row = &model.host_weights[off..off + row_bytes];
                            for i in 0..d_model {
                                let block = i / Q8_0_BLOCK;
                                let idx = i % Q8_0_BLOCK;
                                let base = block * Q8_0_BLOCK_BYTES;
                                let d_bits = u16::from_le_bytes([row[base], row[base + 1]]);
                                let d = f16::from_bits(d_bits).to_f32();
                                let q = row[base + 2 + idx] as i8 as f32;
                                hidden[i] = d * q;
                            }
                        }
                        _ => anyhow::bail!("unsupported embeddings dtype for host path"),
                    }
                    let result =
                        unsafe { model.logits_from_hidden(hidden.as_ptr() as *const c_void) };
                    timing::timing_log!(
                        logits_fn_start.elapsed(),
                        "{log_prefix}.logits_fn.ids_len_{}",
                        ids.len()
                    );
                    result
                }
            };
            let elapsed = timed_logits_fn_start.elapsed();
            if logits_call_count == 0 {
                prompt_prefill_elapsed += elapsed;
                #[cfg(feature = "cuda")]
                {
                    materialized_f32_cache_after_prompt =
                        Some(model.materialized_f32_cache_stats());
                    prompt_forward_sync_diag = decode_session.forward_sync_diag_timings();
                }
                #[cfg(not(feature = "cuda"))]
                {
                    materialized_f32_cache_after_prompt = Some((0usize, 0usize));
                }
                if capture_logits {
                    if let Ok(logits) = &result {
                        prompt_logits_snapshot = Some(logits.clone());
                    }
                }
            } else {
                generated_decode_elapsed += elapsed;
                if capture_logits && logits_call_count == 1 {
                    if let Ok(logits) = &result {
                        first_decode_logits_snapshot = Some(logits.clone());
                    }
                }
            }
            logits_call_count += 1;
            result
        }
    };

    let decode_start = std::time::Instant::now();
    let mut ids = prompt_ids;
    let start_len = ids.len();
    let mut generated = Vec::new();
    let long_decode_log_interval = long_decode_log_interval();
    loop {
        if stopping.should_stop(&generated) {
            break;
        }
        let logits = logits_fn(&ids).with_context(|| {
            format!(
                concat!(
                    "generation failed after {} generated tokens ",
                    "(seq_len={}, prompt_tokens={}, context_remaining={}, ",
                    "kv_mode={:?}, top_blocks={}, exact_old_backing={}, exact_old_attention={})"
                ),
                generated.len(),
                ids.len(),
                prompt_token_len,
                context_len.saturating_sub(ids.len()),
                options.kv_compression.mode,
                options.kv_compression.top_blocks,
                options
                    .kv_compression
                    .effective_exact_old_backing()
                    .as_str(),
                options
                    .kv_compression
                    .effective_exact_old_attention()
                    .as_str()
            )
        })?;
        if logits.is_empty() {
            anyhow::bail!("logits_fn returned empty logits");
        }
        let next = sampler.sample(&logits)? as u32;
        if capture_logit_trace {
            generated_logit_trace.push(GeneratedLogitTraceStep {
                step: generated.len(),
                sampled_token_id: next,
                logits: logits.clone(),
            });
        }
        ids.push(next);
        generated.push(next);
        if std::env::var("M40LLM_DECODE_LOG").ok().as_deref() == Some("1") {
            eprintln!("[decode] sampled token id={next}");
        }
        if long_decode_log_interval.is_some_and(|interval| {
            generated.len() == 1 || generated.len() % interval == 0 || generated.len() == max_tokens
        }) {
            eprintln!(
                concat!(
                    "[{}] long_decode generated={} seq_len={} prompt_tokens={} ",
                    "context_remaining={} sampled_token={} kv_mode={:?} top_blocks={} ",
                    "exact_old_backing={} exact_old_attention={}"
                ),
                options.log_prefix,
                generated.len(),
                ids.len(),
                prompt_token_len,
                context_len.saturating_sub(ids.len()),
                next,
                options.kv_compression.mode,
                options.kv_compression.top_blocks,
                options
                    .kv_compression
                    .effective_exact_old_backing()
                    .as_str(),
                options
                    .kv_compression
                    .effective_exact_old_attention()
                    .as_str()
            );
        }
        if stopping.stop_ids.contains(&next) {
            break;
        }
        if let Some(mt) = stopping.max_tokens {
            if generated.len() >= mt {
                break;
            }
        }
        if stopping.max_tokens.is_none()
            && stopping.eos_id.is_none()
            && generated.len() > 4 * (start_len.max(1))
            && generated.len() > 4096
        {
            anyhow::bail!("decode loop refused to continue without stopping criteria");
        }
    }
    timing::timing_log!(decode_start.elapsed(), "{}.decode_loop", options.log_prefix);
    #[cfg(feature = "cuda")]
    let packed_prefill_sync_diag = decode_session.prefill_sync_diag_timings();
    #[cfg(feature = "cuda")]
    let packed_prefill_sync_wall_ms = packed_prefill_sync_diag.map(|diag| diag.wall_ms);
    #[cfg(feature = "cuda")]
    let packed_prefill_sync_decode_gpu_ms = packed_prefill_sync_diag.map(|diag| diag.decode_gpu_ms);
    #[cfg(feature = "cuda")]
    let packed_prefill_sync_prefill_gpu_ms =
        packed_prefill_sync_diag.map(|diag| diag.prefill_gpu_ms);
    #[cfg(not(feature = "cuda"))]
    let packed_prefill_sync_wall_ms = None;
    #[cfg(not(feature = "cuda"))]
    let packed_prefill_sync_decode_gpu_ms = None;
    #[cfg(not(feature = "cuda"))]
    let packed_prefill_sync_prefill_gpu_ms = None;
    #[cfg(feature = "cuda")]
    let prompt_forward_sync_wall_ms = prompt_forward_sync_diag.map(|diag| diag.wall_ms);
    #[cfg(feature = "cuda")]
    let prompt_forward_sync_decode_gpu_ms = prompt_forward_sync_diag.map(|diag| diag.decode_gpu_ms);
    #[cfg(feature = "cuda")]
    let prompt_forward_sync_prefill_gpu_ms =
        prompt_forward_sync_diag.map(|diag| diag.prefill_gpu_ms);
    #[cfg(not(feature = "cuda"))]
    let prompt_forward_sync_wall_ms = None;
    #[cfg(not(feature = "cuda"))]
    let prompt_forward_sync_decode_gpu_ms = None;
    #[cfg(not(feature = "cuda"))]
    let prompt_forward_sync_prefill_gpu_ms = None;
    let output_decode_start = std::time::Instant::now();
    let text =
        decode_generated_text(&tokenizer, &ids, prompt_token_len).context("decode failed")?;
    timing::timing_log!(
        output_decode_start.elapsed(),
        "{}.output_decode",
        options.log_prefix
    );
    #[cfg(feature = "cuda")]
    {
        model.cuda.synchronize_stream(CudaStream::Decode)?;
        model.cuda.synchronize_stream(CudaStream::Prefill)?;
    }

    #[cfg(feature = "cuda")]
    if decode_session_log_enabled() {
        eprintln!(
            "[mem] (finish) pid={} device_id={} TOTAL_DEVICE_BYTES={}",
            std::process::id(),
            model.cuda.device_id(),
            CudaContext::total_device_bytes()
        );
    }

    timing::timing_log!(
        total_start.elapsed(),
        "{}.generate_text_total",
        options.log_prefix
    );
    #[cfg(feature = "cuda")]
    let (materialized_f32_cache_entries, materialized_f32_cache_bytes) =
        model.materialized_f32_cache_stats();
    #[cfg(not(feature = "cuda"))]
    let (materialized_f32_cache_entries, materialized_f32_cache_bytes) = (0usize, 0usize);
    let (materialized_f32_cache_entries_after_prompt, materialized_f32_cache_bytes_after_prompt) =
        materialized_f32_cache_after_prompt
            .unwrap_or((materialized_f32_cache_entries, materialized_f32_cache_bytes));
    let materialized_f32_cache_entries_added_prompt = materialized_f32_cache_entries_after_prompt
        .saturating_sub(materialized_f32_cache_entries_before);
    let materialized_f32_cache_bytes_added_prompt = materialized_f32_cache_bytes_after_prompt
        .saturating_sub(materialized_f32_cache_bytes_before);
    let materialized_f32_cache_entries_added_total =
        materialized_f32_cache_entries.saturating_sub(materialized_f32_cache_entries_before);
    let materialized_f32_cache_bytes_added_total =
        materialized_f32_cache_bytes.saturating_sub(materialized_f32_cache_bytes_before);
    #[cfg(feature = "cuda")]
    let reported_prefill_chunk_size = if options.kv_compression.is_preferred_batched_runtime() {
        preferred_compressed_prefill_chunk_size
    } else {
        prefill_chunk_size
    };
    #[cfg(not(feature = "cuda"))]
    let reported_prefill_chunk_size = prefill_chunk_size;

    Ok(GeneratedText {
        output: sanitize_output(&text),
        token_ids: ids,
        prompt_token_len,
        prompt_prefill_elapsed_ms: prompt_prefill_elapsed.as_millis(),
        generated_decode_elapsed_ms: generated_decode_elapsed.as_millis(),
        total_elapsed_ms: total_start.elapsed().as_millis(),
        attention_compression_elapsed_ms: None,
        prefill_mode,
        prefill_chunk_size: reported_prefill_chunk_size,
        compressed_prefill_chunk_size,
        temporary_dense_kv_bytes,
        packed_prefill_sync_wall_ms,
        packed_prefill_sync_decode_gpu_ms,
        packed_prefill_sync_prefill_gpu_ms,
        prompt_forward_sync_wall_ms,
        prompt_forward_sync_decode_gpu_ms,
        prompt_forward_sync_prefill_gpu_ms,
        final_kv_allocated_bytes: model.kv_cache.as_ref().map(|kv| kv.actual_bytes()),
        dense_equivalent_kv_bytes: model
            .kv_cache
            .as_ref()
            .map(|kv| kv.dense_equivalent_bytes()),
        materialized_f32_cache_entries: Some(materialized_f32_cache_entries),
        materialized_f32_cache_bytes: Some(materialized_f32_cache_bytes),
        materialized_f32_cache_entries_before: Some(materialized_f32_cache_entries_before),
        materialized_f32_cache_bytes_before: Some(materialized_f32_cache_bytes_before),
        materialized_f32_cache_entries_after_prompt: Some(
            materialized_f32_cache_entries_after_prompt,
        ),
        materialized_f32_cache_bytes_after_prompt: Some(materialized_f32_cache_bytes_after_prompt),
        materialized_f32_cache_entries_added_prompt: Some(
            materialized_f32_cache_entries_added_prompt,
        ),
        materialized_f32_cache_bytes_added_prompt: Some(materialized_f32_cache_bytes_added_prompt),
        materialized_f32_cache_entries_added_total: Some(
            materialized_f32_cache_entries_added_total,
        ),
        materialized_f32_cache_bytes_added_total: Some(materialized_f32_cache_bytes_added_total),
        materialized_f32_warm_row: Some(materialized_f32_cache_bytes_added_total == 0),
        exact_old_backing: model
            .kv_cache
            .as_ref()
            .map(|kv| kv.exact_old_backing().to_string()),
        exact_old_attention_backend: model.kv_cache.as_ref().and_then(|kv| {
            if kv.exact_old_backing() == "q8" {
                Some(exact_old_attention_backend_name().unwrap_or_else(|| "staged-q8".to_string()))
            } else if kv.exact_old_backing() == "fp16-k-q4-v" {
                Some(
                    exact_old_attention_backend_name()
                        .unwrap_or_else(|| "staged-fp16-k-q4-v".to_string()),
                )
            } else {
                None
            }
        }),
        q8_old_backing_bytes: model.kv_cache.as_ref().map(|kv| kv.q8_old_backing_bytes()),
        q8_old_backing_scale_bytes: model
            .kv_cache
            .as_ref()
            .map(|kv| kv.q8_old_backing_scale_bytes()),
        old_k_fp16_bytes: model.kv_cache.as_ref().map(|kv| kv.old_k_fp16_bytes()),
        q4_old_v_payload_bytes: model
            .kv_cache
            .as_ref()
            .map(|kv| kv.q4_old_v_payload_bytes()),
        q4_old_v_scale_bytes: model.kv_cache.as_ref().map(|kv| kv.q4_old_v_scale_bytes()),
        recent_fp16_bytes: model.kv_cache.as_ref().map(|kv| kv.recent_fp16_bytes()),
        summary_index_bytes: model.kv_cache.as_ref().map(|kv| kv.summary_index_bytes()),
        #[cfg(feature = "cuda")]
        staged_workspace_reused: decode_session.exact_block_staging_reused(),
        #[cfg(not(feature = "cuda"))]
        staged_workspace_reused: false,
        #[cfg(feature = "cuda")]
        staged_workspace_bytes: decode_session.exact_block_staging_workspace_bytes(),
        #[cfg(not(feature = "cuda"))]
        staged_workspace_bytes: None,
        #[cfg(feature = "cuda")]
        staged_workspace_capacity_tokens: decode_session.exact_block_staging_capacity_tokens(),
        #[cfg(not(feature = "cuda"))]
        staged_workspace_capacity_tokens: None,
        #[cfg(feature = "cuda")]
        staged_workspace_allocations: decode_session.exact_block_staging_allocations(),
        #[cfg(not(feature = "cuda"))]
        staged_workspace_allocations: 0,
        kv_selection: kv_selection::enabled()
            .then(kv_selection::snapshot)
            .flatten(),
        prompt_logits: prompt_logits_snapshot,
        first_decode_logits: first_decode_logits_snapshot,
        generated_logit_trace: capture_logit_trace.then_some(generated_logit_trace),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_output_drops_nuls_and_byte_markers() {
        let input = "Catalog\0 yesterday \u{0003}<0xAA> Chile <0x7f>";
        let output = sanitize_output(input);
        assert_eq!(output, "Catalog yesterday  Chile ");
    }

    #[test]
    fn sanitize_output_keeps_printable_text() {
        let input = "Hello, world!";
        assert_eq!(sanitize_output(input), "Hello, world!");
    }
}

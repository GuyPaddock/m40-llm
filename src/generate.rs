use anyhow::{Context, Result};

#[cfg(feature = "cuda")]
use crate::cuda::CudaContext;
use crate::decode::{decode_loop_with, StoppingCriteria};
#[cfg(not(feature = "cuda"))]
use crate::gguf::GgmlDType;
use crate::infer::LoadedModel;
use crate::sampling::{Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub log_prefix: &'static str,
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
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeneratedText {
    pub output: String,
    pub token_ids: Vec<u32>,
}

pub fn sanitize_output(text: &str) -> String {
    text.chars().filter(|c| *c != '\0').collect()
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

pub fn generate_text(model: &LoadedModel, options: GenerateOptions) -> Result<GeneratedText> {
    #[cfg(feature = "cuda")]
    eprintln!(
        "[mem] (start) pid={} device_id={} TOTAL_DEVICE_BYTES={}",
        std::process::id(),
        model.cuda.device_id(),
        CudaContext::total_device_bytes()
    );

    let tokenizer = Tokenizer::from_gguf_metadata(&model.gguf.metadata)
        .unwrap_or_else(|_| Tokenizer::byte_level());
    let add_bos = true;
    let prompt_token_len = tokenizer
        .encode_with_specials(&options.prompt, add_bos, false)
        .context("encode prompt")?
        .len();
    let max_tokens = options
        .max_tokens
        .or_else(|| Some(model.model_config.context_length as usize))
        .unwrap_or(32);
    let stopping = StoppingCriteria::new(Some(max_tokens), tokenizer.eos_id());
    let sampler = sampler_from_options(&options)?;

    if model.kv_cache.is_some() {
        model.reset_kv_cache()?;
    }

    #[cfg(feature = "cuda")]
    let (full_decode_d_model, full_decode_hidden_dim) = model
        .validate_full_layer_decode()
        .context("CUDA full-layer decode is unavailable")?;

    let log_prefix = options.log_prefix;
    let mut logits_fn = {
        #[cfg(feature = "cuda")]
        let mut step: usize = 0;
        #[cfg(feature = "cuda")]
        let mut processed_len: usize = 0;
        move |ids: &[u32]| -> anyhow::Result<Vec<f32>> {
            eprintln!("[{log_prefix}] logits_fn called with {} tokens", ids.len());
            #[cfg(feature = "cuda")]
            {
                step += 1;
                eprintln!(
                    "[mem] (token) step={} pid={} device_id={} TOTAL_DEVICE_BYTES={}",
                    step,
                    std::process::id(),
                    model.cuda.device_id(),
                    CudaContext::total_device_bytes()
                );
            }

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

            #[cfg(feature = "cuda")]
            let (d, can_forward) = {
                eprintln!(
                    "[{log_prefix}] mapped {} standard layers d_model={} hidden_dim={}",
                    model.model_config.block_count, full_decode_d_model, full_decode_hidden_dim
                );
                (full_decode_d_model, true)
            };
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
                                eprintln!("[{log_prefix}] derived d_model from lm_head: {}", d_m);
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
            let bytes = d * std::mem::size_of::<f32>();

            #[cfg(not(feature = "cuda"))]
            {
                let _ = (can_forward, bytes);
            }

            #[cfg(feature = "cuda")]
            {
                if ids.is_empty() {
                    return Err(anyhow::anyhow!("empty ids"));
                }
                if processed_len > ids.len() {
                    processed_len = 0;
                    if model.kv_cache.is_some() {
                        model.reset_kv_cache()?;
                    }
                }

                let start = if can_forward {
                    processed_len
                } else {
                    ids.len().saturating_sub(1)
                };
                let mut logits: Option<Vec<f32>> = None;
                for token_idx in start..ids.len() {
                    let tok_id = ids[token_idx] as u64;
                    eprintln!("[{log_prefix}] token id {}", tok_id);

                    let d_x = model
                        .cuda
                        .device_malloc_tagged(bytes, "generate:d_x_embed_f32")?;

                    let token_logits = (|| -> anyhow::Result<Vec<f32>> {
                        unsafe {
                            model.load_token_embedding_to_f32(tok_id, d_x)?;
                        }

                        if can_forward {
                            let d_out = model
                                .cuda
                                .device_malloc_tagged(bytes, "generate:d_out_hidden_f32")?;
                            let result = (|| -> anyhow::Result<Vec<f32>> {
                                unsafe {
                                    let layers = model.forward_one_token_all_layers(
                                        d_x as *const _,
                                        (token_idx + 1) as u32,
                                        d_out,
                                    )?;
                                    if token_idx == start {
                                        eprintln!(
                                            "[{log_prefix}] full-layer forward enabled layers={layers}"
                                        );
                                    }
                                    model.logits_from_hidden(d_out as *const _)
                                }
                            })();
                            unsafe {
                                let _ = model.cuda.device_free(d_out);
                            }
                            result
                        } else {
                            unsafe { model.logits_from_hidden(d_x as *const _) }
                        }
                    })();

                    unsafe {
                        let _ = model.cuda.device_free(d_x);
                    }

                    let token_logits = token_logits?;
                    log_top_logits(&token_logits, 8, log_prefix);
                    logits = Some(token_logits);
                }
                if can_forward {
                    processed_len = ids.len();
                }

                logits.ok_or_else(|| anyhow::anyhow!("no token processed for logits"))
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
                unsafe { model.logits_from_hidden(hidden.as_ptr() as *const c_void) }
            }
        }
    };

    let ids = decode_loop_with(
        &tokenizer,
        &options.prompt,
        add_bos,
        sampler,
        &stopping,
        &mut logits_fn,
    )
    .context("generation failed")?;
    let text =
        decode_generated_text(&tokenizer, &ids, prompt_token_len).context("decode failed")?;

    #[cfg(feature = "cuda")]
    eprintln!(
        "[mem] (finish) pid={} device_id={} TOTAL_DEVICE_BYTES={}",
        std::process::id(),
        model.cuda.device_id(),
        CudaContext::total_device_bytes()
    );

    Ok(GeneratedText {
        output: sanitize_output(&text),
        token_ids: ids,
    })
}

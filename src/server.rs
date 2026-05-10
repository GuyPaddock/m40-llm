// src/server.rs
#![allow(dead_code)]
use axum::{
    body::Body,
    response::IntoResponse,
    response::Response,
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;
use std::{io, sync::Arc};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

#[cfg(feature = "cuda")]
use crate::cuda::CudaContext;
use crate::decode::{decode_loop_with, StoppingCriteria};
use crate::generate::{
    decode_generated_text, generate_text, sampler_from_options, sanitize_output, GenerateOptions,
};
#[cfg(not(feature = "cuda"))]
use crate::gguf::GgmlDType;
use crate::infer::LoadedModel;
use crate::tokenizer::Tokenizer;
use anyhow::Result;
use axum::http::{header::CONTENT_TYPE, HeaderName, HeaderValue, StatusCode};
use tower_http::{cors::CorsLayer, set_header::SetResponseHeaderLayer};

#[derive(serde::Serialize, serde::Deserialize, Default)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct GenerateResponse {
    pub output: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct GenerateStreamChunk {
    pub output: String,
    pub done: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

fn chunk_to_bytes(chunk: &GenerateStreamChunk) -> io::Result<Bytes> {
    let safe_chunk = GenerateStreamChunk {
        output: sanitize_output(&chunk.output),
        done: chunk.done,
    };

    let mut buf: Vec<u8> = Vec::with_capacity(safe_chunk.output.len() + 16);
    serde_json::to_writer(&mut buf, &safe_chunk)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    buf.push(b'\n');
    Ok(Bytes::from(buf))
}

fn options_from_request(req: &GenerateRequest, log_prefix: &'static str) -> GenerateOptions {
    GenerateOptions {
        prompt: req.prompt.clone(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        seed: req.seed,
        log_prefix,
    }
}

fn internal_error(err: anyhow::Error) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: err.to_string(),
        }),
    )
}

#[cfg(feature = "cuda")]
fn log_top_logits(logits: &[f32], k: usize) {
    if std::env::var("M40LLM_LOGITS_LOG").ok().as_deref() != Some("1") {
        return;
    }
    let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    top.sort_by(|a, b| f32::total_cmp(&b.1, &a.1));
    top.truncate(k);
    eprintln!("[server] top_logits={top:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_bytes_strip_nuls_and_terminate() {
        let chunk = GenerateStreamChunk {
            output: "hi\0there".to_string(),
            done: false,
        };
        let bytes = chunk_to_bytes(&chunk).expect("serialize chunk");
        assert_eq!(bytes.last().copied(), Some(b'\n'));
        assert!(bytes[..bytes.len() - 1].iter().all(|b| *b != 0));

        let lines: Vec<&[u8]> = bytes.split(|b| *b == b'\n').collect();
        let parsed: GenerateStreamChunk = serde_json::from_slice(lines[0]).expect("decode chunk");
        assert_eq!(parsed.output, "hithere");
        assert!(!parsed.done);
    }

    #[test]
    fn sanitize_output_passes_clean_text() {
        let clean = sanitize_output("abc");
        assert_eq!(clean, "abc");
    }

    #[test]
    fn sampler_from_request_applies_overrides() {
        let req = GenerateRequest {
            prompt: "hi".to_string(),
            temperature: Some(0.5),
            top_k: Some(1),
            top_p: Some(0.9),
            seed: Some(7),
            ..Default::default()
        };
        let mut sampler =
            sampler_from_options(&options_from_request(&req, "test")).expect("build sampler");
        let logits = [0.2f32, 0.8, 0.1];
        // top_k=1 should force the max logit (index 1)
        for _ in 0..5 {
            assert_eq!(sampler.sample(&logits).unwrap(), 1);
        }
    }

    #[test]
    fn sampler_from_request_rejects_invalid_top_p() {
        let req = GenerateRequest {
            prompt: "".to_string(),
            top_p: Some(1.5),
            ..Default::default()
        };
        assert!(sampler_from_request(&req).is_err());
    }

    #[test]
    fn decode_generated_text_excludes_prompt_tokens() {
        let tokenizer = Tokenizer::byte_level();
        let prompt = "Hello";
        let prompt_ids = tokenizer
            .encode_with_specials(prompt, true, false)
            .expect("encode prompt");
        let mut ids = prompt_ids.clone();
        ids.extend("ijk".bytes().map(u32::from));

        let generated =
            decode_generated_text(&tokenizer, &ids, prompt_ids.len()).expect("decode generated");

        assert_eq!(generated, "ijk");
    }

    #[test]
    fn decode_generated_text_counts_tokens_not_characters() {
        use crate::gguf::{GgufScalar, GgufValue};
        use std::collections::HashMap;

        let mut meta = HashMap::new();
        meta.insert(
            "tokenizer.ggml.model".to_string(),
            GgufValue::Scalar(GgufScalar::Str("spm".to_string())),
        );
        meta.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::Array(vec![
                GgufScalar::Str("<s>".to_string()),
                GgufScalar::Str("▁Hello".to_string()),
                GgufScalar::Str("ijk".to_string()),
            ]),
        );
        meta.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            GgufValue::Scalar(GgufScalar::U32(0)),
        );
        let tokenizer = Tokenizer::from_gguf_metadata(&meta).expect("tokenizer");
        let prompt_ids = vec![0, 1];
        let ids = vec![0, 1, 2];

        let generated =
            decode_generated_text(&tokenizer, &ids, prompt_ids.len()).expect("decode generated");

        assert_eq!(generated, "ijk");
    }
}

pub struct AppState {
    pub model: LoadedModel,
}

// LoadedModel contains raw device pointers behind cfg(feature = "cuda"). For Axum state,
// we assert Send + Sync because LoadedModel is only moved between threads and internal
// mutability is managed by CUDA context APIs.
unsafe impl Send for AppState {}
unsafe impl Sync for AppState {}

pub fn app_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::very_permissive();
    let xfo = SetResponseHeaderLayer::if_not_present(
        HeaderName::from_static("x-frame-options"),
        HeaderValue::from_static("ALLOWALL"),
    );
    let csp = SetResponseHeaderLayer::if_not_present(
        HeaderName::from_static("content-security-policy"),
        HeaderValue::from_static("frame-ancestors *"),
    );

    Router::new()
        .route("/health", get(health))
        .route("/generate", post(generate))
        .layer(csp)
        .layer(xfo)
        .layer(cors)
        .with_state(state)
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status":"ok"}))
}

async fn generate(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    if !req.stream {
        let generated =
            generate_text(&state.model, options_from_request(&req, "server")).map_err(|e| {
                eprintln!("[server] decode_loop failed: {e}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
            })?;
        return Ok(Json(GenerateResponse {
            output: generated.output,
        })
        .into_response());
    }

    #[cfg(feature = "cuda")]
    eprintln!(
        "[mem] (start) pid={} device_id={} TOTAL_DEVICE_BYTES={}",
        std::process::id(),
        state.model.cuda.device_id(),
        CudaContext::total_device_bytes()
    );
    // Build tokenizer from GGUF metadata. Fallback to byte-level.
    let tokenizer: Tokenizer = match Tokenizer::from_gguf_metadata(&state.model.gguf.metadata) {
        Ok(t) => t,
        Err(_) => Tokenizer::byte_level(),
    };

    let eos_id = tokenizer.eos_id();
    let max_tokens = req
        .max_tokens
        .or_else(|| Some(state.model.model_config.context_length as usize))
        .unwrap_or(32);
    let stopping = StoppingCriteria::new(Some(max_tokens), eos_id);
    let sampler =
        sampler_from_options(&options_from_request(&req, "server")).map_err(internal_error)?;

    if state.model.kv_cache.is_some() {
        state.model.reset_kv_cache().map_err(internal_error)?;
    }

    #[cfg(feature = "cuda")]
    let (full_decode_d_model, full_decode_hidden_dim) = state
        .model
        .validate_full_layer_decode()
        .map_err(|e| internal_error(e.context("CUDA full-layer decode is unavailable")))?;

    // logits_fn that uses the minimal forward path to produce next-token logits
    let logits_fn = {
        let state_for_logits = state.clone();
        #[cfg(feature = "cuda")]
        let mut step: usize = 0;
        #[cfg(feature = "cuda")]
        let mut processed_len: usize = 0;
        move |ids: &[u32]| -> anyhow::Result<Vec<f32>> {
            let model = &state_for_logits.model;
            // KV cache is pre-allocated at startup on the real model instance

            eprintln!("[server] logits_fn called with {} tokens", ids.len());
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
            // Determine d_model and whether we can run the full transformer stack.
            // Try to infer d_model from embeddings or lm_head as fallback
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
                .find_map(|n| model.device_tensors.get(*n));
            #[cfg(not(feature = "cuda"))]
            if tok_opt.is_none() {
                eprintln!(
                    "[server] warning: no known embedding tensor found; will try lm_head only"
                );
            }
            #[cfg(not(feature = "cuda"))]
            let d_model_from_tok =
                tok_opt.and_then(|t| t.shape.get(1).copied()).unwrap_or(0) as usize;
            #[cfg(not(feature = "cuda"))]
            if let Some(t) = tok_opt {
                eprintln!("[server] embeddings shape: {:?}", t.shape);
            }

            #[cfg(feature = "cuda")]
            let (d, can_forward) = {
                eprintln!(
                    "[server] mapped {} standard layers d_model={} hidden_dim={}",
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
                            "[server] KV cache cannot address all layers; using embeddings-only logits fallback"
                        );
                    }
                    eprintln!(
                        "[server] mapped {} standard layers d_model={} hidden_dim={}",
                        model.model_config.block_count, d_model, hidden_dim
                    );
                    (d_model, ok)
                }
                Err(e) => {
                    eprintln!("[server] validate_standard_layers failed; falling back to embeddings/logits path: {e}");
                    let d_try = if d_model_from_tok > 0 {
                        d_model_from_tok
                    } else {
                        match model.map_lm_head() {
                            Ok((_name, _lm, d_m, _vocab, _tied)) => {
                                eprintln!("[server] derived d_model from lm_head: {}", d_m);
                                d_m
                            }
                            Err(e2) => {
                                eprintln!("[server] could not derive d_model from lm_head: {e2}");
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
                // Last token id to embed
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
                    eprintln!("[server] token id {}", tok_id);

                    // Allocate device buffer for embedding row (f32)
                    let d_x = model
                        .cuda
                        .device_malloc_tagged(bytes, "server:d_x_embed_f32")?;

                    let token_logits = (|| -> anyhow::Result<Vec<f32>> {
                        // Load embedding row for the token into d_x (f32)
                        unsafe {
                            model.load_token_embedding_to_f32(tok_id, d_x)?;
                        }

                        // If we have layer weights and KV, run prompt tokens as prefill and sampled tokens as decode.
                        if can_forward {
                            let d_out = model
                                .cuda
                                .device_malloc_tagged(bytes, "server:d_out_hidden_f32")?;
                            let result = (|| -> anyhow::Result<Vec<f32>> {
                                unsafe {
                                    let layers = model.forward_one_token_all_layers(
                                        d_x as *const _,
                                        (token_idx + 1) as u32,
                                        d_out,
                                    )?;
                                    if token_idx == start {
                                        eprintln!(
                                            "[server] full-layer forward enabled layers={layers}"
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
                            // Fallback: treat embedding as hidden and compute logits.
                            unsafe { model.logits_from_hidden(d_x as *const _) }
                        }
                    })();

                    unsafe {
                        let _ = model.cuda.device_free(d_x);
                    }

                    let token_logits = token_logits?;
                    log_top_logits(&token_logits, 8);
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
                // Build hidden state from embeddings row (F16 -> F32) for the last token, then compute host logits
                let tok_id = *ids.last().ok_or_else(|| anyhow::anyhow!("empty ids"))? as usize;
                let tok = embed_names
                    .iter()
                    .find_map(|n| model.device_tensors.get(*n))
                    .ok_or_else(|| {
                        anyhow::anyhow!("missing embeddings tensor (tried common names)")
                    })?;
                if tok.shape.len() != 2 {
                    return Err(anyhow::anyhow!("embeddings must be [vocab, d_model]"));
                }
                let vocab = tok.shape[0] as usize;
                if tok_id >= vocab {
                    return Err(anyhow::anyhow!(format!(
                        "token id {} out of range (vocab={})",
                        tok_id, vocab
                    )));
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
                            let blk = i / Q8_0_BLOCK;
                            let idx = i % Q8_0_BLOCK;
                            let base = blk * Q8_0_BLOCK_BYTES;
                            let d_bits = u16::from_le_bytes([row[base], row[base + 1]]);
                            let d = f16::from_bits(d_bits).to_f32();
                            let q = row[base + 2 + idx] as i8 as f32;
                            hidden[i] = d * q;
                        }
                    }
                    _ => {
                        return Err(anyhow::anyhow!(
                            "unsupported embeddings dtype for host path"
                        ))
                    }
                }
                let logits = unsafe { model.logits_from_hidden(hidden.as_ptr() as *const c_void) }?;
                Ok(logits)
            }
        }
    };

    let add_bos = true;
    let base_sampler = sampler;

    if req.stream {
        let mut sampler = base_sampler.clone();
        let tokenizer_stream = tokenizer.clone();
        let prompt = req.prompt.clone();
        #[allow(unused_mut)]
        let mut logits_fn = logits_fn;
        let stopping_stream = stopping.clone();
        let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(8);
        tokio::spawn(async move {
            let mut ids = match tokenizer_stream.encode_with_specials(&prompt, add_bos, false) {
                Ok(ids) => ids,
                Err(e) => {
                    let _ = tx
                        .send(Err(io::Error::new(
                            io::ErrorKind::Other,
                            format!("encode failed: {e}"),
                        )))
                        .await;
                    return;
                }
            };
            let mut generated: Vec<u32> = Vec::new();

            let send_chunk = |chunk: GenerateStreamChunk| {
                let tx = tx.clone();
                async move {
                    match chunk_to_bytes(&chunk) {
                        Ok(bytes) => {
                            let _ = tx.send(Ok(bytes)).await;
                        }
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                        }
                    }
                }
            };

            loop {
                if stopping_stream.should_stop(&generated) {
                    break;
                }
                let logits = match logits_fn(&ids) {
                    Ok(logits) => logits,
                    Err(e) => {
                        let _ = tx
                            .send(Err(io::Error::new(io::ErrorKind::Other, e.to_string())))
                            .await;
                        return;
                    }
                };
                if logits.is_empty() {
                    let _ = tx
                        .send(Err(io::Error::new(
                            io::ErrorKind::Other,
                            "logits_fn returned empty logits",
                        )))
                        .await;
                    return;
                }
                let next = match sampler.sample(&logits) {
                    Ok(n) => n as u32,
                    Err(e) => {
                        let _ = tx
                            .send(Err(io::Error::new(io::ErrorKind::Other, e.to_string())))
                            .await;
                        return;
                    }
                };
                ids.push(next);
                generated.push(next);
                #[cfg(feature = "cuda")]
                if std::env::var("M40LLM_DECODE_LOG").ok().as_deref() == Some("1") {
                    let text = tokenizer_stream
                        .decode_ignoring_specials(&[next])
                        .unwrap_or_else(|_| "<decode-error>".to_string());
                    eprintln!("[server] sampled token id={next} text={text:?}");
                }

                match tokenizer_stream.decode_ignoring_specials(&generated) {
                    Ok(text) => {
                        let _ = send_chunk(GenerateStreamChunk {
                            output: sanitize_output(&text),
                            done: false,
                        })
                        .await;
                    }
                    Err(e) => {
                        let _ = tx
                            .send(Err(io::Error::new(io::ErrorKind::Other, e.to_string())))
                            .await;
                        return;
                    }
                }

                if let Some(eos) = stopping_stream.eos_id {
                    if next == eos {
                        break;
                    }
                }
                if let Some(mt) = stopping_stream.max_tokens {
                    if generated.len() >= mt {
                        break;
                    }
                }
            }

            let final_text = match tokenizer_stream.decode_ignoring_specials(&generated) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx
                        .send(Err(io::Error::new(io::ErrorKind::Other, e.to_string())))
                        .await;
                    return;
                }
            };
            let _ = send_chunk(GenerateStreamChunk {
                output: sanitize_output(&final_text),
                done: true,
            })
            .await;
        });

        let body_stream = ReceiverStream::new(rx);
        let response = Response::builder()
            .status(StatusCode::OK)
            .header(CONTENT_TYPE, "application/jsonl")
            .body(Body::from_stream(body_stream))
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("streaming response build failed: {e}"),
                    }),
                )
            })?;
        return Ok(response);
    }

    let prompt_token_len = match tokenizer.encode_with_specials(&req.prompt, add_bos, false) {
        Ok(ids) => ids.len(),
        Err(e) => {
            eprintln!("[server] prompt encode failed: {e}");
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("encode failed: {e}"),
                }),
            ));
        }
    };

    let ids = match decode_loop_with(
        &tokenizer,
        &req.prompt,
        add_bos,
        base_sampler,
        &stopping,
        logits_fn,
    ) {
        Ok(ids) => ids,
        Err(e) => {
            eprintln!("[server] decode_loop failed: {e}");
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("generation failed: {e}"),
                }),
            ));
        }
    };
    let text = match decode_generated_text(&tokenizer, &ids, prompt_token_len) {
        Ok(t) => sanitize_output(&t),
        Err(e) => {
            #[cfg(feature = "cuda")]
            eprintln!(
                "[mem] (finish-err) pid={} device_id={} TOTAL_DEVICE_BYTES={}",
                std::process::id(),
                state.model.cuda.device_id(),
                CudaContext::total_device_bytes()
            );
            eprintln!("[server] decode failed: {e}");
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("decode failed: {e}"),
                }),
            ));
        }
    };
    #[cfg(feature = "cuda")]
    eprintln!(
        "[mem] (finish) pid={} device_id={} TOTAL_DEVICE_BYTES={}",
        std::process::id(),
        state.model.cuda.device_id(),
        CudaContext::total_device_bytes()
    );
    Ok(Json(GenerateResponse { output: text }).into_response())
}

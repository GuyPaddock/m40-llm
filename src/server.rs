// src/server.rs
#![allow(dead_code)]
use axum::{routing::post, Json, Router};
use std::sync::Arc;

use crate::decode::{decode_loop_with, greedy_sampler, StoppingCriteria};
use crate::infer::LoadedModel;
use crate::tokenizer::Tokenizer;
use anyhow::Result;
use axum::http::{HeaderName, HeaderValue};
use tower_http::{cors::CorsLayer, set_header::SetResponseHeaderLayer};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct GenerateResponse {
    pub output: String,
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
        .route("/generate", post(generate))
        .layer(csp)
        .layer(xfo)
        .layer(cors)
        .with_state(state)
}

async fn generate(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, axum::http::StatusCode> {
    // Build tokenizer from GGUF metadata. Fallback to byte-level.
    let tokenizer: Tokenizer = match Tokenizer::from_gguf_metadata(&state.model.gguf.metadata) {
        Ok(t) => t,
        Err(_) => Tokenizer::byte_level(),
    };

    let eos_id = tokenizer.eos_id();
    let max_tokens = req.max_tokens.unwrap_or(32);
    let stopping = StoppingCriteria::new(Some(max_tokens), eos_id);
    let sampler = greedy_sampler(42);

    // logits_fn that uses the minimal forward path to produce next-token logits
    let logits_fn = {
        let model = &state.model;
        move |ids: &[u32]| -> anyhow::Result<Vec<f32>> {
            // KV cache is pre-allocated at startup on the real model instance

            eprintln!("[server] logits_fn called with {} tokens", ids.len());
            #[cfg(feature = "cuda")]
            let seq_len_now: u32 = ids.len() as u32;

            // Determine d_model and whether we can run the minimal forward layer path
            // Try to infer d_model from embeddings or lm_head as fallback
            let embed_names = [
                "tok_embeddings.weight",
                "token_embd.weight",
                "token_embd",
                "token_embeddings.weight",
            ];
            let tok_opt = embed_names
                .iter()
                .find_map(|n| model.device_tensors.get(*n));
            if tok_opt.is_none() {
                eprintln!("[server] warning: no known embedding tensor found; will try lm_head only");
            }
            let d_model_from_tok = tok_opt.and_then(|t| t.shape.get(1).copied()).unwrap_or(0) as usize;
            if let Some(t) = tok_opt { eprintln!("[server] embeddings shape: {:?}", t.shape); }

            let (d, can_forward) = match model.map_standard_layer(0) {
                Ok(w) => {
                    let ok = model.kv_cache.is_some();
                    if !ok {
                        eprintln!("[server] KV cache not available; using embeddings-only logits fallback");
                    }
                    eprintln!("[server] mapped standard layer d_model={} hidden_dim={}", w.d_model, w.hidden_dim);
                    (w.d_model, ok)
                }
                Err(e) => {
                    eprintln!("[server] map_standard_layer failed; falling back to embeddings/logits path: {e}");
                    let d_try = if d_model_from_tok > 0 { d_model_from_tok } else {
                        match model.map_lm_head() {
                            Ok((_lm, d_m, _vocab, _tied)) => {
                                eprintln!("[server] derived d_model from lm_head: {}", d_m);
                                d_m
                            }
                            Err(e2) => {
                                eprintln!("[server] could not derive d_model from lm_head: {e2}");
                                0
                            }
                        }
                    };
                    if d_try == 0 { return Err(anyhow::anyhow!("could not determine d_model")); }
                    (d_try, false)
                }
            };
            let bytes = d * std::mem::size_of::<f32>();

            #[cfg(not(feature = "cuda"))]
            { let _ = (can_forward, bytes); }

            #[cfg(feature = "cuda")]
            {
                // Last token id to embed
                let tok_id = *ids.last().ok_or_else(|| anyhow::anyhow!("empty ids"))? as u64;
                eprintln!("[server] token id {}", tok_id);

                // Allocate device buffer for embedding row (f32)
                let d_x = model.cuda.device_malloc(bytes)?;

                // Load embedding row for last token into d_x (f32)
                unsafe { model.load_token_embedding_f16_to_f32(tok_id, d_x)?; }

                // If we have FP16 layer weights mapped and KV available, run the minimal forward through one layer.
                let logits = if can_forward {
                    let d_out = model.cuda.device_malloc(bytes)?;
                    unsafe {
                        model.forward_one_token_with_layer(
                            d_x as *const _,
                            0,
                            0,
                            seq_len_now,
                            d_out,
                        )?;
                    }
                    let logits = unsafe { model.logits_from_hidden(d_out as *const _) }?;
                    unsafe { let _ = model.cuda.device_free(d_out); }
                    logits
                } else {
                    // Fallback: treat embedding as hidden and compute logits (uses lm_head when present or tok_embeddings^T)
                    unsafe { model.logits_from_hidden(d_x as *const _) }?
                };

                // Free embedding buffer
                unsafe { let _ = model.cuda.device_free(d_x); }

                Ok(logits)
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
                    .ok_or_else(|| anyhow::anyhow!("missing embeddings tensor (tried common names)"))?;
                if tok.shape.len() != 2 {
                    return Err(anyhow::anyhow!("embeddings must be [vocab, d_model]"));
                }
                let vocab = tok.shape[0] as usize;
                if tok_id >= vocab {
                    return Err(anyhow::anyhow!(format!("token id {} out of range (vocab={})", tok_id, vocab)));
                }
                let d_model = tok.shape[1] as usize;
                let row_bytes = d_model * 2;
                let off = tok.byte_offset as usize + tok_id * row_bytes;
                let row = &model.host_weights[off..off + row_bytes];
                let mut hidden = vec![0f32; d_model];
                for i in 0..d_model {
                    let lo = row[2 * i] as u16;
                    let hi = row[2 * i + 1] as u16;
                    hidden[i] = f16::from_bits(lo | (hi << 8)).to_f32();
                }
                let logits = unsafe { model.logits_from_hidden(hidden.as_ptr() as *const c_void) }?;
                Ok(logits)
            }
        }
    };

    let add_bos = true;
    let ids = match decode_loop_with(
        &tokenizer,
        &req.prompt,
        add_bos,
        sampler,
        &stopping,
        logits_fn,
    ) {
        Ok(ids) => ids,
        Err(e) => {
            eprintln!("[server] decode_loop failed; returning prompt as output: {e}");
            // Fallback: return 200 with the original prompt when generation fails
            return Ok(Json(GenerateResponse { output: req.prompt }));
        }
    };
    let text = match tokenizer.decode_ignoring_specials(&ids) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[server] decode failed; returning prompt as output: {e}");
            req.prompt
        }
    };
    Ok(Json(GenerateResponse { output: text }))
}

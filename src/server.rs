// src/server.rs
#![allow(dead_code)]
use axum::{routing::post, Json, Router};
use serde::Deserialize;
use std::sync::Arc;

use crate::decode::{decode_loop_with, greedy_sampler, StoppingCriteria};
use crate::infer::LoadedModel;
use crate::tokenizer::Tokenizer;
use anyhow::Result;

#[derive(Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
}

#[derive(serde::Serialize)]
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
    // Wrap state in an extractor that is Clone + Send + Sync by using Arc
    Router::new()
        .route("/generate", post(generate))
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
    let mut seq_len: u32 = 0;
    let logits_fn = {
        let model = &state.model;
        move |ids: &[u32]| -> anyhow::Result<Vec<f32>> {
            // KV cache is pre-allocated at startup on the real model instance

            // Prepare device buffers for embeddings and layer output
            let w = model.map_standard_layer(0)?; // use first layer for minimal path
            let d = w.d_model as usize;
            let bytes = d * std::mem::size_of::<f32>();

            #[cfg(feature = "cuda")]
            {
                // Last token id to embed
                let tok_id = *ids.last().ok_or_else(|| anyhow::anyhow!("empty ids"))? as u64;

                // Allocate device buffers
                let d_x = model.cuda.device_malloc(bytes)?;
                let d_out = model.cuda.device_malloc(bytes)?;

                // Load embedding row for last token into d_x (f32)
                unsafe {
                    model.load_token_embedding_f16_to_f32(tok_id, d_x)?;
                }

                // Run minimal forward through one layer (prefill+decode semantics simplified as last-token)
                unsafe {
                    model.forward_one_token_with_layer(
                        d_x as *const _,
                        0,
                        0,
                        seq_len + 1,
                        d_out,
                    )?;
                }

                // Compute logits using lm_head if available (GPU GEMM); fallback to embeddings^T host path
                let logits = unsafe { model.logits_from_hidden(d_out as *const _) }?;

                // Free device buffers
                let _ = model.cuda.device_free(d_x);
                let _ = model.cuda.device_free(d_out);

                seq_len += 1;
                Ok(logits)
            }

            #[cfg(not(feature = "cuda"))]
            {
                use half::f16;
                use std::ffi::c_void;
                // Build hidden state from tok_embeddings row (F16 -> F32) for the last token, then compute host logits
                let tok_id = *ids.last().ok_or_else(|| anyhow::anyhow!("empty ids"))? as usize;
                let tok = model
                    .device_tensors
                    .get("tok_embeddings.weight")
                    .ok_or_else(|| anyhow::anyhow!("missing tok_embeddings.weight"))?;
                if tok.shape.len() != 2 {
                    return Err(anyhow::anyhow!("tok_embeddings must be [vocab, d_model]"));
                }
                let vocab = tok.shape[0] as usize;
                if tok_id >= vocab {
                    return Err(anyhow::anyhow!(format!(
                        "token id {} out of range (vocab={})",
                        tok_id, vocab
                    )));
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
                seq_len += 1;
                Ok(logits)
            }
        }
    };

    let add_bos = true;
    let ids = decode_loop_with(
        &tokenizer,
        &req.prompt,
        add_bos,
        sampler,
        &stopping,
        logits_fn,
    )
    .map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    let text = tokenizer
        .decode_ignoring_specials(&ids)
        .map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(GenerateResponse { output: text }))
}

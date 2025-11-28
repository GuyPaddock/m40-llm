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

                // For logits, do a simple linear head using tok_embeddings^T as a temporary stand-in
                // NOTE: This is a placeholder to let decode loop run; proper lm_head mapping should replace this.
                let tok = model
                    .device_tensors
                    .get("tok_embeddings.weight")
                    .ok_or_else(|| anyhow::anyhow!("missing tok_embeddings.weight"))?;
                let vocab = tok.shape[0] as usize;
                // Compute logits on host: logits[v] = dot(out, E[v]) where E is F16 embeddings
                let mut out_h = vec![0u8; bytes];
                model
                    .cuda
                    .memcpy_d2h(out_h.as_mut_ptr() as *mut _, d_out, bytes)?;
                let mut out_f = vec![0f32; d];
                for (i, ch) in out_h.chunks_exact(4).enumerate() {
                    out_f[i] = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
                }
                use half::f16;
                let mut logits = vec![0f32; vocab];
                let row_bytes = w.d_model * 2;
                for v in 0..vocab {
                    // Copy embedding row v from device to host before reading
                    let mut row_h = vec![0u8; row_bytes as usize];
                    #[cfg(feature = "cuda")]
                    unsafe {
                        let row_dev = (tok.dptr as usize + v * row_bytes as usize) as *const _;
                        model.cuda.memcpy_d2h(
                            row_h.as_mut_ptr() as *mut _,
                            row_dev,
                            row_bytes as usize,
                        )?;
                    }
                    let mut acc = 0f32;
                    for i in 0..d {
                        let lo = row_h[i * 2];
                        let hi = row_h[i * 2 + 1];
                        let val = f16::from_bits(u16::from_le_bytes([lo, hi])).to_f32();
                        acc += out_f[i] * val;
                    }
                    logits[v] = acc;
                }

                // Free device buffers
                let _ = model.cuda.device_free(d_x);
                let _ = model.cuda.device_free(d_out);

                seq_len += 1;
                Ok(logits)
            }

            #[cfg(not(feature = "cuda"))]
            {
                let _ = (ids, bytes);
                // CPU fallback placeholder: emit a simple rising logit on last byte id
                let vocab = 256usize;
                let mut logits = vec![0.0f32; vocab];
                if let Some(&last) = ids.last() {
                    let idx = (last as usize) % vocab;
                    logits[idx] = 1.0;
                }
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

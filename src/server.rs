// src/server.rs
#![allow(dead_code)]
use axum::{routing::post, Json, Router};
use serde::Deserialize;
use std::sync::Arc;

use crate::infer::LoadedModel;

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
    axum::extract::State(_state): axum::extract::State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, axum::http::StatusCode> {
    // TODO: tokenize `req.prompt`, run decode on GPU
    let max_tokens = req.max_tokens.unwrap_or(16);

    // For now, dummy output:
    let mut s = String::new();
    for i in 0..max_tokens {
        s.push_str(&format!("<{}>", i));
    }

    Ok(Json(GenerateResponse { output: s }))
}

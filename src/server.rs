// src/server.rs
#![cfg(feature = "server")]
use axum::{Router, routing::post, Json};
use serde::Deserialize;
use anyhow::Result;
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
    // TODO: tokenize `req.prompt`, run decode on GPU
    let max_tokens = req.max_tokens.unwrap_or(16);

    // For now, dummy output:
    let mut s = String::new();
    for i in 0..max_tokens {
        s.push_str(&format!("<{}>", i));
    }

    Ok(Json(GenerateResponse { output: s }))
}

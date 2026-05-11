#![cfg(feature = "server")]

use anyhow::Result;
use m40_llm::infer::LoadedModel;
use m40_llm::server::{
    app_router, AppState, GenerateRequest, GenerateResponse, GenerateStreamChunk,
};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio_stream::StreamExt;

#[path = "common/tiny_gguf.rs"]
mod tiny_gguf;

fn server_smoke_model() -> (m40_llm::gguf::GgufModel, Vec<u8>) {
    tiny_gguf::make_identity_tiny_gguf(tiny_gguf::TinyGgufConfig {
        vocab: 128,
        d_model: 128,
        ..Default::default()
    })
}

async fn start_test_server(model: LoadedModel) -> Result<SocketAddr> {
    let state = Arc::new(AppState::new(model));
    let router = app_router(state);

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr: SocketAddr = listener.local_addr()?;
    tokio::spawn(async move {
        axum::serve(listener, router.into_make_service())
            .await
            .unwrap();
    });
    Ok(addr)
}

#[tokio::test]
async fn server_generate_smoke() -> Result<()> {
    if std::env::var("M40LLM_ENABLE_NVCC").ok().as_deref() != Some("1") {
        eprintln!("skipping server smoke tests without CUDA upload support");
        return Ok(());
    }
    let (gguf, bytes) = server_smoke_model();

    let mut model = LoadedModel::from_gguf(gguf, bytes, 0)?;
    // KV cache optional for non-CUDA path but harmless to allocate small
    let _ = model.allocate_kv_cache(16, 1);

    let addr = start_test_server(model).await?;

    // Request small generation
    let client = reqwest::Client::new();
    let url = format!("http://{}/generate", addr);
    let req = GenerateRequest {
        prompt: "A".to_string(),
        max_tokens: Some(2),
        stream: false,
        ..Default::default()
    };
    let resp = client.post(&url).json(&req).send().await.unwrap();
    assert!(resp.status().is_success());
    let raw = resp.bytes().await.unwrap();
    assert!(raw.iter().all(|b| *b != 0));
    let jr: GenerateResponse = serde_json::from_slice(&raw).unwrap();
    assert!(!jr.output.starts_with(&req.prompt));
    assert!(!jr.output.is_empty());
    Ok(())
}

#[tokio::test]
async fn server_generate_streaming_nul_free() -> Result<()> {
    if std::env::var("M40LLM_ENABLE_NVCC").ok().as_deref() != Some("1") {
        eprintln!("skipping server smoke tests without CUDA upload support");
        return Ok(());
    }
    let (gguf, bytes) = server_smoke_model();

    let mut model = LoadedModel::from_gguf(gguf, bytes, 0)?;
    let _ = model.allocate_kv_cache(16, 1);

    let addr = start_test_server(model).await?;

    let client = reqwest::Client::new();
    let url = format!("http://{}/generate", addr);
    let req = GenerateRequest {
        prompt: "A".to_string(),
        max_tokens: Some(2),
        stream: true,
        ..Default::default()
    };
    let resp = client.post(&url).json(&req).send().await.unwrap();
    assert!(resp.status().is_success());

    let mut stream = resp.bytes_stream();
    let mut combined: Vec<u8> = Vec::new();
    let mut last: Option<GenerateStreamChunk> = None;
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.unwrap();
        assert!(bytes.iter().all(|b| *b != 0));
        combined.extend_from_slice(&bytes);
        for line in bytes.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
            let part: GenerateStreamChunk = serde_json::from_slice(line).unwrap();
            last = Some(part);
        }
    }

    assert!(!combined.is_empty());
    let last = last.expect("expected at least one streamed chunk");
    assert!(last.done);
    assert!(!last.output.is_empty());
    assert!(!last.output.starts_with(&req.prompt));
    Ok(())
}

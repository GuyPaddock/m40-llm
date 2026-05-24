#![cfg(feature = "server")]

use anyhow::Result;
use m40_llm::infer::LoadedModel;
use m40_llm::kv_compression::{KvCompressMode, KvCompressionConfig};
use m40_llm::profile;
use m40_llm::server::{
    app_router, AppState, GenerateRequest, GenerateResponse, GenerateStreamChunk,
};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio_stream::StreamExt;

#[path = "common/tiny_gguf.rs"]
mod tiny_gguf;

struct EnvVarGuard {
    key: &'static str,
    old: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let old = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self { key, old }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.old {
            Some(value) => std::env::set_var(self.key, value),
            None => std::env::remove_var(self.key),
        }
    }
}

fn server_smoke_model() -> (m40_llm::gguf::GgufModel, Vec<u8>) {
    tiny_gguf::make_identity_tiny_gguf(tiny_gguf::TinyGgufConfig {
        vocab: 128,
        d_model: 128,
        head_count: 2,
        ..Default::default()
    })
}

fn server_smoke_compressed_config(top_blocks: u32) -> KvCompressionConfig {
    KvCompressionConfig {
        recent_window: 4,
        block_size: 4,
        top_blocks,
        ..KvCompressionConfig::default()
    }
}

async fn run_non_stream_generate_smoke(kv_compression: KvCompressionConfig) -> Result<()> {
    let (gguf, bytes) = server_smoke_model();
    let mut model = LoadedModel::from_gguf(gguf, bytes, 0)?;
    if kv_compression.mode == KvCompressMode::Off {
        model.allocate_kv_cache_for_layer_sequences(16, 1)?;
    } else {
        model.allocate_compressed_kv_cache_for_layers(16, &kv_compression)?;
    }
    let server = start_test_server_with_kv_config(model, kv_compression).await?;

    let client = reqwest::Client::new();
    let url = format!("http://{}/generate", server.addr);
    let req = GenerateRequest {
        prompt: "A".to_string(),
        max_tokens: Some(2),
        stream: false,
        ..Default::default()
    };
    let resp = client.post(&url).json(&req).send().await?;
    assert!(resp.status().is_success());
    let raw = resp.bytes().await?;
    assert!(raw.iter().all(|b| *b != 0));
    let jr: GenerateResponse = serde_json::from_slice(&raw)?;
    assert!(!jr.output.is_empty());
    assert!(!jr.output.starts_with(&req.prompt));
    server.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn server_generate_default_compressed_kv_smoke() -> Result<()> {
    if std::env::var("M40LLM_ENABLE_NVCC").ok().as_deref() != Some("1") {
        eprintln!("skipping server smoke tests without CUDA upload support");
        return Ok(());
    }
    let _batch_env = EnvVarGuard::set("M40LLM_SERVER_BATCH_DECODE", "1");
    let kv_compression = server_smoke_compressed_config(8);
    let (gguf, bytes) = server_smoke_model();

    let mut model = LoadedModel::from_gguf(gguf, bytes, 0)?;
    model.allocate_compressed_kv_cache_for_layers(16, &kv_compression)?;

    let state = AppState::new_with_kv_config(model, kv_compression);
    assert!(
        !state.decode_batching_requested,
        "compressed KV should stay on the serialized server path until the scheduler is cache-layout-aware"
    );
    assert!(
        state.decode_sequence_pool.is_none(),
        "compressed KV should not allocate dense scheduler sequence leases"
    );
    let server = start_test_server_with_kv_config_state(state).await?;

    let client = reqwest::Client::new();
    let url = format!("http://{}/generate", server.addr);
    let req = GenerateRequest {
        prompt: "A".to_string(),
        max_tokens: Some(2),
        stream: false,
        ..Default::default()
    };
    let resp = client.post(&url).json(&req).send().await?;
    assert!(resp.status().is_success());
    let raw = resp.bytes().await?;
    assert!(raw.iter().all(|b| *b != 0));
    let jr: GenerateResponse = serde_json::from_slice(&raw)?;
    assert!(!jr.output.is_empty());
    assert!(!jr.output.starts_with(&req.prompt));
    server.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn server_generate_compressed_top_block_overrides_smoke() -> Result<()> {
    if std::env::var("M40LLM_ENABLE_NVCC").ok().as_deref() != Some("1") {
        eprintln!("skipping server smoke tests without CUDA upload support");
        return Ok(());
    }
    run_non_stream_generate_smoke(server_smoke_compressed_config(4)).await?;
    run_non_stream_generate_smoke(server_smoke_compressed_config(16)).await
}

#[tokio::test]
async fn zz_server_generate_batch_decode_leases_sequence_slots() -> Result<()> {
    if std::env::var("M40LLM_ENABLE_NVCC").ok().as_deref() != Some("1") {
        eprintln!("skipping server smoke tests without CUDA upload support");
        return Ok(());
    }
    let _batch_env = EnvVarGuard::set("M40LLM_SERVER_BATCH_DECODE", "1");
    let _prefill_env = EnvVarGuard::set("M40LLM_SERVER_BATCH_PREFILL", "1");
    profile::reset();
    let (gguf, bytes) = server_smoke_model();

    let mut model = LoadedModel::from_gguf(gguf, bytes, 0)?;
    model.allocate_kv_cache_for_layer_sequences(16, 2)?;

    let server =
        start_test_server_with_kv_config(model, KvCompressionConfig::dense_reference()).await?;
    let client = reqwest::Client::new();
    let url = format!("http://{}/generate", server.addr);
    let req_a = GenerateRequest {
        prompt: "A".to_string(),
        max_tokens: Some(2),
        stream: false,
        ..Default::default()
    };
    let req_b = GenerateRequest {
        prompt: "B".to_string(),
        max_tokens: Some(2),
        stream: false,
        ..Default::default()
    };

    let (resp_a, resp_b) = tokio::join!(
        client.post(&url).json(&req_a).send(),
        client.post(&url).json(&req_b).send()
    );
    let resp_a = resp_a?;
    let resp_b = resp_b?;
    assert!(resp_a.status().is_success());
    assert!(resp_b.status().is_success());

    let out_a: GenerateResponse = serde_json::from_slice(&resp_a.bytes().await?)?;
    let out_b: GenerateResponse = serde_json::from_slice(&resp_b.bytes().await?)?;
    assert!(!out_a.output.is_empty());
    assert!(!out_b.output.is_empty());
    let snapshot = profile::snapshot();
    let batched_attention_launches = snapshot
        .by_op
        .get("attention_last_token_f32_gqa_batched")
        .map(|counts| counts.launches)
        .unwrap_or_default();
    assert!(
        batched_attention_launches >= 1,
        "server batch scheduler should use packed GQA decode attention"
    );
    let prefill_attention_launches = snapshot
        .by_op
        .get("attention_prefill_f32_gqa_varlen")
        .map(|counts| counts.launches)
        .unwrap_or_default();
    assert!(
        prefill_attention_launches >= 1,
        "server batch scheduler should use packed varlen prefill attention"
    );
    server.shutdown().await?;
    Ok(())
}

struct TestServer {
    addr: SocketAddr,
    shutdown_tx: Option<oneshot::Sender<()>>,
    handle: tokio::task::JoinHandle<Result<(), std::io::Error>>,
}

impl TestServer {
    async fn shutdown(mut self) -> Result<()> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        self.handle.await??;
        Ok(())
    }
}

async fn start_test_server_with_kv_config(
    model: LoadedModel,
    kv_compression: KvCompressionConfig,
) -> Result<TestServer> {
    start_test_server_with_kv_config_state(AppState::new_with_kv_config(model, kv_compression))
        .await
}

async fn start_test_server_with_kv_config_state(state: AppState) -> Result<TestServer> {
    let state = Arc::new(state);
    let router = app_router(state);

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr: SocketAddr = listener.local_addr()?;
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        axum::serve(listener, router.into_make_service())
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
    });
    Ok(TestServer {
        addr,
        shutdown_tx: Some(shutdown_tx),
        handle,
    })
}

#[tokio::test]
async fn server_streaming_respects_dense_kv_config() -> Result<()> {
    if std::env::var("M40LLM_ENABLE_NVCC").ok().as_deref() != Some("1") {
        eprintln!("skipping server smoke tests without CUDA upload support");
        return Ok(());
    }
    let (gguf, bytes) = server_smoke_model();

    let mut model = LoadedModel::from_gguf(gguf, bytes, 0)?;
    let _ = model.allocate_kv_cache(16, 1);

    let server =
        start_test_server_with_kv_config(model, KvCompressionConfig::dense_reference()).await?;

    let client = reqwest::Client::new();
    let url = format!("http://{}/generate", server.addr);
    let req = GenerateRequest {
        prompt: "A".to_string(),
        max_tokens: Some(2),
        stream: true,
        ..Default::default()
    };
    let resp = client.post(&url).json(&req).send().await.unwrap();
    assert!(resp.status().is_success());

    let mut stream = resp.bytes_stream();
    let mut last: Option<GenerateStreamChunk> = None;
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.unwrap();
        for line in bytes.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
            let part: GenerateStreamChunk = serde_json::from_slice(line).unwrap();
            last = Some(part);
        }
    }

    let last = last.expect("expected at least one streamed chunk");
    assert!(last.done);
    assert!(!last.output.is_empty());
    server.shutdown().await?;
    Ok(())
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

    let server =
        start_test_server_with_kv_config(model, KvCompressionConfig::dense_reference()).await?;

    // Request small generation
    let client = reqwest::Client::new();
    let url = format!("http://{}/generate", server.addr);
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
    server.shutdown().await?;
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

    let server =
        start_test_server_with_kv_config(model, KvCompressionConfig::dense_reference()).await?;

    let client = reqwest::Client::new();
    let url = format!("http://{}/generate", server.addr);
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
    server.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn server_generate_streaming_default_compressed_kv_smoke() -> Result<()> {
    if std::env::var("M40LLM_ENABLE_NVCC").ok().as_deref() != Some("1") {
        eprintln!("skipping server smoke tests without CUDA upload support");
        return Ok(());
    }
    let kv_compression = server_smoke_compressed_config(8);
    let (gguf, bytes) = server_smoke_model();

    let mut model = LoadedModel::from_gguf(gguf, bytes, 0)?;
    model.allocate_compressed_kv_cache_for_layers(16, &kv_compression)?;

    let server = start_test_server_with_kv_config(model, kv_compression).await?;

    let client = reqwest::Client::new();
    let url = format!("http://{}/generate", server.addr);
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
    server.shutdown().await?;
    Ok(())
}

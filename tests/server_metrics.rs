#![cfg(feature = "server")]

use std::{sync::Arc, time::Duration};

use anyhow::Result;
use m40_llm::{
    decode::StopReason,
    generation_metrics::GenerationMetrics,
    gguf::{GgufScalar, GgufValue},
    infer::LoadedModel,
    kv_compression::KvCompressionConfig,
    server::{app_router, AppState},
};
use serde_json::{json, Value};

#[path = "common/tiny_gguf.rs"]
mod tiny_gguf;

struct MetricsServer {
    url: String,
    task: tokio::task::JoinHandle<Result<(), std::io::Error>>,
}

impl Drop for MetricsServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

struct BatchEnv {
    previous: Option<String>,
}

impl BatchEnv {
    fn set(is_batched: bool) -> Self {
        let previous = std::env::var("M40LLM_SERVER_BATCH_DECODE").ok();
        std::env::set_var(
            "M40LLM_SERVER_BATCH_DECODE",
            if is_batched { "1" } else { "0" },
        );
        Self { previous }
    }
}

impl Drop for BatchEnv {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => std::env::set_var("M40LLM_SERVER_BATCH_DECODE", value),
            None => std::env::remove_var("M40LLM_SERVER_BATCH_DECODE"),
        }
    }
}

async fn start_metrics_server(
    config: KvCompressionConfig,
    eos_id: Option<u32>,
    is_batched: bool,
) -> Result<MetricsServer> {
    let (mut gguf, bytes) = tiny_gguf::make_ascii_tiny_gguf(tiny_gguf::TinyGgufConfig {
        vocab: 128,
        d_model: 128,
        head_count: 2,
        context_length: 32,
        ..Default::default()
    });
    if let Some(eos_id) = eos_id {
        gguf.metadata.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            GgufValue::Scalar(GgufScalar::U32(eos_id)),
        );
    }
    let mut model = LoadedModel::from_gguf(gguf, bytes, 0)?;
    if config.mode.is_enabled() {
        model.allocate_compressed_kv_cache_for_layer_sequences(32, 2, &config)?;
    } else {
        model.allocate_kv_cache_for_layer_sequences(32, 2)?;
    }
    let state = AppState::new_with_kv_config(model, config);
    assert_eq!(state.decode_batching_requested, is_batched);
    let router = app_router(Arc::new(state));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let url = format!("http://{}/generate", listener.local_addr()?);
    let task = tokio::spawn(async move { axum::serve(listener, router.into_make_service()).await });
    Ok(MetricsServer { url, task })
}

async fn post_generation(
    client: &reqwest::Client,
    server: &MetricsServer,
    max_tokens: usize,
    should_include_metrics: bool,
) -> Result<Value> {
    let response = client
        .post(&server.url)
        .json(&json!({
            "prompt": "BC",
            "prompt_format": "raw",
            "max_tokens": max_tokens,
            "top_k": 1,
            "include_metrics": should_include_metrics,
        }))
        .send()
        .await?;
    let status = response.status();
    let body: Value = response.json().await?;
    assert!(status.is_success(), "{status}: {body}");
    Ok(body)
}

fn assert_metrics(body: &Value, generated_tokens: usize, stop_reason: StopReason) -> Result<()> {
    let metrics: GenerationMetrics = serde_json::from_value(body["metrics"].clone())?;
    assert_eq!(metrics.prompt_tokens, 2);
    assert_eq!(metrics.generated_tokens, generated_tokens);
    assert_eq!(metrics.stop_reason, stop_reason);
    assert!(metrics.prefill_elapsed_ms.is_finite() && metrics.prefill_elapsed_ms >= 0.0);
    assert!(metrics.decode_elapsed_ms.is_finite() && metrics.decode_elapsed_ms >= 0.0);
    assert!(metrics.total_elapsed_ms.is_finite());
    assert!(metrics.total_elapsed_ms >= metrics.prefill_elapsed_ms + metrics.decode_elapsed_ms);
    if generated_tokens == 0 {
        assert_eq!(metrics.prefill_elapsed_ms, 0.0);
    } else {
        assert!(metrics.prefill_elapsed_ms > 0.0);
    }
    if generated_tokens < 2 {
        assert_eq!(metrics.decode_elapsed_ms, 0.0);
    } else {
        assert!(metrics.decode_elapsed_ms > 0.0);
    }
    Ok(())
}

/// Buffered metrics count sampled IDs, distinguish EOS from limits, and leave
/// ordinary responses unchanged on serial and CUDA scheduler generation paths.
#[tokio::test]
async fn test_buffered_generation_metrics() -> Result<()> {
    #[cfg(feature = "cuda")]
    if std::env::var("M40LLM_ENABLE_NVCC").ok().as_deref() != Some("1") {
        eprintln!("skipping CUDA metrics test without device upload support");
        return Ok(());
    }
    let configs = vec![KvCompressionConfig::dense_reference()];
    #[cfg(feature = "cuda")]
    let configs = {
        let mut configs = configs;
        configs.push(KvCompressionConfig {
            recent_window: 4,
            block_size: 4,
            ..KvCompressionConfig::default()
        });
        configs
    };
    let batching_modes: &[bool] = if cfg!(feature = "cuda") {
        &[false, true]
    } else {
        &[false]
    };
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    for &is_batched in batching_modes {
        let _batch_env = BatchEnv::set(is_batched);
        for config in &configs {
            let server = start_metrics_server(config.clone(), None, is_batched).await?;
            let ordinary = post_generation(&client, &server, 2, false).await?;
            assert_eq!(ordinary, json!({"output": "AA"}));
            let measured = post_generation(&client, &server, 2, true).await?;
            assert_eq!(measured["output"], "AA");
            assert_metrics(&measured, 2, StopReason::Length)?;
            let zero_tokens = post_generation(&client, &server, 0, true).await?;
            assert_eq!(zero_tokens["output"], "");
            assert_metrics(&zero_tokens, 0, StopReason::Length)?;
            if is_batched {
                let (first, second) = tokio::join!(
                    post_generation(&client, &server, 8, true),
                    post_generation(&client, &server, 8, true),
                );
                for concurrent in [first?, second?] {
                    assert_eq!(concurrent["output"], "AAAAAAAA");
                    assert_metrics(&concurrent, 8, StopReason::Length)?;
                }
            }
            let streaming = client
                .post(&server.url)
                .json(&json!({
                    "prompt": "BC", "stream": true, "include_metrics": true,
                }))
                .send()
                .await?;
            assert_eq!(streaming.status(), reqwest::StatusCode::BAD_REQUEST);
            assert!(streaming.json::<Value>().await?["error"]
                .as_str()
                .unwrap()
                .contains("buffered"));
            drop(server);

            let eos_server =
                start_metrics_server(config.clone(), Some(b'A' as u32), is_batched).await?;
            for max_tokens in [1, 4] {
                let eos = post_generation(&client, &eos_server, max_tokens, true).await?;
                assert_eq!(eos["output"], "");
                assert_metrics(&eos, 1, StopReason::Stop)?;
            }
        }
    }
    Ok(())
}

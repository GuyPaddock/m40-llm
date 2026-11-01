use super::{
    protocol, quality,
    run::{self, Args, Report},
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaContext,
    evaluation::identity::file_sha256,
    generate::{PromptFormat, PromptPrefill},
    infer::{LoadedModel, Q8FfnOwnership},
    kv_compression::KvCompressionConfig,
    server::{app_router, AppState, GenerateRequest, GenerateResponse},
};
use serde_json::{json, Value};
use std::{fs, sync::Arc, time::Instant};

pub(super) struct Server {
    pub(super) state: Arc<AppState>,
    url: String,
    pub(super) task: tokio::task::JoinHandle<std::io::Result<()>>,
    pub(super) shutdown: tokio::sync::oneshot::Sender<()>,
}

/// Graceful shutdown drains connection tasks that retain router/model owners.
fn spawn_router(
    listener: tokio::net::TcpListener,
    router: axum::Router,
) -> (
    tokio::sync::oneshot::Sender<()>,
    tokio::task::JoinHandle<std::io::Result<()>>,
) {
    let (shutdown, stopped) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        axum::serve(listener, router.into_make_service())
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
    });
    (shutdown, task)
}

pub(super) fn validate_timing_gpu(state: &Value) -> Result<()> {
    let clocks = state["clocks"]
        .as_str()
        .context("GPU clocks")?
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    ensure!(
        clocks.len() == 4 && clocks[1] == "P0" && clocks[2] == "1113" && clocks[3] == "3004",
        "timing clocks differ from the qualified reference: {clocks:?}"
    );
    ensure!(
        !state["processes"]
            .as_str()
            .context("GPU processes")?
            .is_empty(),
        "missing benchmark GPU process"
    );
    Ok(())
}

pub(super) async fn start(model: LoadedModel) -> Result<Server> {
    let mut state = AppState::new_with_kv_config(model, KvCompressionConfig::dense_reference());
    state.prompt_prefill = PromptPrefill::PackedPrefix {
        max_prompt_tokens: 128,
    };
    ensure!(!state.decode_batching_requested, "disable server batching");
    let state = Arc::new(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let url = format!("http://{}/generate", listener.local_addr()?);
    let router = app_router(state.clone());
    let (shutdown, task) = spawn_router(listener, router);
    Ok(Server {
        state,
        url,
        task,
        shutdown,
    })
}

fn request_body() -> GenerateRequest {
    GenerateRequest {
        prompt: protocol::render_prompt(),
        max_tokens: Some(512),
        should_include_metrics: true,
        top_k: Some(1),
        seed: Some(0),
        prompt_format: PromptFormat::Raw,
        ..GenerateRequest::default()
    }
}

pub(super) async fn request(
    client: &reqwest::Client,
    server: &Server,
    expected: &str,
) -> Result<Value> {
    let counters = m40_llm::profile::snapshot();
    let began = Instant::now();
    let response = client
        .post(&server.url)
        .json(&request_body())
        .send()
        .await?
        .error_for_status()?;
    let bytes = response.bytes().await?;
    let elapsed = began.elapsed().as_secs_f64() * 1000.;
    let response: GenerateResponse = serde_json::from_slice(&bytes)?;
    let metrics = response.metrics.context("missing generation metrics")?;
    ensure!(
        response.output == expected
            && metrics.generated_tokens == 512
            && metrics.prompt_tokens == 65
            && metrics.stop_reason == m40_llm::decode::StopReason::Length,
        "HTTP fixed work/output changed"
    );
    Ok(
        json!({"output":response.output,"metrics":metrics,"http_ms":elapsed,"http_tok_per_sec":512000./elapsed,
        "dispatch":quality::dispatch(&counters)}),
    )
}

pub(super) fn read_quality(args: &Args) -> Result<Vec<Value>> {
    let path = args
        .quality_report
        .as_ref()
        .context("--quality-report required before HTTP")?;
    let evidence: Vec<Value> = fs::read_to_string(path)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    ensure!(
        evidence.last().is_some_and(
            |r| r["kind"] == "joint_only_quality_complete" && r["is_gate_passed"] == true
        ),
        "incomplete quality evidence"
    );
    ensure!(
        evidence[0]["revision"] == run::command("git", &["rev-parse", "HEAD"])?
            && evidence[0]["executable_sha256"] == file_sha256(&std::env::current_exe()?)?,
        "quality source/executable differs"
    );
    Ok(evidence)
}

pub async fn execute(args: &Args, report: &mut Report) -> Result<()> {
    let evidence = read_quality(args)?;
    let path = args
        .quality_report
        .as_ref()
        .expect("validated quality path");
    let generation = evidence
        .iter()
        .find(|r| r["kind"] == "joint_only_generation")
        .context("missing actual quality generation")?;
    let expected = generation["output"].as_str().context("expected output")?;
    let baseline = CudaContext::total_device_bytes();
    let a = start(run::load(&args.model, Q8FfnOwnership::RetainSplit)?).await?;
    let b = start(run::load(&args.model, Q8FfnOwnership::JointOnly)?).await?;
    report.record(json!({"kind":"joint_only_http_binding","quality_sha256":file_sha256(path)?,"control":run::identity(&a.state.model),"candidate":run::identity(&b.state.model)}))?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;
    request(&client, &a, expected).await?;
    request(&client, &b, expected).await?;
    let mut pairs = Vec::new();
    let mut phase = "screen";
    let mut target = 5;
    loop {
        for pair in pairs.len()..target {
            let mut rates = [0.; 2];
            for arm in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                let before = run::gpu()?;
                validate_timing_gpu(&before)?;
                let row = request(&client, if arm == 0 { &a } else { &b }, expected).await?;
                let after = run::gpu()?;
                validate_timing_gpu(&after)?;
                ensure!(
                    row["dispatch"] == generation["control_dispatch"],
                    "ordinary HTTP dispatch/counters differ from completed quality"
                );
                rates[arm] = row["http_tok_per_sec"].as_f64().context("HTTP rate")?;
                report.record(json!({"kind":"joint_only_http_trial","phase":phase,"pair":pair,"arm":arm,"row":row,"gpu_before":before,"gpu_after":after}))?;
            }
            pairs.push(protocol::Pair {
                control_tps: rates[0],
                candidate_tps: rates[1],
            });
        }
        let comparison = protocol::compare_pairs(&pairs)?;
        let reverse: Vec<_> = pairs
            .iter()
            .map(|p| protocol::Pair {
                control_tps: p.candidate_tps,
                candidate_tps: p.control_tps,
            })
            .collect();
        let regression = protocol::compare_pairs(&reverse)?;
        report.record(json!({"kind":"joint_only_http_summary","phase":phase,"comparison":comparison,"reverse_comparison":regression,
            "scope":"storage-only non-regression; not exact-dense speed-goal qualification"}))?;
        if phase == "screen" && comparison.needs_variation_extension && target == 5 {
            target = 10;
            continue;
        }
        ensure!(
            !comparison.needs_variation_extension,
            "unstable paired timing"
        );
        if phase == "screen"
            && (comparison.needs_fresh_confirmation || regression.needs_fresh_confirmation)
        {
            pairs.clear();
            phase = "confirmation";
            target = 20;
            continue;
        }
        ensure!(
            comparison.gain > -comparison.promotion_threshold
                && comparison.bootstrap_gain_ci95[0] > -comparison.promotion_threshold,
            "meaningful regression not excluded"
        );
        break;
    }
    for server in [&a, &b] {
        let replay = quality::generate(&server.state.model, &protocol::render_prompt(), 512)?;
        ensure!(
            serde_json::to_value(&replay.token_ids[replay.prompt_token_len..])?
                == generation["generated_ids"],
            "post-HTTP actual IDs differ"
        );
    }
    drop(client);
    let _ = a.shutdown.send(());
    let _ = b.shutdown.send(());
    tokio::time::timeout(std::time::Duration::from_secs(5), a.task).await???;
    tokio::time::timeout(std::time::Duration::from_secs(5), b.task).await???;
    drop(a.state);
    drop(b.state);
    ensure!(
        CudaContext::total_device_bytes() == baseline,
        "HTTP teardown leaked"
    );
    report.record(json!({"kind":"joint_only_http_complete","is_storage_gate_passed":true,"device_bytes_after":CudaContext::total_device_bytes(),"requires_speedup":false}))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The actual benchmark request must ask for fixed-work metrics; idle
    /// keep-alive connections must release captured state after shutdown.
    #[tokio::test]
    async fn test_graceful_shutdown_releases_connection_owners() -> Result<()> {
        let owner = Arc::new(());
        let captured = owner.clone();
        let router = axum::Router::new().route(
            "/",
            axum::routing::post(move |axum::Json(request): axum::Json<GenerateRequest>| {
                let state = captured.clone();
                async move {
                    drop(state);
                    assert!(request.should_include_metrics);
                    assert_eq!(request.max_tokens, Some(512));
                    assert_eq!(request.top_k, Some(1));
                    assert_eq!(request.seed, Some(0));
                    assert!(matches!(request.prompt_format, PromptFormat::Raw));
                    "ok"
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let url = format!("http://{}", listener.local_addr()?);
        let (shutdown, task) = spawn_router(listener, router);
        let client = reqwest::Client::new();
        assert_eq!(
            client
                .post(url)
                .json(&request_body())
                .send()
                .await?
                .text()
                .await?,
            "ok"
        );
        let _ = shutdown.send(());
        tokio::time::timeout(std::time::Duration::from_secs(5), task).await???;
        assert_eq!(Arc::strong_count(&owner), 1);
        Ok(())
    }
}

use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{ensure, Context, Result};
use clap::Parser;
use m40_llm::{
    generate::{generate_text, GenerateOptions, PromptFormat, PromptPrefill},
    generation_metrics::GenerationMetrics,
    gguf,
    infer::{
        execution_weights::{Q8ExecutionConfig, Q8ExecutionFamilies},
        LoadedModel,
    },
    kv_compression::KvCompressionConfig,
    server::{app_router, AppState, GenerateResponse},
    tokenizer::Tokenizer,
};
use serde::Deserialize;
use serde_json::{json, Value};

use super::protocol::{self, CacheEvidence, Pair, CONTEXT_TOKENS, MAX_TOKENS};
use super::q8_runtime::{
    AttentionProjectionChoice, ControlChoice, FfnChoice, PrefillChoice, RuntimeConfig,
};

#[path = "q8.rs"]
mod q8;
#[path = "runtime_quality.rs"]
mod runtime_quality;

#[derive(Parser)]
#[command(about = "Controlled warm HTTP Qwen Q8_0 comparison; requires an isolated Ollama server")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "qwen2.5:3b-instruct-q8_0")]
    ollama_model: String,
    #[arg(long, default_value = "http://127.0.0.1:18435")]
    ollama_url: String,
    #[arg(long)]
    ollama_log: PathBuf,
    #[arg(long)]
    ollama_pid: u32,
    /// New report file; refusing overwrite keeps earlier evidence intact.
    #[arg(long)]
    output: PathBuf,
    /// Compare five unprofiled/profiled m40-llm pairs instead of timing Ollama.
    #[arg(long = "compare-stage-profile")]
    should_compare_stage_profile: bool,
    /// Compare legacy/parallel m40-llm attention, not Ollama throughput.
    #[arg(
        long = "compare-head128-attention",
        conflicts_with = "should_compare_stage_profile"
    )]
    should_compare_head128: bool,
    /// Compare native/prepared m40 models; Ollama supplies only an untimed oracle.
    #[arg(long = "compare-q8-execution", value_parser = ["split", "mixed"],
        conflicts_with_all = ["should_compare_stage_profile", "should_compare_head128"])]
    q8_execution_mode: Option<String>,
    #[arg(long, default_value = "all", value_parser = Q8ExecutionFamilies::parse,
        requires = "q8_execution_mode")]
    q8_families: Q8ExecutionFamilies,
    #[arg(long, default_value_t = 6144, requires = "q8_execution_mode")]
    q8_budget_mb: usize,
    /// M08 candidate; both models retain the qualified down/vocabulary control.
    #[arg(long = "compare-q8-runtime", value_enum, requires="q8_runtime_quality_report",
        conflicts_with_all=["q8_execution_mode", "should_compare_stage_profile", "should_compare_head128"])]
    q8_runtime_candidate: Option<FfnChoice>,
    #[arg(long, value_enum, requires = "q8_runtime_candidate")]
    q8_runtime_prefill: Option<PrefillChoice>,
    #[arg(long, value_enum, requires = "q8_runtime_candidate")]
    q8_runtime_control: Option<ControlChoice>,
    #[arg(long, value_enum, requires = "q8_runtime_candidate")]
    q8_runtime_attention: Option<AttentionProjectionChoice>,
    /// Completed model_quality report for this exact candidate/source/corpus.
    #[arg(long, requires = "q8_runtime_candidate")]
    q8_runtime_quality_report: Option<PathBuf>,
    /// Time the selected qualified runtime against Ollama, not native FFN.
    #[arg(long = "time-ollama", requires = "q8_runtime_candidate")]
    should_time_ollama: bool,
    /// Compare profiling off/on on the selected qualified runtime only.
    #[arg(
        long = "profile-q8-runtime",
        requires = "q8_runtime_candidate",
        conflicts_with = "should_time_ollama"
    )]
    should_profile_runtime: bool,
}

impl Args {
    fn control_runtime_config(&self) -> RuntimeConfig {
        self.q8_runtime_control.unwrap_or_default().config()
    }

    fn candidate_runtime_config(&self) -> Option<RuntimeConfig> {
        self.q8_runtime_candidate.map(|ffn| RuntimeConfig {
            ffn,
            prefill: self.q8_runtime_prefill.unwrap_or_default(),
            attention_projection: self.q8_runtime_attention.unwrap_or_default(),
        })
    }

    fn primary_runtime_config(&self) -> Option<RuntimeConfig> {
        self.candidate_runtime_config().map(|candidate| {
            if self.should_time_ollama || self.should_profile_runtime {
                candidate
            } else {
                self.control_runtime_config()
            }
        })
    }

    fn primary_execution_config(&self) -> Result<Q8ExecutionConfig> {
        if let Some(config) = self.primary_runtime_config() {
            config.execution_config()
        } else if self.q8_execution_mode.is_some() {
            Ok(Q8ExecutionConfig::disabled())
        } else {
            Q8ExecutionConfig::from_env()
        }
    }

    fn candidate_q8_config(&self) -> Result<Option<Q8ExecutionConfig>> {
        self.q8_execution_mode
            .as_deref()
            .map(|mode| {
                let mut config =
                    Q8ExecutionConfig::parse(mode, Some(&self.q8_budget_mb.to_string()), None)?;
                config.families = self.q8_families;
                Ok(config)
            })
            .transpose()
    }
}

#[derive(Debug, Clone, Copy)]
enum ServerVariant {
    Control,
    Prepared,
}

impl ServerVariant {
    fn as_str(self) -> &'static str {
        match self {
            Self::Control => "control",
            Self::Prepared => "prepared",
        }
    }
}

struct AttentionSwitch {
    previous: Option<std::ffi::OsString>,
}

impl AttentionSwitch {
    fn new() -> Self {
        Self {
            previous: std::env::var_os("M40LLM_ATTN_HEAD128_KERNEL"),
        }
    }
    fn set(&self, is_parallel: bool) {
        std::env::set_var(
            "M40LLM_ATTN_HEAD128_KERNEL",
            if is_parallel { "parallel" } else { "fast" },
        );
    }
}

impl Drop for AttentionSwitch {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => std::env::set_var("M40LLM_ATTN_HEAD128_KERNEL", value),
            None => std::env::remove_var("M40LLM_ATTN_HEAD128_KERNEL"),
        }
    }
}

struct StageProfileSwitch {
    previous: Option<std::ffi::OsString>,
}

impl StageProfileSwitch {
    fn new() -> Self {
        Self {
            previous: std::env::var_os("M40LLM_STAGE_PROFILE"),
        }
    }
    fn set(&self, is_enabled: bool) {
        std::env::set_var("M40LLM_STAGE_PROFILE", if is_enabled { "1" } else { "0" });
    }
}

impl Drop for StageProfileSwitch {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => std::env::set_var("M40LLM_STAGE_PROFILE", value),
            None => std::env::remove_var("M40LLM_STAGE_PROFILE"),
        }
    }
}

struct Report {
    file: File,
}

impl Report {
    fn create(path: &Path) -> Result<Self> {
        Ok(Self {
            file: OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(path)
                .with_context(|| format!("create new report {}", path.display()))?,
        })
    }

    fn record(&mut self, value: Value) -> Result<()> {
        serde_json::to_writer(&mut self.file, &value)?;
        self.file.write_all(b"\n")?;
        self.file.flush()?;
        Ok(())
    }
}

struct HttpServer {
    url: String,
    state: Arc<AppState>,
    task: tokio::task::JoinHandle<Result<(), std::io::Error>>,
}

impl Drop for HttpServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
    done: bool,
    done_reason: String,
    prompt_eval_count: usize,
    prompt_eval_cached_count: Option<usize>,
    eval_count: usize,
    total_duration: u64,
    load_duration: u64,
    prompt_eval_duration: u64,
    eval_duration: u64,
}

struct Measurement {
    tps: f64,
    output: String,
    is_fixed_workload: bool,
}

struct ComparisonRun {
    args: Args,
    client: reqwest::Client,
    server: HttpServer,
    prepared_server: Option<HttpServer>,
    prompt: String,
    prompt_ids: Vec<u32>,
    model_sha256: String,
    gpu_uuid: String,
    runner_url: String,
    report: Report,
    selected_quality: Option<runtime_quality::QualityEvidence>,
}

fn checked_command(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("run {program}"))?;
    ensure!(
        output.status.success(),
        "{program} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn sha256_bytes(bytes: &[u8]) -> Result<String> {
    let mut child = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .context("sha256sum stdin missing")?
        .write_all(bytes)?;
    let result = child.wait_with_output()?;
    ensure!(result.status.success(), "sha256sum failed");
    Ok(String::from_utf8(result.stdout)?
        .split_whitespace()
        .next()
        .context("missing SHA256")?
        .to_owned())
}

fn sha256_file(path: &Path) -> Result<String> {
    let output = Command::new("sha256sum").arg(path).output()?;
    ensure!(output.status.success(), "could not hash {}", path.display());
    Ok(String::from_utf8(output.stdout)?
        .split_whitespace()
        .next()
        .context("missing file SHA256")?
        .to_owned())
}

fn is_descendant_of(mut pid: u32, ancestor: u32) -> bool {
    for _ in 0..16 {
        if pid == ancestor {
            return true;
        }
        let Ok(status) = std::fs::read_to_string(format!("/proc/{pid}/status")) else {
            return false;
        };
        let Some(parent) = status
            .lines()
            .find_map(|line| line.strip_prefix("PPid:")?.trim().parse::<u32>().ok())
        else {
            return false;
        };
        if parent <= 1 || parent == pid {
            return false;
        }
        pid = parent;
    }
    false
}

fn gpu_snapshot(ollama_pid: u32, expected_uuid: &str) -> Result<Value> {
    let gpu = checked_command("nvidia-smi", &[
        "--query-gpu=uuid,name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])?;
    ensure!(
        gpu.lines().any(|line| line.starts_with(expected_uuid)),
        "selected GPU disappeared"
    );
    let processes = checked_command(
        "nvidia-smi",
        &[
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
    )?;
    for process in processes
        .lines()
        .filter(|line| line.starts_with(expected_uuid))
    {
        let pid = process
            .split(',')
            .nth(1)
            .context("invalid GPU process record")?
            .trim()
            .parse::<u32>()?;
        ensure!(
            pid == std::process::id() || is_descendant_of(pid, ollama_pid),
            "unrelated GPU process: {process}"
        );
    }
    Ok(json!({"gpus_csv": gpu, "compute_processes_csv": processes}))
}

fn read_log_after(path: &Path, offset: u64) -> Result<(String, u64)> {
    let mut file = File::open(path)?;
    ensure!(
        file.metadata()?.len() >= offset,
        "Ollama log rotated during trial"
    );
    file.seek(SeekFrom::Start(offset))?;
    let mut log = String::new();
    file.read_to_string(&mut log)?;
    Ok((log, file.stream_position()?))
}

async fn post_ollama(
    client: &reqwest::Client,
    args: &Args,
    prompt: &str,
    max_tokens: usize,
) -> Result<(OllamaResponse, f64)> {
    let request = json!({
        "model": args.ollama_model, "prompt": prompt, "raw": true, "stream": false,
        "keep_alive": "30m", "options": {
            "num_ctx": CONTEXT_TOKENS, "num_predict": max_tokens, "num_gpu": 99,
            "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
            "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "seed": 0,
        },
    });
    let started_at = Instant::now();
    let response = client
        .post(format!("{}/api/generate", args.ollama_url))
        .json(&request)
        .send()
        .await?;
    let status = response.status();
    let bytes = response.bytes().await?;
    let wall_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    ensure!(
        status.is_success(),
        "Ollama {status}: {}",
        String::from_utf8_lossy(&bytes)
    );
    let response: OllamaResponse = serde_json::from_slice(&bytes)?;
    ensure!(
        response.done,
        "Ollama returned an incomplete buffered response"
    );
    Ok((response, wall_ms))
}

async fn start_server(model: LoadedModel) -> Result<HttpServer> {
    start_server_with_prefill(model, PromptPrefill::Environment).await
}

async fn start_server_with_prefill(
    model: LoadedModel,
    prefill: PromptPrefill,
) -> Result<HttpServer> {
    let mut state = AppState::new_with_kv_config(model, KvCompressionConfig::dense_reference());
    state.prompt_prefill = prefill;
    let state = Arc::new(state);
    ensure!(
        !state.decode_batching_requested,
        "single-request comparison does not use the scheduler"
    );
    let router = app_router(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let url = format!("http://{}/generate", listener.local_addr()?);
    let task = tokio::spawn(async move { axum::serve(listener, router.into_make_service()).await });
    Ok(HttpServer { url, state, task })
}

#[derive(Deserialize)]
struct TokenizerResponse {
    tokens: Vec<u32>,
}

async fn tokenize_with_oracle(
    client: &reqwest::Client,
    runner_url: &str,
    content: &str,
) -> Result<Vec<u32>> {
    let response: TokenizerResponse = client
        .post(format!("{runner_url}/tokenize"))
        .json(&json!({"content": content, "add_special": false, "parse_special": true}))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(response.tokens)
}

impl ComparisonRun {
    async fn measure_m40(&mut self, phase: &str, pair: usize) -> Result<Measurement> {
        self.measure_m40_variant(phase, pair, ServerVariant::Control)
            .await
    }

    async fn measure_m40_variant(
        &mut self,
        phase: &str,
        pair: usize,
        variant: ServerVariant,
    ) -> Result<Measurement> {
        let server = match variant {
            ServerVariant::Control => &self.server,
            ServerVariant::Prepared => self
                .prepared_server
                .as_ref()
                .context("prepared server not loaded")?,
        };
        let requested_runtime = match variant {
            ServerVariant::Control => self.args.primary_runtime_config(),
            ServerVariant::Prepared => self.args.candidate_runtime_config(),
        };
        if let Some(config) = requested_runtime {
            config.validate_loaded_model(&server.state.model)?;
        }
        let is_stage_profile_enabled =
            std::env::var("M40LLM_STAGE_PROFILE").ok().as_deref() == Some("1");
        let before = gpu_snapshot(self.args.ollama_pid, &self.gpu_uuid)?;
        let counters_before = m40_llm::profile::snapshot();
        let started_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
        let started_at = Instant::now();
        let response = self
            .client
            .post(&server.url)
            .json(&json!({
                "prompt": self.prompt, "prompt_format": "raw", "max_tokens": MAX_TOKENS,
                "top_k": 1, "seed": 0, "include_metrics": true,
            }))
            .send()
            .await?;
        let status = response.status();
        let bytes = response.bytes().await?;
        let wall_ms = started_at.elapsed().as_secs_f64() * 1000.0;
        let finished_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
        let counters_after = m40_llm::profile::snapshot();
        ensure!(
            status.is_success(),
            "m40-llm {status}: {}",
            String::from_utf8_lossy(&bytes)
        );
        let response: GenerateResponse = serde_json::from_slice(&bytes)?;
        let metrics: GenerationMetrics = response
            .metrics
            .context("server omitted requested metrics")?;
        let stop = match metrics.stop_reason {
            m40_llm::decode::StopReason::Length => "length",
            m40_llm::decode::StopReason::Stop => "stop",
        };
        let validation = protocol::validate_fixed_workload(
            metrics.prompt_tokens,
            metrics.generated_tokens,
            stop,
            self.prompt_ids.len(),
        )
        .and_then(|()| {
            if requested_runtime.is_some_and(|config| {
                config.attention_projection == AttentionProjectionChoice::Warp256
            }) {
                let layers = u64::from(server.state.model.model_config.block_count);
                let forwards = metrics.generated_tokens as u64;
                for op in ["qkv_f32xq8_0_gguf_warp", "gemm_f32xq8_0_gguf_coltile4_warp"] {
                    let launches = counters_after.by_op.get(op).map_or(0, |v| v.launches)
                        - counters_before.by_op.get(op).map_or(0, |v| v.launches);
                    ensure!(
                        launches == layers * forwards,
                        "requested projection backend did not execute every one-row layer forward"
                    );
                }
            }
            if let Some(evidence) = &self.selected_quality {
                ensure!(
                    evidence.expected_output == response.output,
                    "timed runtime output differs from completed quality evidence"
                );
            }
            Ok(())
        });
        let tps = metrics.generated_tokens as f64 * 1000.0 / wall_ms;
        let execution = server.state.model.q8_execution_stats();
        let (activation_bytes, temporary_weight_bytes) = server
            .state
            .model
            .forward_workspace
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned forward workspace"))?
            .as_ref()
            .map(|workspace| {
                let pointers = workspace.ptrs();
                (
                    pointers.q8_activation.map_or(0, |s| s.bytes),
                    pointers.q8_prefill_weight.map_or(0, |s| s.bytes),
                )
            })
            .unwrap_or((0, 0));
        self.report.record(json!({
            "kind": "trial", "runtime": "m40-llm", "phase": phase, "pair": pair,
            "model_variant": variant.as_str(),
            "client_started_unix_ms": started_unix_ms, "client_finished_unix_ms": finished_unix_ms,
            "q8_execution_mode": execution.mode, "q8_execution_families": execution.families,
            "q8_prepared_bytes": execution.resident_bytes,
            "q8_ffn":server.state.model.q8_ffn,"q8_prefill_backend":server.state.model.q8_prefill,
            "q8_attention_projection_backend":server.state.model.q8_attention_projection.backend(),
            "prompt_prefill":server.state.prompt_prefill,
            "resident_q8_activation_scratch_bytes_after_request":activation_bytes,
            "resident_prefill_weight_scratch_bytes_after_request":temporary_weight_bytes,
            "scratch_scope":"actual remaining workspace capacity; released prefill temporary capacity is recorded by the matching untimed token trace",
            "enqueue_counters":m40_llm::profile::counter_delta(&counters_before,&counters_after),
            "launch_deltas":counters_after.by_op.iter().filter_map(|(name,counts)| {
                let delta=counts.launches-counters_before.by_op.get(name).map_or(0,|v|v.launches);
                (delta>0).then_some((*name,delta))
            }).collect::<BTreeMap<_,_>>(),
            "client_elapsed_ms": wall_ms, "e2e_tps": tps, "metrics": metrics,
            "stage_profile_enabled": is_stage_profile_enabled,
            "head128_kernel": std::env::var("M40LLM_ATTN_HEAD128_KERNEL").ok(),
            "head128_partitions": std::env::var("M40LLM_ATTN_HEAD128_PARTITIONS").ok(),
            "head128_threads": std::env::var("M40LLM_ATTN_HEAD128_THREADS").ok(),
            "head128_workspace_bytes": server.state.model.kv_cache.as_ref().map(|kv| kv.head128_attention_workspace_bytes()),
            "is_fixed_workload": validation.is_ok(), "invalid_reason": validation.as_ref().err().map(|e| e.to_string()),
            "matches_completed_quality_output":self.selected_quality.as_ref().map(|e|e.expected_output==response.output),
            "cache": {"reused_prompt_tokens": 0, "evaluated_prompt_tokens": metrics.prompt_tokens, "reset": "generate_text reset_kv_cache=true"},
            "gpu_before": before, "gpu_after": gpu_snapshot(self.args.ollama_pid, &self.gpu_uuid)?,
            "output": response.output, "output_sha256": sha256_bytes(response.output.as_bytes())?,
            "final_kv_bytes": server.state.model.kv_cache.as_ref().map(|kv| kv.actual_bytes()),
        }))?;
        Ok(Measurement {
            tps,
            output: response.output,
            is_fixed_workload: validation.is_ok(),
        })
    }

    async fn measure_ollama(&mut self, phase: &str, pair: usize) -> Result<Measurement> {
        post_ollama(&self.client, &self.args, protocol::REPLACEMENT_PROMPT, 1).await?;
        let offset = std::fs::metadata(&self.args.ollama_log)?.len();
        let before = gpu_snapshot(self.args.ollama_pid, &self.gpu_uuid)?;
        let (response, wall_ms) =
            post_ollama(&self.client, &self.args, &self.prompt, MAX_TOKENS).await?;
        let (log, log_end) = read_log_after(&self.args.ollama_log, offset)?;
        let cache: Result<CacheEvidence> =
            protocol::verify_cache_reset(&log, self.prompt_ids.len());
        let validation = protocol::validate_fixed_workload(
            response.prompt_eval_count,
            response.eval_count,
            &response.done_reason,
            self.prompt_ids.len(),
        );
        let is_valid =
            validation.is_ok() && cache.is_ok() && response.prompt_eval_cached_count == Some(0);
        let tps = response.eval_count as f64 * 1000.0 / wall_ms;
        self.report.record(json!({
            "kind": "trial", "runtime": "ollama", "phase": phase, "pair": pair,
            "client_elapsed_ms": wall_ms, "e2e_tps": tps,
            "prompt_tokens": response.prompt_eval_count, "generated_tokens": response.eval_count,
            "stop_reason": response.done_reason, "total_elapsed_ms": response.total_duration as f64 / 1e6,
            "load_elapsed_ms": response.load_duration as f64 / 1e6,
            "prefill_elapsed_ms": response.prompt_eval_duration as f64 / 1e6,
            "decode_elapsed_ms": response.eval_duration as f64 / 1e6,
            "is_fixed_workload": is_valid, "workload_error": validation.err().map(|e| e.to_string()),
            "cache": cache.as_ref().ok(), "cache_error": cache.as_ref().err().map(|e| e.to_string()),
            "prompt_eval_cached_count": response.prompt_eval_cached_count,
            "runner_log": self.args.ollama_log, "runner_log_offset": offset, "runner_log_end": log_end,
            "gpu_before": before, "gpu_after": gpu_snapshot(self.args.ollama_pid, &self.gpu_uuid)?,
            "output": response.response, "output_sha256": sha256_bytes(response.response.as_bytes())?,
        }))?;
        Ok(Measurement {
            tps,
            output: response.response,
            is_fixed_workload: is_valid,
        })
    }

    async fn run_pairs(
        &mut self,
        phase: &str,
        start: usize,
        end: usize,
        expected_outputs: &(String, String),
        pairs: &mut Vec<Pair>,
    ) -> Result<(String, String)> {
        let mut last_outputs = (String::new(), String::new());
        for pair in start..end {
            let (m40, ollama) = if pair.is_multiple_of(2) {
                let control = self.measure_ollama(phase, pair).await?;
                let candidate = self.measure_m40(phase, pair).await?;
                (candidate, control)
            } else {
                let candidate = self.measure_m40(phase, pair).await?;
                let control = self.measure_ollama(phase, pair).await?;
                (candidate, control)
            };
            ensure!(
                m40.is_fixed_workload && ollama.is_fixed_workload,
                "invalid pair {pair}; raw rows retained, no fixed-workload claim permitted"
            );
            ensure!(
                m40.output == expected_outputs.0 && ollama.output == expected_outputs.1,
                "engine output changed after warmup; raw rows retained"
            );
            eprintln!(
                "[{phase} pair {pair}] m40={:.3} ollama={:.3} HTTP E2E tok/s",
                m40.tps, ollama.tps
            );
            pairs.push(Pair {
                control_tps: ollama.tps,
                candidate_tps: m40.tps,
            });
            last_outputs = (m40.output, ollama.output);
        }
        Ok(last_outputs)
    }

    async fn capture_m40_trace(&mut self, expected_output: &str) -> Result<Vec<u32>> {
        self.capture_m40_trace_variant(expected_output, ServerVariant::Control)
            .await
    }

    async fn capture_m40_trace_variant(
        &mut self,
        expected_output: &str,
        variant: ServerVariant,
    ) -> Result<Vec<u32>> {
        let server = match variant {
            ServerVariant::Control => &self.server,
            ServerVariant::Prepared => self
                .prepared_server
                .as_ref()
                .context("prepared server not loaded")?,
        };
        let _generation_guard = server.state.generation_lock.lock().await;
        let generated = generate_text(
            &server.state.model,
            GenerateOptions {
                prompt: self.prompt.clone(),
                prompt_format: PromptFormat::Raw,
                max_tokens: Some(MAX_TOKENS),
                top_k: Some(1),
                seed: Some(0),
                kv_compression: KvCompressionConfig::dense_reference(),
                prompt_prefill: server.state.prompt_prefill,
                ..Default::default()
            },
        )?;
        ensure!(
            generated.output == expected_output,
            "untimed m40 continuation differs from measured configuration"
        );
        ensure!(
            generated.prompt_token_len <= generated.token_ids.len(),
            "invalid token trace bounds"
        );
        protocol::validate_prompt_ids(
            &self.prompt_ids,
            &generated.token_ids[..generated.prompt_token_len],
        )?;
        ensure!(
            generated.token_ids.len() - generated.prompt_token_len == MAX_TOKENS,
            "untimed m40 trace omitted fixed-workload generated IDs"
        );
        if let Some(evidence) = &self.selected_quality {
            evidence.verify_generation(
                &generated.token_ids[generated.prompt_token_len..],
                &generated.output,
            )?;
        }
        self.report.record(json!({
            "kind": "token_trace", "runtime": "m40-llm", "is_timed": false,
            "model_variant": variant.as_str(),
            "prompt_token_ids": &generated.token_ids[..generated.prompt_token_len],
            "generated_token_ids": &generated.token_ids[generated.prompt_token_len..],
            "matches_measured_output": true, "output": generated.output,
            "prefill_mode":generated.prefill_mode,
            "prefill_chunk_size":generated.prefill_chunk_size,
            "q8_prefill_temporary_weight_bytes":generated.q8_prefill_temporary_weight_bytes,
            "q8_ffn":server.state.model.q8_ffn,"q8_prefill_backend":server.state.model.q8_prefill,
            "stage_profile_enabled": std::env::var("M40LLM_STAGE_PROFILE").ok().as_deref() == Some("1"),
            "head128_kernel": std::env::var("M40LLM_ATTN_HEAD128_KERNEL").ok(),
        }))?;
        Ok(generated.token_ids[generated.prompt_token_len..].to_vec())
    }

    async fn capture_traces(&mut self, last_outputs: &(String, String)) -> Result<Vec<u32>> {
        let m40_ids = self.capture_m40_trace(&last_outputs.0).await?;
        let response: Value = self
            .client
            .post(format!("{}/completion", self.runner_url))
            .json(&json!({
                "prompt": self.prompt, "stream": false, "cache_prompt": false,
                "return_tokens": true, "n_predict": MAX_TOKENS,
                "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
                "repeat_penalty": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0, "seed": 0,
            }))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        ensure!(
            response["content"].as_str() == Some(last_outputs.1.as_str()),
            "untimed Ollama continuation differs from measured configuration"
        );
        let tokens: Vec<u32> = serde_json::from_value(response["tokens"].clone())?;
        ensure!(
            tokens.len() == MAX_TOKENS,
            "native runner omitted complete generated-token trace"
        );
        ensure!(
            response["generation_settings"]["ignore_eos"] == false,
            "native trace suppresses EOS"
        );
        self.report.record(json!({
            "kind": "token_trace", "runtime": "ollama", "is_timed": false,
            "prompt_token_ids": self.prompt_ids, "generated_token_ids": tokens,
            "matches_measured_output": true, "output": response["content"],
            "trace_source": "same Ollama native runner /completion return_tokens=true",
            "generation_settings": response["generation_settings"],
        }))?;
        Ok(m40_ids)
    }

    async fn run_attention_pairs(
        &mut self,
        switch: &AttentionSwitch,
        phase: &str,
        start: usize,
        end: usize,
        pairs: &mut Vec<Pair>,
    ) -> Result<(String, String)> {
        let mut outputs = (String::new(), String::new());
        for pair in start..end {
            let (control, candidate) = if pair.is_multiple_of(2) {
                switch.set(false);
                let control = self.measure_m40(&format!("{phase}-legacy"), pair).await?;
                switch.set(true);
                let candidate = self.measure_m40(&format!("{phase}-parallel"), pair).await?;
                (control, candidate)
            } else {
                switch.set(true);
                let candidate = self.measure_m40(&format!("{phase}-parallel"), pair).await?;
                switch.set(false);
                let control = self.measure_m40(&format!("{phase}-legacy"), pair).await?;
                (control, candidate)
            };
            ensure!(
                control.is_fixed_workload && candidate.is_fixed_workload,
                "invalid attention comparison pair; raw rows retained"
            );
            eprintln!("[attention {phase} pair {pair}] legacy={:.3} parallel={:.3} HTTP E2E tok/s, same_output={}",
                control.tps, candidate.tps, control.output == candidate.output);
            pairs.push(Pair {
                control_tps: control.tps,
                candidate_tps: candidate.tps,
            });
            outputs = (control.output, candidate.output);
        }
        Ok(outputs)
    }

    fn record_attention_comparison(
        &mut self,
        phase: &str,
        comparison: &protocol::Comparison,
    ) -> Result<()> {
        let mut summary = serde_json::to_value(comparison)?;
        // Here the control is m40-llm, not Ollama. Do not reinterpret the
        // protocol's 1.30x-Ollama target field against this different control.
        summary
            .as_object_mut()
            .context("comparison object")?
            .remove("reaches_original_target");
        self.report
            .record(json!({"kind":"attention_comparison_summary", "phase":phase,
            "comparison":summary,"is_ollama_comparison":false,
            "requires_quality_parity_for_promotion":true}))
    }

    async fn compare_head128_attention(&mut self) -> Result<()> {
        let switch = AttentionSwitch::new();
        for is_parallel in [false, true] {
            switch.set(is_parallel);
            ensure!(
                self.measure_m40(
                    if is_parallel {
                        "attention-warmup-parallel"
                    } else {
                        "attention-warmup-legacy"
                    },
                    0
                )
                .await?
                .is_fixed_workload,
                "invalid attention warmup"
            );
        }
        let mut pairs = Vec::new();
        let mut outputs = self
            .run_attention_pairs(&switch, "screen", 0, 5, &mut pairs)
            .await?;
        let mut comparison = protocol::compare_pairs(&pairs)?;
        self.record_attention_comparison("screen5", &comparison)?;
        let mut next = protocol::next_trial_count(&comparison, false)?;
        if next == Some(10) {
            outputs = self
                .run_attention_pairs(&switch, "screen", 5, 10, &mut pairs)
                .await?;
            comparison = protocol::compare_pairs(&pairs)?;
            self.record_attention_comparison("screen10", &comparison)?;
            next = protocol::next_trial_count(&comparison, false)?;
        }
        if next == Some(20) {
            let mut confirmation = Vec::new();
            outputs = self
                .run_attention_pairs(&switch, "confirmation", 0, 20, &mut confirmation)
                .await?;
            comparison = protocol::compare_pairs(&confirmation)?;
            protocol::next_trial_count(&comparison, true)?;
            self.record_attention_comparison("confirmation20", &comparison)?;
        }
        switch.set(false);
        let control_ids = self.capture_m40_trace(&outputs.0).await?;
        switch.set(true);
        let candidate_ids = self.capture_m40_trace(&outputs.1).await?;
        self.report.record(json!({"kind":"attention_token_comparison",
            "is_equal":control_ids==candidate_ids,
            "first_different_generated_step":control_ids.iter().zip(&candidate_ids).position(|(a,b)|a!=b),
            "different_id_count":control_ids.iter().zip(&candidate_ids).filter(|(a,b)|a!=b).count()}))?;
        Ok(())
    }

    async fn compare_stage_profile(&mut self) -> Result<()> {
        ensure!(
            std::env::var_os("M40LLM_STAGE_PROFILE_REPORT").is_some(),
            "set M40LLM_STAGE_PROFILE_REPORT for the profile comparison"
        );
        if self.args.should_profile_runtime {
            let path =
                std::env::var_os("M40LLM_STAGE_PROFILE_REPORT").context("stage profile path")?;
            ensure!(
                !Path::new(&path).exists(),
                "selected-runtime stage report already exists; choose a new path"
            );
        }
        let switch = StageProfileSwitch::new();
        switch.set(false);
        ensure!(
            self.measure_m40("profile-warmup-off", 0)
                .await?
                .is_fixed_workload,
            "invalid unprofiled warmup"
        );
        switch.set(true);
        ensure!(
            self.measure_m40("profile-warmup-on", 0)
                .await?
                .is_fixed_workload,
            "invalid profiled warmup"
        );
        let mut ordinary = Vec::new();
        let mut profiled = Vec::new();
        let mut last_output = String::new();
        for pair in 0_usize..5 {
            let (plain, traced) = if pair.is_multiple_of(2) {
                switch.set(false);
                let plain = self.measure_m40("profile-off", pair).await?;
                switch.set(true);
                let traced = self.measure_m40("profile-on", pair).await?;
                (plain, traced)
            } else {
                switch.set(true);
                let traced = self.measure_m40("profile-on", pair).await?;
                switch.set(false);
                let plain = self.measure_m40("profile-off", pair).await?;
                (plain, traced)
            };
            ensure!(
                plain.is_fixed_workload && traced.is_fixed_workload,
                "invalid stage-profile pair"
            );
            ensure!(
                plain.output == traced.output,
                "stage profiling changed generated output"
            );
            eprintln!(
                "[stage-profile pair {pair}] off={:.3} on={:.3} HTTP E2E tok/s",
                plain.tps, traced.tps
            );
            ordinary.push(plain.tps);
            profiled.push(traced.tps);
            last_output = plain.output;
        }
        let ordinary = protocol::distribution(&ordinary)?;
        let profiled = protocol::distribution(&profiled)?;
        self.report.record(json!({
            "kind": "profile_comparison_summary", "pairs": 5,
            "profile_off": ordinary, "profile_on": profiled,
            "throughput_change": profiled.median / ordinary.median - 1.0,
            "report_path": std::env::var_os("M40LLM_STAGE_PROFILE_REPORT").map(PathBuf::from),
            "selected_runtime_config":self.args.primary_runtime_config(),
            "is_optimization_claim": false,
        }))?;
        switch.set(false);
        let plain_ids = self.capture_m40_trace(&last_output).await?;
        switch.set(true);
        let profiled_ids = self.capture_m40_trace(&last_output).await?;
        ensure!(
            plain_ids == profiled_ids,
            "stage profiling changed actual generated token IDs"
        );
        self.report
            .record(json!({"kind":"profile_token_comparison",
            "actual_ids_equal":true,"actual_generated_tokens":plain_ids.len(),
            "matches_completed_quality":self.selected_quality.as_ref().map(|_|true),
            "is_optimization_claim":false}))?;
        eprintln!(
            "Stage-profile comparison and token-ID parity complete: {}",
            self.args.output.display()
        );
        Ok(())
    }
}

pub async fn compare() -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "timing comparison requires a release Rust build"
    );
    let args = Args::parse();
    let candidate_q8_config = args.candidate_q8_config()?;
    let candidate_runtime = args.candidate_runtime_config();
    let primary_runtime = args.primary_runtime_config();
    ensure!(
        Path::new(&format!("/proc/{}", args.ollama_pid)).exists(),
        "Ollama PID is not live"
    );
    let ollama_environment: BTreeMap<String, String> =
        std::fs::read(format!("/proc/{}/environ", args.ollama_pid))?
            .split(|byte| *byte == 0)
            .filter_map(|entry| {
                let entry = std::str::from_utf8(entry).ok()?;
                let (key, value) = entry.split_once('=')?;
                [
                    "OLLAMA_NUM_PARALLEL",
                    "OLLAMA_MAX_LOADED_MODELS",
                    "OLLAMA_FLASH_ATTENTION",
                    "OLLAMA_KV_CACHE_TYPE",
                    "LLAMA_ARG_CACHE_RAM",
                    "CUDA_VISIBLE_DEVICES",
                ]
                .contains(&key)
                .then(|| (key.to_owned(), value.to_owned()))
            })
            .collect();
    ensure!(
        ollama_environment
            .get("OLLAMA_NUM_PARALLEL")
            .map(String::as_str)
            == Some("1"),
        "start Ollama with OLLAMA_NUM_PARALLEL=1"
    );
    ensure!(
        ollama_environment
            .get("LLAMA_ARG_CACHE_RAM")
            .map(String::as_str)
            == Some("0"),
        "start Ollama with LLAMA_ARG_CACHE_RAM=0 to disable saved prompt restoration"
    );
    let ollama_executable = std::fs::read_link(format!("/proc/{}/exe", args.ollama_pid))?;
    for key in [
        "M40LLM_SERVER_BATCH_DECODE",
        "M40LLM_DECODE_GRAPH",
        "M40LLM_TIMING_LOG",
        "M40LLM_ALLOC_LOG",
        "M40LLM_LAUNCH_LOG",
        "M40LLM_SYNC_LOG",
        "M40LLM_COPY_LOG",
        "M40LLM_Q8_DECODE_EVENT_LOG",
        "M40LLM_STAGE_PROFILE",
    ] {
        ensure!(
            std::env::var(key).map_or(true, |v| v == "0" || v.is_empty()),
            "disable {key} for controlled ordinary timing"
        );
    }
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?;
    let mut report = Report::create(&args.output)?;
    let environment: BTreeMap<String, String> = std::env::vars()
        .filter(|(k, _)| {
            k.starts_with("M40LLM_")
                || [
                    "CXX",
                    "NVCC_CCBIN",
                    "CUDAHOSTCXX",
                    "RUSTFLAGS",
                    "CUDA_VISIBLE_DEVICES",
                ]
                .contains(&k.as_str())
        })
        .collect();
    if candidate_runtime.is_some() {
        super::q8_runtime::validate_environment(&environment)?;
    }
    let ollama_version: Value = client
        .get(format!("{}/api/version", args.ollama_url))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let show: Value = client
        .post(format!("{}/api/show", args.ollama_url))
        .json(&json!({"model": args.ollama_model}))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let modelfile = show["modelfile"]
        .as_str()
        .context("Ollama show lacks modelfile")?;
    let ollama_model_path = modelfile
        .lines()
        .find_map(|line| line.strip_prefix("FROM "))
        .context("Ollama model blob path missing")?
        .trim_matches('"');
    let model_sha256 = sha256_file(&args.model)?;
    ensure!(
        model_sha256 == sha256_file(Path::new(ollama_model_path))?,
        "runtime model GGUF hashes differ"
    );
    let prompt = protocol::render_prompt();
    let gguf_model = gguf::load_gguf(&args.model)?;
    let tensor_manifest = serde_json::to_value(&gguf_model.tensors)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf_model.metadata)?;
    ensure!(
        tokenizer.is_qwen2(),
        "comparison requires Qwen2 tokenizer metadata"
    );
    let prompt_ids = tokenizer.encode_with_specials(&prompt, false, false)?;
    let replacement_ids =
        tokenizer.encode_with_specials(protocol::REPLACEMENT_PROMPT, false, false)?;
    protocol::validate_replacement_ids(&prompt_ids, &replacement_ids)?;
    let selected_quality = if args.should_time_ollama || args.should_profile_runtime {
        let evidence = runtime_quality::load(
            args.q8_runtime_quality_report
                .as_deref()
                .context("selected quality report")?,
            candidate_runtime.context("selected runtime")?,
            args.control_runtime_config(),
            &model_sha256,
            &prompt_ids,
        )?;
        ensure!(evidence.passes_exact_gate,
            "selected-runtime timing/profiling requires a passing exact quality report; use candidate-pair mode for numerical/failed-gate diagnostics");
        Some(evidence)
    } else {
        None
    };
    let load_start = Instant::now();
    let bytes = std::fs::read(&args.model)?;
    let control_execution = args.primary_execution_config()?;
    let mut model = if let Some(config) = primary_runtime {
        config.load_weights(gguf_model, bytes)?
    } else {
        LoadedModel::from_gguf_with_q8_execution(gguf_model, bytes, -1, control_execution)?
    };
    let props = model.cuda.current_device_props()?;
    ensure!(
        props.major == 5 && props.minor == 2,
        "selected device is not sm_52"
    );
    ensure!(
        model.model_config.context_length >= CONTEXT_TOKENS,
        "model context is too short"
    );
    model.allocate_kv_cache_for_layers(CONTEXT_TOKENS)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let gpu_uuid = checked_command(
        "nvidia-smi",
        &[
            "-i",
            &props.device_id.to_string(),
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
    )?;
    let gpu = gpu_snapshot(args.ollama_pid, &gpu_uuid)?;
    let mut manifest = json!({
        "kind": "manifest", "created_unix_s": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "model_path": args.model, "model_sha256": model_sha256, "ollama_model_path": ollama_model_path,
        "ollama_model": args.ollama_model, "ollama_version": ollama_version,
        "ollama_executable": ollama_executable, "ollama_executable_sha256": sha256_file(&ollama_executable)?,
        "ollama_environment": ollama_environment,
        "revision": checked_command("git", &["rev-parse", "HEAD"])?, "worktree": checked_command("git", &["status", "--short"])?,
        "benchmark_executable_sha256": sha256_file(&std::env::current_exe()?)?,
        "rustc": checked_command("rustc", &["--version"])?, "nvcc": checked_command("nvcc", &["--version"])?,
        "host_compiler": checked_command("g++", &["--version"])?, "build_profile": "release",
        "cuda_flags": ["-O3", "-gencode=arch=compute_52,code=sm_52", "-gencode=arch=compute_52,code=compute_52"],
        "environment": environment, "tensor_manifest": tensor_manifest, "model_config": format!("{:?}", model.model_config),
        "prompt": prompt, "user_prompt": protocol::USER_PROMPT, "prompt_sha256": sha256_bytes(prompt.as_bytes())?,
        "prompt_token_ids": prompt_ids, "prompt_token_ids_sha256": sha256_bytes(&serde_json::to_vec(&prompt_ids)?)?,
        "replacement_prompt": protocol::REPLACEMENT_PROMPT, "replacement_token_ids": replacement_ids,
        "m40_stop_token_ids": tokenizer.stop_ids(), "kv_mode": "off", "context_tokens": CONTEXT_TOKENS,
        "max_tokens": MAX_TOKENS, "sampling": "greedy argmax top_k=1, penalties=1/0/0, seed=0; Ollama temperature=0, m40 default temperature ignored by greedy path",
        "comparison_kind": if args.should_profile_runtime { "selected-runtime-stage-profile" } else if args.should_time_ollama { "selected-runtime-vs-ollama" } else if candidate_runtime.is_some() { "q8-runtime" } else if candidate_q8_config.is_some() { "q8-execution" } else if args.should_compare_stage_profile { "stage-profile-perturbation" } else if args.should_compare_head128 { "head128-attention" } else { "ollama-baseline" },
        "candidate_runtime_config":candidate_runtime,
        "actual_q8_attention_projection_backend":model.q8_attention_projection.backend(),
        "control_q8_execution": model.q8_execution_stats(), "candidate_q8_config": candidate_q8_config,
        "m40_cold_load_elapsed_ms": load_ms, "gpu_uuid": gpu_uuid, "gpu": gpu,
        "primary_boundary": "client send through complete buffered HTTP body, resident weights, prompt KV reset",
    });
    manifest["primary_runtime_config"] = serde_json::to_value(primary_runtime)?;
    manifest["quality_control_runtime_config"] =
        serde_json::to_value(args.control_runtime_config())?;
    manifest["prefill_execution_contract"] = json!(super::q8_runtime::PREFILL_EXECUTION_CONTRACT);
    report.record(manifest)?;
    if let Some(evidence) = &selected_quality {
        report.record(evidence.record.clone())?;
    }
    let (oracle, _) = post_ollama(&client, &args, &prompt, 1).await?;
    let startup_log = std::fs::read_to_string(&args.ollama_log)?;
    let runner_url = protocol::discover_runner_url(&startup_log)?;
    let oracle_prompt_ids = tokenize_with_oracle(&client, &runner_url, &prompt).await?;
    let oracle_replacement_ids =
        tokenize_with_oracle(&client, &runner_url, protocol::REPLACEMENT_PROMPT).await?;
    protocol::validate_prompt_ids(&prompt_ids, &oracle_prompt_ids)?;
    protocol::validate_prompt_ids(&replacement_ids, &oracle_replacement_ids)?;
    ensure!(
        oracle.prompt_eval_count == prompt_ids.len(),
        "actual raw prompt processing differs from tokenizer oracle"
    );
    let layer_count = model.model_config.block_count + 1;
    ensure!(
        startup_log.contains(&format!(
            "offloaded {layer_count}/{layer_count} layers to GPU"
        )),
        "Ollama full-model GPU offload is not verified"
    );
    ensure!(
        startup_log.contains("K (f16):") && startup_log.contains("V (f16):"),
        "Ollama dense FP16 K/V is not verified"
    );
    let offload_lines: Vec<&str> = startup_log
        .lines()
        .filter(|line| {
            line.contains("offloaded ")
                || line.contains("KV buffer size")
                || line.contains("inference compute")
        })
        .collect();
    report.record(json!({"kind": "ollama_backend_verification", "runner_lines": offload_lines, "gpu": gpu_snapshot(args.ollama_pid, &gpu_uuid)?}))?;
    report.record(json!({"kind": "tokenizer_oracle", "prompt_token_ids": oracle_prompt_ids, "replacement_token_ids": oracle_replacement_ids, "is_equal": true, "cold_ollama_load_ms": oracle.load_duration as f64 / 1e6, "source": runner_url}))?;
    let server = if let Some(config) = primary_runtime {
        start_server_with_prefill(model, config.prompt_prefill()).await?
    } else {
        start_server(model).await?
    };
    let mut run = ComparisonRun {
        args,
        client,
        server,
        prepared_server: None,
        prompt,
        prompt_ids,
        model_sha256,
        gpu_uuid,
        runner_url,
        report,
        selected_quality,
    };
    if run.args.should_profile_runtime {
        return run.compare_stage_profile().await;
    }
    if run.args.q8_execution_mode.is_some()
        || (run.args.q8_runtime_candidate.is_some() && !run.args.should_time_ollama)
    {
        return run.compare_q8_execution().await;
    }
    if run.args.should_compare_stage_profile {
        return run.compare_stage_profile().await;
    }
    if run.args.should_compare_head128 {
        return run.compare_head128_attention().await;
    }
    let m40_warmup = run.measure_m40("warmup", 0).await?;
    ensure!(
        m40_warmup.is_fixed_workload,
        "m40 warmup did not match the fixed workload; inspect retained row before timing"
    );
    let ollama_warmup = run.measure_ollama("warmup", 0).await?;
    ensure!(
        ollama_warmup.is_fixed_workload,
        "Ollama warmup did not verify the workload/cache reset; inspect retained row before timing"
    );
    let expected_outputs = (m40_warmup.output, ollama_warmup.output);
    let mut pairs = Vec::new();
    let mut last_outputs = run
        .run_pairs("screen", 0, 5, &expected_outputs, &mut pairs)
        .await?;
    let mut comparison = protocol::compare_pairs(&pairs)?;
    run.report
        .record(json!({"kind": "summary", "phase": "screen5", "comparison": comparison}))?;
    let mut next = protocol::next_trial_count(&comparison, false)?;
    if next == Some(10) {
        last_outputs = run
            .run_pairs("screen", 5, 10, &expected_outputs, &mut pairs)
            .await?;
        comparison = protocol::compare_pairs(&pairs)?;
        run.report
            .record(json!({"kind": "summary", "phase": "screen10", "comparison": comparison}))?;
        next = protocol::next_trial_count(&comparison, false)?;
    }
    if next == Some(20) {
        eprintln!("Borderline screen; collecting exactly twenty fresh confirmation pairs.");
        let mut confirmation = Vec::new();
        last_outputs = run
            .run_pairs("confirmation", 0, 20, &expected_outputs, &mut confirmation)
            .await?;
        comparison = protocol::compare_pairs(&confirmation)?;
        protocol::next_trial_count(&comparison, true)?;
        run.report.record(
            json!({"kind": "summary", "phase": "confirmation20", "comparison": comparison}),
        )?;
    }
    let m40_ids = run.capture_traces(&last_outputs).await?;
    if let Some(evidence) = &run.selected_quality {
        evidence.verify_generation(&m40_ids, &last_outputs.0)?;
        run.report.record(json!({"kind":"selected_runtime_ollama_qualification",
            "runtime_config":candidate_runtime,"passes_completed_exact_quality":true,
            "actual_ids_match_completed_quality":true,"actual_generated_tokens":m40_ids.len(),
            "reaches_original_target":comparison.reaches_original_target,
            "required_absolute_http_tps":53.06,"required_relative_http_tps":1.30*comparison.control.median,
            "changes_runtime_defaults":false}))?;
    }
    eprintln!("{}", serde_json::to_string_pretty(&comparison)?);
    eprintln!("Raw evidence: {}", run.args.output.display());
    if comparison.reaches_original_target {
        eprintln!("Both unchanged-model throughput targets reached; broader quality audit still required.");
    }
    Ok(())
}

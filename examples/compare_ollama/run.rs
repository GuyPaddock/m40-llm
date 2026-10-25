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
    generate::{generate_text, GenerateOptions, PromptFormat},
    generation_metrics::GenerationMetrics,
    gguf,
    infer::LoadedModel,
    kv_compression::KvCompressionConfig,
    server::{app_router, AppState, GenerateResponse},
    tokenizer::Tokenizer,
};
use serde::Deserialize;
use serde_json::{json, Value};

use super::protocol::{self, CacheEvidence, Pair, CONTEXT_TOKENS, MAX_TOKENS};

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
    prompt: String,
    prompt_ids: Vec<u32>,
    gpu_uuid: String,
    runner_url: String,
    report: Report,
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
    let state = Arc::new(AppState::new_with_kv_config(
        model,
        KvCompressionConfig::dense_reference(),
    ));
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
        let before = gpu_snapshot(self.args.ollama_pid, &self.gpu_uuid)?;
        let started_at = Instant::now();
        let response = self
            .client
            .post(&self.server.url)
            .json(&json!({
                "prompt": self.prompt, "prompt_format": "raw", "max_tokens": MAX_TOKENS,
                "top_k": 1, "seed": 0, "include_metrics": true,
            }))
            .send()
            .await?;
        let status = response.status();
        let bytes = response.bytes().await?;
        let wall_ms = started_at.elapsed().as_secs_f64() * 1000.0;
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
        );
        let tps = metrics.generated_tokens as f64 * 1000.0 / wall_ms;
        self.report.record(json!({
            "kind": "trial", "runtime": "m40-llm", "phase": phase, "pair": pair,
            "client_elapsed_ms": wall_ms, "e2e_tps": tps, "metrics": metrics,
            "is_fixed_workload": validation.is_ok(), "invalid_reason": validation.as_ref().err().map(|e| e.to_string()),
            "cache": {"reused_prompt_tokens": 0, "evaluated_prompt_tokens": metrics.prompt_tokens, "reset": "generate_text reset_kv_cache=true"},
            "gpu_before": before, "gpu_after": gpu_snapshot(self.args.ollama_pid, &self.gpu_uuid)?,
            "output": response.output, "output_sha256": sha256_bytes(response.output.as_bytes())?,
            "final_kv_bytes": self.server.state.model.kv_cache.as_ref().map(|kv| kv.actual_bytes()),
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

    async fn capture_traces(&mut self, last_outputs: &(String, String)) -> Result<()> {
        let _generation_guard = self.server.state.generation_lock.lock().await;
        let generated = generate_text(
            &self.server.state.model,
            GenerateOptions {
                prompt: self.prompt.clone(),
                prompt_format: PromptFormat::Raw,
                max_tokens: Some(MAX_TOKENS),
                top_k: Some(1),
                seed: Some(0),
                kv_compression: KvCompressionConfig::dense_reference(),
                ..Default::default()
            },
        )?;
        ensure!(
            generated.output == last_outputs.0,
            "untimed m40 continuation differs from measured configuration"
        );
        self.report.record(json!({
            "kind": "token_trace", "runtime": "m40-llm", "is_timed": false,
            "prompt_token_ids": &generated.token_ids[..generated.prompt_token_len],
            "generated_token_ids": &generated.token_ids[generated.prompt_token_len..],
            "matches_measured_output": true, "output": generated.output,
        }))?;
        drop(_generation_guard);
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
        Ok(())
    }
}

pub async fn compare() -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "timing comparison requires a release Rust build"
    );
    let args = Args::parse();
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
    let load_start = Instant::now();
    let bytes = std::fs::read(&args.model)?;
    let mut model = LoadedModel::from_gguf(gguf_model, bytes, -1)?;
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
    report.record(json!({
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
        "m40_cold_load_elapsed_ms": load_ms, "gpu_uuid": gpu_uuid, "gpu": gpu,
        "primary_boundary": "client send through complete buffered HTTP body, resident weights, prompt KV reset",
    }))?;
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
    let server = start_server(model).await?;
    let mut run = ComparisonRun {
        args,
        client,
        server,
        prompt,
        prompt_ids,
        gpu_uuid,
        runner_url,
        report,
    };
    ensure!(
        run.measure_m40("warmup", 0).await?.is_fixed_workload,
        "m40 warmup did not match the fixed workload; inspect retained row before timing"
    );
    ensure!(
        run.measure_ollama("warmup", 0).await?.is_fixed_workload,
        "Ollama warmup did not verify the workload/cache reset; inspect retained row before timing"
    );
    let mut pairs = Vec::new();
    let mut last_outputs = run.run_pairs("screen", 0, 5, &mut pairs).await?;
    let mut comparison = protocol::compare_pairs(&pairs)?;
    run.report
        .record(json!({"kind": "summary", "phase": "screen5", "comparison": comparison}))?;
    let mut next = protocol::next_trial_count(&comparison, false)?;
    if next == Some(10) {
        last_outputs = run.run_pairs("screen", 5, 10, &mut pairs).await?;
        comparison = protocol::compare_pairs(&pairs)?;
        run.report
            .record(json!({"kind": "summary", "phase": "screen10", "comparison": comparison}))?;
        next = protocol::next_trial_count(&comparison, false)?;
    }
    if next == Some(20) {
        eprintln!("Borderline screen; collecting exactly twenty fresh confirmation pairs.");
        let mut confirmation = Vec::new();
        last_outputs = run
            .run_pairs("confirmation", 0, 20, &mut confirmation)
            .await?;
        comparison = protocol::compare_pairs(&confirmation)?;
        protocol::next_trial_count(&comparison, true)?;
        run.report.record(
            json!({"kind": "summary", "phase": "confirmation20", "comparison": comparison}),
        )?;
    }
    run.capture_traces(&last_outputs).await?;
    eprintln!("{}", serde_json::to_string_pretty(&comparison)?);
    eprintln!("Raw evidence: {}", run.args.output.display());
    if comparison.reaches_original_target {
        eprintln!("Both unchanged-model throughput targets reached; broader quality audit still required.");
    }
    Ok(())
}

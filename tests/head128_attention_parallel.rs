#![cfg(all(feature = "cuda", nvcc))]

use std::{fs::OpenOptions, io::Write, process::Command, sync::Mutex, time::Instant};

use anyhow::{ensure, Result};
use half::f16;
use m40_llm::{
    cuda::{
        CudaContext, CudaStream, DeviceBuffer, Head128AttentionConfig, Head128AttentionWorkspace,
        KVCache,
    },
    profile,
};

static TEST_LOCK: Mutex<()> = Mutex::new(());
const Q_HEADS: u32 = 16;
const KV_HEADS: usize = 2;
const DIM: usize = 128;

struct EnvRestore {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvRestore {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        std::env::set_var(key, value);
        Self { key, previous }
    }
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => std::env::set_var(self.key, value),
            None => std::env::remove_var(self.key),
        }
    }
}

fn rotate_absolute(values: &mut [f32], position: usize) {
    for head in values.chunks_exact_mut(DIM) {
        for d in 0..DIM / 2 {
            let angle = position as f64 / 1_000_000_f64.powf((2 * d) as f64 / DIM as f64);
            let (sin, cos) = angle.sin_cos();
            let a = head[d] as f64;
            let b = head[d + DIM / 2] as f64;
            head[d] = (a * cos - b * sin) as f32;
            head[d + DIM / 2] = (a * sin + b * cos) as f32;
        }
    }
}

struct Fixture {
    ctx: CudaContext,
    kv: KVCache,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    q: DeviceBuffer,
    output: DeviceBuffer,
    reference: DeviceBuffer,
    workspace: Head128AttentionWorkspace,
}

impl Fixture {
    fn new(valid_tokens: usize) -> Result<Self> {
        let ctx = CudaContext::new(-1)?;
        let props = ctx.current_device_props()?;
        ensure!(
            props.major == 5 && props.minor == 2,
            "head128 tests require an M40"
        );
        let kv = KVCache::new_with_context(
            &ctx,
            valid_tokens as u32 + 3,
            3,
            KV_HEADS as u32,
            DIM as u32,
        )?;
        let k_device = DeviceBuffer::new_tagged(&ctx, KV_HEADS * DIM * 4, "head128_test:k")?;
        let v_device = DeviceBuffer::new_tagged(&ctx, KV_HEADS * DIM * 4, "head128_test:v")?;
        let mut keys = vec![Vec::new(); 3];
        let mut values = vec![Vec::new(); 3];
        for slot in [0, 2] {
            for token in 0..=valid_tokens {
                let mut k: Vec<f32> = (0..KV_HEADS * DIM)
                    .map(|d| ((token * 17 + d * 13 + slot * 19) % 257) as f32 / 128.0 - 1.0)
                    .collect();
                let mut v: Vec<f32> = (0..KV_HEADS * DIM)
                    .map(|d| ((token * 31 + d * 7 + slot * 43) % 251) as f32 / 125.0 - 1.0)
                    .collect();
                rotate_absolute(&mut k, token + 17);
                if token == valid_tokens {
                    k.fill(f32::NAN);
                    v.fill(f32::NAN);
                } else {
                    keys[slot].extend(k.iter().map(|&x| f16::from_f32(x).to_f32()));
                    values[slot].extend(v.iter().map(|&x| f16::from_f32(x).to_f32()));
                }
                unsafe {
                    ctx.memcpy_h2d(k_device.as_mut_ptr(), k.as_ptr().cast(), KV_HEADS * DIM * 4)?;
                    ctx.memcpy_h2d(v_device.as_mut_ptr(), v.as_ptr().cast(), KV_HEADS * DIM * 4)?;
                    kv.append_token_f32(&ctx, slot as u32, k_device.as_ptr(), v_device.as_ptr())?;
                }
            }
        }
        let q = DeviceBuffer::new_tagged(&ctx, Q_HEADS as usize * DIM * 4, "head128_test:q")?;
        let output = DeviceBuffer::new_tagged(&ctx, q.bytes(), "head128_test:output")?;
        let reference = DeviceBuffer::new_tagged(&ctx, q.bytes(), "head128_test:reference")?;
        let workspace = Head128AttentionWorkspace::new(&ctx, Q_HEADS)?;
        Ok(Self {
            ctx,
            kv,
            keys,
            values,
            q,
            output,
            reference,
            workspace,
        })
    }

    fn upload_query(&self, seq_len: usize, scale: f32) -> Result<Vec<f32>> {
        let mut query: Vec<f32> = (0..Q_HEADS as usize * DIM)
            .map(|d| (((d * 23 + 5) % 257) as f32 / 128.0 - 1.0) * scale)
            .collect();
        rotate_absolute(&mut query, seq_len + 16);
        unsafe {
            self.ctx
                .memcpy_h2d(self.q.as_mut_ptr(), query.as_ptr().cast(), self.q.bytes())?;
        }
        Ok(query)
    }

    fn read(&self, buffer: &DeviceBuffer) -> Result<Vec<f32>> {
        let mut output = vec![0.0; Q_HEADS as usize * DIM];
        unsafe {
            self.ctx
                .memcpy_d2h(output.as_mut_ptr().cast(), buffer.as_ptr(), buffer.bytes())?;
        }
        Ok(output)
    }

    fn cpu_attention(&self, slot: usize, seq_len: usize, query: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; query.len()];
        for head in 0..Q_HEADS as usize {
            let kv_head = head / (Q_HEADS as usize / KV_HEADS);
            let mut scores: Vec<f64> = (0..seq_len)
                .map(|token| {
                    let base = token * KV_HEADS * DIM + kv_head * DIM;
                    (0..DIM)
                        .map(|d| query[head * DIM + d] as f64 * self.keys[slot][base + d] as f64)
                        .sum::<f64>()
                        / (DIM as f64).sqrt()
                })
                .collect();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for score in &mut scores {
                *score = (*score - maximum).exp();
            }
            let sum: f64 = scores.iter().sum();
            for d in 0..DIM {
                let weighted_sum = scores
                    .iter()
                    .enumerate()
                    .map(|(token, &weight)| {
                        weight
                            * self.values[slot][token * KV_HEADS * DIM + kv_head * DIM + d] as f64
                    })
                    .sum::<f64>();
                output[head * DIM + d] = (weighted_sum / sum) as f32;
            }
        }
        output
    }

    unsafe fn enqueue(
        &self,
        seq_len: u32,
        slot: u32,
        config: Head128AttentionConfig,
    ) -> Result<()> {
        self.kv.attention_head128_parallel_async(
            &self.ctx,
            slot,
            self.q.as_ptr(),
            Q_HEADS,
            seq_len,
            self.output.as_mut_ptr(),
            config,
            Some(&self.workspace),
        )
    }
}

fn check_close(actual: &[f32], expected: &[f32], label: &str) -> (f32, f64) {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0_f32;
    let mut total = 0.0_f64;
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= 1e-3 + 1e-4 * b.abs(),
            "{label} element {index}: actual={a} expected={b}"
        );
        maximum = maximum.max((a - b).abs());
        total += f64::from((a - b).abs());
    }
    (maximum, total / actual.len() as f64)
}

/// Every geometry preserves GQA, nonzero slots, absolute RoPE, causal prefix
/// lengths, and stable softmax for empty partitions, uniform and extreme scores.
/// A poisoned future KV entry must never contribute to any output.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_parallel_head128_matches_dense_and_cpu() -> Result<()> {
    let _lock = TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let _backend = EnvRestore::set("M40LLM_ATTN_HEAD128_KERNEL", "fast");
    let fixture = Fixture::new(1025)?;
    let mut cpu_max = 0.0_f32;
    let mut legacy_max = 0.0_f32;
    let mut cpu_mean_sum = 0.0;
    let mut legacy_mean_sum = 0.0;
    let mut comparisons = 0;
    for seq_len in [
        1, 7, 32, 33, 64, 127, 128, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025,
    ] {
        let scale = match seq_len {
            33 => 200.0,
            257 => 0.0,
            _ => 1.0,
        };
        let query = fixture.upload_query(seq_len, scale)?;
        for slot in [0, 2] {
            let cpu = fixture.cpu_attention(slot, seq_len, &query);
            unsafe {
                fixture.kv.attention_last_token_f32_gqa(
                    &fixture.ctx,
                    slot as u32,
                    fixture.q.as_ptr(),
                    Q_HEADS,
                    seq_len as u32,
                    fixture.reference.as_mut_ptr(),
                )?;
            }
            let reference = fixture.read(&fixture.reference)?;
            check_close(&reference, &cpu, "legacy vs CPU");
            for threads in [128, 256] {
                for partitions in [1, 2, 4, 8] {
                    let before = profile::snapshot();
                    unsafe {
                        fixture.enqueue(
                            seq_len as u32,
                            slot as u32,
                            Head128AttentionConfig {
                                partitions,
                                threads,
                            },
                        )?;
                    }
                    let delta = profile::counter_delta(&before, &profile::snapshot());
                    assert_eq!(delta.launches, if partitions == 1 { 1 } else { 2 });
                    assert_eq!(delta.stream_syncs, 0);
                    assert_eq!(delta.device_allocs, 0);
                    assert_eq!(delta.device_frees, 0);
                    fixture.ctx.synchronize_stream(CudaStream::Decode)?;
                    let actual = fixture.read(&fixture.output)?;
                    let (maximum, mean) = check_close(
                        &actual,
                        &cpu,
                        &format!("CPU s{seq_len} slot{slot} p{partitions} t{threads}"),
                    );
                    cpu_max = cpu_max.max(maximum);
                    cpu_mean_sum += mean;
                    let (maximum, mean) = check_close(&actual, &reference, "parallel vs legacy");
                    legacy_max = legacy_max.max(maximum);
                    legacy_mean_sum += mean;
                    comparisons += 1;
                }
            }
        }
    }
    eprintln!(
        "[head128-parity] {}",
        serde_json::json!({
            "comparisons":comparisons,"elements_per_comparison":Q_HEADS as usize*DIM,
            "max_abs_diff_vs_cpu":cpu_max,"mean_abs_diff_vs_cpu":cpu_mean_sum/comparisons as f64,
            "max_abs_diff_vs_legacy":legacy_max,"mean_abs_diff_vs_legacy":legacy_mean_sum/comparisons as f64,
        })
    );
    Ok(())
}

/// Opt-in runtime dispatch owns one tracked reusable partial buffer, while the
/// reference/one-block paths allocate none. Invalid configurations are errors.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_runtime_reuses_workspace_and_rejects_invalid_configs() -> Result<()> {
    let _lock = TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let _kernel = EnvRestore::set("M40LLM_ATTN_HEAD128_KERNEL", "parallel");
    let _threads = EnvRestore::set("M40LLM_ATTN_HEAD128_THREADS", "128");
    let _parts = EnvRestore::set("M40LLM_ATTN_HEAD128_PARTITIONS", "4");
    let bytes_before = CudaContext::total_device_bytes();
    let fixture = Fixture::new(33)?;
    fixture.upload_query(33, 1.0)?;
    assert_eq!(fixture.kv.head128_attention_workspace_bytes(), 0);
    {
        let _one = EnvRestore::set("M40LLM_ATTN_HEAD128_PARTITIONS", "1");
        let before = profile::snapshot();
        unsafe {
            fixture.kv.attention_last_token_f32_gqa_async(
                &fixture.ctx,
                2,
                fixture.q.as_ptr(),
                Q_HEADS,
                33,
                fixture.output.as_mut_ptr(),
            )?;
        }
        let delta = profile::counter_delta(&before, &profile::snapshot());
        assert_eq!(delta.launches, 1);
        assert_eq!(delta.device_allocs, 0);
        assert_eq!(fixture.kv.head128_attention_workspace_bytes(), 0);
        fixture.ctx.synchronize_stream(CudaStream::Decode)?;
    }
    for round in 0..3 {
        let before = profile::snapshot();
        unsafe {
            fixture.kv.attention_last_token_f32_gqa_async(
                &fixture.ctx,
                2,
                fixture.q.as_ptr(),
                Q_HEADS,
                33,
                fixture.output.as_mut_ptr(),
            )?;
        }
        let delta = profile::counter_delta(&before, &profile::snapshot());
        assert_eq!(delta.device_allocs, u64::from(round == 0));
        assert_eq!(delta.device_frees, 0);
        assert_eq!(delta.stream_syncs, 0);
        assert_eq!(
            fixture.kv.head128_attention_workspace_bytes(),
            16 * 8 * 130 * 4
        );
        fixture.ctx.synchronize_stream(CudaStream::Decode)?;
    }
    for config in [
        Head128AttentionConfig {
            partitions: 3,
            threads: 128,
        },
        Head128AttentionConfig {
            partitions: 4,
            threads: 64,
        },
    ] {
        let before = profile::snapshot();
        assert!(unsafe { fixture.enqueue(33, 0, config) }.is_err());
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()).launches,
            0
        );
    }
    let config = Head128AttentionConfig {
        partitions: 4,
        threads: 128,
    };
    unsafe {
        assert!(fixture
            .kv
            .attention_head128_parallel_async(
                &fixture.ctx,
                0,
                fixture.q.as_ptr(),
                Q_HEADS,
                33,
                fixture.output.as_mut_ptr(),
                config,
                None
            )
            .is_err());
        assert!(fixture.enqueue(0, 0, config).is_err());
        assert!(fixture.enqueue(34, 3, config).is_err());
    }
    let small_workspace = Head128AttentionWorkspace::new(&fixture.ctx, 1)?;
    let foreign_context = CudaContext::new(-1)?;
    let foreign_workspace = Head128AttentionWorkspace::new(&foreign_context, Q_HEADS)?;
    for workspace in [&small_workspace, &foreign_workspace] {
        unsafe {
            assert!(fixture
                .kv
                .attention_head128_parallel_async(
                    &fixture.ctx,
                    0,
                    fixture.q.as_ptr(),
                    Q_HEADS,
                    33,
                    fixture.output.as_mut_ptr(),
                    config,
                    Some(workspace)
                )
                .is_err());
        }
    }
    drop(small_workspace);
    drop(foreign_workspace);
    drop(foreign_context);
    drop(fixture);
    assert_eq!(CudaContext::total_device_bytes(), bytes_before);
    Ok(())
}

/// An explicitly supplied Qwen model must preserve arithmetic, instruction,
/// and config-lookup answers using the benchmark's explicit Qwen system/ChatML
/// format through ordinary full-layer dense generation.
/// No cache search or download occurs when the optional model is absent.
#[test]
fn test_optional_qwen_generation_smokes() -> Result<()> {
    let _lock = TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let Some(path) = std::env::var_os("M40LLM_HEAD128_GENERATION_MODEL") else {
        eprintln!("not run: set M40LLM_HEAD128_GENERATION_MODEL to a Qwen head128 GGUF");
        return Ok(());
    };
    let path = std::path::PathBuf::from(path);
    let _settings = [
        EnvRestore::set("M40LLM_PROJECTION_BACKEND", "large-model"),
        EnvRestore::set("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
        EnvRestore::set("M40LLM_DECODE_GRAPH", "0"),
        EnvRestore::set("M40LLM_ATTN_HEAD128_PARTITIONS", "8"),
        EnvRestore::set("M40LLM_ATTN_HEAD128_THREADS", "256"),
    ];
    let mut model = m40_llm::infer::LoadedModel::from_gguf(
        m40_llm::gguf::load_gguf(&path)?,
        std::fs::read(&path)?,
        -1,
    )?;
    ensure!(
        model.model_config.architecture.starts_with("qwen")
            && model.model_config.attention_key_length == 128,
        "requested generation fixture must be Qwen with head_dim=128"
    );
    model.allocate_kv_cache_for_layers(256)?;
    for kernel in ["fast", "parallel"] {
        let _kernel = EnvRestore::set("M40LLM_ATTN_HEAD128_KERNEL", kernel);
        for (name, prompt, expected) in [
            ("arithmetic", "What is 2+2? Answer with one digit.", "4"),
            ("instruction", "Reply with exactly the word OK.", "OK"),
            ("config", "Configuration note: the active value is M40_BATCH_LIMIT=37. What is M40_BATCH_LIMIT? Answer with the number only.", "37"),
        ] {
            let result = m40_llm::generate::generate_text(&model,
                m40_llm::generate::GenerateOptions {
                    prompt: format!("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"),
                    prompt_format: m40_llm::generate::PromptFormat::Raw,
                    max_tokens: Some(16), top_k: Some(1),
                    kv_compression: m40_llm::kv_compression::KvCompressionConfig::dense_reference(),
                    ..Default::default()
                })?;
            eprintln!("[head128-smoke] {}", serde_json::json!({
                "kernel":kernel,"case":name,"model":path,"expected":expected,
                "output":result.output,"passed":result.output.trim()==expected,
                "prompt_tokens":result.prompt_token_len,
                "generated_token_ids":&result.token_ids[result.prompt_token_len..],
                "kv_bytes":model.kv_cache.as_ref().map(KVCache::actual_bytes),
                "workspace_bytes":model.kv_cache.as_ref().map(KVCache::head128_attention_workspace_bytes),
            }));
            assert_eq!(result.output.trim(), expected, "{kernel} {name}");
        }
    }
    Ok(())
}

/// Explicit timing only: include all partition/merge launches and one completion
/// per group, with scratch setup outside timings and no competing GPU work.
#[test]
#[ignore = "isolated release hardware benchmark"]
fn benchmark_head128_geometries() -> Result<()> {
    let _lock = TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    ensure!(!cfg!(debug_assertions), "use --release for head128 timing");
    let _backend = EnvRestore::set("M40LLM_ATTN_HEAD128_KERNEL", "fast");
    let fixture = Fixture::new(1025)?;
    let mut report = std::env::var_os("M40LLM_HEAD128_BENCH_REPORT")
        .map(|path| OpenOptions::new().create_new(true).write(true).open(path))
        .transpose()?;
    let run_command = |program: &str, args: &[&str]| -> Result<String> {
        let output = Command::new(program).args(args).output()?;
        ensure!(
            output.status.success(),
            "benchmark metadata command {program} failed"
        );
        Ok(String::from_utf8(output.stdout)?.trim().to_owned())
    };
    let executable = std::env::current_exe()?;
    let gpu_processes = run_command(
        "nvidia-smi",
        &[
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
    )?;
    ensure!(
        gpu_processes.lines().all(|line| line
            .split(',')
            .next()
            .and_then(|pid| pid.trim().parse::<u32>().ok())
            == Some(std::process::id())),
        "another GPU process is live; run attention timing in isolation"
    );
    let manifest = serde_json::json!({"kind":"manifest","build_profile":"release",
        "revision":run_command("git", &["rev-parse","HEAD"] )?,
        "worktree":run_command("git", &["status","--short"] )?,
        "executable_sha256":run_command("sha256sum", &[executable.to_str().unwrap()] )?,
        "rustc":run_command("rustc", &["--version"] )?,
        "nvcc":run_command("nvcc", &["--version"] )?,
        "cuda_flags":["-O3","-gencode=arch=compute_52,code=sm_52","-gencode=arch=compute_52,code=compute_52"],
        "gpu":run_command("nvidia-smi", &["--query-gpu=uuid,name,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw","--format=csv,noheader"] )?,
        "gpu_processes":gpu_processes,
        "q_heads":Q_HEADS,"kv_heads":KV_HEADS,"head_dim":DIM,
        "timing_boundary":"40 repeated calls between CUDA events; one stream completion; GPU cache-hot; no model weights",
        "seed":"deterministic token/dimension/slot modular values; absolute split-half RoPE offset17; FP16 KV"});
    if let Some(file) = &mut report {
        writeln!(file, "{manifest}")?;
    }
    eprintln!("{manifest}");
    for seq_len in [1, 32, 64, 128, 256, 512, 1024, 319, 575, 1025] {
        fixture.upload_query(seq_len, 1.0)?;
        for trial in 0..5 {
            let mut configs = vec![None];
            for threads in [128, 256] {
                for partitions in [1, 2, 4, 8] {
                    configs.push(Some(Head128AttentionConfig {
                        partitions,
                        threads,
                    }));
                }
            }
            if trial % 2 != 0 {
                configs.reverse();
            }
            for config in configs {
                let enqueue = || unsafe {
                    match config {
                        None => fixture.kv.attention_last_token_f32_gqa_async(
                            &fixture.ctx,
                            2,
                            fixture.q.as_ptr(),
                            Q_HEADS,
                            seq_len as u32,
                            fixture.output.as_mut_ptr(),
                        ),
                        Some(config) => fixture.enqueue(seq_len as u32, 2, config),
                    }
                };
                for _ in 0..4 {
                    enqueue()?;
                }
                fixture.ctx.synchronize_stream(CudaStream::Decode)?;
                let start = fixture.ctx.create_event()?;
                let stop = fixture.ctx.create_event()?;
                let wall = Instant::now();
                start.record(CudaStream::Decode)?;
                for _ in 0..40 {
                    enqueue()?;
                }
                stop.record(CudaStream::Decode)?;
                fixture.ctx.synchronize_stream(CudaStream::Decode)?;
                let wall_us = wall.elapsed().as_secs_f64() * 1e6 / 40.0;
                let gpu_us = f64::from(start.elapsed_ready(&stop)?.expect("completed timing"))
                    * 1000.0
                    / 40.0;
                let row = serde_json::json!({"kind":"trial","seq_len":seq_len,"slot":2,"trial":trial,
                    "partitions":config.map(|c|c.partitions),"threads":config.map(|c|c.threads),
                    "gpu_us":gpu_us,"wall_us":wall_us,"repetitions":40,
                    "workspace_capacity_bytes":fixture.workspace.bytes(),
                    "active_partial_bytes":config.filter(|c|c.partitions>1).map_or(0,|c|Q_HEADS*c.partitions*130*4),
                    "launches_per_call":if config.is_some_and(|c|c.partitions>1){2}else{1}});
                if let Some(file) = &mut report {
                    writeln!(file, "{row}")?;
                }
                eprintln!("{row}");
            }
        }
    }
    Ok(())
}

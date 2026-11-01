//! Offline host-adapter cost on qualified actual IDs, never model inference.
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaContext,
    evaluation::identity::file_sha256,
    sampling::{Sampler, SamplerConfig},
};
use serde_json::{json, Value};
use std::{
    fs::{self, OpenOptions},
    hint::black_box,
    io::Write,
    path::PathBuf,
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const TRACE_SHA: &str = "8b499787179f484aeb32e37caf837208e0994da4995918e18e34ba922d3d949e";
const ITERATIONS: usize = 256;

fn parse_tokens(records: &[Value]) -> Result<Vec<u32>> {
    let one = |kind: &str| -> Result<&Value> {
        let mut matches = records.iter().filter(|row| row["kind"] == kind);
        let row = matches.next().context("missing trace record")?;
        ensure!(matches.next().is_none(), "duplicate trace record");
        Ok(row)
    };
    let manifest = one("q8_runtime_manifest")?;
    let summary = one("q8_runtime_summary")?;
    ensure!(
        manifest["prefill_execution_contract"] == "prefix-decode-completed-v1"
            && manifest["original_exact_goal_eligible"] == true
            && summary["passes_exact_quality_gate"] == true,
        "adapter study requires qualified exact completed-prefix evidence"
    );
    let mut generations = records.iter().filter(|row| {
        row["kind"] == "q8_runtime_browser_generation" && row["label"] == "candidate"
    });
    let row = generations.next().context("missing actual generation")?;
    ensure!(generations.next().is_none(), "duplicate actual generation");
    let tokens: Vec<u32> =
        serde_json::from_value(row["generation"]["generated_token_ids"].clone())?;
    ensure!(
        tokens.len() == 512
            && tokens.iter().all(|&token| token < 151936)
            && row["generation"]["metrics"]["generated_tokens"] == 512
            && row["generation"]["metrics"]["stop_reason"] == "length",
        "adapter study requires actual512 bounded Qwen token IDs"
    );
    Ok(tokens)
}

fn replay(tokens: &[u32], is_synthetic: bool, iterations: usize) -> Result<u64> {
    let mut sampler = Sampler::new(SamplerConfig {
        top_k: Some(1),
        ..SamplerConfig::default()
    });
    let mut checksum = 0u64;
    for _ in 0..iterations {
        for &token in tokens {
            let token = black_box(token);
            let next = if is_synthetic {
                let mut logits = vec![f32::NEG_INFINITY; token as usize + 1];
                logits[token as usize] = 0.0;
                sampler.sample(black_box(&logits))? as u32
            } else {
                token
            };
            checksum = checksum.wrapping_add(u64::from(black_box(next)));
        }
    }
    Ok(black_box(checksum))
}

fn command_text(program: &str, arguments: &[&str]) -> Result<String> {
    let output = Command::new(program).args(arguments).output()?;
    ensure!(output.status.success(), "identity command failed");
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

/// Replay the real synthetic-logits adapter versus direct forwarding without
/// model/CUDA construction. Reject missing explicit inputs, debug timing and
/// visible GPUs; require bounded fixed pairs, preserved IDs and zero device use.
#[test]
#[ignore = "CPU-only release study; explicit trace/report paths and hidden CUDA required"]
fn test_greedy_adapter_cost_from_frozen_tokens() -> Result<()> {
    ensure!(!cfg!(debug_assertions), "adapter timing requires release");
    ensure!(
        std::env::var("CUDA_VISIBLE_DEVICES").as_deref() == Ok("-1"),
        "hide GPUs with CUDA_VISIBLE_DEVICES=-1 for this CPU-only study"
    );
    let input = PathBuf::from(std::env::var("M40LLM_GREEDY_ADAPTER_TRACE")?);
    let output = PathBuf::from(std::env::var("M40LLM_GREEDY_ADAPTER_REPORT")?);
    ensure!(
        file_sha256(&input)? == TRACE_SHA,
        "qualified trace SHA mismatch"
    );
    let records = fs::read_to_string(&input)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<Vec<Value>, _>>()?;
    let tokens = parse_tokens(&records)?;
    let requested_bytes: u64 = tokens.iter().map(|&id| 4 * (u64::from(id) + 1)).sum();
    let token_sum: u64 = tokens.iter().map(|&id| u64::from(id)).sum();
    let checksum = token_sum * ITERATIONS as u64;
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)?;
    let mut record = |value: Value| -> Result<()> {
        serde_json::to_writer(&mut output, &value)?;
        writeln!(&mut output)?;
        output.flush()?;
        Ok(())
    };
    ensure!(
        CudaContext::total_device_bytes() == 0,
        "unexpected resident device bytes"
    );
    record(json!({"kind":"greedy_adapter_manifest","schema_version":1,
        "revision":command_text("git", &["rev-parse","HEAD"])?,
        "worktree":command_text("git", &["status","--short"])?,
        "executable_sha256":file_sha256(&std::env::current_exe()?)?,
        "rustc":command_text("rustc", &["--version"])?,"profile":"release",
        "cpu_model":fs::read_to_string("/proc/cpuinfo")?.lines().find(|line|line.starts_with("model name")),
        "cpu_affinity":fs::read_to_string("/proc/self/status")?.lines().find(|line|line.starts_with("Cpus_allowed_list")),
        "trace":input,"trace_sha256":TRACE_SHA,"generated_ids":tokens,
        "tokens":512,"iterations":ITERATIONS,"pairs":5,"cuda_visible_devices":"-1",
        "synthetic_vector_requested_bytes_per_trace":requested_bytes,
        "scope":"isolated CPU adapter wall cost, not GPU/HTTP latency, allocation traffic or an inference bound"}))?;
    for &token in &tokens {
        for is_synthetic in [true, false] {
            ensure!(
                replay(&[token], is_synthetic, 1)? == u64::from(token),
                "adapter changed an individual actual token"
            );
        }
    }
    for is_synthetic in [true, false] {
        ensure!(
            replay(&tokens, is_synthetic, 2)? == token_sum * 2,
            "warmup changed token sum"
        );
    }
    for trial in 0..5 {
        let arms = if trial % 2 == 0 {
            [true, false]
        } else {
            [false, true]
        };
        let mut measurements = Vec::new();
        for is_synthetic in arms {
            let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
            let start = Instant::now();
            let actual = replay(&tokens, is_synthetic, ITERATIONS)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            let completed = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
            ensure!(actual == checksum, "adapter changed token IDs");
            measurements.push(json!({"is_synthetic":is_synthetic,"elapsed_ms":elapsed_ms,
                "per_trace_ms":elapsed_ms / ITERATIONS as f64,"checksum":actual,
                "started_unix_us":started,"completed_unix_us":completed}));
        }
        record(json!({"kind":"greedy_adapter_pair","trial":trial,"measurements":measurements}))?;
    }
    ensure!(
        CudaContext::total_device_bytes() == 0,
        "adapter allocated device storage"
    );
    record(json!({"kind":"greedy_adapter_complete","is_complete":true,"tracked_device_bytes":0}))?;
    Ok(())
}

fn trace_fixture() -> Vec<Value> {
    vec![
        json!({"kind":"q8_runtime_manifest","prefill_execution_contract":"prefix-decode-completed-v1","original_exact_goal_eligible":true}),
        json!({"kind":"q8_runtime_summary","passes_exact_quality_gate":true}),
        json!({"kind":"q8_runtime_browser_generation","label":"candidate",
            "generation":{"generated_token_ids":vec![7;512],"metrics":{"generated_tokens":512,"stop_reason":"length"}}}),
    ]
}

/// Both adapter arms preserve token zero, signed-float argmax positions and
/// the highest supported Qwen index through the real production sampler.
#[test]
fn test_replay_preserves_zero_and_high_token_ids() -> Result<()> {
    for token in [0, 1, 32768, 151935] {
        for is_synthetic in [true, false] {
            assert_eq!(replay(&[token], is_synthetic, 3)?, u64::from(token) * 3);
        }
    }
    Ok(())
}

/// CPU measurement refuses unqualified, stale, truncated or out-of-range
/// generations instead of manufacturing a cost from a different workload.
#[test]
fn test_trace_validation_rejects_ineligible_inputs() -> Result<()> {
    assert_eq!(parse_tokens(&trace_fixture())?, vec![7; 512]);
    let mut stale = trace_fixture();
    stale[0]["prefill_execution_contract"] = json!("old");
    assert!(parse_tokens(&stale).is_err());
    let mut unqualified = trace_fixture();
    unqualified[1]["passes_exact_quality_gate"] = json!(false);
    assert!(parse_tokens(&unqualified).is_err());
    let mut truncated = trace_fixture();
    truncated[2]["generation"]["generated_token_ids"] = json!([7]);
    assert!(parse_tokens(&truncated).is_err());
    let mut invalid = trace_fixture();
    invalid[2]["generation"]["generated_token_ids"][0] = json!(151936);
    assert!(parse_tokens(&invalid).is_err());
    let mut duplicate = trace_fixture();
    duplicate.push(duplicate[0].clone());
    assert!(parse_tokens(&duplicate).is_err());
    Ok(())
}

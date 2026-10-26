//! CPU-only cost of disabled forward wrappers; never enqueues GPU work.
use crate::{cuda::CudaContext, evaluation::identity::file_sha256, profile};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    fs::{self, OpenOptions},
    hint::black_box,
    io::Write,
    path::PathBuf,
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const TRACE_SHA: &str = "4eeec0020ed346341cfd816cc3ef28ae1e98ae88f2b7a6f24a378b75927b126c";
const STAGES: [&str; 11] = [
    "attn_norm",
    "qkv_project",
    "rope_q",
    "kv_append_rope_k",
    "attention",
    "out_project",
    "attn_residual",
    "ffn_norm",
    "mlp_gate_up_swiglu",
    "mlp_down",
    "mlp_residual",
];
const TOKENS: usize = 512;
const ITERATIONS: usize = 8;

struct Operation {
    label: String,
    stage: String,
}

fn parse_operations(records: &[Value]) -> Result<Vec<Operation>> {
    let mut matches = records
        .iter()
        .filter(|row| row["session_id"] == 2 && row["generated_step"] == 255);
    let record = matches
        .next()
        .context("missing primary middle-token profile")?;
    ensure!(
        matches.next().is_none() && record["absolute_position"] == 319,
        "ambiguous middle-token profile"
    );
    let stages: Vec<_> = record["stages"]
        .as_array()
        .context("missing stages")?
        .iter()
        .filter(|stage| !stage["layer_id"].is_null())
        .collect();
    ensure!(stages.len() == 36 * STAGES.len(), "incomplete layer stages");
    stages
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let layer = index / STAGES.len();
            let stage = row["stage"].as_str().context("missing stage identity")?;
            ensure!(
                row["layer_id"] == layer && stage == STAGES[index % STAGES.len()],
                "unexpected layer/stage order"
            );
            Ok(Operation {
                label: format!("forward.layer.{layer}.seq_len.320"),
                stage: stage.to_owned(),
            })
        })
        .collect()
}

fn replay(operations: &[Operation], has_diagnostics: bool, iterations: usize) -> u64 {
    let mut checksum = 0u64;
    for _ in 0..iterations {
        for _ in 0..TOKENS {
            for (index, operation) in operations.iter().enumerate() {
                if has_diagnostics {
                    let before = profile::snapshot_if_enabled();
                    let began = Instant::now();
                    super::log_profiled_op(
                        black_box(&operation.label),
                        black_box(&operation.stage),
                        before.as_ref(),
                        began.elapsed(),
                    );
                }
                checksum = checksum.wrapping_add(black_box(index as u64 + 1));
            }
        }
    }
    black_box(checksum)
}

fn command_text(program: &str, arguments: &[&str]) -> Result<String> {
    let output = Command::new(program).args(arguments).output()?;
    ensure!(output.status.success(), "identity command failed");
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn ensure_logs_disabled() -> Result<()> {
    for key in [
        "M40LLM_LAUNCH_LOG",
        "M40LLM_SYNC_LOG",
        "M40LLM_COPY_LOG",
        "M40LLM_PROFILE_LOG",
        "M40LLM_TIMING_LOG",
    ] {
        ensure!(
            std::env::var(key).as_deref() != Ok("1"),
            "{key} must be disabled"
        );
    }
    Ok(())
}

/// Run the actual private disabled wrapper with frozen layer/stage coverage.
/// Both arms traverse the same identities without constructing a CUDA model.
#[test]
#[ignore = "CPU-only release study; explicit trace/report and hidden CUDA required"]
fn test_disabled_diagnostic_cost_from_frozen_operations() -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "diagnostic timing requires release"
    );
    ensure!(
        std::env::var("CUDA_VISIBLE_DEVICES").as_deref() == Ok("-1"),
        "hide GPUs for CPU study"
    );
    ensure_logs_disabled()?;
    let input = PathBuf::from(std::env::var("M40LLM_DIAGNOSTICS_COST_TRACE")?);
    let output = PathBuf::from(std::env::var("M40LLM_DIAGNOSTICS_COST_REPORT")?);
    ensure!(
        file_sha256(&input)? == TRACE_SHA,
        "qualified profile SHA mismatch"
    );
    let records = fs::read_to_string(&input)?
        .lines()
        .map(serde_json::from_str)
        .collect::<serde_json::Result<Vec<Value>>>()?;
    let operations = parse_operations(&records)?;
    let expected = (operations.len() * (operations.len() + 1) / 2 * TOKENS * ITERATIONS) as u64;
    let before = profile::snapshot();
    let device_before = CudaContext::total_device_bytes();
    ensure!(
        device_before == 0,
        "CPU study must not own device allocations"
    );
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&output)?;
    writeln!(
        file,
        "{}",
        json!({"kind":"disabled_diagnostics_manifest","schema_version":1,
        "source":command_text("git", &["rev-parse","HEAD"])?,"worktree":command_text("git", &["status","--short"])?,
        "executable_sha256":file_sha256(&std::env::current_exe()?)?,"build_profile":"release",
        "rustc":command_text("rustc", &["--version"])?,"cpu":fs::read_to_string("/proc/cpuinfo")?
            .lines().find(|line|line.starts_with("model name")).unwrap_or("unknown"),
        "affinity":command_text("taskset", &["-pc", &std::process::id().to_string()])?,
        "trace_sha256":TRACE_SHA,"session_id":2,"generated_step":255,"absolute_position":319,
        "operations_per_nominal_token":operations.len(),"stages_per_layer":STAGES,"layers":36,
        "nominal_tokens_per_replay":TOKENS,"iterations_per_arm":ITERATIONS,"pairs":5,"expected_checksum":expected,
        "cuda_visible_devices":"-1","logs_disabled":true,"device_bytes_before":device_before,
        "counters_before":&before.by_op,
        "scope":"CPU wrapper/traversal replay, not inference, full enqueue cost or latency bound",
        "excluded":"CUDA enqueue/counter recording; layer label construction; model loading; actual token generation",
        "qualified_http_ms":10417.843301})
    )?;
    file.flush()?;
    for has_diagnostics in [false, true] {
        ensure!(
            replay(&operations, has_diagnostics, ITERATIONS) == expected,
            "warmup work mismatch"
        );
    }
    for trial in 0..5 {
        let mut arms = Vec::new();
        for has_diagnostics in if trial % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
            let began = Instant::now();
            let checksum = replay(&operations, has_diagnostics, ITERATIONS);
            let elapsed = began.elapsed();
            ensure!(checksum == expected, "measured replay work mismatch");
            arms.push(
                json!({"has_diagnostics":has_diagnostics,"elapsed_ns":elapsed.as_nanos(),
                "elapsed_ms_per_replay":elapsed.as_secs_f64()*1000.0/ITERATIONS as f64,
                "checksum":checksum,"started_unix_us":started,
                "completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros()}),
            );
        }
        ensure_logs_disabled()?;
        ensure!(
            profile::snapshot() == before && CudaContext::total_device_bytes() == device_before,
            "CPU replay changed counters/device allocation"
        );
        writeln!(
            file,
            "{}",
            json!({"kind":"disabled_diagnostics_pair","trial":trial,"arms":arms})
        )?;
        file.flush()?;
    }
    writeln!(
        file,
        "{}",
        json!({"kind":"disabled_diagnostics_complete","pairs":5,
        "counters_unchanged":profile::snapshot()==before,"counters_after":profile::snapshot().by_op,
        "device_bytes_after":CudaContext::total_device_bytes()})
    )?;
    Ok(())
}

fn profile_fixture() -> Value {
    let stages: Vec<_> = (0..36)
        .flat_map(|layer| {
            STAGES
                .iter()
                .map(move |stage| json!({"layer_id":layer,"stage":stage}))
        })
        .collect();
    json!({"session_id":2,"generated_step":255,"absolute_position":319,"stages":stages})
}

/// Disabled diagnostic calls preserve traversal work. Counter isolation is
/// checked separately by the single-threaded ignored timing test.
#[test]
fn test_disabled_diagnostic_replay_preserves_traversal_work() -> Result<()> {
    ensure_logs_disabled()?;
    let operations = parse_operations(&[profile_fixture()])?;
    let expected = (396 * 397 / 2 * TOKENS) as u64;
    assert_eq!(replay(&operations, true, 1), expected);
    assert_eq!(replay(&operations, false, 1), expected);
    Ok(())
}

/// Incomplete, duplicated or reordered recorded layer stages cannot define
/// a nominal full-token replay and must be rejected before timing.
#[test]
fn test_diagnostic_replay_rejects_invalid_coverage() -> Result<()> {
    let original = profile_fixture();
    assert_eq!(
        parse_operations(std::slice::from_ref(&original))?.len(),
        396
    );
    assert!(parse_operations(&[]).is_err());
    assert!(parse_operations(&[original.clone(), original.clone()]).is_err());
    let mut incomplete = original.clone();
    incomplete["stages"].as_array_mut().unwrap().pop();
    assert!(parse_operations(&[incomplete]).is_err());
    let mut reordered = original;
    reordered["stages"].as_array_mut().unwrap().swap(0, 1);
    assert!(parse_operations(&[reordered]).is_err());
    Ok(())
}

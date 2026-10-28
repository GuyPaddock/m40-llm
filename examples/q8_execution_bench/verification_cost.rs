//! Observe the rejected SGEMM verifier; no target backend is selected here.
use super::{command_output, inspect_gpu, write_record, Args};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaContext,
    evaluation::identity::{file_sha256, sha256},
    gguf,
    gpu_profile::{DecodeGpuProfiler, GpuProfileConfig},
    infer::{
        execution_weights::Q8ExecutionConfig, q8_prefill::Q8PrefillBackend, LoadedModel,
        Q8AttentionProjectionBackend, Q8FfnConfig, Q8FfnOwnership,
    },
    profile,
    verification::VerificationSession,
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, time::Instant};

const REFERENCE: &str = "docs/experiments/2026-10-27-compound-reference.json";
const QUALITY: &str = "docs/experiments/2026-10-27-maxwell-compound-runtime-quality.jsonl";
const EVENT_CAPACITY: usize = 4096;

struct Trace {
    prompt: Vec<u32>,
    tokens: Vec<u32>,
    stop_ids: Vec<u32>,
    vocab: usize,
}

fn read_trace(reference: &Value) -> Result<Trace> {
    ensure!(
        file_sha256(std::path::Path::new(QUALITY))? == reference["artifact_sha256"][QUALITY],
        "frozen quality hash changed"
    );
    let rows: Vec<Value> = fs::read_to_string(QUALITY)?
        .lines()
        .map(serde_json::from_str)
        .collect::<serde_json::Result<_>>()?;
    let manifest = &rows[0];
    let row = rows
        .iter()
        .find(|r| r["kind"] == "q8_runtime_browser_generation" && r["label"] == "candidate")
        .context("missing frozen actual-token trace")?;
    let trace = Trace {
        prompt: serde_json::from_value(row["prompt_token_ids"].clone())?,
        tokens: serde_json::from_value(row["generation"]["generated_token_ids"].clone())?,
        stop_ids: serde_json::from_value(manifest["candidate"]["stop_ids"].clone())?,
        vocab: 151936,
    };
    ensure!(
        trace.prompt.len() == 65 && trace.tokens.len() == 512,
        "wrong frozen workload"
    );
    Ok(trace)
}

fn rejection_positions(width: usize) -> Vec<usize> {
    let mut positions = vec![0, width / 2, width - 1];
    positions.sort_unstable();
    positions.dedup();
    positions
}

fn validate_gpu() -> Result<Value> {
    let state = inspect_gpu()?;
    validate_gpu_state(&state)?;
    Ok(state)
}

fn validate_gpu_state(state: &Value) -> Result<()> {
    let fields = state["device"]
        .as_str()
        .context("device snapshot")?
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    ensure!(
        fields.len() == 9
            && fields[0] == "GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852"
            && fields[3] == "P0"
            && fields[5].strip_suffix(" MHz").unwrap_or(fields[5]) == "1113"
            && fields[6].strip_suffix(" MHz").unwrap_or(fields[6]) == "3004",
        "diagnostic GPU/clocks differ: {fields:?}"
    );
    Ok(())
}

fn prefix_hash(session: &VerificationSession<'_>, layers: u32, prefix: usize) -> Result<String> {
    let mut bytes = Vec::new();
    for layer in 0..layers {
        for position in 0..prefix {
            let (k, v) = session.debug_kv_token(layer, position as u32)?;
            bytes.extend(k.iter().chain(&v).flat_map(|value| value.to_le_bytes()));
        }
    }
    sha256(&bytes)
}

fn check_suffix(
    session: &VerificationSession<'_>,
    layers: u32,
    start: usize,
    end: usize,
) -> Result<()> {
    for layer in 0..layers {
        for position in start..end {
            let (k, v) = session.debug_kv_token(layer, position as u32)?;
            ensure!(
                k.iter().chain(&v).all(|&bits| bits == 0),
                "rejected KV was not cleared"
            );
        }
    }
    Ok(())
}

fn new_profiler(ctx: &CudaContext) -> Result<DecodeGpuProfiler> {
    DecodeGpuProfiler::new_with_event_capacity(
        ctx,
        GpuProfileConfig {
            generated_steps: vec![0],
            report_path: None,
            should_profile_prefill: false,
        },
        EVENT_CAPACITY,
    )
}

/// All debug readbacks and report serialization are outside the batch interval.
fn measure(
    session: &mut VerificationSession<'_>,
    profiler: &mut DecodeGpuProfiler,
    trace: &Trace,
    width: usize,
    accepted: Option<usize>,
    is_profiled: bool,
    layers: u32,
) -> Result<Value> {
    ensure!(
        matches!(width, 2 | 4 | 8)
            && accepted.is_none_or(|n| n < width)
            && trace.tokens.len() > width,
        "invalid diagnostic case"
    );
    ensure!(
        session.prefill(&trace.prompt)? == trace.tokens[0],
        "prefill token changed"
    );
    let prefix = prefix_hash(session, layers, trace.prompt.len())?;
    let mut candidates = trace.tokens[..width].to_vec();
    if let Some(index) = accepted {
        candidates[index] = (candidates[index] + 1) % trace.vocab as u32;
    }
    let scope = if is_profiled {
        Some(profiler.begin_token(0, trace.prompt.len())?)
    } else {
        None
    };
    let before = profile::snapshot();
    let began = Instant::now();
    let attempt = session.begin(&candidates, false)?;
    let begin_ms = began.elapsed().as_secs_f64() * 1000.;
    let after_begin = profile::snapshot();
    let matched = attempt.accepted_len();
    let predictions = attempt.predictions().to_vec();
    let committed = Instant::now();
    let outcome = attempt.commit()?;
    let commit_ms = committed.elapsed().as_secs_f64() * 1000.;
    let after_commit = profile::snapshot();
    let mut correction_ms = 0.;
    let mut correction_token = None;
    if accepted.is_some() {
        let corrected = Instant::now();
        correction_token = Some(session.serial_step(outcome.next_token)?);
        correction_ms = corrected.elapsed().as_secs_f64() * 1000.;
    }
    let elapsed_ms = began.elapsed().as_secs_f64() * 1000.;
    let after = profile::snapshot();
    let counters = profile::counter_delta(&before, &after);
    let gpu_profile = scope.map(|scope| scope.finish()).transpose()?;
    let expected = accepted.unwrap_or(width);
    ensure!(
        matched == expected && !outcome.has_stopped,
        "accepted-prefix or stop changed"
    );
    let expected_next = trace.tokens[expected];
    ensure!(
        outcome.next_token == expected_next,
        "committed greedy token changed"
    );
    if let Some(index) = accepted {
        ensure!(
            correction_token == Some(trace.tokens[index + 1]),
            "correction token changed"
        );
        // Correction overwrites one rejected slot; every remaining slot is zero.
        check_suffix(
            session,
            layers,
            trace.prompt.len() + index + 1,
            trace.prompt.len() + width,
        )?;
    } else {
        ensure!(
            predictions == trace.tokens[1..=width],
            "verification predictions changed"
        );
    }
    let progress = if accepted.is_some() {
        expected + 1
    } else {
        width
    };
    let lengths = session.debug_lengths()?;
    ensure!(
        lengths
            .iter()
            .all(|&length| length as usize == trace.prompt.len() + progress),
        "committed layer lengths changed"
    );
    let after_prefix = prefix_hash(session, layers, trace.prompt.len())?;
    ensure!(prefix == after_prefix, "retained prefix bits changed");
    let gpu_profile = gpu_profile.map(|p| {
        json!({
            "event_pool_capacity":p.event_pool_capacity,"event_pool_setup_ms":p.event_pool_setup_ms,
            "event_pool_reused":p.event_pool_reused,"gpu_stage_sum_ms":p.gpu_stage_sum_ms,
            "gpu_stage_union_ms":p.gpu_stage_union_ms,"gpu_timeline_span_ms":p.gpu_timeline_span_ms,
            "wall_minus_gpu_union_ms":p.wall_minus_gpu_union_ms,"stages":p.stages,
        })
    });
    Ok(
        json!({"prefix_tokens":trace.prompt.len(),"width":width,"accepted_prefix":accepted,
        "is_profiled":is_profiled,"begin_ms":begin_ms,"commit_ms":commit_ms,
        "correction_ms":correction_ms,"elapsed_ms":elapsed_ms,"counters":counters,
        "begin_counters":profile::counter_delta(&before,&after_begin),
        "commit_counters":profile::counter_delta(&after_begin,&after_commit),
        "correction_counters":profile::counter_delta(&after_commit,&after),
        "predictions":predictions,"next_token":outcome.next_token,"correction_token":correction_token,
        "useful_progress":progress,"lengths":lengths,"prefix_sha256":prefix,"after_prefix_sha256":after_prefix,
        "has_exact_output":true,"has_cleared_suffix":true,"gpu_profile":gpu_profile,
        "suffix_check_scope":"remaining rejected slots after correction; pre-correction clearing covered by transaction tests",
        "scope":"existing SGEMM verification batch; prefill/debug/report excluded; not HTTP or direct-Q8 timing"}),
    )
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

pub(super) fn benchmark(args: &Args) -> Result<()> {
    ensure!(cfg!(have_cublas), "cost diagnostics require cuBLAS");
    ensure!(
        !cfg!(debug_assertions)
            && args.trials == 5
            && args.rows == [2, 4, 8]
            && args.iterations == 10,
        "bounded feasibility requires release, five trials and --rows 2,4,8; no iteration override"
    );
    let reference: Value = serde_json::from_slice(&fs::read(REFERENCE)?)?;
    let environment: BTreeMap<_, _> = std::env::vars()
        .filter(|(key, _)| key.starts_with("M40LLM_") || key == "CUDA_VISIBLE_DEVICES")
        .collect();
    ensure!(
        serde_json::to_value(&environment)? == reference["configuration"]["environment"],
        "restore the exact frozen environment; no diagnostics/batching flags"
    );
    ensure!(
        file_sha256(&args.model)? == reference["model"]["sha256"],
        "model hash changed"
    );
    let trace = read_trace(&reference)?;
    let worktree = command_output("git", &["status", "--short"])?;
    ensure!(
        worktree.is_empty(),
        "diagnostics require a clean committed source"
    );
    let mut output = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.output)?;
    write_record(
        &mut output,
        json!({"kind":"direct_q8_cost_manifest","schema_version":1,
        "reference_id":reference["reference_id"],"revision":command_output("git",&["rev-parse","HEAD"])?,
        "worktree":worktree,"executable_sha256":file_sha256(&std::env::current_exe()?)?,
        "model_sha256":file_sha256(&args.model)?,"quality_sha256":file_sha256(std::path::Path::new(QUALITY))?,
        "environment":environment,"scope":"diagnostic only; old verifier remains rejected"}),
    )?;
    let initial = CudaContext::total_device_bytes();
    let mut model = LoadedModel::from_gguf_with_q8_runtime(
        gguf::load_gguf(&args.model)?,
        fs::read(&args.model)?,
        -1,
        Q8ExecutionConfig::parse("split", Some("6144"), Some("gate-up,down,vocabulary"))?,
        Q8FfnConfig::joint_shared_scales(),
        Q8PrefillBackend::DequantSgemm,
        Q8AttentionProjectionBackend::PackedWarp256,
    )?;
    ensure!(
        model.q8_ffn.ownership() == Q8FfnOwnership::JointOnly
            && model.model_config.block_count == 36
            && model.model_config.embedding_length == 2048
            && model.model_config.attention_key_length == 128,
        "wrong compound model"
    );
    model.allocate_kv_cache_for_layers(1024)?;
    let prepared = serde_json::to_value(model.q8_execution_stats())?;
    let joint = serde_json::to_value(model.q8_ffn_pair_stats())?;
    write_record(
        &mut output,
        json!({"kind":"direct_q8_cost_ownership","split":prepared,"joint":joint,
        "ownership":model.q8_ffn.ownership(),"logical_sources":model.q8_execution.logical_sources(),
        "source_bytes":model.weights_len,"dense_kv_bytes":model.kv_cache.as_ref().unwrap().actual_bytes()}),
    )?;
    let mut profiler = new_profiler(&model.cuda)?;
    {
        let mut session = VerificationSession::new(&mut model, &trace.stop_ids)?;
        // First use and cold event publication are excluded and never pooled.
        for width in [2, 4, 8] {
            measure(&mut session, &mut profiler, &trace, width, None, false, 36)?;
            measure(&mut session, &mut profiler, &trace, width, None, true, 36)?;
        }
        for width in [2, 4, 8] {
            let mut off = Vec::new();
            let mut on = Vec::new();
            for pair in 0..5 {
                let mut counts = Vec::new();
                for is_profiled in if pair % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let before = validate_gpu()?;
                    let row = measure(
                        &mut session,
                        &mut profiler,
                        &trace,
                        width,
                        None,
                        is_profiled,
                        36,
                    )?;
                    let after = validate_gpu()?;
                    let elapsed = row["elapsed_ms"].as_f64().unwrap();
                    if is_profiled {
                        on.push(elapsed)
                    } else {
                        off.push(elapsed)
                    };
                    counts.push(row["counters"].clone());
                    write_record(
                        &mut output,
                        json!({"kind":"direct_q8_cost_trial","pair":pair,
                        "gpu_before":before,"gpu_after":after,"row":row}),
                    )?;
                }
                ensure!(
                    counts[0] == counts[1],
                    "profiling changed ordinary work counters"
                );
            }
            let off = median(off);
            let on = median(on);
            let overhead = on / off - 1.;
            write_record(
                &mut output,
                json!({"kind":"direct_q8_cost_observer","width":width,
                "off_ms":off,"on_ms":on,"latency_overhead":overhead,"limit":0.03,
                "is_attribution_valid":overhead<=0.03}),
            )?;
            ensure!(
                overhead <= 0.03,
                "observer exceeds unchanged three-percent allowance"
            );
        }
        for width in [2, 4, 8] {
            for accepted in rejection_positions(width) {
                for trial in 0..5 {
                    let before = validate_gpu()?;
                    let row = measure(
                        &mut session,
                        &mut profiler,
                        &trace,
                        width,
                        Some(accepted),
                        false,
                        36,
                    )?;
                    write_record(
                        &mut output,
                        json!({"kind":"direct_q8_cost_rejection","trial":trial,
                        "gpu_before":before,"gpu_after":validate_gpu()?,"row":row}),
                    )?;
                }
            }
        }
    }
    ensure!(
        serde_json::to_value(model.q8_execution_stats())? == prepared
            && serde_json::to_value(model.q8_ffn_pair_stats())? == joint,
        "prepared ownership changed"
    );
    drop(profiler);
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "diagnostic teardown leaked"
    );
    write_record(
        &mut output,
        json!({"kind":"direct_q8_cost_complete","is_diagnostic_valid":true,
        "device_bytes_after":CudaContext::total_device_bytes(),"is_http_claim":false,
        "is_new_kernel_measured":false}),
    )
}

#[cfg(test)]
#[path = "../../tests/common/q8_model.rs"]
#[allow(dead_code)]
mod fixture;

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// NVIDIA's default CSV includes units. Normalizing that presentation
    /// must not accept different clocks, a different GPU or an idle P-state.
    #[test]
    fn test_gpu_snapshot_accepts_units_but_rejects_clock_changes() {
        let line = "GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, Tesla M40 24GB, 580.159.04, P0, 32, 1113 MHz, 3004 MHz, 6155 MiB, 89 %";
        assert!(validate_gpu_state(&json!({"device":line})).is_ok());
        assert!(validate_gpu_state(&json!({"device":line.replace(" MHz","")})).is_ok());
        for changed in [
            line.replace("1113", "822"),
            line.replace("3004", "2505"),
            line.replace("P0", "P8"),
            line.replace("e590a24b", "00000000"),
            line.replace(" MHz", " kHz"),
        ] {
            assert!(validate_gpu_state(&json!({"device":changed})).is_err());
        }
    }

    /// The observation mode must be explicit and cannot select an existing
    /// verifier or FFN experiment in the same invocation.
    #[test]
    fn test_cost_mode_is_separate_from_old_experiments() {
        let args = [
            "bench",
            "--model",
            "model.gguf",
            "--output",
            "report.jsonl",
            "--rows",
            "2,4,8",
            "--direct-q8-feasibility",
        ];
        assert!(Args::try_parse_from(args).unwrap().direct_q8_feasibility);
        let mut conflicting = args.to_vec();
        conflicting.extend(["--verification-trace", "old.jsonl"]);
        assert!(Args::try_parse_from(conflicting).is_err());
    }

    /// The diagnostic is fixed to endpoint and middle rejection positions,
    /// without duplicating the two-row endpoint or expanding into a sweep.
    #[test]
    fn test_rejection_positions_are_bounded() {
        assert_eq!(rejection_positions(2), vec![0, 1]);
        assert_eq!(rejection_positions(4), vec![0, 2, 3]);
        assert_eq!(rejection_positions(8), vec![0, 4, 7]);
    }

    /// Profiling cannot change actual greedy results, prefix bits, lengths,
    /// cleared suffixes or work counters, including the bounded larger pool.
    #[test]
    #[ignore = "requires isolated M40; run --include-ignored --test-threads=1"]
    fn test_cost_observer_preserves_verification() -> Result<()> {
        use m40_llm::kv_compression::{KvCompressionConfig, ScopedRuntimeConfig};
        let _scope = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
        let initial = CudaContext::total_device_bytes();
        for is_qwen in [false, true] {
            let (gguf, bytes) = fixture::build_prefill_fixture(is_qwen);
            let mut model = LoadedModel::from_gguf_with_q8_runtime(
                gguf,
                bytes,
                -1,
                Q8ExecutionConfig::parse("split", Some("6144"), Some("gate-up,down,vocabulary"))?,
                Q8FfnConfig::joint_shared_scales(),
                Q8PrefillBackend::DequantSgemm,
                Q8AttentionProjectionBackend::PackedWarp256,
            )?;
            model.allocate_kv_cache_for_layers(32)?;
            let mut profiler = new_profiler(&model.cuda)?;
            let mut trace = Trace {
                prompt: vec![1, 7, 3, 9, 4],
                tokens: Vec::new(),
                stop_ids: Vec::new(),
                vocab: model.map_lm_head()?.3,
            };
            let mut session = VerificationSession::new(&mut model, &[])?;
            let mut next = session.prefill(&trace.prompt)?;
            trace.tokens.push(next);
            for _ in 0..8 {
                next = session.serial_step(next)?;
                trace.tokens.push(next);
            }
            for width in [2, 4, 8] {
                let off = measure(&mut session, &mut profiler, &trace, width, None, false, 2)?;
                let on = measure(&mut session, &mut profiler, &trace, width, None, true, 2)?;
                assert_eq!(off["predictions"], on["predictions"]);
                assert_eq!(off["counters"], on["counters"]);
                assert_eq!(off["prefix_sha256"], on["after_prefix_sha256"]);
                assert_eq!(on["gpu_profile"]["event_pool_capacity"], EVENT_CAPACITY);
                for accepted in rejection_positions(width) {
                    let row = measure(
                        &mut session,
                        &mut profiler,
                        &trace,
                        width,
                        Some(accepted),
                        false,
                        2,
                    )?;
                    assert_eq!(row["useful_progress"], accepted + 1);
                    assert_eq!(row["has_exact_output"], true);
                    assert_eq!(row["has_cleared_suffix"], true);
                }
            }
        }
        assert_eq!(CudaContext::total_device_bytes(), initial);
        Ok(())
    }
}

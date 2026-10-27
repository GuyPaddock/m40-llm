//! Perfect-drafter target costs, independent of any real drafting algorithm.
use super::{command_output, inspect_gpu, write_record, Args};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, CudaStream},
    evaluation::identity::file_sha256,
    profile,
    verification::VerificationSession,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
#[path = "../compare_ollama/protocol.rs"]
#[allow(dead_code)]
mod protocol;
#[path = "../common/q8_runtime.rs"]
#[allow(dead_code)]
mod runtime;

struct FrozenTrace {
    prompt: Vec<u32>,
    tokens: Vec<u32>,
    stop_ids: Vec<u32>,
    manifest: Value,
}
fn token_ids(value: &Value) -> Result<Vec<u32>> {
    Ok(serde_json::from_value(value.clone())?)
}
fn read_trace(path: &std::path::Path) -> Result<FrozenTrace> {
    ensure!(
        file_sha256(path)? == "8b499787179f484aeb32e37caf837208e0994da4995918e18e34ba922d3d949e",
        "verification requires the qualified packed-projection actual512 evidence"
    );
    let records = fs::read_to_string(path)?
        .lines()
        .map(serde_json::from_str)
        .collect::<serde_json::Result<Vec<Value>>>()?;
    let manifest = records
        .iter()
        .find(|r| r["kind"] == "q8_runtime_manifest")
        .context("missing frozen manifest")?
        .clone();
    let generated = records
        .iter()
        .find(|r| r["kind"] == "q8_runtime_browser_generation" && r["label"] == "candidate")
        .context("missing qualified trace")?;
    let prompt = token_ids(&generated["prompt_token_ids"])?;
    let tokens = token_ids(&generated["generation"]["generated_token_ids"])?;
    ensure!(
        prompt.len() == 65 && tokens.len() == 512,
        "verification requires actual65 prompt/512 generated IDs"
    );
    Ok(FrozenTrace {
        prompt,
        tokens,
        stop_ids: token_ids(&manifest["candidate"]["stop_ids"])?,
        manifest,
    })
}

fn run_arm(
    model: &mut m40_llm::infer::LoadedModel,
    trace: &FrozenTrace,
    width: usize,
) -> Result<Value> {
    let ctx = model.cuda.clone();
    let kv_bytes = model
        .kv_cache
        .as_ref()
        .context("missing dense cache")?
        .actual_bytes();
    let total_start = ctx.create_event()?;
    let decode_start = ctx.create_event()?;
    let stop = ctx.create_event()?;
    let mut session = VerificationSession::new(model, &trace.stop_ids)?;
    let before = profile::snapshot();
    let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
    total_start.record(CudaStream::Decode)?;
    let total_began = Instant::now();
    let first = session.prefill(&trace.prompt)?;
    let prefill_ms = total_began.elapsed().as_secs_f64() * 1000.0;
    decode_start.record(CudaStream::Decode)?;
    let decode_began = Instant::now();
    let mut actual = vec![first];
    let mut steps = 0;
    let mut batches = 0;
    let mut tail = 0;
    let mut batch_ms = 0.0;
    let mut observed_bytes = CudaContext::total_device_bytes();
    let mut failure = None;
    while actual == trace.tokens[..actual.len()] && steps < 511 {
        let count = width.min(511 - steps);
        if width == 1 || count < width {
            let next = session.serial_step(*actual.last().unwrap())?;
            actual.push(next);
            steps += 1;
            tail += 1;
        } else {
            let began = Instant::now();
            let attempt = session.begin(&trace.tokens[steps..steps + width], false)?;
            let accepted = attempt.accepted_len();
            let predictions = attempt.predictions().to_vec();
            if accepted != width {
                failure = Some(
                    json!({"input_offset":steps,"accepted":accepted,"predictions":predictions}),
                );
                drop(attempt);
                break;
            }
            let out = attempt.commit()?;
            ensure!(!out.has_stopped, "perfect trace unexpectedly stopped");
            batch_ms += began.elapsed().as_secs_f64() * 1000.0;
            actual.extend(predictions);
            steps += width;
            batches += 1;
        }
        observed_bytes = observed_bytes.max(CudaContext::total_device_bytes());
    }
    let decode_ms = decode_began.elapsed().as_secs_f64() * 1000.0;
    let total_ms = total_began.elapsed().as_secs_f64() * 1000.0;
    stop.record(CudaStream::Decode)?;
    let gpu_total_ms = total_start.elapsed_sync(&stop, "verification_total")?;
    let gpu_decode_ms = decode_start.elapsed_sync(&stop, "verification_decode")?;
    let counters = profile::counter_delta(&before, &profile::snapshot());
    let first_divergence = actual.iter().zip(&trace.tokens).position(|(a, b)| a != b);
    let exact = actual == trace.tokens;
    let lengths = session.debug_lengths()?;
    ensure!(
        lengths.iter().all(|&n| n == session.committed_len()),
        "verification layer lengths differ"
    );
    Ok(
        json!({"width":width,"prompt_tokens":65,"generated_tokens":actual.len(),"target_input_rows":steps,
        "prefill_ms":prefill_ms,"decode_ms":decode_ms,"total_ms":total_ms,"gpu_total_interval_ms":gpu_total_ms,
        "gpu_decode_interval_ms":gpu_decode_ms,"resident_total_tps":actual.len() as f64*1000.0/total_ms,
        "verification_batches":batches,"serial_steps":tail,"verification_batches_ms":batch_ms,
        "mean_verification_batch_ms":if batches>0 {Some(batch_ms/batches as f64)} else {None},
        "has_exact_output":exact,"first_divergence_step":first_divergence,"failure":failure,"generated_token_ids":actual,
        "stop_reason":if exact {"length"} else {"exactness-failure"},
        "committed_len":session.committed_len(),"layer_lengths":lengths,"session_scratch_bytes":session.scratch_bytes(),
        "observed_tracked_device_high_water_bytes":observed_bytes,"resident_kv_bytes":kv_bytes,
        "memory_scope":"Rust-tracked boundary high-water, not allocator peak; native dense KV separate; driver/library overhead excluded",
        "counters":counters,"started_unix_us":started,"completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros(),
        "scope":"resident complete generation with identical prompt prep and 511 target input rows plus cached first token; no HTTP or real drafter"}),
    )
}

fn rejection_cost(
    model: &mut m40_llm::infer::LoadedModel,
    trace: &FrozenTrace,
    width: usize,
    accepted: usize,
) -> Result<Value> {
    let mut session = VerificationSession::new(model, &trace.stop_ids)?;
    ensure!(
        session.prefill(&trace.prompt)? == trace.tokens[0],
        "rejection reference first token differs"
    );
    let mut candidates = trace.tokens[..width].to_vec();
    candidates[accepted] = (candidates[accepted] + 1) % 151936;
    let began = Instant::now();
    let attempt = session.begin(&candidates, false)?;
    ensure!(
        attempt.accepted_len() == accepted,
        "unexpected rejection-prefix mismatch"
    );
    let result = attempt.commit()?;
    ensure!(
        result.next_token == trace.tokens[accepted],
        "incorrect rejection correction token"
    );
    let next = session.serial_step(result.next_token)?;
    let elapsed_ms = began.elapsed().as_secs_f64() * 1000.0;
    ensure!(
        next == trace.tokens[accepted + 1],
        "corrected prefix changed greedy next token"
    );
    ensure!(
        session
            .debug_lengths()?
            .iter()
            .all(|&n| n == 65 + accepted as u32 + 1),
        "rejection KV commit differs"
    );
    Ok(
        json!({"width":width,"accepted_prefix":accepted,"emitted_tokens":accepted+1,
        "elapsed_ms":elapsed_ms,"has_exact_correction":true,"prefix_tokens":65,
        "scope":"target verification plus suffix rollback and one ordinary correction step; drafting/prefill excluded"}),
    )
}

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    if v.len() % 2 == 1 {
        v[v.len() / 2]
    } else {
        (v[v.len() / 2 - 1] + v[v.len() / 2]) * 0.5
    }
}
fn acceptance_cost(probability: f64, costs: &[f64]) -> (f64, f64) {
    let k = costs.len() - 1;
    let mut cost = 0.0;
    let mut tokens = 0.0;
    for (accepted, &c) in costs.iter().enumerate() {
        let p = probability.powi(accepted as i32)
            * if accepted == k {
                1.0
            } else {
                1.0 - probability
            };
        cost += p * c;
        tokens += p * if accepted == k {
            k as f64
        } else {
            (accepted + 1) as f64
        };
    }
    (cost, tokens)
}

pub(super) fn benchmark(args: &Args) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions) && args.rows == [2, 4, 8],
        "verification requires release and --rows 2,4,8"
    );
    let path = args
        .verification_trace
        .as_ref()
        .context("missing verification trace")?;
    let trace = read_trace(path)?;
    let environment = std::env::vars()
        .filter(|(k, _)| k.starts_with("M40LLM_") || k == "CUDA_VISIBLE_DEVICES")
        .collect::<BTreeMap<_, _>>();
    runtime::validate_environment(&environment)?;
    ensure!(
        file_sha256(&args.model)? == trace.manifest["candidate"]["sha256"].as_str().unwrap_or(""),
        "verification model hash differs"
    );
    let config = runtime::RuntimeConfig {
        ffn: runtime::FfnChoice::SplitF32x4,
        prefill: runtime::PrefillChoice::DequantSgemm,
        attention_projection: runtime::AttentionProjectionChoice::PackedWarp256,
    };
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.output)?;
    let gpu = inspect_gpu()?;
    let initial = CudaContext::total_device_bytes();
    let setup = Instant::now();
    let (mut model, tokenizer) = config.load_model(&args.model, 1024)?;
    let original_stats = model.q8_execution_stats();
    let prepared = serde_json::to_value(&original_stats)?;
    runtime::validate_prepared_control(
        &original_stats,
        &original_stats,
        runtime::ControlChoice::QualifiedWarpSgemm.config(),
        config,
    )?;
    write_record(
        &mut output,
        json!({"kind":"verification_manifest","revision":command_output("git",&["rev-parse","HEAD"])?,
        "worktree":command_output("git",&["status","--short"])?,"build_profile":"release",
        "executable_sha256":file_sha256(&std::env::current_exe()?)?,"model_sha256":file_sha256(&args.model)?,
        "trace":path,"trace_sha256":file_sha256(path)?,"runtime_config":config,"environment":environment,"gpu_before":gpu,
        "prepared_weights":prepared,"setup_ms":setup.elapsed().as_secs_f64()*1000.0,
        "widths":[1,2,4,8],"trials":args.trials,"prompt_token_ids":trace.prompt,"reference_token_ids":trace.tokens,
        "scope":"perfect-drafter resident-generation upper bound for this verifier; not HTTP qualification or an original-goal claim"}),
    )?;
    for width in [1, 2, 4, 8] {
        let result = run_arm(&mut model, &trace, width)?;
        let exact = result["has_exact_output"] == true;
        write_record(
            &mut output,
            json!({"kind":"verification_preflight","result":result}),
        )?;
        ensure!(
            exact,
            "width {width} does not reproduce frozen target greedy IDs; stop before primary timing"
        );
    }
    if !args.verification_only_preflight {
        let mut trials = Vec::new();
        let mut rejections = Vec::new();
        for trial in 0..args.trials {
            let mut widths = [1, 2, 4, 8];
            widths.rotate_left(trial % 4);
            if trial % 2 == 1 {
                widths.reverse();
            }
            let mut measurements = Vec::new();
            for width in widths {
                let gpu_before = inspect_gpu()?;
                let result = run_arm(&mut model, &trace, width)?;
                let exact = result["has_exact_output"] == true;
                write_record(
                    &mut output,
                    json!({"kind":"verification_trial","trial":trial,"width":width,
                    "gpu_before":gpu_before,"gpu_after":inspect_gpu()?,"output":tokenizer.decode(&trace.tokens)?,"result":result}),
                )?;
                ensure!(
                    exact,
                    "primary verification width {width} changed frozen IDs"
                );
                measurements.push(result);
            }
            trials.push(measurements);
        }
        for trial in 0..args.trials {
            for width in [2, 4, 8] {
                for accepted in 0..width {
                    let gpu_before = inspect_gpu()?;
                    let result = rejection_cost(&mut model, &trace, width, accepted)?;
                    write_record(
                        &mut output,
                        json!({"kind":"verification_rejection_cost","trial":trial,"gpu_before":gpu_before,
                "gpu_after":inspect_gpu()?,"result":result}),
                    )?;
                    rejections.push(result);
                }
            }
        }
        for width in [2, 4, 8] {
            let find = |trial: &Vec<Value>, n: usize| -> f64 {
                trial.iter().find(|r| r["width"] == n).unwrap()["total_ms"]
                    .as_f64()
                    .unwrap()
            };
            let pairs = trials
                .iter()
                .map(|t| protocol::Pair {
                    control_tps: 512000.0 / find(t, 1),
                    candidate_tps: 512000.0 / find(t, width),
                })
                .collect::<Vec<_>>();
            let comparison = protocol::compare_pairs(&pairs)?;
            let serial_ms = median(
                &trials
                    .iter()
                    .map(|t| {
                        t.iter().find(|r| r["width"] == 1).unwrap()["decode_ms"]
                            .as_f64()
                            .unwrap()
                            / 511.0
                    })
                    .collect::<Vec<_>>(),
            );
            let full = median(
                &trials
                    .iter()
                    .map(|t| {
                        t.iter().find(|r| r["width"] == width).unwrap()
                            ["mean_verification_batch_ms"]
                            .as_f64()
                            .unwrap()
                    })
                    .collect::<Vec<_>>(),
            );
            let mut costs = (0..width)
                .map(|a| {
                    median(
                        &rejections
                            .iter()
                            .filter(|r| r["width"] == width && r["accepted_prefix"] == a)
                            .map(|r| r["elapsed_ms"].as_f64().unwrap())
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            costs.push(full);
            let curves = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
                .into_iter()
                .map(|p| {
                    let (cost, tokens) = acceptance_cost(p, &costs);
                    json!({"per_token_acceptance_probability":p,"expected_tokens_per_cycle":tokens,
                    "target_ms_per_cycle":cost,"max_drafter_ms_to_break_even":serial_ms*tokens-cost,
                    "max_drafter_ms_for_three_percent_gain":serial_ms*tokens/1.03-cost})
                })
                .collect::<Vec<_>>();
            write_record(
                &mut output,
                json!({"kind":"verification_summary","width":width,"pairs":pairs.iter().map(|p|json!({"control_tps":p.control_tps,"candidate_tps":p.candidate_tps})).collect::<Vec<_>>(),"gain":comparison.gain,
                "bootstrap_gain_ci95":comparison.bootstrap_gain_ci95,"promotion_threshold":comparison.promotion_threshold,
                "needs_variation_extension":comparison.needs_variation_extension,"needs_fresh_confirmation":comparison.needs_fresh_confirmation,
                "passes_resident_screen":comparison.passes_promotion_gate,"control":comparison.control,"candidate":comparison.candidate,
                "serial_mean_step_ms":serial_ms,"mean_full_verification_ms":full,"rejection_cycle_ms_by_accepted_prefix":costs,
                "acceptance_curves":curves,"has_real_drafter":false,"is_http_claim":false,
                "curve_scope":"iid acceptance sensitivity, not observed drafter acceptance; rejection costs measured at 65-token prefix, full acceptance averaged across 511 rows"}),
            )?;
        }
    }
    ensure!(
        serde_json::to_value(model.q8_execution_stats())? == prepared,
        "verification changed immutable prepared cache"
    );
    runtime::validate_prepared_control(
        &original_stats,
        &model.q8_execution_stats(),
        runtime::ControlChoice::QualifiedWarpSgemm.config(),
        config,
    )?;
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "verification leaked tracked device bytes"
    );
    write_record(
        &mut output,
        json!({"kind":"verification_complete","is_complete":true,"is_preflight_only":args.verification_only_preflight,
        "tracked_device_bytes_before":initial,"tracked_device_bytes_after":CudaContext::total_device_bytes(),"gpu_after_release":inspect_gpu()?}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Full acceptance has no correction cycle; zero acceptance emits exactly
    /// one corrected token and includes the measured rejection cost.
    #[test]
    fn test_acceptance_cost_accounts_for_corrections_and_full_acceptance() {
        assert_eq!(acceptance_cost(1.0, &[9.0, 10.0, 7.0]), (7.0, 2.0));
        assert_eq!(acceptance_cost(0.0, &[9.0, 10.0, 7.0]), (9.0, 1.0));
        assert_eq!(acceptance_cost(0.5, &[9.0, 10.0, 7.0]), (8.75, 1.5));
    }
}

use super::{check_result, execute, gpu, protocol, records, runtime, ResultRow, Trace};
use anyhow::{ensure, Context, Result};
use m40_llm::{cuda::CudaContext, infer::LoadedModel, verification::VerificationDownProjection};
use serde_json::{json, Value};
use std::{
    fs::{File, OpenOptions},
    path::Path,
    time::Instant,
};

const ARMS: [Option<VerificationDownProjection>; 3] = [
    None,
    Some(VerificationDownProjection::MultiRow),
    Some(VerificationDownProjection::SharedWeights),
];
const WORK_FIELDS: [&str; 5] = [
    "attempts",
    "rejections",
    "lookup_count",
    "accepted_lengths_1_to_4",
    "cleared_suffix_words",
];

fn storage_stats(mut stats: Value) -> Value {
    stats
        .as_object_mut()
        .unwrap()
        .retain(|key, _| !key.ends_with("_elapsed_ms"));
    stats
}

fn evaluate(
    model: &mut LoadedModel,
    trace: &Trace,
    arm: usize,
    is_correctness: bool,
    oracle: Option<&[Vec<f32>]>,
) -> Result<ResultRow> {
    let result = execute(
        model,
        &trace.prompt_token_ids,
        trace.max_tokens,
        &trace.stop_ids,
        ARMS[arm],
        is_correctness,
        oracle,
    )?;
    check_result(&result, trace)?;
    Ok(result)
}

fn validate_arms(arms: &[Value], trace: &Trace, is_correctness: bool) -> Result<()> {
    ensure!(arms.len() == ARMS.len(), "missing live schedule");
    for (row, schedule) in arms.iter().zip(ARMS) {
        ensure!(
            row["down_projection"] == json!(schedule)
                && row["generated_token_ids"] == json!(trace.generated_token_ids)
                && row["stop_reason"] == trace.stop_reason
                && row["includes_debug_readbacks"] == is_correctness,
            "schedule, IDs, stop, or capture mismatch for {}",
            trace.id
        );
    }
    for key in WORK_FIELDS {
        ensure!(
            arms[1][key] == arms[2][key],
            "live draft work differs: {key}"
        );
    }
    if is_correctness {
        for row in &arms[1..] {
            ensure!(
                row["compared_full_logit_rows"] == trace.generated_token_ids.len() - 1
                    && row["has_full_logit_tolerance_passed"] == true,
                "full live logit coverage missing for {}",
                trace.id
            );
        }
    }
    Ok(())
}

fn validate_correctness(path: &Path, manifest: &Value, traces: &[Trace]) -> Result<()> {
    let rows = records::rows(path)?;
    ensure!(rows.len() == traces.len() + 2, "incorrect cohort size");
    for key in [
        "kind",
        "revision",
        "executable_sha256",
        "heldout_sha256",
        "model_sha256",
        "environment",
        "arms",
    ] {
        ensure!(
            rows[0][key] == manifest[key],
            "correctness provenance differs: {key}"
        );
    }
    for key in ["prepared", "joint"] {
        ensure!(
            rows[0][key].is_object()
                && storage_stats(rows[0][key].clone()) == storage_stats(manifest[key].clone()),
            "correctness weight storage differs: {key}"
        );
    }
    ensure!(
        rows[0]["is_correctness_only"] == true
            && rows[rows.len() - 1]["does_release_all_device_bytes"] == true
            && rows[rows.len() - 1]["is_correctness_passed"] == true,
        "complete executable-bound correctness required"
    );
    for (trace, row) in traces.iter().zip(&rows[1..rows.len() - 1]) {
        ensure!(
            row["kind"] == "copy_schedule_correctness"
                && row["id"] == trace.id
                && row["has_exact_valid_kv"] == true
                && row["valid_kv_words"].as_u64().is_some_and(|n| n > 0),
            "missing exact valid KV evidence for {}",
            trace.id
        );
        validate_arms(
            row["results"].as_array().context("missing arms")?,
            trace,
            true,
        )?;
    }
    Ok(())
}

fn collect(
    model: &mut LoadedModel,
    trace: &Trace,
    out: &mut File,
    phase: &str,
    rounds: &mut Vec<[f64; 3]>,
    count: usize,
) -> Result<()> {
    for round in rounds.len()..count {
        let start = round % ARMS.len();
        let mut order = [start, (start + 1) % ARMS.len(), (start + 2) % ARMS.len()];
        if round % 2 != 0 {
            order.reverse();
        }
        let mut results = vec![Value::Null; ARMS.len()];
        let mut rates = [0.; 3];
        for arm in order {
            let before = gpu()?;
            let result = evaluate(model, trace, arm, false, None)?;
            rates[arm] = result.data["resident_tps"].as_f64().unwrap();
            records::write(
                out,
                json!({"kind":"copy_schedule_trial","id":trace.id,"round":round,"phase":phase,
                    "arm":arm,"gpu_before":before,"gpu_after":gpu()?,"result":result.data}),
            )?;
            results[arm] = result.data;
        }
        validate_arms(&results, trace, false)?;
        ensure!(
            results[2]["attempts"].as_u64().is_some_and(|n| n > 0),
            "empty copy workload"
        );
        rounds.push(rates);
    }
    Ok(())
}

fn measure(model: &mut LoadedModel, trace: &Trace, out: &mut File) -> Result<&'static str> {
    collect(model, trace, out, "warmup", &mut Vec::new(), 1)?;
    let mut rounds = Vec::new();
    let mut phase = "screen";
    let mut count = 5;
    loop {
        collect(model, trace, out, phase, &mut rounds, count)?;
        let compare = |control, candidate| {
            protocol::compare_pairs(
                &rounds
                    .iter()
                    .map(|rates| protocol::Pair {
                        control_tps: rates[control],
                        candidate_tps: rates[candidate],
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let ordinary = compare(0, 2)?;
        let existing = compare(0, 1)?;
        let multi = compare(1, 2)?;
        let incremental_threshold = 2.0
            * multi
                .control
                .mad_over_median
                .max(multi.candidate.mad_over_median);
        let is_unstable = ordinary.needs_variation_extension || multi.needs_variation_extension;
        let needs_confirmation = ordinary.needs_fresh_confirmation
            || (multi.gain - incremental_threshold).abs() <= 0.01
            || (multi.gain > incremental_threshold && multi.bootstrap_gain_ci95[0] <= 0.0);
        let passes_incremental =
            multi.gain > incremental_threshold && multi.bootstrap_gain_ci95[0] > 0.;
        records::write(
            out,
            json!({"kind":"copy_schedule_comparison","id":trace.id,"phase":phase,
                "existing_vs_ordinary":existing,"shared_vs_ordinary":ordinary,"shared_vs_multi_row":multi,
                "is_unstable":is_unstable,"needs_confirmation":needs_confirmation,
                "incremental_threshold":incremental_threshold,"passes_incremental":passes_incremental}),
        )?;
        if is_unstable {
            if count == 5 {
                count = 10;
                continue;
            }
            return Ok("INCONCLUSIVE");
        }
        if phase == "screen" && needs_confirmation {
            phase = "confirmation";
            rounds.clear();
            count = 20;
            continue;
        }
        return Ok(if ordinary.passes_promotion_gate && passes_incremental {
            "PASS"
        } else {
            "FAIL"
        });
    }
}

pub fn run(
    model_path: &Path,
    heldout: &Path,
    output: &Path,
    correctness: Option<&Path>,
) -> Result<()> {
    let traces = records::read_traces(Some(heldout))?;
    ensure!(traces.len() == 21, "wrong frozen correctness cohort");
    let copy_cases: Vec<_> = traces
        .iter()
        .filter(|trace| trace.split == "heldout" && trace.class == "copy")
        .collect();
    ensure!(copy_cases.len() == 2, "wrong frozen copy cohort");
    let initial = CudaContext::total_device_bytes();
    let load = Instant::now();
    let runtime::Runtime {
        mut model,
        reference,
        environment,
        ..
    } = runtime::load(model_path)?;
    let load_ms = load.elapsed().as_secs_f64() * 1000.;
    let manifest = json!({
        "kind":"copy_schedule_manifest","revision":records::command("git",&["rev-parse","HEAD"])?,
        "is_correctness_only":correctness.is_none(),"arms":ARMS,"environment":environment,
        "heldout_sha256":records::hash_file(heldout)?,"model_sha256":reference["model"]["sha256"],
        "executable_sha256":records::hash_file(&std::env::current_exe()?)?,
        "prepared":model.q8_execution_stats(),"joint":model.q8_ffn_pair_stats(),
        "driver_gpu":gpu()?, "preflight_and_load_ms":load_ms
    });
    if let Some(path) = correctness {
        validate_correctness(path, &manifest, &traces)?;
    }
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    records::write(&mut out, manifest.clone())?;
    if correctness.is_none() {
        let mut cleared = 0u64;
        for trace in &traces {
            let serial = evaluate(&mut model, trace, 0, true, None)?;
            let mut arms = vec![serial.data];
            for arm in 1..ARMS.len() {
                let candidate = evaluate(&mut model, trace, arm, true, Some(&serial.logits))?;
                ensure!(
                    serial.kv == candidate.kv,
                    "valid KV mismatch for {}",
                    trace.id
                );
                arms.push(candidate.data);
            }
            validate_arms(&arms, trace, true)?;
            cleared += arms[2]["cleared_suffix_words"].as_u64().unwrap();
            records::write(
                &mut out,
                json!({"kind":"copy_schedule_correctness","id":trace.id,
                "has_exact_valid_kv":true,"valid_kv_words":serial.kv.len(),"results":arms}),
            )?;
            eprintln!("copy schedule correctness {}", trace.id);
        }
        ensure!(cleared > 0, "no natural draft rollback exercised");
    } else {
        for trace in copy_cases {
            let disposition = measure(&mut model, trace, &mut out)?;
            records::write(
                &mut out,
                json!({"kind":"copy_schedule_gate","id":trace.id,
                "disposition":disposition}),
            )?;
            eprintln!("copy schedule gate {} {disposition}", trace.id);
        }
    }
    ensure!(
        serde_json::to_value(model.q8_execution_stats())? == manifest["prepared"]
            && serde_json::to_value(model.q8_ffn_pair_stats())? == manifest["joint"],
        "copy study weight storage changed"
    );
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "copy study leaked device bytes"
    );
    records::write(
        &mut out,
        json!({"kind":"copy_schedule_complete",
        "is_correctness_passed":correctness.is_none(),"does_release_all_device_bytes":true,
        "device_bytes_after":CudaContext::total_device_bytes()}),
    )
}

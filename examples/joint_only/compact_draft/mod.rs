//! Synthetic proposal-cost admission only. No learned artifact or teacher data.
mod pipeline;
mod reference;

use super::run::{gpu, identity, load, Args, Report};
use anyhow::{ensure, Result};
use m40_llm::{cuda::CudaContext, infer::Q8FfnOwnership, profile};
use pipeline::Proposal;
use serde_json::json;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const TRIALS: usize = 5;
const DECISIONS: usize = 200;
const ADMISSION_MS: f64 = 3.884;

fn percentile(values: &[f64], fraction: f64) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let position = fraction * (values.len() - 1) as f64;
    let lo = position.floor() as usize;
    let hi = position.ceil() as usize;
    values[lo] + (values[hi] - values[lo]) * (position - lo as f64)
}

fn check_gpu() -> Result<serde_json::Value> {
    let state = gpu()?;
    ensure!(
        state["clocks"] == "GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852, P0, 1113, 3004",
        "M40/clock state changed: {state}"
    );
    Ok(state)
}

pub(super) fn execute(args: &Args, report: &mut Report) -> Result<()> {
    ensure!(
        args.ownership == "joint-only",
        "compact cost requires qualified joint-only ownership"
    );
    let began = Instant::now();
    let initial = CudaContext::total_device_bytes();
    let model = load(&args.model, Q8FfnOwnership::JointOnly)?;
    let model_load_ms = began.elapsed().as_secs_f64() * 1000.;
    let target_bytes = CudaContext::total_device_bytes();
    let parameter_start = Instant::now();
    let data = reference::Parameters::new();
    let parameter_creation_ms = parameter_start.elapsed().as_secs_f64() * 1000.;
    let mut probe = Proposal::new(&model.cuda, &data)?;
    let ids = probe.propose()?;
    let validation = reference::validate(&data, &probe.read_hidden()?, &probe.read_logits()?, ids)?;
    report.record(json!({"kind":"compact_cost_setup","architecture":"three affine D2048->rank256 SiLU heads; shared rank256->151936 full vocabulary",
        "is_trained":false,"is_acceptance_measured":false,"parameter_seed":"fixed integer mixer v1",
        "parameter_bytes":data.bytes(),"parameter_sha256":data.sha256(),"parameter_creation_ms":parameter_creation_ms,"proposal_setup":probe.setup,
        "target":identity(&model),"target_load_ms":model_load_ms,"additional_device_bytes":CudaContext::total_device_bytes()-target_bytes,
        "synthetic_feature_width":2048,"feature_source":"already-available device target-shaped synthetic hidden; D2D snapshot charged",
        "same_stream_contract":"target producer and proposal on decode stream; no target forward or teacher generation",
        "validation":validation,"trials":TRIALS,"decisions_per_trial":DECISIONS,"admission_median_ms":ADMISSION_MS,
        "primary_boundary":"wall clock around feature snapshot, two cuBLAS calls, bias, SiLU, three argmax, decode synchronization and 12-byte D2H",
        "excludes":"one-time synthetic parameter creation/upload and target load, reported separately; unmeasured live feature extraction/target interference/acceptance",
        "build_profile":"release","cuda_kernels_added":0}))?;
    // Warm the exact fixed proposal, not a geometry or implementation search.
    let start = Instant::now();
    let mut warmup_calls = 0;
    while start.elapsed().as_secs_f64() < 1. {
        ensure!(probe.propose()? == ids, "proposal changed during warmup");
        warmup_calls += 1;
    }
    report.record(json!({"kind":"compact_cost_warmup","calls":warmup_calls,"elapsed_ms":start.elapsed().as_secs_f64()*1000.,"gpu":check_gpu()?}))?;
    let mut all_times = Vec::new();
    let mut trial_medians = Vec::new();
    for trial in 0..TRIALS {
        ensure!(
            began.elapsed().as_secs_f64() < 4. * 3600.,
            "cost-probe GPU budget exhausted"
        );
        let before_gpu = check_gpu()?;
        let before = profile::snapshot();
        let memory = CudaContext::total_device_bytes();
        let started_unix_us = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let mut times = Vec::with_capacity(DECISIONS);
        for _ in 0..DECISIONS {
            let start = Instant::now();
            let actual = probe.propose()?;
            times.push(start.elapsed().as_secs_f64() * 1000.);
            ensure!(actual == ids, "proposal IDs changed");
        }
        let counters = profile::counter_delta(&before, &profile::snapshot());
        ensure!(
            counters
                == profile::OpCounters {
                    launches: 5 * DECISIONS as u64,
                    cublas_calls: 2 * DECISIONS as u64,
                    stream_syncs: DECISIONS as u64,
                    d2h_copies: DECISIONS as u64,
                    d2h_bytes: 12 * DECISIONS as u64,
                    ..Default::default()
                },
            "proposal work differs from fixed schedule: {counters:?}"
        );
        ensure!(
            CudaContext::total_device_bytes() == memory,
            "proposal allocation changed"
        );
        let median = percentile(&times, 0.5);
        trial_medians.push(median);
        all_times.extend_from_slice(&times);
        report.record(json!({"kind":"compact_cost_trial","trial":trial,"started_unix_us":started_unix_us,
            "completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros(),
            "all_in_proposal_ms":times,"median_ms":median,"p95_ms":percentile(&times,0.95),"p99_ms":percentile(&times,0.99),
            "counters":counters,"gpu_before":before_gpu,"gpu_after":check_gpu()?,"tracked_device_bytes":memory}))?;
    }
    // Event observations are outside primary all-in timing, including their
    // measurement overhead. They do not replace complete wall-clock costs.
    for diagnostic in 0..5 {
        report.record(json!({"kind":"compact_cost_stage_diagnostic","diagnostic":diagnostic,"stages":probe.diagnostic()?}))?;
    }
    let median = percentile(&all_times, 0.5);
    let p95 = percentile(&all_times, 0.95);
    let is_admitted = median <= ADMISSION_MS;
    probe.verify_owners(&data)?;
    drop(probe);
    ensure!(
        CudaContext::total_device_bytes() == target_bytes,
        "proposal owners leaked"
    );
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "target owners leaked"
    );
    report.record(json!({"kind":"compact_cost_decision","median_ms":median,"p95_ms":p95,"p99_ms":percentile(&all_times,0.99),
        "min_ms":percentile(&all_times,0.),"max_ms":percentile(&all_times,1.),"trial_medians_ms":trial_medians,
        "admission_median_ms":ADMISSION_MS,"passes_cost_admission":is_admitted,
        "is_acceptance_measured":false,"is_http_measured":false,"is_training_authorized":false,
        "next":if is_admitted {"review causal dataset/licensing/split and small-pilot plan before teacher generation or training"}else{"stop fixed pilot; no architecture/geometry rescue"},
        "tracked_device_bytes_after":CudaContext::total_device_bytes(),"elapsed_gpu_session_hours":began.elapsed().as_secs_f64()/3600.,
        "is_complete":true}))
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Interpolated percentiles expose variation rather than rounding away tails.
    #[test]
    fn test_percentiles_preserve_endpoints_and_median() {
        assert_eq!(percentile(&[4., 1., 3., 2.], 0.), 1.);
        assert_eq!(percentile(&[4., 1., 3., 2.], 0.5), 2.5);
        assert_eq!(percentile(&[4., 1., 3., 2.], 1.), 4.);
    }
}

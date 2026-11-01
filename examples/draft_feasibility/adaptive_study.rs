use super::{
    adaptive_replay::{self, Policy},
    cost::CostModel,
    records,
    replay::Replay,
};
use anyhow::{ensure, Result};
use serde_json::json;
use std::{fs::OpenOptions, path::Path};

pub fn run(heldout: &Path, output: &Path) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "CPU study requires a release build"
    );
    ensure!(
        records::command("git", &["status", "--porcelain"])?.is_empty(),
        "commit source first"
    );
    let processes = records::command(
        "nvidia-smi",
        &["--query-compute-apps=pid", "--format=csv,noheader"],
    )?;
    ensure!(
        processes.is_empty(),
        "do not overlap CPU timing with GPU inference"
    );
    let traces = records::read_traces(Some(heldout))?;
    ensure!(
        traces.len() == 21,
        "all 21 retrospective traces are required"
    );
    let model = CostModel::load()?;
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)?;
    records::write(
        &mut output,
        json!({"kind":"adaptive_manifest","revision":records::command("git",&["rev-parse","HEAD"])?,
        "executable_sha256":records::hash_file(&std::env::current_exe()?)?,
        "heldout_source_sha256":records::hash_file(heldout)?,"target_sha256":records::hash_file(records::TARGET.as_ref())?,
        "reference_sha256":records::hash_file(records::REFERENCE.as_ref())?,"traces":traces,
        "protocol":"2026-10-31-adaptive-drafting-plan.md","is_retrospective":true,
        "policy_proposal_allowance_ms":0.1,"minimum_gain":0.03,"recovery_rows":32,
        "cpu_timing_scope":"construction, source lookup, controller, candidate assembly, committed maintenance, teardown; excludes future scorer and diagnostic serialization",
        "cost_table_scope":"historical hardware calibration prepared once, before request timing",
        "is_measured_gpu_speedup":false,"is_measured_http":false}),
    )?;
    let mut cases = 0;
    for scale in [1., 1.25, 2.] {
        let table = model.prepare_decision_costs(scale)?;
        records::write(
            &mut output,
            json!({"kind":"adaptive_cost_table","scale":scale,"first_position":65,"last_position":569,"costs":table}),
        )?;
        for trace in &traces {
            for policy in [Policy::AlwaysAttempt, Policy::Adaptive] {
                let _warmup = adaptive_replay::run(trace, policy, &table)?;
                let trials = (0..5)
                    .map(|_| adaptive_replay::run(trace, policy, &table))
                    .collect::<Result<Vec<_>>>()?;
                for trial in &trials {
                    ensure!(
                        trial.replay.signature() == trials[0].replay.signature()
                            && trial.decisions == trials[0].decisions,
                        "nondeterministic policy"
                    );
                }
                let mut row = json!({"kind":"adaptive_case","id":trace.id,"original_split":trace.split,
                    "interpretation":"retrospective regression, not fresh held-out evaluation","scale":scale,"policy":policy,
                    "trials":trials,"visited_stats":trials[0].replay.statistics()});
                let paths: Vec<Replay> = trials.into_iter().map(|t| t.replay).collect();
                row["projections"] = model.project_scaled(trace, &paths, scale)?;
                records::write(&mut output, row)?;
                cases += 1;
            }
        }
    }
    ensure!(cases == 126, "missing trace/policy/scale cases");
    ensure!(
        records::command(
            "nvidia-smi",
            &["--query-compute-apps=pid", "--format=csv,noheader"]
        )?
        .is_empty(),
        "GPU workload appeared during CPU study"
    );
    records::write(
        &mut output,
        json!({"kind":"adaptive_complete","cases":cases,"trials":cases*5,
        "is_complete":true,"is_measured_gpu_speedup":false,"is_measured_http":false}),
    )
}

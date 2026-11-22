//! Measure complete four-token transactions with unchanged weights, inputs and KV storage.
use super::*;
use std::fs::File;

const SCHEDULES: [(VerificationDownProjection, &str); 3] = [
    (VerificationDownProjection::Serial, "serial"),
    (VerificationDownProjection::MultiRow, "multi-row"),
    (VerificationDownProjection::SharedWeights, "shared-weights"),
];

pub(super) fn validate_reports(dir: &Path, frozen: &mut Frozen, executable: &str) -> Result<()> {
    let mut control = None;
    for (schedule, name) in SCHEDULES {
        frozen.down = schedule;
        let input = fs::read_to_string(dir.join(format!("{name}.jsonl")))?;
        let rows = input
            .lines()
            .map(serde_json::from_str)
            .collect::<serde_json::Result<Vec<Value>>>()?;
        validate_correctness(&rows, frozen, executable)?;
        let browser = rows
            .iter()
            .find(|r| r["kind"] == "direct_verifier_browser_correctness")
            .context("missing browser evidence")?;
        let digests = browser["full_logit_comparisons"]
            .as_array()
            .context("missing logit rows")?
            .iter()
            .map(|r| {
                r["comparison"]["actual_fp32le_sha256"]
                    .as_str()
                    .filter(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
                    .map(str::to_owned)
                    .context("missing raw logit digest")
            })
            .collect::<Result<Vec<_>>>()?;
        let signature = (
            digests,
            browser["valid_kv_sha256"]
                .as_str()
                .filter(|s| s.len() == 64)
                .context("missing valid KV digest")?
                .to_owned(),
        );
        ensure!(
            control.get_or_insert_with(|| signature.clone()) == &signature,
            "four-row logits or KV differ for {name}"
        );
    }
    frozen.down = VerificationDownProjection::Serial;
    Ok(())
}

fn order(round: usize) -> [usize; 3] {
    let mut slots = [0, 1, 2];
    slots.rotate_left((round / 2) % 3);
    if round % 2 == 1 {
        slots.reverse();
    }
    slots
}

fn rounds(
    model: &mut LoadedModel,
    frozen: &mut Frozen,
    out: &mut File,
    phase: &str,
    rates: &mut Vec<[f64; 3]>,
    count: usize,
) -> Result<()> {
    for round in rates.len()..count {
        let mut measured = [0.; 3];
        for (position, slot) in order(round).into_iter().enumerate() {
            frozen.down = SCHEDULES[slot].0;
            let before = check_gpu()?;
            let result = measurement::run_arm(model, frozen, true)?;
            let batches = result["batch_ms"]
                .as_array()
                .context("missing complete target timings")?;
            ensure!(
                batches.len() == 127,
                "wrong number of four-row transactions"
            );
            let elapsed = batches
                .iter()
                .map(|v| {
                    v.as_f64()
                        .filter(|t| t.is_finite() && *t > 0.)
                        .context("invalid target timing")
                })
                .sum::<Result<f64>>()?;
            measured[slot] = 127000. / elapsed;
            write_record(
                out,
                json!({"kind":"down_target_trial","phase":phase,"round":round,"order":position,
                "down_projection":frozen.down,"batches_per_second":measured[slot],"gpu_before":before,"gpu_after":check_gpu()?,"result":result}),
            )?;
        }
        rates.push(measured);
    }
    Ok(())
}

struct TargetDecision {
    unstable: bool,
    needs_confirmation: bool,
    passed: bool,
}

fn summarize(out: &mut File, phase: &str, rates: &[[f64; 3]]) -> Result<TargetDecision> {
    let mut unstable = false;
    let mut confirm = false;
    let mut passed = true;
    for (slot, (control, _)) in SCHEDULES[..2].iter().enumerate() {
        let pairs: Vec<_> = rates
            .iter()
            .map(|r| protocol::Pair {
                control_tps: r[slot],
                candidate_tps: r[2],
            })
            .collect();
        let stats = protocol::compare_pairs(&pairs)?;
        let saving = stats.gain / (1. + stats.gain);
        let threshold = 0.05_f64.max(
            2. * stats
                .control
                .mad_over_median
                .max(stats.candidate.mad_over_median),
        );
        let lower = stats.bootstrap_gain_ci95[0];
        unstable |= stats.needs_variation_extension;
        confirm |= stats.needs_fresh_confirmation || (saving - threshold).abs() <= 0.01;
        passed &= !stats.needs_variation_extension && saving >= threshold && lower > 0.;
        write_record(
            out,
            json!({"kind":"down_target_summary","phase":phase,"control":control,"rate_comparison":stats,
            "median_target_latency_saving_fraction":saving,"required_latency_saving_fraction":threshold}),
        )?;
    }
    Ok(TargetDecision {
        unstable,
        needs_confirmation: confirm,
        passed,
    })
}

pub(super) fn measure(model: &mut LoadedModel, frozen: &mut Frozen, out: &mut File) -> Result<()> {
    for (schedule, _) in SCHEDULES {
        frozen.down = schedule;
        let result = measurement::run_arm(model, frozen, true)?;
        write_record(
            out,
            json!({"kind":"down_target_warmup","down_projection":schedule,"result":result,"gpu_after":check_gpu()?}),
        )?;
    }
    let mut rates = Vec::new();
    rounds(model, frozen, out, "screen", &mut rates, 5)?;
    let mut decision = summarize(out, "screen", &rates)?;
    if decision.unstable {
        rounds(model, frozen, out, "extension", &mut rates, 10)?;
        decision = summarize(out, "extended", &rates)?;
    }
    if !decision.unstable && decision.needs_confirmation {
        let mut fresh = Vec::new();
        rounds(model, frozen, out, "confirmation", &mut fresh, 20)?;
        decision = summarize(out, "confirmation", &fresh)?;
    }
    for round in 0..5 {
        for accepted in 0..4 {
            for slot in order(round + accepted) {
                frozen.down = SCHEDULES[slot].0;
                let before = check_gpu()?;
                let result = measurement::rejection(model, frozen, accepted)?;
                write_record(
                    out,
                    json!({"kind":"down_target_rejection","round":round,"down_projection":frozen.down,
                    "gpu_before":before,"gpu_after":check_gpu()?,"result":result}),
                )?;
            }
        }
    }
    write_record(
        out,
        json!({"kind":"down_target_gate","unstable":decision.unstable,"passes_complete_target_gate":decision.passed,
        "scope":"perfect four-token target plus commit versus both controls; partial rejection and correction reported separately"}),
    )
}

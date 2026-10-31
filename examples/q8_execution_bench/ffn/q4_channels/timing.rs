use super::pipeline::{Arm, Pipeline};
use crate::run::{inspect_gpu, write_record, Args};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::fs::File;

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    (sorted[(n - 1) / 2] + sorted[n / 2]) * 0.5
}

fn variation(times: &[f64]) -> f64 {
    let rates: Vec<_> = times.iter().map(|t| 1.0 / t).collect();
    let middle = median(&rates);
    median(&rates.iter().map(|v| (v - middle).abs()).collect::<Vec<_>>()) / middle
}

fn summary(pairs: &[(f64, f64)]) -> Result<Value> {
    let mut report = super::super::ablation::summarize(pairs)?;
    let a = variation(&pairs.iter().map(|v| v.0).collect::<Vec<_>>());
    let b = variation(&pairs.iter().map(|v| v.1).collect::<Vec<_>>());
    let threshold = 0.05f64.max(2.0 * a.max(b));
    let gain = report["throughput_gain"].as_f64().context("missing gain")?;
    let lower = report["bootstrap_gain_ci95"][0]
        .as_f64()
        .context("missing bound")?;
    report["throughput_mad_over_median"] = json!([a, b]);
    report["is_stable"] = json!(a.max(b) <= 0.05);
    report["required_gain"] = json!(threshold);
    report["passes_micro_screen"] = json!(a.max(b) <= 0.05 && gain >= threshold && lower > 0.0);
    report["needs_confirmation"] = json!(
        a.max(b) <= 0.05
            && ((gain - threshold).abs() <= 0.01 || (gain >= threshold && lower <= 0.0))
    );
    Ok(report)
}

fn collect(
    output: &mut File,
    pipeline: &Pipeline<'_>,
    args: &Args,
    arms: [Arm; 2],
    case: &Value,
    pairs: &mut Vec<(f64, f64)>,
    phase: (&str, usize),
) -> Result<()> {
    let (phase, total) = phase;
    while pairs.len() < total {
        let index = pairs.len();
        let order = if index.is_multiple_of(2) {
            [0, 1]
        } else {
            [1, 0]
        };
        let mut times = [Value::Null, Value::Null];
        for slot in order {
            times[slot] = pipeline.time(arms[slot], args.iterations)?;
        }
        let a = times[0]["gpu_ms"]
            .as_f64()
            .context("missing control timing")?;
        let b = times[1]["gpu_ms"]
            .as_f64()
            .context("missing candidate timing")?;
        write_record(
            output,
            json!({"kind":"q4_channel_pair","case":case,"phase":phase,"pair":index,"order":order,"times":times}),
        )?;
        pairs.push((a, b));
    }
    Ok(())
}

pub(super) fn compare(
    output: &mut File,
    pipeline: &Pipeline<'_>,
    args: &Args,
    arms: [Arm; 2],
    case: &Value,
) -> Result<()> {
    let before = inspect_gpu()?;
    let mut pairs = Vec::new();
    collect(
        output,
        pipeline,
        args,
        arms,
        case,
        &mut pairs,
        ("screen", 5),
    )?;
    if summary(&pairs)?["is_stable"] == false {
        collect(
            output,
            pipeline,
            args,
            arms,
            case,
            &mut pairs,
            ("screen", 10),
        )?;
    }
    let screen = summary(&pairs)?;
    let mut confirmation = Value::Null;
    if screen["needs_confirmation"] == true {
        let mut fresh = Vec::new();
        collect(
            output,
            pipeline,
            args,
            arms,
            case,
            &mut fresh,
            ("confirmation", 20),
        )?;
        confirmation = summary(&fresh)?;
    }
    let decision = if confirmation.is_null() {
        &screen
    } else {
        &confirmation
    };
    write_record(
        output,
        json!({"kind":"q4_channel_comparison","case":case,"arms":arms,
        "screen":screen,"confirmation":confirmation,"passes_operator_screen":decision["passes_micro_screen"],
        "gpu_before":before,"gpu_after":inspect_gpu()?,"has_selector_cost":false,
        "interpretation":"supplied-index execution upper bound; no model quality or E2E claim"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strong gains pass, borderline gains request one fresh confirmation, noisy pairs fail.
    #[test]
    fn test_decisions() -> Result<()> {
        let strong = summary(&[(2.0, 1.0); 5])?;
        assert_eq!(strong["passes_micro_screen"], true);
        assert_eq!(strong["needs_confirmation"], false);
        assert_eq!(summary(&[(1.05, 1.0); 5])?["needs_confirmation"], true);
        let unstable = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)];
        assert_eq!(summary(&unstable)?["is_stable"], false);
        assert_eq!(summary(&unstable)?["passes_micro_screen"], false);
        Ok(())
    }
}

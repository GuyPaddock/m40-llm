use crate::run::{inspect_gpu, write_record};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::fs::File;

pub(super) fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    (sorted[(n - 1) / 2] + sorted[n / 2]) * 0.5
}

pub(super) fn variation(times: &[f64]) -> f64 {
    let rates: Vec<_> = times.iter().map(|t| 1.0 / t).collect();
    let middle = median(&rates);
    median(&rates.iter().map(|v| (v - middle).abs()).collect::<Vec<_>>()) / middle
}

fn summary(pairs: &[(f64, f64)]) -> Result<Value> {
    let mut report = super::ablation::summarize(pairs)?;
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
    case: &Value,
    pairs: &mut Vec<(f64, f64)>,
    phase: (&str, usize),
    sample: &mut impl FnMut(usize) -> Result<Value>,
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
            times[slot] = sample(slot)?;
        }
        let a = times[0]["gpu_ms"]
            .as_f64()
            .context("missing control timing")?;
        let b = times[1]["gpu_ms"]
            .as_f64()
            .context("missing candidate timing")?;
        write_record(
            output,
            json!({"kind":"ffn_fusion_pair","case":case,"phase":phase,"pair":index,"order":order,"times":times}),
        )?;
        pairs.push((a, b));
    }
    Ok(())
}

/// Reuse the channel experiment's pair ordering, variation and confirmation
/// rules through a sampler callback; numerical qualification remains separate.
pub(super) fn compare_samples(
    output: &mut File,
    case: &Value,
    arms: Value,
    mut sample: impl FnMut(usize) -> Result<Value>,
) -> Result<()> {
    let before = inspect_gpu()?;
    let mut pairs = Vec::new();
    collect(output, case, &mut pairs, ("screen", 5), &mut sample)?;
    if summary(&pairs)?["is_stable"] == false {
        collect(output, case, &mut pairs, ("screen", 10), &mut sample)?;
    }
    let screen = summary(&pairs)?;
    let mut confirmation = Value::Null;
    if screen["needs_confirmation"] == true {
        let mut fresh = Vec::new();
        collect(output, case, &mut fresh, ("confirmation", 20), &mut sample)?;
        confirmation = summary(&fresh)?;
    }
    let decision = if confirmation.is_null() {
        &screen
    } else {
        &confirmation
    };
    write_record(
        output,
        json!({"kind":"ffn_fusion_comparison","case":case,"arms":arms,
        "screen":screen,"confirmation":confirmation,"passes_operator_screen":decision["passes_micro_screen"],
        "gpu_before":before,"gpu_after":inspect_gpu()?,
        "interpretation":"operation scope is explicit in case.stage; no whole-model claim"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Extending a screen preserves arm identity and continues alternating order.
    #[test]
    fn test_pair_callback_order_and_failure() -> Result<()> {
        let mut output = std::fs::OpenOptions::new().write(true).open("/dev/null")?;
        let mut order = Vec::new();
        let mut pairs = Vec::new();
        let case = json!({"test":"callback"});
        let mut sample = |slot| {
            order.push(slot);
            Ok(json!({"gpu_ms":if slot == 0 {10.0} else {1.0}}))
        };
        collect(&mut output, &case, &mut pairs, ("screen", 3), &mut sample)?;
        collect(&mut output, &case, &mut pairs, ("screen", 5), &mut sample)?;
        assert_eq!(order, [0, 1, 1, 0, 0, 1, 1, 0, 0, 1]);
        assert_eq!(pairs, [(10.0, 1.0); 5]);
        let failure = collect(&mut output, &case, &mut pairs, ("screen", 6), &mut |_| {
            anyhow::bail!("sample failed")
        });
        assert!(failure.is_err());
        assert_eq!(pairs.len(), 5);
        Ok(())
    }

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

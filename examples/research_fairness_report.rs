//! Reconstruct active decode and arriving prefill from the common host clock.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use m40_llm::generation_metrics::GenerationMetrics;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Args {
    directory: PathBuf,
}

#[derive(Deserialize)]
struct Load {
    client_origin_unix_ns: String,
    success: bool,
    records: Vec<Request>,
}
#[derive(Deserialize)]
struct Request {
    slot: usize,
    submitted_ms: f64,
    done_ms: f64,
    response: Response,
}
#[derive(Deserialize)]
struct Response {
    metrics: GenerationMetrics,
    output: String,
}

fn load(dir: &Path, name: &str) -> Result<Load> {
    let data: Load = serde_json::from_slice(&fs::read(dir.join(format!("{name}.json")))?)?;
    ensure!(
        data.success && !data.records.is_empty(),
        "failed or empty load"
    );
    for (slot, r) in data.records.iter().enumerate() {
        ensure!(
            r.slot == slot && r.done_ms > r.submitted_ms && r.response.metrics.generated_tokens > 0,
            "invalid client accounting"
        );
    }
    Ok(data)
}

fn percentile(values: &[f64], pct: usize) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(sorted[(values.len() * pct).div_ceil(100) - 1])
}

fn samples(load: &Load, request: &Request) -> Result<(f64, Vec<f64>)> {
    let m = &request.response.metrics;
    let clock = m
        .sample_origin_unix_ns
        .context("server sample origin absent")?;
    let origin = (i128::from(clock) - load.client_origin_unix_ns.parse::<i128>()?) as f64 / 1e6;
    let times = m.token_elapsed_ms.as_ref().context("missing token clock")?;
    let ids = m.sampled_ids.as_ref().context("missing sampled IDs")?;
    ensure!(
        times.len() == m.generated_tokens
            && ids.len() == times.len()
            && !times.is_empty()
            && times.iter().all(|v| v.is_finite() && *v >= 0.)
            && times.windows(2).all(|pair| pair[0] <= pair[1]),
        "bad sample trace"
    );
    let absolute: Vec<_> = times.iter().map(|value| origin + value).collect();
    // Client and server share the host clock. Exclude inversions beyond one millisecond.
    ensure!(
        origin + 1.0 >= request.submitted_ms
            && absolute.last().unwrap() <= &(request.done_ms + 1.0),
        "clock inversion"
    );
    Ok((origin, absolute))
}

fn overlap_gap(first: &[f64], arrival: f64, admission: f64, first_sample: f64) -> Option<f64> {
    if first.len() < 2
        || arrival <= first[0]
        || arrival >= *first.last()?
        || first_sample <= admission
    {
        return None;
    }
    first
        .windows(2)
        .filter(|p| p[0] < first_sample && p[1] > admission)
        .map(|p| p[1] - p[0])
        .max_by(f64::total_cmp)
}

fn summarize(load: &Load, diagnostic: bool) -> Result<Value> {
    let mut requests = Vec::new();
    let mut tokens = 0;
    let begin = load
        .records
        .iter()
        .map(|r| r.submitted_ms)
        .min_by(f64::total_cmp)
        .unwrap();
    let end = load
        .records
        .iter()
        .map(|r| r.done_ms)
        .max_by(f64::total_cmp)
        .unwrap();
    for request in &load.records {
        let m = &request.response.metrics;
        tokens += m.generated_tokens;
        let mut row = json!({"slot":request.slot,"wall_ms":request.done_ms-request.submitted_ms,
            "prompt_tokens":m.prompt_tokens,"generated_tokens":m.generated_tokens,"stop":m.stop_reason,
            "server_total_ms":m.total_elapsed_ms,"server_prefill_ms":m.prefill_elapsed_ms,
            "decode_ms":m.decode_elapsed_ms,"completion_ms":request.done_ms});
        if diagnostic {
            let (admitted, events) = samples(load, request)?;
            let gaps: Vec<_> = events.windows(2).map(|p| p[1] - p[0]).collect();
            row["queue_and_transport_ms"] = json!(admitted - request.submitted_ms);
            row["client_to_first_sample_ms"] = json!(events[0] - request.submitted_ms);
            row["itl_median_ms"] = json!(percentile(&gaps, 50));
            row["itl_p95_ms"] = json!(percentile(&gaps, 95));
            row["itl_max_ms"] = json!(percentile(&gaps, 100));
        } else {
            ensure!(
                m.token_elapsed_ms.is_none() && m.sampled_ids.is_none(),
                "tracing leaked into primary"
            );
        }
        requests.push(row);
    }
    Ok(
        json!({"wall_ms":end-begin,"aggregate_tps":tokens as f64*1000./(end-begin),"requests":requests}),
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut summary = Vec::new();
    let mut overlaps = Vec::new();
    let mut passing = 0;
    let mut valid = 0;
    for scenario in ["single-short", "single-medium", "four"]
        .into_iter()
        .map(str::to_owned)
        .chain((0..5).flat_map(|i| [format!("{i}-control"), format!("{i}-treatment")]))
    {
        let diagnostic = load(&args.directory, &format!("diagnostic-{scenario}"))?;
        let primary = load(&args.directory, &format!("primary-{scenario}"))?;
        let n = if scenario == "four" {
            4
        } else if scenario.starts_with("single") {
            1
        } else {
            2
        };
        ensure!(
            diagnostic.records.len() == n && primary.records.len() == n,
            "wrong client count"
        );
        let mut equivalent = true;
        for (slot, (a, b)) in diagnostic.records.iter().zip(&primary.records).enumerate() {
            let expected =
                if scenario == "single-medium" || (scenario.ends_with("treatment") && slot == 1) {
                    2060
                } else {
                    65
                };
            ensure!(
                a.response.metrics.prompt_tokens == expected
                    && b.response.metrics.prompt_tokens == expected,
                "unexpected prompt count"
            );
            equivalent &= a.response.output == b.response.output
                && a.response.metrics.generated_tokens == b.response.metrics.generated_tokens
                && a.response.metrics.stop_reason == b.response.metrics.stop_reason;
        }
        summary.push(
            json!({"scenario":scenario,"diagnostic":summarize(&diagnostic,true)?,
            "primary":summarize(&primary,false)?,"primary_trace_work_matches":equivalent}),
        );
    }
    for pair in 0..5 {
        let mut gaps = [None, None];
        let mut first_ids = Vec::new();
        for (i, arm) in ["control", "treatment"].iter().enumerate() {
            let load = load(&args.directory, &format!("diagnostic-{pair}-{arm}"))?;
            let first = samples(&load, &load.records[0])?.1;
            let second = &load.records[1];
            let (admission, other) = samples(&load, second)?;
            gaps[i] = overlap_gap(&first, second.submitted_ms, admission, other[0]);
            first_ids.push(load.records[0].response.metrics.sampled_ids.clone());
        }
        let equal = first_ids[0] == first_ids[1];
        let work_matches = summary
            .iter()
            .filter(|row| {
                row["scenario"]
                    .as_str()
                    .is_some_and(|s| s.starts_with(&format!("{pair}-")))
            })
            .all(|r| r["primary_trace_work_matches"] == true);
        let qualifies = match gaps {
            [Some(control), Some(treatment)] if equal && work_matches => {
                valid += 1;
                treatment > 2. * control && treatment - control > 100.
            }
            _ => false,
        };
        passing += usize::from(qualifies);
        overlaps.push(json!({"pair":pair,"control_gap_ms":gaps[0],"treatment_gap_ms":gaps[1],"same_short_ids":equal,
            "primary_trace_work_matches":work_matches,"qualifies":qualifies}));
    }
    let decision = if passing >= 4 {
        "PASS"
    } else if valid < 4 || passing + (5 - valid) >= 4 {
        "INCONCLUSIVE"
    } else {
        "FAIL"
    };
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"scenarios":summary,"overlaps":overlaps,"valid_pairs":valid,
        "passing_pairs":passing,"stall_admission":decision})
        )?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    /// A queued request admitted after the first decode ends cannot cause a measured decode gap.
    #[test]
    fn excludes_queue_only_and_pre_decode_arrivals() {
        let first = [10., 20., 30., 140., 150.];
        assert_eq!(overlap_gap(&first, 25., 151., 175.), None);
        assert_eq!(overlap_gap(&first, 5., 6., 40.), None);
        assert_eq!(overlap_gap(&first, 25., 32., 130.), Some(110.));
    }
}

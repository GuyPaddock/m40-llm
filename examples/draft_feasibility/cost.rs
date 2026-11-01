use super::{
    records::{self, Trace},
    replay::Replay,
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};

pub struct CostModel {
    pairs: Vec<(Value, Value)>,
    rejection: Vec<Value>,
    historical_fixed_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{replay::Round, source::Proposal};

    fn model() -> CostModel {
        let control = json!({"serial_step_ms":vec![20.;511],"total_ms":10720.,"decode_ms":10220.,
            "session_setup_ms":0.1,"session_teardown_ms":0.1});
        let direct = json!({"batch_ms":vec![60.;127],"decode_other_ms":0.,"session_setup_ms":0.2,"session_teardown_ms":0.1});
        CostModel {
            pairs: vec![(control, direct)],
            rejection: vec![
                json!({"target_ms":60.,"rollback_ms":0.1,
            "correction_ms":20.,"elapsed_ms":80.1});
                4
            ],
            historical_fixed_ms: 500.,
        }
    }
    fn trace(ids: Vec<u32>, stop: &str) -> Trace {
        Trace {
            id: "unit".into(),
            class: "unit".into(),
            split: "unit".into(),
            prompt_token_ids: vec![1; 65],
            generated_token_ids: ids,
            stop_ids: vec![99],
            stop_reason: stop.into(),
            output: String::new(),
            serial_fixed_ms: Some(500.),
            max_tokens: 128,
        }
    }
    fn replay(guess: Option<Proposal>, accepted: usize, progress: usize) -> Replay {
        Replay {
            construction_ns: 1_000_000,
            cpu_elapsed_ns: 2_000_000,
            logical_payload_bytes: 0,
            rounds: vec![Round {
                input_row: 0,
                absolute_position: 65,
                known_seed: 1,
                guesses: guess,
                accepted_guesses: accepted,
                progress,
                proposal_ns: 500_000,
                maintenance_ns: 500_000,
            }],
        }
    }

    /// Ineligible positions pay measured proposal/maintenance costs and receive no reuse credit.
    #[test]
    fn test_miss_pays_all_in_cpu_cost() {
        let t = trace(vec![1, 99], "stop");
        let p = model().project(&t, &[replay(None, 0, 1)]).unwrap();
        let row = &p["scenarios"][0]["model_samples"][0];
        assert_eq!(row["serial_decode_model_ms"], 20.);
        assert_eq!(row["candidate_decode_model_ms"], 20.);
        assert_eq!(row["all_in_cpu_ms"], 2.);
        assert!(row["resident_gain"].as_f64().unwrap() < 0.);
    }
    /// A verified terminal token ends after rollback without a fictitious correction step.
    #[test]
    fn test_verified_stop_avoids_only_correction_not_target() {
        let t = trace(vec![1, 2, 99], "stop");
        let guess = Some(Proposal {
            guesses: [2, 7, 8],
            match_start: None,
            suffix_len: None,
        });
        let p = model().project(&t, &[replay(guess, 1, 2)]).unwrap();
        let actual = p["scenarios"][0]["model_samples"][0]["candidate_decode_model_ms"]
            .as_f64()
            .unwrap();
        assert!((actual - 60.1).abs() < 1e-8);
    }
}
fn number(v: &Value, key: &str) -> f64 {
    v[key].as_f64().unwrap()
}
fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<_> = values.collect();
    v.iter().sum::<f64>() / v.len() as f64
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.len() % 2 == 1 {
        v[v.len() / 2]
    } else {
        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.
    }
}
fn distribution(values: &[f64]) -> Value {
    json!({"min":values.iter().copied().fold(f64::INFINITY,f64::min),
        "median":median(values.to_vec()),"max":values.iter().copied().fold(f64::NEG_INFINITY,f64::max)})
}
fn at_serial(control: &Value, position: usize) -> f64 {
    control["serial_step_ms"][position.saturating_sub(65).min(510)]
        .as_f64()
        .unwrap()
}
fn at_batch(direct: &Value, position: usize) -> f64 {
    let offset = position.saturating_sub(65).min(504);
    let index = offset / 4;
    let fraction = (offset % 4) as f64 / 4.;
    let v = direct["batch_ms"][index].as_f64().unwrap();
    if index == 126 {
        v
    } else {
        v + (direct["batch_ms"][index + 1].as_f64().unwrap() - v) * fraction
    }
}
impl CostModel {
    pub fn load() -> Result<Self> {
        let data = records::rows(records::TARGET.as_ref())?;
        ensure!(
            data.last()
                .is_some_and(|r| r["is_complete"] == true && r["device_bytes_after"] == 0)
                && data
                    .iter()
                    .any(|r| r["kind"] == "direct_verifier_gate"
                        && r["passes_resident_gate"] == true),
            "target evidence incomplete"
        );
        let mut pairs = Vec::new();
        for i in 0..5 {
            let find = |is_direct| -> Result<Value> {
                Ok(data
                    .iter()
                    .find(|r| {
                        r["kind"] == "direct_verifier_trial"
                            && r["trial"] == i
                            && r["is_direct"] == is_direct
                    })
                    .context("target pair missing")?["result"]
                    .clone())
            };
            pairs.push((find(false)?, find(true)?));
        }
        let mut rejection = Vec::new();
        for accepted in 0..4 {
            let r: Vec<_> = data
                .iter()
                .filter(|r| {
                    r["kind"] == "direct_verifier_rejection"
                        && r["result"]["accepted_prefix"] == accepted
                })
                .map(|r| &r["result"])
                .collect();
            ensure!(r.len() == 5, "rejection costs missing");
            let mut v = json!({});
            for field in ["target_ms", "rollback_ms", "correction_ms", "elapsed_ms"] {
                v[field] = json!(mean(r.iter().map(|r| number(r, field))));
            }
            rejection.push(v);
        }
        let old: Value = serde_json::from_slice(&std::fs::read(
            "docs/experiments/2026-10-28-direct-verifier-model.json",
        )?)?;
        Ok(Self {
            pairs,
            rejection,
            historical_fixed_ms: old["historical_http_assumption"]["fixed_overhead_ms"]
                .as_f64()
                .unwrap(),
        })
    }

    /// Replay observed eligibility/rejection jointly, not an IID mean acceptance.
    pub fn project(&self, trace: &Trace, trials: &[Replay]) -> Result<Value> {
        let path = &trials[0];
        let cpu_ms = trials
            .iter()
            .map(|r| r.cpu_elapsed_ns as f64 / 1e6)
            .fold(0., f64::max);
        let mut scenarios = Vec::new();
        for extra in [0., 1., 5.] {
            let mut rows = Vec::new();
            for (control, direct) in &self.pairs {
                let mut serial = 0.;
                let mut target = 0.;
                let mut extrapolated = 0;
                let mut extrapolated_serial = 0;
                for input in 0..trace.generated_token_ids.len().saturating_sub(1) {
                    if !(65..=575).contains(&(trace.prompt_token_ids.len() + input)) {
                        extrapolated_serial += 1;
                    }
                    serial += at_serial(control, trace.prompt_token_ids.len() + input);
                }
                for r in &path.rounds {
                    if r.guesses.is_none() {
                        target += at_serial(control, r.absolute_position);
                    } else {
                        if !(65..=569).contains(&r.absolute_position) {
                            extrapolated += 1;
                        }
                        target += at_batch(direct, r.absolute_position) + extra;
                        if r.accepted_guesses < 3 {
                            let accepted = r.accepted_guesses + 1;
                            let reject = &self.rejection[accepted];
                            let residual = (number(reject, "elapsed_ms")
                                - number(reject, "target_ms")
                                - number(reject, "rollback_ms")
                                - number(reject, "correction_ms"))
                            .max(0.);
                            target += (number(reject, "target_ms")
                                - direct["batch_ms"][0].as_f64().unwrap())
                            .max(0.)
                                + number(reject, "rollback_ms")
                                + residual
                                + if r.progress > accepted {
                                    number(reject, "correction_ms")
                                        .max(at_serial(control, r.absolute_position + accepted))
                                } else {
                                    0.
                                };
                        }
                    }
                }
                target += number(direct, "decode_other_ms") / 130. * path.rounds.len() as f64;
                let construction_delta = (number(direct, "session_setup_ms")
                    + number(direct, "session_teardown_ms")
                    - number(control, "session_setup_ms")
                    - number(control, "session_teardown_ms"))
                .max(0.);
                let fixed = if trace.id == "browser" {
                    Some(number(control, "total_ms") - number(control, "decode_ms"))
                } else {
                    trace.serial_fixed_ms
                };
                let baseline = fixed.map(|f| f + serial);
                let candidate = fixed.map(|f| f + target + cpu_ms + construction_delta);
                let resident_gain = baseline.zip(candidate).map(|(b, c)| b / c - 1.);
                let http = fixed.map(|f| {
                    let extra_http = if trace.id == "browser" {
                        (self.historical_fixed_ms - f).max(0.)
                    } else {
                        (self.historical_fixed_ms
                            - (number(control, "total_ms") - number(control, "decode_ms")))
                        .max(0.)
                    };
                    (
                        f + extra_http + serial,
                        f + extra_http + target + cpu_ms + construction_delta,
                    )
                });
                rows.push(json!({"serial_decode_model_ms":serial,"candidate_decode_model_ms":target,
                    "candidate_zero_draft_gain":serial/target-1.,"all_in_cpu_ms":cpu_ms,
                    "extra_session_ms":construction_delta,"resident_serial_ms":baseline,"resident_candidate_ms":candidate,
                    "resident_gain":resident_gain,
                    "projected_resident_tok_per_sec":candidate.map(|ms|1000.*trace.generated_token_ids.len() as f64/ms),
                    "projected_http_tok_per_sec":http.map(|(_,c)|1000.*trace.generated_token_ids.len() as f64/c),
                    "projected_http_gain":http.map(|(b,c)|b/c-1.),"extrapolated_verifier_attempts":extrapolated,
                    "extrapolated_serial_rows":extrapolated_serial,
                    "max_all_in_cpu_ms_to_break_even":serial-target-construction_delta,
                    "max_all_in_cpu_ms_for_three_percent":fixed.map(|f|(f+serial)/1.03-f-target-construction_delta)}));
            }
            let gains: Vec<_> = rows
                .iter()
                .filter_map(|r| r["resident_gain"].as_f64())
                .collect();
            let decode_gains: Vec<_> = rows
                .iter()
                .map(|r| {
                    r["serial_decode_model_ms"].as_f64().unwrap()
                        / (r["candidate_decode_model_ms"].as_f64().unwrap() + cpu_ms)
                        - 1.
                })
                .collect();
            scenarios.push(json!({"extra_per_attempt_ms":extra,"model_samples":rows,
                "resident_gain":if gains.is_empty() {None} else {Some(distribution(&gains))},
                "decode_only_gain":distribution(&decode_gains)}));
        }
        Ok(
            json!({"scope":"actual causal visit path; measured old target samples, observed CPU cost, correction/rollback, finite tails; projections not measured E2E",
            "is_http_measured":false,"historical_http_fixed_overhead_ms":self.historical_fixed_ms,
            "unmeasured_late_rejections":true,"new_prompt_prefill_anchor_available":trace.serial_fixed_ms.is_some(),
            "cpu_cost_charged":"maximum of five all-in replays including initialization and ineligible proposals",
            "scenarios":scenarios}),
        )
    }
}

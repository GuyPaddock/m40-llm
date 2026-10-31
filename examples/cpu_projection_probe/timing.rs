use crate::{
    kernel::Q8Matrix,
    protocol::{self, Pair},
    Report,
};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{hint::black_box, time::Instant};

pub struct Eviction(Vec<u64>);
impl Eviction {
    pub fn new() -> Self {
        Self(vec![0x123456789abcdef; 128 * 1024 * 1024 / 8])
    }
    pub fn bytes(&self) -> usize {
        self.0.len() * 8
    }
    fn evict(&self) {
        // Read a word from each cache line. Non-zero initialization first-touches
        // real pages on the bound NUMA node; a calloc-only zero mapping would not.
        let sum = black_box(&self.0)
            .iter()
            .step_by(8)
            .fold(0u64, |sum, value| sum.wrapping_add(*value));
        black_box(sum);
    }
}

pub fn validate_output(output: &[f32], reference: &[f64]) -> Result<Value> {
    ensure!(
        output.len() == reference.len() && !output.is_empty(),
        "validation shape mismatch"
    );
    let mut max_absolute = 0.0f64;
    let mut max_tolerance_ratio = 0.0f64;
    let mut squared_error = 0.0;
    for (&actual, &expected) in output.iter().zip(reference) {
        ensure!(
            actual.is_finite() && expected.is_finite(),
            "non-finite projection result"
        );
        let error = (f64::from(actual) - expected).abs();
        let ratio = error / (1e-3 + 2e-5 * expected.abs());
        max_absolute = max_absolute.max(error);
        max_tolerance_ratio = max_tolerance_ratio.max(ratio);
        squared_error += error * error;
    }
    ensure!(
        max_tolerance_ratio <= 1.0,
        "projection exceeds FP64 tolerance: {max_tolerance_ratio}"
    );
    Ok(
        json!({"elements":output.len(),"max_absolute_error":max_absolute,
        "max_tolerance_ratio":max_tolerance_ratio,"mean_squared_error":squared_error/output.len() as f64}),
    )
}

fn measure(
    matrix: &Q8Matrix,
    input: &[f32],
    batch: usize,
    iterations: usize,
    is_avx2: bool,
    output: &mut [f32],
    eviction: &Eviction,
) -> Result<(f64, f64)> {
    let mut elapsed = 0.0;
    let mut eviction_elapsed = 0.0;
    for _ in 0..iterations {
        let began = Instant::now();
        eviction.evict();
        eviction_elapsed += began.elapsed().as_secs_f64();
        let began = Instant::now();
        if is_avx2 {
            matrix.project_avx2(black_box(input), batch, black_box(output))?;
        } else {
            matrix.project_scalar(black_box(input), batch, black_box(output))?;
        }
        elapsed += began.elapsed().as_secs_f64();
        black_box(&*output);
    }
    ensure!(elapsed > 0.0, "zero timer interval");
    Ok((
        elapsed * 1000.0 / iterations as f64,
        eviction_elapsed * 1000.0,
    ))
}

pub fn run_case(
    matrix: &Q8Matrix,
    input: &[f32],
    batch: usize,
    iterations: usize,
    key: &Value,
    eviction: &mut Eviction,
    report: &mut Report,
) -> Result<()> {
    let setup = Instant::now();
    let mut output = vec![0.0; batch * matrix.rows];
    matrix.project_scalar(input, batch, &mut output)?;
    matrix.project_avx2(input, batch, &mut output)?;
    report.record(json!({"kind":"cpu_projection_timing_setup","case":key,
        "allocation_warmup_ms":setup.elapsed().as_secs_f64()*1000.0,
        "timing_output_bytes":output.len()*4}))?;
    let mut pairs = Vec::new();
    let mut target = 5;
    let mut is_confirmation = false;
    loop {
        for index in pairs.len()..target {
            let mut elapsed = [0.0; 2];
            let mut evicted = [0.0; 2];
            let order = if index % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for is_avx2 in order {
                (elapsed[usize::from(is_avx2)], evicted[usize::from(is_avx2)]) = measure(
                    matrix,
                    input,
                    batch,
                    iterations,
                    is_avx2,
                    &mut output,
                    eviction,
                )?;
            }
            pairs.push(Pair {
                control_tps: 1000.0 / elapsed[0],
                candidate_tps: 1000.0 / elapsed[1],
            });
            report.record(json!({"kind":"cpu_projection_pair","case":key,"pair":index,
                "is_confirmation":is_confirmation,"avx2_first":order[0],"iterations":iterations,
                "scalar_ms":elapsed[0],"avx2_ms":elapsed[1],"eviction_total_ms":evicted,
                "timing_boundary":"one complete projection including shape/input validation; allocation and eviction excluded"}))?;
        }
        let comparison = protocol::compare_pairs(&pairs)?;
        let decision = protocol::next_trial_count(&comparison, is_confirmation);
        let mut summary = serde_json::to_value(&comparison)?;
        let fields = summary.as_object_mut().unwrap();
        fields.remove("reaches_original_target");
        if let Some(value) = fields.remove("passes_promotion_gate") {
            fields.insert("passes_operator_speed_gate".into(), value);
        }
        report.record(json!({"kind":"cpu_projection_comparison","case":key,
            "is_confirmation":is_confirmation,"comparison":summary,"units":"complete projections/second, not tokens/second",
            "timing_error":decision.as_ref().err().map(|e|format!("{e:#}")),"is_model_speedup":false}))?;
        match decision {
            Ok(Some(10)) => target = 10,
            Ok(Some(20)) => {
                target = 20;
                pairs.clear();
                is_confirmation = true;
            }
            Ok(None) | Err(_) => break,
            Ok(Some(_)) => anyhow::bail!("unexpected trial policy"),
        }
    }
    Ok(())
}

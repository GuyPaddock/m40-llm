use super::pipeline::{Pipeline, Stage, ARMS, CHANNELS};
use anyhow::{ensure, Result};
use half::f16;
use m40_llm::infer::q8_activation::{quantize_reference, Q8ActivationBlock, Q8ActivationShape};
use serde_json::{json, Value};

/// Scalar encoded dot product and a cancellation-aware FP32 roundoff bound.
/// This is the same numerical contract as the producer's hardware stress test.
fn project(
    input: &[Q8ActivationBlock],
    weights: &[u8],
    row: usize,
    channel: usize,
    columns: usize,
) -> (f64, f64) {
    let blocks = columns / 32;
    let mut value = 0.0;
    let mut absolute_sum = 0.0;
    for block in 0..blocks {
        let activation = &input[row * blocks + block];
        let base = (channel * blocks + block) * 34;
        let scale = f16::from_bits(activation.scale_bits).to_f64()
            * f16::from_le_bytes([weights[base], weights[base + 1]]).to_f64();
        for lane in 0..32 {
            let term = scale
                * f64::from(activation.quants[lane])
                * f64::from(weights[base + 2 + lane] as i8);
            value += term;
            absolute_sum += term.abs();
        }
    }
    let roundings = (blocks * 4).div_ceil(128) + 7 + 2;
    let epsilon = roundings as f64 * 2.0f64.powi(-24);
    (value, epsilon / (1.0 - epsilon) * absolute_sum)
}

pub(super) fn validate(
    pipeline: &Pipeline<'_>,
    reference: &[f64],
    weights: &[Vec<u8>; 2],
) -> Result<Value> {
    let columns = pipeline.producer.columns();
    let rows = pipeline.producer.rows();
    let mut expected_bits = None;
    let mut reports = Vec::new();
    for arm in ARMS {
        pipeline.enqueue(arm, Stage::Complete)?;
        let (rotated, encoded, result) = pipeline.read()?;
        ensure!(
            rotated.len() == reference.len() && rotated.len() == rows * columns,
            "rotation length mismatch"
        );
        let mut max_rotation_error = 0.0f64;
        for (&actual, &expected) in rotated.iter().zip(reference) {
            let error = (f64::from(actual) - expected).abs();
            ensure!(
                actual.is_finite() && expected.is_finite() && error <= 1e-5 + 2e-6 * expected.abs(),
                "rotated FP64 mismatch: {actual} vs {expected}"
            );
            max_rotation_error = max_rotation_error.max(error);
        }
        let shape = Q8ActivationShape::new(1, rows * columns, rows * columns)?;
        let blocks = quantize_reference(&rotated, shape)?;
        ensure!(
            encoded == blocks.iter().flat_map(|b| b.to_bytes()).collect::<Vec<_>>(),
            "Q8_1 CPU bytes mismatch"
        );
        let mut max_output_error = 0.0f64;
        let mut max_output_bound_fraction = 0.0f64;
        for row in 0..rows {
            for channel in 0..CHANNELS {
                let (g, ge) = project(&blocks, &weights[0], row, channel, columns);
                let (u, ue) = project(&blocks, &weights[1], row, channel, columns);
                let expected = g / (1.0 + (-g).exp()) * u;
                let bound = 1.1 * ge * (u.abs() + ue) + g.abs() * ue + 1e-3 + 2e-5 * expected.abs();
                let actual = f64::from(result[row * CHANNELS + channel]);
                let error = (actual - expected).abs();
                ensure!(
                    actual.is_finite() && expected.is_finite() && error <= bound,
                    "consumer FP64 mismatch"
                );
                max_output_error = max_output_error.max(error);
                max_output_bound_fraction = max_output_bound_fraction.max(error / bound);
            }
        }
        let bits = (
            rotated.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            encoded,
            result.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        );
        if let Some(expected) = &expected_bits {
            ensure!(
                &bits == expected,
                "fused preparation changed canonical/encoded/consumer bits"
            );
        }
        expected_bits = Some(bits);
        reports.push(json!({"arm":arm,"max_rotation_error":max_rotation_error,
            "rotation_values":rows*columns,"consumer_values":rows*CHANNELS,
            "max_output_error":max_output_error,"max_output_bound_fraction":max_output_bound_fraction}));
    }
    Ok(
        json!({"passed":true,"canonical_bits_equal":true,"encoded_bytes_equal":true,
        "consumer_bits_equal":true,"all_arms":reports}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encoded integer values and stored scales define the scalar dot reference;
    /// a constant exactly representable fixture has a known value and finite bound.
    #[test]
    fn test_encoded_dot_reference() {
        let activation = Q8ActivationBlock {
            scale_bits: f16::from_f32(0.5).to_bits(),
            sum_bits: 0,
            quants: [2; 32],
        };
        let mut weight = vec![3; 34];
        weight[..2].copy_from_slice(&f16::from_f32(0.25).to_le_bytes());
        let (actual, bound) = project(&[activation], &weight, 0, 0, 32);
        assert_eq!(actual, 24.0);
        assert!(bound > 0.0 && bound < 0.001);
    }
}

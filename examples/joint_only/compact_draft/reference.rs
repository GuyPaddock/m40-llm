//! Reproducible untrained coefficients and an independent FP64 cost-probe oracle.
use anyhow::{ensure, Result};
use serde_json::{json, Value};

pub const D: usize = 2048;
pub const R: usize = 256;
pub const HEADS: usize = 3;
pub const V: usize = 151936;

fn value(index: usize, seed: u32) -> f32 {
    let mut x = (index as u32).wrapping_add(seed);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    ((x & 65535) as f32 - 32768.) / 32768.
}
pub(super) struct Parameters {
    pub maps: Vec<f32>,
    pub bias: Vec<f32>,
    pub vocabulary: Vec<f32>,
    pub feature: Vec<f32>,
}
impl Parameters {
    pub fn new() -> Self {
        Self {
            maps: (0..D * R * HEADS).map(|i| value(i, 1) * 0.02).collect(),
            bias: (0..R * HEADS).map(|i| value(i, 2) * 0.02).collect(),
            vocabulary: (0..R * V).map(|i| value(i, 3) * 0.02).collect(),
            feature: (0..D).map(|i| value(i, 4)).collect(),
        }
    }
    pub fn bytes(&self) -> usize {
        (self.maps.len() + self.bias.len() + self.vocabulary.len()) * 4
    }
    pub fn sha256(&self) -> String {
        let mut digest = ring::digest::Context::new(&ring::digest::SHA256);
        for value in self.maps.iter().chain(&self.bias).chain(&self.vocabulary) {
            digest.update(&value.to_le_bytes());
        }
        digest
            .finish()
            .as_ref()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

fn error(actual: &[f32], expected: &[f64]) -> Result<Value> {
    ensure!(actual.len() == expected.len(), "cost oracle shape differs");
    let mut max = 0f64;
    let mut sum = 0.;
    for (&a, &b) in actual.iter().zip(expected) {
        let e = (f64::from(a) - b).abs();
        ensure!(
            a.is_finite() && b.is_finite() && e < 1e-3 + 2e-5 * b.abs(),
            "cost oracle numerical mismatch: {e}"
        );
        max = max.max(e);
        sum += e;
    }
    Ok(json!({"max_abs_diff":max,"mean_abs_diff":sum/actual.len() as f64,"values":actual.len()}))
}
pub(super) fn validate(
    p: &Parameters,
    hidden: &[f32],
    logits: &[f32],
    ids: [u32; 3],
) -> Result<Value> {
    let expected_hidden: Vec<f64> = (0..HEADS * R)
        .map(|out| {
            let sum = (0..D)
                .map(|k| f64::from(p.feature[k]) * f64::from(p.maps[k * HEADS * R + out]))
                .sum::<f64>()
                + f64::from(p.bias[out]);
            sum / (1. + (-sum).exp())
        })
        .collect();
    let mut expected_logits = vec![0f64; HEADS * V];
    for head in 0..HEADS {
        for k in 0..R {
            for out in 0..V {
                expected_logits[head * V + out] +=
                    expected_hidden[head * R + k] * f64::from(p.vocabulary[k * V + out]);
            }
        }
    }
    let expected_ids: [u32; 3] = std::array::from_fn(|h| {
        (0..V)
            .max_by(|&a, &b| {
                expected_logits[h * V + a]
                    .total_cmp(&expected_logits[h * V + b])
                    .then_with(|| b.cmp(&a))
            })
            .unwrap() as u32
    });
    ensure!(
        ids == expected_ids,
        "synthetic full-vocabulary argmax mismatch"
    );
    Ok(
        json!({"hidden":error(hidden,&expected_hidden)?,"logits":error(logits,&expected_logits)?,"ids":ids,"is_passed":true,
        "scope":"untrained computation/layout check, not target-token acceptance"}),
    )
}

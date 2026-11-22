//! Independent quantization, block ranking and FP64 attention for the probe.

use anyhow::{ensure, Context, Result};
use half::f16;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Backing {
    Q8,
    Mixed,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Shape {
    pub length: usize,
    pub head_dim: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub recent: usize,
    pub block: usize,
    pub top: usize,
}

impl Shape {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.length > 0
                && self.head_dim > 0
                && self.kv_heads > 0
                && self.query_heads > 0
                && self.query_heads.is_multiple_of(self.kv_heads),
            "invalid attention shape"
        );
        ensure!(
            self.recent > 0 && self.block > 0 && (1..=64).contains(&self.top),
            "invalid selection configuration"
        );
        self.length
            .checked_mul(self.head_dim)
            .and_then(|v| v.checked_mul(self.kv_heads))
            .context("KV size overflow")?;
        self.query_heads
            .checked_mul(self.head_dim)
            .context("query size overflow")?;
        self.top
            .checked_mul(self.block)
            .context("selection size overflow")?;
        Ok(())
    }
    pub fn kv_width(self) -> usize {
        self.kv_heads * self.head_dim
    }
    pub fn query_width(self) -> usize {
        self.query_heads * self.head_dim
    }
    pub fn old_len(self) -> usize {
        self.length.saturating_sub(self.recent)
    }
    pub fn capacity(self) -> usize {
        self.length.min(self.recent) + self.old_len().min(self.top * self.block)
    }
}

pub struct Reconstructed {
    /// Q8 block selection reads code * FP32 scale before FP16 rounding.
    pub selection_keys: Vec<f32>,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
}

/// The existing append API performs an adjacent-pair RoPE expression even at
/// frequency zero. Numeric values stay unchanged, but IEEE signed zeros can
/// change. Model that append before comparing reconstructed FP16 bits.
pub fn zero_frequency_append(keys: &[f32]) -> Result<Vec<f32>> {
    ensure!(
        keys.len().is_multiple_of(2) && keys.iter().all(|v| v.is_finite()),
        "invalid identity-append keys"
    );
    let mut output = Vec::with_capacity(keys.len());
    for pair in keys.chunks_exact(2) {
        output.push(pair[0] - pair[1] * 0.0);
        output.push(pair[0] * 0.0 + pair[1]);
    }
    Ok(output)
}

pub fn reconstruct(
    shape: Shape,
    backing: Backing,
    keys: &[f32],
    values: &[f32],
) -> Result<Reconstructed> {
    shape.validate()?;
    ensure!(
        keys.len() == shape.length * shape.kv_width() && keys.len() == values.len(),
        "KV fixture shape differs"
    );
    ensure!(
        keys.iter()
            .chain(values)
            .all(|v| v.is_finite() && f16::from_f32(*v).to_f32() == *v),
        "KV fixture is not finite FP16"
    );
    let mut result = Reconstructed {
        selection_keys: keys.to_vec(),
        keys: keys.to_vec(),
        values: values.to_vec(),
    };
    for token in 0..shape.old_len() {
        for head in 0..shape.kv_heads {
            let start = token * shape.kv_width() + head * shape.head_dim;
            let range = start..start + shape.head_dim;
            if backing == Backing::Q8 {
                quantize_vector(
                    &keys[range.clone()],
                    127.0,
                    &mut result.selection_keys[range.clone()],
                );
                for index in range.clone() {
                    result.keys[index] = f16::from_f32(result.selection_keys[index]).to_f32();
                }
            }
            quantize_vector(
                &values[range.clone()],
                if backing == Backing::Q8 { 127.0 } else { 7.0 },
                &mut result.values[range.clone()],
            );
            for index in range {
                result.values[index] = f16::from_f32(result.values[index]).to_f32();
            }
        }
    }
    Ok(result)
}

fn quantize_vector(input: &[f32], maximum_code: f32, output: &mut [f32]) {
    let maximum = input
        .iter()
        .fold(0.0f32, |value, next| value.max(next.abs()));
    let scale = if maximum > 0.0 {
        maximum / maximum_code
    } else {
        1.0 / maximum_code
    };
    for (&value, output) in input.iter().zip(output) {
        let code = (value / scale)
            .round_ties_even()
            .clamp(-maximum_code, maximum_code) as i32;
        *output = code as f32 * scale;
    }
}

/// Rank mathematical mean-key dot products in FP64, breaking exact ties by
/// the earliest block. GPU selection must reproduce these positions to qualify.
pub fn select_positions(
    shape: Shape,
    keys: &[f32],
    query: &[f32],
    query_head: usize,
) -> Result<Vec<usize>> {
    validate_attention(shape, keys, keys, query, query_head)?;
    let kv_head = query_head / (shape.query_heads / shape.kv_heads);
    let mut scores = Vec::new();
    for block in 0..shape.old_len().div_ceil(shape.block) {
        let start = block * shape.block;
        let end = (start + shape.block).min(shape.old_len());
        let mut score = 0.0;
        for token in start..end {
            for channel in 0..shape.head_dim {
                score += f64::from(query[query_head * shape.head_dim + channel])
                    * f64::from(
                        keys[token * shape.kv_width() + kv_head * shape.head_dim + channel],
                    );
            }
        }
        scores.push((block, score / (end - start) as f64));
    }
    scores.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(scores
        .iter()
        .take(shape.top)
        .flat_map(|(block, _)| {
            block * shape.block..((block + 1) * shape.block).min(shape.old_len())
        })
        .chain(shape.old_len()..shape.length)
        .collect())
}

pub fn attention(
    shape: Shape,
    keys: &[f32],
    values: &[f32],
    query: &[f32],
    query_head: usize,
    positions: &[usize],
) -> Result<Vec<f64>> {
    validate_attention(shape, keys, values, query, query_head)?;
    ensure!(
        !positions.is_empty() && positions.iter().all(|&t| t < shape.length),
        "invalid attention positions"
    );
    let kv_head = query_head / (shape.query_heads / shape.kv_heads);
    let mut scores = Vec::with_capacity(positions.len());
    for &token in positions {
        let mut dot = 0.0;
        for channel in 0..shape.head_dim {
            dot += f64::from(query[query_head * shape.head_dim + channel])
                * f64::from(keys[token * shape.kv_width() + kv_head * shape.head_dim + channel]);
        }
        scores.push(dot / (shape.head_dim as f64).sqrt());
    }
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let probabilities: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
    let total: f64 = probabilities.iter().sum();
    let mut output = vec![0.0; shape.head_dim];
    for (&token, probability) in positions.iter().zip(probabilities) {
        for (channel, output) in output.iter_mut().enumerate() {
            *output += probability / total
                * f64::from(values[token * shape.kv_width() + kv_head * shape.head_dim + channel]);
        }
    }
    Ok(output)
}

fn validate_attention(
    shape: Shape,
    keys: &[f32],
    values: &[f32],
    query: &[f32],
    head: usize,
) -> Result<()> {
    shape.validate()?;
    ensure!(
        keys.len() == shape.length * shape.kv_width()
            && values.len() == keys.len()
            && query.len() == shape.query_width()
            && head < shape.query_heads,
        "attention input shape differs"
    );
    ensure!(
        keys.iter()
            .chain(values)
            .chain(query)
            .all(|v| v.is_finite()),
        "non-finite attention input"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> Shape {
        Shape {
            length: 7,
            head_dim: 2,
            query_heads: 4,
            kv_heads: 2,
            recent: 2,
            block: 2,
            top: 1,
        }
    }

    /// Zero-frequency append preserves nonzero values and has the documented
    /// IEEE zero signs for both adjacent coordinates.
    #[test]
    fn test_zero_frequency_append_preserves_numeric_identity() {
        let input = [-0.0, 1.0, 1.0, -0.0, -1.0, -0.0, -0.0, -1.0];
        let output = zero_frequency_append(&input).unwrap();
        assert_eq!(output, input);
        let expected = [-0.0f32, 1.0, 1.0, 0.0, -1.0, -0.0, 0.0, -1.0];
        assert_eq!(
            output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.map(f32::to_bits)
        );
        assert!(zero_frequency_append(&[1.0]).is_err());
        assert!(zero_frequency_append(&[f32::NAN, 0.0]).is_err());
    }

    /// Rank full and partial blocks by mean, preserve every recent token, and
    /// map grouped query heads to the correct KV head with stable exact ties.
    #[test]
    fn test_selection_and_gqa_mapping() {
        let shape = shape();
        let keys: Vec<_> = (0..7)
            .flat_map(|t| [t as f32, 0.0, -(t as f32), 0.0])
            .collect();
        let query = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        assert_eq!(
            select_positions(shape, &keys, &query, 0).unwrap(),
            [4, 5, 6]
        );
        assert_eq!(
            select_positions(shape, &keys, &query, 1).unwrap(),
            [4, 5, 6]
        );
        assert_eq!(
            select_positions(shape, &keys, &query, 2).unwrap(),
            [0, 1, 5, 6]
        );
        assert_eq!(
            select_positions(shape, &keys, &query, 3).unwrap(),
            [0, 1, 5, 6]
        );
        assert_eq!(shape.capacity(), 4);
    }

    /// Zero queries yield an arithmetic mean over exactly the selected tokens,
    /// independent of key values; full-context and selected references differ.
    #[test]
    fn test_attention_matches_hand_mean() {
        let shape = shape();
        let keys = vec![1.0; 28];
        let values: Vec<_> = (0..7)
            .flat_map(|t| [t as f32, 2.0 * t as f32, 10.0 + t as f32, 20.0])
            .collect();
        let query = [0.0; 8];
        assert_eq!(
            attention(shape, &keys, &values, &query, 0, &[0, 2, 4]).unwrap(),
            [2.0, 4.0]
        );
        let result = attention(shape, &keys, &values, &query, 3, &[1, 5]).unwrap();
        assert_eq!(result, [13.0, 20.0]);
    }

    /// Old mixed V uses signed Q4 with ties-to-even; recent values preserve
    /// FP16 bits. Q8 selection keeps pre-FP16 reconstruction separately.
    #[test]
    fn test_quantization_and_recent_boundary() {
        let shape = Shape {
            length: 2,
            head_dim: 4,
            query_heads: 1,
            kv_heads: 1,
            recent: 1,
            block: 1,
            top: 1,
        };
        let keys = vec![1.0, 0.25, -0.5, -0.0, 1.0, 2.0, 3.0, -0.0];
        let values = vec![7.0, 0.5, 1.5, -1.5, -0.0, 0.5, 1.5, 7.0];
        let mixed = reconstruct(shape, Backing::Mixed, &keys, &values).unwrap();
        assert_eq!(&mixed.values[..4], &[7.0, 0.0, 2.0, -2.0]);
        assert_eq!(mixed.values[4].to_bits(), (-0.0f32).to_bits());
        assert_eq!(mixed.keys, keys);
        let q8 = reconstruct(shape, Backing::Q8, &keys, &values).unwrap();
        assert_eq!(q8.selection_keys[1], 32.0 * (1.0f32 / 127.0));
        assert_eq!(q8.keys[1], f16::from_f32(q8.selection_keys[1]).to_f32());
        assert_ne!(q8.keys[1].to_bits(), q8.selection_keys[1].to_bits());
        assert_eq!(q8.keys[3].to_bits(), 0);
        assert_eq!(&q8.values[4..], &values[4..]);
    }

    /// The no-old-token control retains the entire history; invalid shapes,
    /// non-FP16 source values and non-finite queries are rejected.
    #[test]
    fn test_no_old_tokens_and_invalid_inputs() {
        let shape = Shape {
            length: 2,
            recent: 2,
            ..shape()
        };
        let keys = vec![0.0; 8];
        assert_eq!(
            select_positions(shape, &keys, &[0.0; 8], 0).unwrap(),
            [0, 1]
        );
        assert!(attention(shape, &keys, &keys, &[f32::NAN; 8], 0, &[0]).is_err());
        assert!(reconstruct(shape, Backing::Mixed, &[0.1; 8], &keys).is_err());
        assert!(Shape { block: 0, ..shape }.validate().is_err());
        assert!(Shape {
            query_heads: 3,
            ..shape
        }
        .validate()
        .is_err());
    }
}

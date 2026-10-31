//! Weight-only rotated ternary sketches for the bounded Maxwell research probe.

use super::{identity::sha256, quantized_fixture::decode_q4_values};
use anyhow::{ensure, Context, Result};
use half::f16;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;

pub const ROTATION_BLOCK: usize = 128;

fn validate_signs(signs: &[i8]) -> Result<()> {
    ensure!(
        !signs.is_empty() && signs.len().is_multiple_of(ROTATION_BLOCK),
        "rotation requires complete 128-coordinate blocks"
    );
    ensure!(
        signs.iter().all(|&s| s == -1 || s == 1),
        "rotation signs must be -1 or 1"
    );
    Ok(())
}

pub fn generate_signs(input_dim: usize, seed: u64) -> Result<Vec<i8>> {
    ensure!(
        input_dim > 0 && input_dim.is_multiple_of(ROTATION_BLOCK),
        "invalid rotation width"
    );
    let mut rng = StdRng::seed_from_u64(seed);
    Ok((0..input_dim)
        .map(|_| if rng.gen_bool(0.5) { 1 } else { -1 })
        .collect())
}

fn rotate_block(values: &mut [f64], signs: &[i8]) {
    for (value, &sign) in values.iter_mut().zip(signs) {
        *value *= f64::from(sign);
    }
    let mut stride = 1;
    while stride < ROTATION_BLOCK {
        for group in values.chunks_exact_mut(2 * stride) {
            for i in 0..stride {
                let a = group[i];
                let b = group[i + stride];
                group[i] = a + b;
                group[i + stride] = a - b;
            }
        }
        stride *= 2;
    }
    let normalization = 1.0 / (ROTATION_BLOCK as f64).sqrt();
    for value in values {
        *value *= normalization;
    }
}

/// Independent dense Walsh-matrix reference, not the butterfly implementation
/// used when encoding weights or executing the CUDA input transform.
pub fn rotate_reference(input: &[f32], signs: &[i8]) -> Result<Vec<f64>> {
    validate_signs(signs)?;
    ensure!(
        input.len() == signs.len() && input.iter().all(|v| v.is_finite()),
        "invalid rotation input"
    );
    let mut output = vec![0.0; input.len()];
    let normalization = 1.0 / (ROTATION_BLOCK as f64).sqrt();
    for (block, values) in input.chunks_exact(ROTATION_BLOCK).enumerate() {
        for coordinate in 0..ROTATION_BLOCK {
            let mut sum = 0.0;
            for (source, &value) in values.iter().enumerate() {
                let walsh = if (coordinate & source).count_ones().is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                sum += walsh * f64::from(signs[block * ROTATION_BLOCK + source]) * f64::from(value);
            }
            output[block * ROTATION_BLOCK + coordinate] = sum * normalization;
        }
    }
    Ok(output)
}

/// Rank absolute FP32 values in each block, breaking ties by coordinate index.
pub fn select_coordinates(rotated: &[f32], count: usize) -> Result<Vec<u8>> {
    ensure!(
        !rotated.is_empty()
            && rotated.len().is_multiple_of(ROTATION_BLOCK)
            && (1..=ROTATION_BLOCK).contains(&count)
            && rotated.iter().all(|x| x.is_finite()),
        "invalid coordinate selection"
    );
    let mut selected = Vec::with_capacity(rotated.len() / ROTATION_BLOCK * count);
    for block in rotated.chunks_exact(ROTATION_BLOCK) {
        let mut order: Vec<_> = (0..ROTATION_BLOCK as u8).collect();
        order.sort_by(|&a, &b| {
            block[b as usize]
                .abs()
                .total_cmp(&block[a as usize].abs())
                .then(a.cmp(&b))
        });
        selected.extend_from_slice(&order[..count]);
    }
    Ok(selected)
}

fn fit_block(values: &[f64]) -> Result<(f16, [u8; ROTATION_BLOCK])> {
    ensure!(
        values
            .iter()
            .all(|v| v.is_finite() && v.abs() <= f16::MAX.to_f64()),
        "rotated sketch values exceed supported FP16 magnitude"
    );
    let mut order: Vec<_> = (0..ROTATION_BLOCK).collect();
    order.sort_by(|&a, &b| values[b].abs().total_cmp(&values[a].abs()).then(a.cmp(&b)));
    let mut sum = 0.0;
    let mut best_reduction = 0.0;
    let mut best_count = 0;
    let mut best_scale = f16::ZERO;
    for (rank, &index) in order.iter().enumerate() {
        sum += values[index].abs();
        let count = rank + 1;
        let scale = f16::from_f64(sum / count as f64);
        if !scale.is_finite() {
            continue;
        }
        let scale_f64 = scale.to_f64();
        let reduction = 2.0 * scale_f64 * sum - count as f64 * scale_f64 * scale_f64;
        if reduction > best_reduction {
            best_reduction = reduction;
            best_count = count;
            best_scale = scale;
        }
    }
    ensure!(
        values.iter().all(|&v| v == 0.0) || best_scale.to_f64() > 0.0,
        "ternary block has no supported nonzero FP16 scale"
    );
    let mut codes = [0; ROTATION_BLOCK];
    for &index in &order[..best_count] {
        codes[index] = if values[index] > 0.0 { 1 } else { 2 };
    }
    Ok((best_scale, codes))
}

#[derive(Debug, Serialize)]
pub struct TernarySketchReport {
    pub encoding: &'static str,
    pub source_sha256: String,
    pub signs_sha256: String,
    pub scales_sha256: String,
    pub codes_sha256: String,
    pub input_dim: usize,
    pub channels: usize,
    pub source_bytes: usize,
    pub scales_bytes: usize,
    pub codes_bytes: usize,
    pub zero_codes: usize,
    pub max_abs_error: f64,
    pub mean_squared_error: f64,
}

pub struct TernarySketch {
    input_dim: usize,
    channels: usize,
    signs: Vec<i8>,
    scales: Vec<u8>,
    codes: Vec<u8>,
    report: TernarySketchReport,
}

impl TernarySketch {
    /// Encode native Q4 rows into coordinate-major two-bit codes and FP16 scales.
    /// Quantization error refers to the rotated Q4 values, not the original Q8 model.
    pub fn encode_q4(
        source: &[u8],
        input_dim: usize,
        channels: usize,
        signs: &[i8],
    ) -> Result<Self> {
        validate_signs(signs)?;
        ensure!(
            input_dim == signs.len() && channels > 0,
            "invalid sketch shape/signs"
        );
        let expected = input_dim
            .checked_div(32)
            .and_then(|v| v.checked_mul(channels))
            .and_then(|v| v.checked_mul(18))
            .context("sketch source overflow")?;
        ensure!(source.len() == expected, "sketch Q4 source size mismatch");
        let stride = channels.div_ceil(4);
        let mut codes = vec![
            0;
            input_dim
                .checked_mul(stride)
                .context("sketch code size overflow")?
        ];
        let mut scales = vec![0; input_dim / ROTATION_BLOCK * channels * 2];
        let mut max_error = 0.0f64;
        let mut squared_error = 0.0;
        let mut zero_codes = 0;
        for (index, block) in source.chunks_exact(ROTATION_BLOCK / 32 * 18).enumerate() {
            let channel = index / (input_dim / ROTATION_BLOCK);
            let group = index % (input_dim / ROTATION_BLOCK);
            let mut values = decode_q4_values(block)?;
            rotate_block(
                &mut values,
                &signs[group * ROTATION_BLOCK..(group + 1) * ROTATION_BLOCK],
            );
            let (scale, block_codes) = fit_block(&values)?;
            let scale_offset = (group * channels + channel) * 2;
            scales[scale_offset..scale_offset + 2].copy_from_slice(&scale.to_le_bytes());
            for (coordinate, (&code, &value)) in block_codes.iter().zip(&values).enumerate() {
                codes[(group * ROTATION_BLOCK + coordinate) * stride + channel / 4] |=
                    code << (2 * (channel % 4));
                let ternary = match code {
                    0 => 0.0,
                    1 => 1.0,
                    _ => -1.0,
                };
                let error = scale.to_f64() * ternary - value;
                max_error = max_error.max(error.abs());
                squared_error += error * error;
                zero_codes += usize::from(code == 0);
            }
        }
        let report = TernarySketchReport {
            encoding: "q4-walsh128-ternary2-fp16-prefix-v1",
            source_sha256: sha256(source)?,
            signs_sha256: sha256(&signs.iter().map(|&s| s as u8).collect::<Vec<_>>())?,
            scales_sha256: sha256(&scales)?,
            codes_sha256: sha256(&codes)?,
            input_dim,
            channels,
            source_bytes: source.len(),
            scales_bytes: scales.len(),
            codes_bytes: codes.len(),
            zero_codes,
            max_abs_error: max_error,
            mean_squared_error: squared_error / (input_dim as f64 * channels as f64),
        };
        Ok(Self {
            input_dim,
            channels,
            signs: signs.to_vec(),
            scales,
            codes,
            report,
        })
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
    pub fn channels(&self) -> usize {
        self.channels
    }
    pub fn signs(&self) -> &[i8] {
        &self.signs
    }
    pub fn planes(&self) -> [&[u8]; 2] {
        [&self.scales, &self.codes]
    }
    pub fn bytes(&self) -> usize {
        self.scales.len() + self.codes.len()
    }
    pub fn report(&self) -> &TernarySketchReport {
        &self.report
    }

    fn read_ternary(&self, coordinate: usize, channel: usize) -> f64 {
        let code = (self.codes[coordinate * self.channels.div_ceil(4) + channel / 4]
            >> (2 * (channel % 4)))
            & 3;
        match code {
            0 => 0.0,
            1 => 1.0,
            2 => -1.0,
            _ => unreachable!("encoder never emits reserved code"),
        }
    }

    fn read_scale(&self, block: usize, channel: usize) -> f64 {
        let offset = (block * self.channels + channel) * 2;
        f16::from_le_bytes([self.scales[offset], self.scales[offset + 1]]).to_f64()
    }

    pub fn decode_weight(&self, coordinate: usize, channel: usize) -> Result<f64> {
        ensure!(
            coordinate < self.input_dim && channel < self.channels,
            "sketch index out of bounds"
        );
        Ok(self.read_ternary(coordinate, channel)
            * self.read_scale(coordinate / ROTATION_BLOCK, channel))
    }

    /// Scalar FP64 accumulation on the encoded sketch and explicitly selected
    /// coordinates. Callers validate the rotation independently before using it.
    pub fn project_reference(
        &self,
        rotated: &[f32],
        coordinates: &[u8],
        count: usize,
    ) -> Result<Vec<f64>> {
        ensure!(
            (1..=ROTATION_BLOCK).contains(&count)
                && rotated.len() == self.input_dim
                && coordinates.len() == self.input_dim / ROTATION_BLOCK * count
                && rotated.iter().all(|x| x.is_finite()),
            "invalid sketch projection shape"
        );
        for block in coordinates.chunks_exact(count) {
            let mut seen = [false; ROTATION_BLOCK];
            for &coordinate in block {
                ensure!(
                    usize::from(coordinate) < ROTATION_BLOCK && !seen[usize::from(coordinate)],
                    "sketch coordinates duplicate/out of bounds"
                );
                seen[usize::from(coordinate)] = true;
            }
        }
        let mut output = vec![0.0; self.channels];
        for (block, selected) in coordinates.chunks_exact(count).enumerate() {
            for (channel, sum) in output.iter_mut().enumerate() {
                let partial: f64 = selected
                    .iter()
                    .map(|&coordinate| {
                        let coordinate = block * ROTATION_BLOCK + usize::from(coordinate);
                        f64::from(rotated[coordinate]) * self.read_ternary(coordinate, channel)
                    })
                    .sum();
                *sum += self.read_scale(block, channel) * partial;
            }
        }
        Ok(output)
    }
}

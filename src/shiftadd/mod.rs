//! Experiment-only SA4_128 weights. This module does not add GGUF types or
//! change inference dispatch. See the October 31 ShiftAdd research audit.
//!
//! Format and fitting method adapted from ggml revision
//! 93a0b5111eb71c90f302f1268f1535f83e2e1f13. The upstream MIT notice is in
//! LICENSE.txt in this directory and applies to the adapted implementation.

mod fit;
#[cfg(test)]
mod tests;

use anyhow::{ensure, Context, Result};
use half::f16;

pub const BLOCK_WEIGHTS: usize = 256;
pub const BLOCK_BYTES: usize = 138;
pub const GROUP_WEIGHTS: usize = 128;
pub const SIGN_PLANES: usize = 4;
const SCALE_OFFSET: usize = 136;

/// Row-major output weights with 256 consecutive input weights per SA4_128
/// block. Payload ownership is immutable after validation.
#[derive(Clone, Debug)]
pub struct ShiftAddMatrix {
    input_width: usize,
    output_width: usize,
    bytes: Vec<u8>,
}

impl ShiftAddMatrix {
    /// Fit finite FP32 source weights with FP64 offline arithmetic. This uses
    /// the upstream weight-only greedy initialization and four least-squares
    /// refinements, not checkpoint calibration or activation-aware fitting.
    pub fn fit(input_width: usize, output_width: usize, source: &[f32]) -> Result<Self> {
        let (weights, bytes) = validate_shape(input_width, output_width)?;
        ensure!(source.len() == weights, "ShiftAdd source shape mismatch");
        ensure!(
            source.iter().all(|v| v.is_finite()),
            "non-finite ShiftAdd source"
        );
        let mut payload = vec![0; bytes];
        for (block, (source, destination)) in source
            .chunks_exact(BLOCK_WEIGHTS)
            .zip(payload.chunks_exact_mut(BLOCK_BYTES))
            .enumerate()
        {
            fit::encode_block(source, destination)
                .with_context(|| format!("ShiftAdd block {block}"))?;
        }
        Self::from_bytes(input_width, output_width, payload)
    }

    /// Validate a packed SA4_128 payload with finite nonnegative block scales.
    /// Every sign pattern and unsigned amplitude code is valid, including zero.
    pub fn from_bytes(input_width: usize, output_width: usize, bytes: Vec<u8>) -> Result<Self> {
        let (_, expected) = validate_shape(input_width, output_width)?;
        ensure!(bytes.len() == expected, "ShiftAdd payload length mismatch");
        for (index, block) in bytes.chunks_exact(BLOCK_BYTES).enumerate() {
            let scale = read_scale(block);
            ensure!(
                scale.is_finite() && scale >= 0.0,
                "invalid ShiftAdd scale at block {index}"
            );
        }
        Ok(Self {
            input_width,
            output_width,
            bytes,
        })
    }

    pub fn get_input_width(&self) -> usize {
        self.input_width
    }

    pub fn get_output_width(&self) -> usize {
        self.output_width
    }

    pub fn get_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Decode stored weights in output-major order using only the format rule.
    /// Every product of a finite FP16 scale and u8 amplitude, and their four
    /// signed sums, is exactly representable in FP32.
    pub fn decode(&self) -> Vec<f32> {
        let mut weights = Vec::with_capacity(self.input_width * self.output_width);
        for block in self.bytes.chunks_exact(BLOCK_BYTES) {
            let scale = read_scale(block);
            for index in 0..BLOCK_WEIGHTS {
                let group = index / GROUP_WEIGHTS;
                let mut weight = 0.0;
                for plane in 0..SIGN_PLANES {
                    let amplitude = scale * f32::from(block[128 + group * SIGN_PLANES + plane]);
                    let is_positive = block[plane * 32 + index / 8] & (1 << (index % 8)) != 0;
                    weight += if is_positive { amplitude } else { -amplitude };
                }
                weights.push(weight);
            }
        }
        weights
    }

    /// FP64 projection of decoded weights. Inputs contain distinct contiguous
    /// rows; output order is input row then output channel. This is a CPU
    /// correctness reference and deliberately materializes decoded weights.
    pub fn project_reference(&self, input: &[f32]) -> Result<Vec<f64>> {
        ensure!(
            !input.is_empty() && input.len().is_multiple_of(self.input_width),
            "ShiftAdd input shape mismatch"
        );
        ensure!(
            input.iter().all(|value| value.is_finite()),
            "non-finite ShiftAdd input"
        );
        let rows = input.len() / self.input_width;
        let count = rows
            .checked_mul(self.output_width)
            .context("ShiftAdd output size overflow")?;
        let weights = self.decode();
        let mut output = Vec::with_capacity(count);
        for row in input.chunks_exact(self.input_width) {
            for channel in weights.chunks_exact(self.input_width) {
                output.push(
                    row.iter()
                        .zip(channel)
                        .map(|(&x, &w)| f64::from(x) * f64::from(w))
                        .sum(),
                );
            }
        }
        Ok(output)
    }
}

fn read_scale(block: &[u8]) -> f32 {
    f16::from_le_bytes([block[SCALE_OFFSET], block[SCALE_OFFSET + 1]]).to_f32()
}

fn validate_shape(input_width: usize, output_width: usize) -> Result<(usize, usize)> {
    ensure!(
        input_width > 0 && input_width.is_multiple_of(BLOCK_WEIGHTS) && output_width > 0,
        "ShiftAdd requires positive dimensions and input width divisible by 256"
    );
    let weights = input_width
        .checked_mul(output_width)
        .context("ShiftAdd weight size overflow")?;
    let bytes = (weights / BLOCK_WEIGHTS)
        .checked_mul(BLOCK_BYTES)
        .context("ShiftAdd payload size overflow")?;
    ensure!(
        weights <= isize::MAX as usize / std::mem::size_of::<f32>(),
        "ShiftAdd shape exceeds addressable storage"
    );
    Ok((weights, bytes))
}

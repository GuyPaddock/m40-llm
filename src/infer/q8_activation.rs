//! Bounded Q8_1 activation storage for the Maxwell FFN experiment, not weights.
//! Contract adapted from llama.cpp 9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e,
//! ggml-common.h block_q8_1 and ggml-cuda/quantize.cu quantize_q8_1.
//! Copyright (c) 2023-2026 The ggml authors; MIT, see licenses/llama.cpp-MIT.txt.

use anyhow::{ensure, Context, Result};
use half::f16;

pub const Q8_ACTIVATION_BLOCK_VALUES: usize = 32;
pub const Q8_ACTIVATION_BLOCK_BYTES: usize = 36;

/// Row-major FP32 input and explicitly padded Q8_1 output. This experiment
/// supports genuine rows 1/4; padding must preserve complete 32-value warps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q8ActivationShape {
    rows: usize,
    columns: usize,
    padded_columns: usize,
    bytes: usize,
}

impl Q8ActivationShape {
    pub fn new(rows: usize, columns: usize, padded_columns: usize) -> Result<Self> {
        ensure!(
            matches!(rows, 1 | 4)
                && columns > 0
                && padded_columns >= columns
                && padded_columns <= i32::MAX as usize - 255
                && padded_columns.is_multiple_of(Q8_ACTIVATION_BLOCK_VALUES),
            "Q8_1 activations require rows 1/4, positive columns and bounded complete-block padding"
        );
        let bytes = (padded_columns / Q8_ACTIVATION_BLOCK_VALUES)
            .checked_mul(rows)
            .and_then(|count| count.checked_mul(Q8_ACTIVATION_BLOCK_BYTES))
            .context("Q8_1 activation byte overflow")?;
        Ok(Self {
            rows,
            columns,
            padded_columns,
            bytes,
        })
    }

    pub fn rows(self) -> usize {
        self.rows
    }

    pub fn columns(self) -> usize {
        self.columns
    }

    pub fn padded_columns(self) -> usize {
        self.padded_columns
    }

    pub fn bytes(self) -> usize {
        self.bytes
    }

    pub fn block_count(self) -> usize {
        self.bytes / Q8_ACTIVATION_BLOCK_BYTES
    }
}

/// FP16 scale, FP16 original-input sum and 32 signed bytes, with no hidden
/// serialization padding. Quantization uses the FP32 scale before FP16 storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Q8ActivationBlock {
    pub scale_bits: u16,
    pub sum_bits: u16,
    pub quants: [i8; Q8_ACTIVATION_BLOCK_VALUES],
}

impl Q8ActivationBlock {
    pub fn to_bytes(&self) -> [u8; Q8_ACTIVATION_BLOCK_BYTES] {
        let mut bytes = [0; Q8_ACTIVATION_BLOCK_BYTES];
        bytes[..2].copy_from_slice(&self.scale_bits.to_le_bytes());
        bytes[2..4].copy_from_slice(&self.sum_bits.to_le_bytes());
        for (target, &q) in bytes[4..].iter_mut().zip(&self.quants) {
            *target = q as u8;
        }
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() == Q8_ACTIVATION_BLOCK_BYTES,
            "invalid Q8_1 block length"
        );
        Ok(Self {
            scale_bits: u16::from_le_bytes([bytes[0], bytes[1]]),
            sum_bits: u16::from_le_bytes([bytes[2], bytes[3]]),
            quants: std::array::from_fn(|i| bytes[4 + i] as i8),
        })
    }

    pub fn reconstruct(&self) -> [f32; Q8_ACTIVATION_BLOCK_VALUES] {
        let scale = f16::from_bits(self.scale_bits).to_f32();
        self.quants.map(|q| scale * f32::from(q))
    }
}

/// Independent CPU encoding of the pinned CUDA contract. XOR-tree addition
/// matches lane zero's original-input sum, including its FP32 association.
/// Nonfinite inputs and FP32 scale underflow are outside the valid contract.
pub fn quantize_reference(
    input: &[f32],
    shape: Q8ActivationShape,
) -> Result<Vec<Q8ActivationBlock>> {
    ensure!(
        input.len() == shape.rows * shape.columns,
        "Q8_1 input length mismatch"
    );
    ensure!(
        input.iter().all(|x| x.is_finite()),
        "Q8_1 input must be finite"
    );
    let mut output = Vec::with_capacity(shape.block_count());
    for row in 0..shape.rows {
        for offset in (0..shape.padded_columns).step_by(Q8_ACTIVATION_BLOCK_VALUES) {
            let values: [f32; 32] = std::array::from_fn(|lane| {
                if offset + lane < shape.columns {
                    input[row * shape.columns + offset + lane]
                } else {
                    0.0
                }
            });
            let maximum = values.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = maximum / 127.0;
            ensure!(maximum == 0.0 || scale > 0.0, "Q8_1 FP32 scale underflow");
            let mut sums = values;
            for distance in [16, 8, 4, 2, 1] {
                let before = sums;
                for lane in 0..32 {
                    sums[lane] = before[lane] + before[lane ^ distance];
                }
            }
            output.push(Q8ActivationBlock {
                scale_bits: f16::from_f32(scale).to_bits(),
                sum_bits: f16::from_f32(sums[0]).to_bits(),
                quants: values.map(|x| {
                    if maximum == 0.0 {
                        0
                    } else {
                        (x / scale).round() as i8
                    }
                }),
            });
        }
    }
    Ok(output)
}

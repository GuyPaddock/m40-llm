//! Research-only Q4_0 down weights arranged for selective channel reads.
//! Packing preserves every original scale and code; it never requantizes.

use anyhow::{ensure, Context, Result};
use half::f16;

pub const Q4_REPACKED_INPUT_DIM: usize = 11008;
pub const Q4_REPACKED_OUTPUT_DIM: usize = 2048;

/// The two weight layouts admitted by the fixed sparse-Q4 operator study.
#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Q4ResearchLayout {
    Packed,
    Zigzag,
}

/// Channel-major nibbles plus block-major original scale bits.
///
/// Each input channel has ceil(N/2) bytes, with adjacent outputs occupying
/// low/high nibbles. Scales are [input_channel / 32][output][two LE bytes].
/// An odd output count leaves one zero padding nibble per input channel.
#[derive(Debug)]
pub struct Q4ChannelWeights {
    input_dim: usize,
    output_dim: usize,
    scales: Vec<u8>,
    codes: Vec<u8>,
}

impl Q4ChannelWeights {
    /// Reorder native GGUF Q4_0 [output][input-block] bytes losslessly.
    pub fn pack(source: &[u8], input_dim: usize, output_dim: usize) -> Result<Self> {
        ensure!(
            input_dim > 0 && input_dim.is_multiple_of(32),
            "Q4 input must contain complete 32-value blocks"
        );
        ensure!(output_dim > 0, "Q4 output must be nonzero");
        let blocks = input_dim / 32;
        let block_count = blocks
            .checked_mul(output_dim)
            .context("Q4 block count overflow")?;
        let source_bytes = block_count
            .checked_mul(18)
            .context("Q4 byte count overflow")?;
        ensure!(
            source.len() == source_bytes,
            "Q4 source byte count mismatch"
        );
        let stride = output_dim.div_ceil(2);
        let code_bytes = input_dim
            .checked_mul(stride)
            .context("Q4 code count overflow")?;
        let mut packed = Self {
            input_dim,
            output_dim,
            scales: vec![0; block_count * 2],
            codes: vec![0; code_bytes],
        };
        for output in 0..output_dim {
            for block in 0..blocks {
                let src = (output * blocks + block) * 18;
                let dst = (block * output_dim + output) * 2;
                packed.scales[dst..dst + 2].copy_from_slice(&source[src..src + 2]);
                for channel in 0..32 {
                    let code = (source[src + 2 + channel % 16] >> (4 * (channel / 16))) & 15;
                    packed.codes[(block * 32 + channel) * stride + output / 2] |=
                        code << (4 * (output % 2));
                }
            }
        }
        Ok(packed)
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Raw upload planes, in scale/code order. Dimensions are separate metadata.
    pub fn planes(&self) -> [&[u8]; 2] {
        [&self.scales, &self.codes]
    }

    /// Payload including odd-output padding, excluding host allocation headers.
    pub fn bytes(&self) -> usize {
        self.scales.len() + self.codes.len()
    }

    /// Reconstruct native bytes for independent source digest verification.
    pub fn unpack(&self) -> Vec<u8> {
        let blocks = self.input_dim / 32;
        let mut source = vec![0; blocks * self.output_dim * 18];
        for (block_index, block) in source.chunks_exact_mut(18).enumerate() {
            let output = block_index / blocks;
            let group = block_index % blocks;
            let scale = (group * self.output_dim + output) * 2;
            block[..2].copy_from_slice(&self.scales[scale..scale + 2]);
            for pair in 0..16 {
                let low = self.read_code(group * 32 + pair, output);
                let high = self.read_code(group * 32 + pair + 16, output);
                block[2 + pair] = low | (high << 4);
            }
        }
        source
    }

    fn read_code(&self, channel: usize, output: usize) -> u8 {
        (self.codes[channel * self.output_dim.div_ceil(2) + output / 2] >> (4 * (output % 2))) & 15
    }

    /// Decode one channel/output weight for CPU numerical comparisons.
    pub fn decode_weight(&self, channel: usize, output: usize) -> Result<f32> {
        ensure!(
            channel < self.input_dim && output < self.output_dim,
            "Q4 index out of bounds"
        );
        let scale = (channel / 32 * self.output_dim + output) * 2;
        let bits = u16::from_le_bytes([self.scales[scale], self.scales[scale + 1]]);
        Ok(half::f16::from_bits(bits).to_f32()
            * (self.read_code(channel, output) as i32 - 8) as f32)
    }

    /// Requantize 32 adjacent outputs per input and store by output tile, then input.
    pub fn repack_zigzag(&self) -> Result<(Vec<u8>, f64, f64)> {
        ensure!(
            self.input_dim == Q4_REPACKED_INPUT_DIM && self.output_dim == Q4_REPACKED_OUTPUT_DIM,
            "zigzag requires the frozen Q4 down-projection shape"
        );
        let mut blocks = Vec::with_capacity(self.bytes());
        let mut squared_error = 0.0_f64;
        let mut max_error = 0.0_f64;
        for base in (0..self.output_dim).step_by(32) {
            for input in 0..self.input_dim {
                let mut values = [0.0_f32; 32];
                let mut extreme = 0.0_f32;
                for (lane, value) in values.iter_mut().enumerate() {
                    *value = self.decode_weight(input, base + lane)?;
                    ensure!(value.is_finite(), "nonfinite native weight");
                    if value.abs() > extreme.abs() {
                        extreme = *value;
                    }
                }
                let scale = extreme / -8.0;
                let inverse = if scale == 0.0 { 0.0 } else { scale.recip() };
                let stored = f16::from_f32(scale);
                blocks.extend_from_slice(&stored.to_bits().to_le_bytes());
                let codes = values.map(|value| ((value * inverse + 8.5) as u8).min(15));
                for lane in 0..16 {
                    blocks.push(codes[lane] | (codes[16 + lane] << 4));
                }
                for (code, weight) in codes.into_iter().zip(values) {
                    let error = f64::from(stored.to_f32() * (f32::from(code) - 8.0) - weight);
                    squared_error += error * error;
                    max_error = max_error.max(error.abs());
                }
            }
        }
        Ok((
            blocks,
            (squared_error / (self.input_dim * self.output_dim) as f64).sqrt(),
            max_error,
        ))
    }
}

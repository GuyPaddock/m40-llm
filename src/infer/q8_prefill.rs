//! Bounded temporary execution storage, not a persistent FP32 weight cache.

use anyhow::{ensure, Context, Result};

pub const Q8_PREFILL_MAX_WEIGHT_BYTES: usize = 128 * 1024 * 1024;

/// One logical GGUF [input_dim, output_dim] Q8_0 tensor, converted to
/// column-major [output_dim, input_dim] FP32 for the existing SGEMM wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q8PrefillWeightShape {
    input_dim: usize,
    output_dim: usize,
    f32_bytes: usize,
    q8_bytes: usize,
}

impl Q8PrefillWeightShape {
    pub fn new(input_dim: usize, output_dim: usize) -> Result<Self> {
        ensure!(
            input_dim > 0 && input_dim.is_multiple_of(32) && output_dim > 0,
            "Q8 prefill requires positive dimensions and complete 32-value input blocks"
        );
        let elements = input_dim
            .checked_mul(output_dim)
            .context("Q8 prefill weight shape overflow")?;
        let f32_bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .context("Q8 prefill FP32 bytes overflow")?;
        ensure!(
            f32_bytes <= Q8_PREFILL_MAX_WEIGHT_BYTES,
            "Q8 prefill weight needs {f32_bytes} bytes, exceeding the 128 MiB temporary budget"
        );
        let q8_bytes = (elements / 32)
            .checked_mul(34)
            .context("Q8 prefill source bytes overflow")?;
        Ok(Self {
            input_dim,
            output_dim,
            f32_bytes,
            q8_bytes,
        })
    }

    pub fn input_dim(self) -> usize {
        self.input_dim
    }

    pub fn output_dim(self) -> usize {
        self.output_dim
    }

    pub fn f32_bytes(self) -> usize {
        self.f32_bytes
    }

    pub fn q8_bytes(self) -> usize {
        self.q8_bytes
    }
}

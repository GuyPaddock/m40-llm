use anyhow::{ensure, Context, Result};
use half::f16;

/// Original row-contiguous GGUF Q8_0 blocks. No requantization or FP32 cache.
pub struct Q8Matrix {
    pub columns: usize,
    pub rows: usize,
    pub bytes: Vec<u8>,
}

impl Q8Matrix {
    pub fn new(columns: usize, rows: usize, bytes: Vec<u8>) -> Result<Self> {
        ensure!(
            columns > 0 && columns.is_multiple_of(32) && rows > 0,
            "invalid Q8 dimensions"
        );
        let size = columns
            .checked_div(32)
            .and_then(|v| v.checked_mul(rows))
            .and_then(|v| v.checked_mul(34))
            .context("Q8 matrix size overflow")?;
        ensure!(bytes.len() == size, "incorrect Q8 payload size");
        ensure!(
            bytes.chunks_exact(34).all(|block| scale(block).is_finite()),
            "non-finite Q8 scale"
        );
        Ok(Self {
            columns,
            rows,
            bytes,
        })
    }

    fn validate_job(&self, input: &[f32], batch: usize, output_len: usize) -> Result<()> {
        ensure!([1, 2, 4, 8].contains(&batch), "unsupported batch");
        ensure!(
            self.columns.checked_mul(batch) == Some(input.len())
                && self.rows.checked_mul(batch) == Some(output_len),
            "projection shape mismatch"
        );
        ensure!(input.iter().all(|v| v.is_finite()), "non-finite input");
        Ok(())
    }

    pub fn project_scalar(&self, input: &[f32], batch: usize, output: &mut [f32]) -> Result<()> {
        self.validate_job(input, batch, output.len())?;
        let row_bytes = self.columns / 32 * 34;
        for row in 0..self.rows {
            let weights = &self.bytes[row * row_bytes..(row + 1) * row_bytes];
            for b in 0..batch {
                let x = &input[b * self.columns..(b + 1) * self.columns];
                let mut sum = 0.0f32;
                for (block, values) in weights.chunks_exact(34).zip(x.chunks_exact(32)) {
                    let d = scale(block);
                    for (&code, &value) in block[2..].iter().zip(values) {
                        sum += (d * f32::from(code as i8)) * value;
                    }
                }
                output[b * self.rows + row] = sum;
            }
        }
        Ok(())
    }

    pub fn project_avx2(&self, input: &[f32], batch: usize, output: &mut [f32]) -> Result<()> {
        self.validate_job(input, batch, output.len())?;
        ensure!(has_avx2(), "AVX2 is unavailable; no scalar substitution");
        #[cfg(target_arch = "x86_64")]
        // SAFETY: runtime AVX2 admission and all buffer dimensions checked above.
        unsafe {
            self.project_avx2_unchecked(input, batch, output);
        }
        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn project_avx2_unchecked(&self, input: &[f32], batch: usize, output: &mut [f32]) {
        use std::arch::x86_64::*;
        let row_bytes = self.columns / 32 * 34;
        for row in 0..self.rows {
            for b in 0..batch {
                let mut accum = _mm256_setzero_ps();
                for block in 0..self.columns / 32 {
                    let offset = row * row_bytes + block * 34;
                    let d = _mm256_set1_ps(scale(&self.bytes[offset..offset + 34]));
                    for group in 0..4 {
                        let codes =
                            _mm_loadl_epi64(self.bytes.as_ptr().add(offset + 2 + group * 8).cast());
                        let weights =
                            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(codes)), d);
                        let values = _mm256_loadu_ps(
                            input
                                .as_ptr()
                                .add(b * self.columns + block * 32 + group * 8),
                        );
                        accum = _mm256_add_ps(accum, _mm256_mul_ps(weights, values));
                    }
                }
                let mut lanes = [0.0f32; 8];
                _mm256_storeu_ps(lanes.as_mut_ptr(), accum);
                output[b * self.rows + row] = lanes.into_iter().sum();
            }
        }
    }

    /// Independent scalar FP64 sum over explicitly decoded original blocks.
    pub fn reference(&self, input: &[f32], batch: usize) -> Result<Vec<f64>> {
        let length = self.rows.checked_mul(batch).context("reference overflow")?;
        self.validate_job(input, batch, length)?;
        let mut result = vec![0.0; length];
        for b in 0..batch {
            for row in 0..self.rows {
                for column in 0..self.columns {
                    let block = (row * self.columns / 32 + column / 32) * 34;
                    let d = f64::from(
                        f16::from_bits(u16::from_le_bytes([
                            self.bytes[block],
                            self.bytes[block + 1],
                        ]))
                        .to_f32(),
                    );
                    let code = f64::from(self.bytes[block + 2 + column % 32] as i8);
                    result[b * self.rows + row] +=
                        d * code * f64::from(input[b * self.columns + column]);
                }
            }
        }
        Ok(result)
    }
}

fn scale(block: &[u8]) -> f32 {
    f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32()
}

pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

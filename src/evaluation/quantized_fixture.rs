//! Deterministic research fixtures derived from existing Q8_0 checkpoint bytes.
//! This additional quantization is distinct from lossless execution repacking.

use super::identity::sha256;
use anyhow::{ensure, Result};
use half::f16;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Q4FixtureReport {
    pub encoding: &'static str,
    pub source_sha256: String,
    pub q4_sha256: String,
    pub source_bytes: usize,
    pub q4_bytes: usize,
    pub values: usize,
    pub max_abs_error: f64,
    pub mean_squared_error: f64,
}

/// Encode each original Q8 block independently with positive absmax/7 scale,
/// ties-away FP32 rounding and [-8,7] clamping. Quantize with the FP32 scale
/// before storing its FP16 rounding, matching the existing Q4 test fixture
/// convention. This is not a claim to reproduce a released Q4 checkpoint.
pub fn requantize_q8_to_q4(source: &[u8]) -> Result<(Vec<u8>, Q4FixtureReport)> {
    ensure!(
        !source.is_empty() && source.len().is_multiple_of(34),
        "fixture requires complete Q8_0 blocks"
    );
    let mut output = Vec::with_capacity(source.len() / 34 * 18);
    let mut max_error = 0.0f64;
    let mut squared_error = 0.0;
    for block in source.chunks_exact(34) {
        let source_scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        ensure!(source_scale.is_finite(), "nonfinite Q8 source scale");
        let mut values = [0.0f32; 32];
        for (value, &code) in values.iter_mut().zip(&block[2..]) {
            *value = source_scale * f32::from(code as i8);
        }
        let maximum = values.iter().fold(0.0f32, |v, &x| v.max(x.abs()));
        let scale = maximum / 7.0;
        let stored_scale = f16::from_f32(scale);
        ensure!(
            stored_scale.is_finite() && (maximum == 0.0 || stored_scale.to_f32() > 0.0),
            "Q4 fixture scale is not representable in FP16"
        );
        let mut codes = [8u8; 32];
        for (code, &value) in codes.iter_mut().zip(&values) {
            let quantized = if scale == 0.0 {
                0
            } else {
                (value / scale).round().clamp(-8.0, 7.0) as i32
            };
            *code = (quantized + 8) as u8;
            let reconstructed = stored_scale.to_f32() * quantized as f32;
            let error = f64::from(reconstructed) - f64::from(value);
            max_error = max_error.max(error.abs());
            squared_error += error * error;
        }
        output.extend_from_slice(&stored_scale.to_le_bytes());
        for pair in 0..16 {
            output.push(codes[pair] | (codes[pair + 16] << 4));
        }
    }
    let values = source.len() / 34 * 32;
    let report = Q4FixtureReport {
        encoding: "q8-f32-q4-absmax7-v1",
        source_sha256: sha256(source)?,
        q4_sha256: sha256(&output)?,
        source_bytes: source.len(),
        q4_bytes: output.len(),
        values,
        max_abs_error: max_error,
        mean_squared_error: squared_error / values as f64,
    };
    Ok((output, report))
}

/// Independently reconstruct native Q4_0 blocks for scalar numerical oracles.
pub fn decode_q4_values(source: &[u8]) -> Result<Vec<f64>> {
    ensure!(
        !source.is_empty() && source.len().is_multiple_of(18),
        "fixture requires complete Q4_0 blocks"
    );
    let mut values = Vec::with_capacity(source.len() / 18 * 32);
    for block in source.chunks_exact(18) {
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f64();
        ensure!(scale.is_finite(), "nonfinite Q4 scale");
        for index in 0..32 {
            let byte = block[2 + index % 16];
            let code = if index < 16 { byte & 15 } else { byte >> 4 };
            values.push(scale * f64::from(i32::from(code) - 8));
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact levels and halfway values preserve the documented ties-away rule.
    #[test]
    fn test_known_values_and_rounding() -> Result<()> {
        let mut raw = vec![0u8; 34];
        raw[..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        raw[2..6].copy_from_slice(&[14, (-14i8) as u8, 7, (-7i8) as u8]);
        let (bytes, report) = requantize_q8_to_q4(&raw)?;
        assert_eq!(&bytes[..6], &[0, 0x3c, 0x8f, 0x81, 0x8c, 0x84]);
        assert!(bytes[6..].iter().all(|&v| v == 0x88));
        assert_eq!(&decode_q4_values(&bytes)?[..4], &[7.0, -7.0, 4.0, -4.0]);
        assert_eq!(report.max_abs_error, 0.5);
        assert_eq!(report.mean_squared_error, 0.5 / 32.0);
        assert_eq!(report.source_sha256, sha256(&raw)?);
        assert_eq!(report.q4_sha256, sha256(&bytes)?);
        assert_eq!(requantize_q8_to_q4(&raw)?.0, bytes);
        Ok(())
    }

    /// Negative Q8 scales retain value signs and each block gets its own scale.
    #[test]
    fn test_signed_scales_and_zero_blocks() -> Result<()> {
        let mut raw = vec![0u8; 68];
        raw[..2].copy_from_slice(&f16::from_f32(-2.0).to_le_bytes());
        raw[2] = 7;
        raw[18] = (-7i8) as u8;
        let (bytes, report) = requantize_q8_to_q4(&raw)?;
        let values = decode_q4_values(&bytes)?;
        assert_eq!(values[0], -14.0);
        assert_eq!(values[16], 14.0);
        assert!(values[32..].iter().all(|&v| v == 0.0));
        assert_eq!(report.values, 64);
        assert_eq!(report.source_bytes, 68);
        assert_eq!(report.q4_bytes, 36);
        assert_eq!(report.max_abs_error, 0.0);
        Ok(())
    }

    /// Malformed inputs and unrepresentable scales fail instead of hiding drift.
    #[test]
    fn test_rejects_invalid_quantization() {
        for size in [0, 1, 33, 35] {
            assert!(requantize_q8_to_q4(&vec![0; size]).is_err());
        }
        for scale in [f16::NAN, f16::INFINITY, f16::MAX, f16::from_bits(1)] {
            let mut raw = vec![0u8; 34];
            raw[..2].copy_from_slice(&scale.to_le_bytes());
            raw[2] = if scale == f16::MAX { 127 } else { 1 };
            assert!(requantize_q8_to_q4(&raw).is_err());
        }
        assert!(decode_q4_values(&[0; 17]).is_err());
    }
}

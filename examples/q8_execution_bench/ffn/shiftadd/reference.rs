//! CPU reference and fit-error reporting, separate from GPU arithmetic error.
use anyhow::{ensure, Result};
use half::f16;
use m40_llm::shiftadd::ShiftAddMatrix;
use serde_json::{json, Value};

pub(super) fn decode_q8(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(
        !bytes.is_empty() && bytes.len().is_multiple_of(34),
        "incomplete Q8 source"
    );
    let mut values = Vec::with_capacity(bytes.len() / 34 * 32);
    for block in bytes.chunks_exact(34) {
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        ensure!(scale.is_finite(), "nonfinite source scale");
        values.extend(block[2..].iter().map(|v| scale * f32::from(*v as i8)));
    }
    Ok(values)
}

pub(super) fn fit_error(reference: &[f64], candidate: &[f64]) -> Result<Value> {
    ensure!(
        !reference.is_empty() && reference.len() == candidate.len(),
        "fit error shape mismatch"
    );
    let mut square_error = 0.0;
    let mut square_reference = 0.0;
    let mut maximum = 0.0_f64;
    for (&a, &b) in reference.iter().zip(candidate) {
        ensure!(a.is_finite() && b.is_finite(), "nonfinite fit value");
        let delta = (a - b).abs();
        maximum = maximum.max(delta);
        square_error += delta * delta;
        square_reference += a * a;
    }
    Ok(json!({"values":reference.len(),"max_abs_error":maximum,
        "mean_squared_error":square_error/reference.len() as f64,
        "relative_l2_error":if square_reference>0.0 {Some((square_error/square_reference).sqrt())} else {None}}))
}

pub(super) struct Reference {
    pub values: Vec<f64>,
    pub bounds: Vec<f64>,
}

/// Independent scalar FP64 dot, with a conservative sequential FP32 bound
/// for ordinary consumers. ShiftAdd uses plane sums before cancellation.
pub(super) fn project(
    weights: &[f64],
    input: &[f32],
    k: usize,
    matrix: Option<&ShiftAddMatrix>,
) -> Result<Reference> {
    ensure!(
        k > 0
            && !weights.is_empty()
            && weights.len().is_multiple_of(k)
            && !input.is_empty()
            && input.len().is_multiple_of(k),
        "invalid reference shape"
    );
    ensure!(
        weights.iter().all(|v| v.is_finite()) && input.iter().all(|v| v.is_finite()),
        "nonfinite reference"
    );
    if let Some(matrix) = matrix {
        ensure!(
            matrix.get_input_width() == k && matrix.get_output_width() == weights.len() / k,
            "ShiftAdd bound shape mismatch"
        );
    }
    let operations = if matrix.is_some() {
        12 + k / 16
    } else {
        2 * k + 32
    };
    let error_unit = operations as f64 * 2.0_f64.powi(-24);
    ensure!(error_unit < 1.0, "unbounded FP32 error estimate");
    let gamma = error_unit / (1.0 - error_unit);
    let mut values = Vec::new();
    let mut bounds = Vec::new();
    for row in input.chunks_exact(k) {
        for (channel, weight) in weights.chunks_exact(k).enumerate() {
            let mut sum = 0.0;
            let mut absolute_sum = 0.0;
            for (&x, &w) in row.iter().zip(weight) {
                let product = f64::from(x) * w;
                sum += product;
                absolute_sum += product.abs();
            }
            if let Some(matrix) = matrix {
                absolute_sum = 0.0;
                for (block_index, block) in matrix.get_bytes()[channel * (k / 256) * 138..]
                    [..(k / 256) * 138]
                    .chunks_exact(138)
                    .enumerate()
                {
                    let scale = f16::from_le_bytes([block[136], block[137]]).to_f64();
                    for group in 0..2 {
                        let amplitude = block[128 + group * 4..132 + group * 4]
                            .iter()
                            .map(|&v| f64::from(v) * scale)
                            .sum::<f64>();
                        let start = block_index * 256 + group * 128;
                        absolute_sum += amplitude
                            * row[start..start + 128]
                                .iter()
                                .map(|&v| f64::from(v).abs())
                                .sum::<f64>();
                    }
                }
            }
            values.push(sum);
            bounds.push(gamma * absolute_sum + operations as f64 * f64::from(f32::MIN_POSITIVE));
        }
    }
    Ok(Reference { values, bounds })
}

impl Reference {
    pub fn check(&self, actual: &[f32]) -> Result<Value> {
        ensure!(
            actual.len() == self.values.len(),
            "projection result length mismatch"
        );
        let mut maximum_fraction = 0.0_f64;
        let mut maximum_error = 0.0_f64;
        for (index, (&a, (&b, &bound))) in actual
            .iter()
            .zip(self.values.iter().zip(&self.bounds))
            .enumerate()
        {
            let error = (f64::from(a) - b).abs();
            ensure!(
                a.is_finite() && error <= bound,
                "projection {index}: {a} versus {b}, error {error} exceeds {bound}"
            );
            maximum_fraction = maximum_fraction.max(error / bound);
            maximum_error = maximum_error.max(error);
        }
        Ok(
            json!({"values":actual.len(),"maximum_bound_fraction":maximum_fraction,"maximum_abs_error":maximum_error}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Opposing sign planes can cancel the stored weight while their reduction
    /// roundoff bound stays positive. The arithmetic oracle remains a zero dot.
    #[test]
    fn test_plane_cancellation_bound() -> Result<()> {
        let mut bytes = vec![0; 138];
        bytes[..32].fill(255);
        bytes[128..136].copy_from_slice(&[3, 3, 0, 0, 3, 3, 0, 0]);
        bytes[136..].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        let matrix = ShiftAddMatrix::from_bytes(256, 1, bytes)?;
        let values: Vec<_> = matrix.decode().into_iter().map(f64::from).collect();
        assert!(values.iter().all(|&v| v == 0.0));
        let reference = project(&values, &[1.0; 256], 256, Some(&matrix))?;
        assert_eq!(reference.values, [0.0]);
        let roundings = 28.0 * 2.0_f64.powi(-24);
        let expected = roundings / (1.0 - roundings) * 768.0;
        assert!((reference.bounds[0] - expected).abs() < 1e-12);
        reference.check(&[0.0])?;
        assert!(reference.check(&[1.0]).is_err());
        Ok(())
    }

    /// Distinct rows and output channels use independent dots; a corrupted
    /// result fails arithmetic qualification instead of becoming fit error.
    #[test]
    fn test_reference_rows_and_corruption() -> Result<()> {
        let reference = project(&[1.0, -2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, None)?;
        assert_eq!(reference.values, [-7.0, 39.0, -9.0, 53.0]);
        reference.check(&[-7.0, 39.0, -9.0, 53.0])?;
        assert!(reference.check(&[-7.0, 39.0, -8.0, 53.0]).is_err());
        assert!(reference.check(&[f32::NAN; 4]).is_err());
        assert!(project(&[1.0], &[1.0], 2, None).is_err());
        let error = fit_error(&[1.0, 2.0], &[2.0, 4.0])?;
        assert_eq!(error["mean_squared_error"], 2.5);
        assert_eq!(error["relative_l2_error"], 1.0);
        Ok(())
    }

    /// Signed Q8 scales and codes reconstruct exactly before either fit.
    #[test]
    fn test_decode_signed_q8() -> Result<()> {
        let mut bytes = vec![0; 34];
        bytes[..2].copy_from_slice(&f16::from_f32(-0.5).to_le_bytes());
        bytes[2] = (-4_i8) as u8;
        bytes[3] = 6;
        let values = decode_q8(&bytes)?;
        assert_eq!(&values[..2], &[2.0, -3.0]);
        assert!(values[2..].iter().all(|&v| v == 0.0));
        assert!(decode_q8(&bytes[..33]).is_err());
        Ok(())
    }
}

use anyhow::{ensure, Result};
use half::f16;
use m40_llm::evaluation::identity::sha256;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{json, Value};

pub(super) fn generate_signs(columns: usize) -> Vec<i8> {
    let mut rng = StdRng::seed_from_u64(0);
    (0..columns)
        .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
        .collect()
}

/// FP32 fixture transform, independently checked against dense FP64 Walsh.
fn rotate_weights(input: &[f32], signs: &[i8]) -> Result<Vec<f32>> {
    ensure!(
        !signs.is_empty()
            && signs.len().is_multiple_of(128)
            && input.len().is_multiple_of(signs.len()),
        "invalid rotation shape"
    );
    ensure!(
        signs.iter().all(|s| matches!(s, -1 | 1)) && input.iter().all(|v| v.is_finite()),
        "invalid rotation values"
    );
    let mut output: Vec<_> = input
        .iter()
        .enumerate()
        .map(|(i, v)| v * f32::from(signs[i % signs.len()]))
        .collect();
    for block in output.chunks_exact_mut(128) {
        for distance in [1, 2, 4, 8, 16, 32, 64] {
            for group in block.chunks_exact_mut(distance * 2) {
                for i in 0..distance {
                    let a = group[i];
                    let b = group[i + distance];
                    group[i] = a + b;
                    group[i + distance] = a - b;
                }
            }
        }
        for value in block {
            *value *= 0.088_388_346;
        }
    }
    ensure!(output.iter().all(|v| v.is_finite()), "rotation overflow");
    Ok(output)
}

/// Dense coefficient reference shares neither butterfly nor normalization code.
pub(super) fn rotate_reference(input: &[f32], signs: &[i8]) -> Vec<f64> {
    input
        .chunks_exact(signs.len())
        .flat_map(|row| {
            row.chunks_exact(128)
                .enumerate()
                .flat_map(|(block, values)| {
                    (0..128usize).map(move |output| {
                        values
                            .iter()
                            .enumerate()
                            .map(|(input, &value)| {
                                let coefficient = if (input & output).count_ones().is_multiple_of(2)
                                {
                                    1.0
                                } else {
                                    -1.0
                                };
                                f64::from(value)
                                    * f64::from(signs[block * 128 + input])
                                    * coefficient
                            })
                            .sum::<f64>()
                            / 128.0f64.sqrt()
                    })
                })
        })
        .collect()
}

pub(super) fn encode_rotated_q8(source: &[u8], signs: &[i8]) -> Result<(Vec<u8>, Value)> {
    ensure!(
        !source.is_empty() && source.len().is_multiple_of(34),
        "invalid Q8 source"
    );
    let mut values = Vec::new();
    for block in source.chunks_exact(34) {
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        values.extend(block[2..].iter().map(|&q| scale * f32::from(q as i8)));
    }
    let rotated = rotate_weights(&values, signs)?;
    let mut bytes = Vec::with_capacity(source.len());
    let mut max_error = 0.0f64;
    let mut squared_error = 0.0;
    for block in rotated.chunks_exact(32) {
        let maximum = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = maximum / 127.0;
        let stored_scale = f16::from_f32(scale);
        ensure!(
            stored_scale.is_finite() && (maximum == 0.0 || scale > 0.0),
            "invalid Q8 fitted scale"
        );
        bytes.extend_from_slice(&stored_scale.to_le_bytes());
        for &value in block {
            let quantized = if maximum == 0.0 {
                0
            } else {
                (value / scale).round().clamp(-127.0, 127.0) as i8
            };
            bytes.push(quantized as u8);
            let error = stored_scale.to_f64() * f64::from(quantized) - f64::from(value);
            max_error = max_error.max(error.abs());
            squared_error += error * error;
        }
    }
    let rotation_bytes: Vec<_> = rotated.iter().flat_map(|v| v.to_le_bytes()).collect();
    let sign_bytes: Vec<_> = signs.iter().map(|&s| s as u8).collect();
    Ok((
        bytes.clone(),
        json!({"encoding":"q8-walsh128-f32-absmax127-v1",
        "source_sha256":sha256(source)?,"source_bytes":source.len(),
        "sign_sha256":sha256(&sign_bytes)?,"rotated_f32_sha256":sha256(&rotation_bytes)?,
        "encoded_sha256":sha256(&bytes)?,"encoded_bytes":bytes.len(),
        "max_abs_error":max_error,"mean_squared_error":squared_error/rotated.len() as f64}),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixture rotation agrees with independent coefficients, preserves dot
    /// products and keeps distinct rows in their original blocks.
    #[test]
    fn test_rotation_matches_walsh_and_preserves_dot_products() -> Result<()> {
        let signs = generate_signs(256);
        let input: Vec<_> = (0..512).map(|i| (i % 31) as f32 * 0.125 - 1.0).collect();
        let actual = rotate_weights(&input, &signs)?;
        let expected = rotate_reference(&input, &signs);
        for (a, b) in actual.iter().zip(expected) {
            assert!((f64::from(*a) - b).abs() < 1e-5);
        }
        let before: f64 = input[..256]
            .iter()
            .zip(&input[256..])
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let after: f64 = actual[..256]
            .iter()
            .zip(&actual[256..])
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        assert!((before - after).abs() < 1e-4);
        Ok(())
    }

    /// Fitting retains valid native Q8 blocks and reports measured decode error;
    /// zeros, invalid dimensions and nonfinite source scales are explicit.
    #[test]
    fn test_rotated_encoding_roundtrip_and_rejections() -> Result<()> {
        let signs = generate_signs(128);
        let (zero, report) = encode_rotated_q8(&[0; 136], &signs)?;
        assert_eq!(zero, [0; 136]);
        assert_eq!(report["max_abs_error"], 0.0);
        let mut source = vec![0; 136];
        for (index, block) in source.chunks_exact_mut(34).enumerate() {
            block[..2].copy_from_slice(&f16::from_f32(0.03125).to_le_bytes());
            for (lane, value) in block[2..].iter_mut().enumerate() {
                *value = (index * 32 + lane) as u8;
            }
        }
        let values: Vec<_> = (0..128).map(|i| i as f32 * 0.03125).collect();
        let rotated = rotate_weights(&values, &signs)?;
        let (bytes, report) = encode_rotated_q8(&source, &signs)?;
        let mut maximum = 0.0f64;
        for (block, reference) in bytes.chunks_exact(34).zip(rotated.chunks_exact(32)) {
            let scale = f16::from_le_bytes([block[0], block[1]]).to_f64();
            for (&q, &value) in block[2..].iter().zip(reference) {
                maximum = maximum.max((scale * f64::from(q as i8) - f64::from(value)).abs());
            }
        }
        assert_eq!(maximum, report["max_abs_error"].as_f64().unwrap());
        assert!(maximum > 0.0 && maximum < 0.2);
        assert!(encode_rotated_q8(&[0; 135], &signs).is_err());
        assert!(encode_rotated_q8(&source, &[1; 32]).is_err());
        source[..2].copy_from_slice(&f16::INFINITY.to_le_bytes());
        assert!(encode_rotated_q8(&source, &signs).is_err());
        Ok(())
    }
}

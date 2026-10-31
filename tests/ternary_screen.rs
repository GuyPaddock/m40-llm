use anyhow::Result;
use half::f16;
use m40_llm::evaluation::ternary_screen::{
    generate_signs, rotate_reference, select_coordinates, TernarySketch,
};

fn encode_native(values: &[i8], scale: f32) -> Vec<u8> {
    let mut bytes = Vec::new();
    for block in values.chunks_exact(32) {
        bytes.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
        for i in 0..16 {
            bytes.push((block[i] + 8) as u8 | (((block[i + 16] + 8) as u8) << 4));
        }
    }
    bytes
}

/// The sign/Walsh reference preserves dot products across independent blocks.
#[test]
fn test_rotation_preserves_geometry() -> Result<()> {
    let signs = generate_signs(256, 0)?;
    assert_eq!(signs, generate_signs(256, 0)?);
    assert_ne!(signs, generate_signs(256, 1)?);
    let a: Vec<_> = (0..256).map(|i| (i % 17) as f32 / 8.0 - 1.0).collect();
    let b: Vec<_> = (0..256).map(|i| (i % 29) as f32 / 16.0 - 0.5).collect();
    let ra = rotate_reference(&a, &signs)?;
    let rb = rotate_reference(&b, &signs)?;
    let original: f64 = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum();
    let rotated: f64 = ra.iter().zip(&rb).map(|(x, y)| x * y).sum();
    assert!((original - rotated).abs() < 1e-10);
    let constant = rotate_reference(&[0.5; 128], &[1; 128])?;
    assert!((constant[0] - 128.0f64.sqrt() * 0.5).abs() < 1e-12);
    assert!(constant[1..].iter().all(|&v| v == 0.0));
    Ok(())
}

/// Signed constant rows become one rotated coefficient; odd channel padding stays zero.
#[test]
fn test_known_sketch_values_and_partial_projection() -> Result<()> {
    let values: Vec<_> = [1, -1, 0, 1, -1]
        .into_iter()
        .flat_map(|v| [v; 128])
        .collect();
    let source = encode_native(&values, 0.5);
    let sketch = TernarySketch::encode_q4(&source, 128, 5, &[1; 128])?;
    assert_eq!(sketch.input_dim(), 128);
    assert_eq!(sketch.channels(), 5);
    assert_eq!(sketch.bytes(), 266);
    assert_eq!(&sketch.planes()[1][..2], &[0x49, 0x02]);
    assert!(sketch.planes()[1][2..].iter().all(|&v| v == 0));
    let scale = f16::from_f64(128.0f64.sqrt() * 0.5).to_f64();
    assert_eq!(sketch.decode_weight(0, 0)?, scale);
    assert_eq!(sketch.decode_weight(0, 1)?, -scale);
    assert_eq!(sketch.decode_weight(0, 2)?, 0.0);
    assert_eq!(sketch.report().zero_codes, 636);
    let rotated: Vec<_> = rotate_reference(&[0.5; 128], &[1; 128])?
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let indices = select_coordinates(&rotated, 80)?;
    assert_eq!(indices, (0..80).collect::<Vec<u8>>());
    let output = sketch.project_reference(&rotated, &indices, 80)?;
    for (actual, expected) in output.iter().zip([32.0, -32.0, 0.0, 32.0, -32.0]) {
        assert!((actual - expected).abs() < 0.005);
    }
    assert!(sketch.decode_weight(128, 0).is_err());
    assert!(sketch.decode_weight(0, 5).is_err());
    Ok(())
}

/// Encoded error and selected-coordinate projections agree with reconstructed values.
#[test]
fn test_mixed_rows_error_and_coordinate_accounting() -> Result<()> {
    let signs = generate_signs(256, 0)?;
    let values: Vec<_> = (0..768)
        .map(|i| ((i * 7 + i / 256 * 3) % 15) as i8 - 7)
        .collect();
    let source = encode_native(&values, 0.125);
    let sketch = TernarySketch::encode_q4(&source, 256, 3, &signs)?;
    let mut squared_error = 0.0;
    let mut maximum = 0.0f64;
    let mut zero_error = 0.0;
    for channel in 0..3 {
        let source: Vec<_> = values[channel * 256..(channel + 1) * 256]
            .iter()
            .map(|&v| f32::from(v) * 0.125)
            .collect();
        let rotated = rotate_reference(&source, &signs)?;
        for (i, &value) in rotated.iter().enumerate() {
            let error = sketch.decode_weight(i, channel)? - value;
            squared_error += error * error;
            maximum = maximum.max(error.abs());
            zero_error += value * value;
        }
    }
    assert!(squared_error < zero_error);
    assert!((sketch.report().mean_squared_error - squared_error / 768.0).abs() < 1e-12);
    assert!((sketch.report().max_abs_error - maximum).abs() < 1e-12);
    let input: Vec<_> = (0..256).map(|i| (i % 19) as f32 / 16.0 - 0.5).collect();
    let rotated: Vec<_> = rotate_reference(&input, &signs)?
        .into_iter()
        .map(|v| v as f32)
        .collect();
    for count in [1, 80, 128] {
        let coordinates = select_coordinates(&rotated, count)?;
        let output = sketch.project_reference(&rotated, &coordinates, count)?;
        for (channel, &actual) in output.iter().enumerate() {
            let mut expected = 0.0;
            for (block, indices) in coordinates.chunks_exact(count).enumerate() {
                for &i in indices {
                    let coordinate = block * 128 + usize::from(i);
                    expected += f64::from(rotated[coordinate])
                        * sketch.decode_weight(coordinate, channel)?;
                }
            }
            assert!((actual - expected).abs() < 1e-11);
        }
    }
    let mut ties = [1.0; 256];
    ties[7] = -4.0;
    ties[8] = 4.0;
    assert_eq!(select_coordinates(&ties, 3)?, [7, 8, 0, 0, 1, 2]);
    Ok(())
}

/// Invalid dimensions, values, signs and coordinate lists fail before indexing or allocation.
#[test]
fn test_rejects_unsupported_inputs() -> Result<()> {
    let source = encode_native(&[1; 128], 0.125);
    let signs = [1; 128];
    assert!(generate_signs(0, 0).is_err());
    assert!(generate_signs(129, 0).is_err());
    assert!(TernarySketch::encode_q4(&source[..71], 128, 1, &signs).is_err());
    assert!(TernarySketch::encode_q4(&source, 128, 0, &signs).is_err());
    assert!(TernarySketch::encode_q4(&[], 128, usize::MAX, &signs).is_err());
    assert!(TernarySketch::encode_q4(&source, 128, 1, &[0; 128]).is_err());
    let mut invalid = source.clone();
    invalid[..2].copy_from_slice(&f16::NAN.to_le_bytes());
    assert!(TernarySketch::encode_q4(&invalid, 128, 1, &signs).is_err());
    assert!(TernarySketch::encode_q4(&encode_native(&[7; 128], 65504.0), 128, 1, &signs).is_err());
    let zero = TernarySketch::encode_q4(&encode_native(&[0; 128], 0.0), 128, 1, &signs)?;
    assert_eq!(zero.report().max_abs_error, 0.0);
    assert_eq!(zero.report().zero_codes, 128);
    assert!(rotate_reference(&[f32::INFINITY; 128], &signs).is_err());
    assert!(select_coordinates(&[0.0; 128], 0).is_err());
    assert!(select_coordinates(&[0.0; 128], 129).is_err());
    assert!(select_coordinates(&[f32::NAN; 128], 80).is_err());
    assert!(zero.project_reference(&[0.0; 128], &[0, 0], 2).is_err());
    assert!(zero.project_reference(&[0.0; 128], &[128], 1).is_err());
    assert!(zero.project_reference(&[0.0; 127], &[0], 1).is_err());
    Ok(())
}

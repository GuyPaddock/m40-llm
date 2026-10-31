use super::{ShiftAddMatrix, BLOCK_BYTES};
use half::f16;

/// A manually specified payload exercises all 16 sign combinations, both
/// amplitude groups, adjacent 138-byte blocks and separate output channels.
fn make_manual_matrix() -> ShiftAddMatrix {
    let mut bytes = vec![0; BLOCK_BYTES * 4];
    for (block_index, block) in bytes.chunks_exact_mut(BLOCK_BYTES).enumerate() {
        for byte in 0..32 {
            block[byte] = 0xaa;
            block[32 + byte] = 0xcc;
            block[64 + byte] = 0xf0;
            block[96 + byte] = if byte % 2 == 0 { 0 } else { 0xff };
        }
        block[128..136].copy_from_slice(&[1, 2, 4, 8, 2, 4, 8, 16]);
        block[136..]
            .copy_from_slice(&f16::from_f32(0.125 * (block_index + 1) as f32).to_le_bytes());
    }
    ShiftAddMatrix::from_bytes(512, 2, bytes).unwrap()
}

/// Evaluate the manual fixture arithmetically without reading packed bytes or
/// calling the production decoder.
fn calculate_manual_weight(index: usize) -> f64 {
    let block = index / 256;
    let within = index % 256;
    let group_scale = if within < 128 { 1.0 } else { 2.0 };
    (2.0 * (within % 16) as f64 - 15.0) * 0.125 * (block + 1) as f64 * group_scale
}

/// Decoded values must match the externally constructed bit-plane payload,
/// including group, block and output-row boundaries.
#[test]
fn test_decode_manual_payload() {
    let matrix = make_manual_matrix();
    let retained = matrix.get_bytes().to_vec();
    for (index, value) in matrix.decode().iter().enumerate() {
        assert_eq!(f64::from(*value), calculate_manual_weight(index));
    }
    assert_eq!(matrix.get_input_width(), 512);
    assert_eq!(matrix.get_output_width(), 2);
    assert_eq!(matrix.get_bytes(), retained);
}

/// Reference projection respects distinct input rows and output channels and
/// leaves both input values and immutable encoded weights unchanged.
#[test]
fn test_reference_projection_distinct_rows() {
    let matrix = make_manual_matrix();
    let input: Vec<f32> = (0..4 * 512)
        .map(|i| ((i * 19 % 97) as f32 - 48.0) * 0.125)
        .collect();
    let retained = input.clone();
    let actual = matrix.project_reference(&input).unwrap();
    for row in 0..4 {
        for channel in 0..2 {
            let expected: f64 = (0..512)
                .map(|index| {
                    f64::from(input[row * 512 + index])
                        * calculate_manual_weight(channel * 512 + index)
                })
                .sum();
            assert_eq!(actual[row * 2 + channel], expected);
        }
    }
    assert_eq!(input, retained);
}

/// Singular least-squares systems for zero and constant groups retain a
/// finite fit. FP16-scale rounding bounds the constant reconstruction error.
#[test]
fn test_fit_zero_constant_and_singular_groups() {
    for constant in [0.0_f32, -0.0, 0.5, -0.5, 127.0, -127.0] {
        let mut source = vec![constant; 512];
        source[128..256].fill(-constant);
        source[384..].fill(-constant);
        let matrix = ShiftAddMatrix::fit(256, 2, &source).unwrap();
        assert_eq!(matrix.get_bytes().len(), 276);
        let tolerance = constant.abs() * 0.0005 + 2e-5;
        for (&actual, &expected) in matrix.decode().iter().zip(&source) {
            assert!(actual.is_finite());
            assert!(
                (actual - expected).abs() <= tolerance,
                "{actual} != {expected}"
            );
        }
        if constant == 0.0 {
            assert!(matrix.get_bytes().iter().all(|&value| value == 0));
        }
    }
}

/// A balanced sixteen-level source is represented closely after shared FP16
/// and u8 amplitude rounding. Multiple fits must produce the same bytes.
#[test]
fn test_fit_sixteen_levels_is_deterministic() {
    let source: Vec<f32> = (0..1024)
        .map(|i| ((i % 16) as f32 * 2.0 - 15.0) * if i % 256 < 128 { 0.125 } else { 0.25 })
        .collect();
    let original = source.clone();
    let matrix = ShiftAddMatrix::fit(512, 2, &source).unwrap();
    assert_eq!(
        matrix.get_bytes(),
        ShiftAddMatrix::fit(512, 2, &source).unwrap().get_bytes()
    );
    let decoded = matrix.decode();
    let maximum = decoded
        .iter()
        .zip(&source)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(maximum <= 0.02, "balanced levels fit error {maximum}");
    assert_eq!(source, original);
}

/// Mixed signs and rare large weights must remain finite and improve squared
/// reconstruction error over discarding the source. This is a numerical
/// sanity gate, not a model-quality acceptance threshold.
#[test]
fn test_fit_outliers_and_mixed_groups() {
    let mut source: Vec<f32> = (0..1024)
        .map(|i| ((i * 17 % 113) as f32 - 56.0) * 0.003)
        .collect();
    source[0] = 64.0;
    source[511] = -32.0;
    source[700] = 16.0;
    let matrix = ShiftAddMatrix::fit(512, 2, &source).unwrap();
    let decoded = matrix.decode();
    let error: f64 = decoded
        .iter()
        .zip(&source)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    let zero_error: f64 = source.iter().map(|&x| f64::from(x).powi(2)).sum();
    assert!(decoded.iter().all(|x| x.is_finite()));
    assert!(
        error < zero_error,
        "fit {error} does not improve zero {zero_error}"
    );
}

/// Scale underflow yields an explicit zero representation. Source values too
/// large for a finite FP16 block scale are rejected instead of emitting NaNs.
#[test]
fn test_fit_scale_limits() {
    let tiny = ShiftAddMatrix::fit(256, 1, &[f32::MIN_POSITIVE; 256]).unwrap();
    assert!(tiny.decode().iter().all(|&value| value == 0.0));
    assert!(ShiftAddMatrix::fit(256, 1, &[f32::MAX; 256]).is_err());
    let large = ShiftAddMatrix::fit(256, 1, &[1_000_000.0; 256]).unwrap();
    assert!(large
        .decode()
        .iter()
        .all(|&value| (value - 1_000_000.0).abs() < 500.0));
}

/// Shape and value rejection prevents partial-block interpretation, overflow
/// and non-finite arithmetic in both CPU and future CUDA consumers.
#[test]
fn test_reject_invalid_inputs() {
    for (input, output) in [(0, 1), (256, 0), (255, 1), (256, usize::MAX)] {
        assert!(ShiftAddMatrix::fit(input, output, &[]).is_err());
        assert!(ShiftAddMatrix::from_bytes(input, output, vec![]).is_err());
    }
    assert!(ShiftAddMatrix::fit(256, 1, &[0.0; 255]).is_err());
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(ShiftAddMatrix::fit(256, 1, &[value; 256]).is_err());
    }
    for size in [0, 137, 139, 276] {
        assert!(ShiftAddMatrix::from_bytes(256, 1, vec![0; size]).is_err());
    }
    for scale in [
        f16::NAN,
        f16::INFINITY,
        f16::NEG_INFINITY,
        f16::from_f32(-1.0),
    ] {
        let mut payload = vec![0; 138];
        payload[136..].copy_from_slice(&scale.to_le_bytes());
        assert!(ShiftAddMatrix::from_bytes(256, 1, payload).is_err());
    }
    let matrix = make_manual_matrix();
    assert!(matrix.project_reference(&[]).is_err());
    assert!(matrix.project_reference(&[0.0; 511]).is_err());
    assert!(matrix.project_reference(&[f32::NAN; 512]).is_err());
}

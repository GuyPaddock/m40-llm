use crate::kernel::{has_avx2, Q8Matrix};
use half::f16;

fn matrix() -> Q8Matrix {
    let mut bytes = Vec::new();
    for block in 0..9 {
        bytes.extend_from_slice(
            &f16::from_f32((block as f32 - 4.0) / 8.0)
                .to_bits()
                .to_le_bytes(),
        );
        bytes.extend((0..32).map(|i| ((i * 31 + block * 17) % 256) as u8));
    }
    Q8Matrix::new(96, 3, bytes).unwrap()
}

fn check_projection(is_avx2: bool) {
    let matrix = matrix();
    for batch in [1, 2, 4, 8] {
        let input: Vec<_> = (0..batch * 96)
            .map(|i| ((i * 17 % 97) as f32 - 48.0) / 64.0)
            .collect();
        let reference = matrix.reference(&input, batch).unwrap();
        let mut guarded = vec![f32::from_bits(0x4a123456); batch * 3 + 2];
        let output = &mut guarded[1..1 + batch * 3];
        if is_avx2 {
            matrix.project_avx2(&input, batch, output).unwrap();
        } else {
            matrix.project_scalar(&input, batch, output).unwrap();
        }
        // All products and sums in this fixture are dyadic and exactly representable.
        assert_eq!(
            output.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
            reference
        );
        assert_eq!(guarded[0].to_bits(), 0x4a123456);
        assert_eq!(guarded.last().unwrap().to_bits(), 0x4a123456);
        let mut repeated = vec![0.0; batch * 3];
        if is_avx2 {
            matrix.project_avx2(&input, batch, &mut repeated).unwrap();
        } else {
            matrix.project_scalar(&input, batch, &mut repeated).unwrap();
        }
        assert_eq!(repeated, guarded[1..1 + batch * 3]);
    }
}

/// Multiple distinct rows, signed codes and signed/zero scales produce the
/// independent FP64 result without overwriting either output guard.
#[test]
fn test_scalar_projection_matches_reference() {
    check_projection(false);
}

/// Execute the actual AVX2 consumer and verify lane unpacking, batching and
/// output bounds. An unavailable ISA fails this explicit hardware test.
#[test]
#[ignore = "requires an AVX2 CPU"]
fn test_avx2_projection_matches_reference() {
    assert!(has_avx2(), "AVX2 hardware is required");
    check_projection(true);
}

/// Invalid dimensions, payloads, scales and job shapes fail before output mutation.
#[test]
fn test_invalid_data_preserves_output() {
    assert!(Q8Matrix::new(33, 1, vec![0; 34]).is_err());
    assert!(Q8Matrix::new(32, 1, vec![0; 33]).is_err());
    let mut invalid = vec![0; 34];
    invalid[..2].copy_from_slice(&f16::INFINITY.to_bits().to_le_bytes());
    assert!(Q8Matrix::new(32, 1, invalid).is_err());
    let matrix = matrix();
    let mut output = vec![123.0; 3];
    assert!(matrix.project_scalar(&[0.0; 95], 1, &mut output).is_err());
    assert!(matrix
        .project_scalar(&[f32::NAN; 96], 1, &mut output)
        .is_err());
    assert!(matrix.project_avx2(&[0.0; 96], 3, &mut output).is_err());
    assert_eq!(output, [123.0; 3]);
}

/// A one-block scalar example has an externally calculable sum and checks
/// the negative int8 endpoint independently of the FP64 decoder.
#[test]
fn test_signed_endpoint_sum() {
    let mut bytes = vec![0; 34];
    bytes[..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
    bytes[2] = 128;
    bytes[3] = 127;
    let matrix = Q8Matrix::new(32, 1, bytes).unwrap();
    let mut output = [0.0];
    matrix.project_scalar(&[2.0; 32], 1, &mut output).unwrap();
    assert_eq!(output, [-1.0]);
}

/// Numerical admission accepts errors within the fixed absolute/relative
/// bound and rejects non-finite or excessive errors before timing can begin.
#[test]
fn test_projection_error_admission() {
    assert!(crate::timing::validate_output(&[0.0009, 1_000_020.0], &[0.0, 1_000_000.0]).is_ok());
    assert!(crate::timing::validate_output(&[0.0011], &[0.0]).is_err());
    assert!(crate::timing::validate_output(&[1_000_022.0], &[1_000_000.0]).is_err());
    assert!(crate::timing::validate_output(&[f32::NAN], &[0.0]).is_err());
    assert!(crate::timing::validate_output(&[0.0], &[f64::INFINITY]).is_err());
}

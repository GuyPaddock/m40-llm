use super::{ActivationThreshold, SparseInputSelection};

/// Magnitude selection preserves original order/value bits and never hides
/// infinities or NaNs. Equal-to-threshold entries are omitted deterministically.
#[test]
fn test_selection_preserves_indices_values_and_nonfinite_inputs() {
    let nan = f32::from_bits(0x7fc0_1234);
    let input = [0.0, -3.0, 2.0, -2.0, f32::INFINITY, nan, 4.0];
    let selection =
        SparseInputSelection::select(&input, ActivationThreshold::new(2.0).unwrap()).unwrap();
    assert_eq!(selection.input_dim(), input.len());
    assert_eq!(selection.indices(), &[1, 4, 5, 6]);
    assert_eq!(
        selection
            .values()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        [input[1], input[4], input[5], input[6]]
            .map(f32::to_bits)
            .to_vec()
    );
    let reconstructed = selection.reconstruct();
    for (index, &value) in reconstructed.iter().enumerate() {
        let expected = if selection.indices().contains(&(index as u32)) {
            input[index].to_bits()
        } else {
            0.0f32.to_bits()
        };
        assert_eq!(value.to_bits(), expected);
    }
    assert_eq!(selection.sparsity(), 1.0 - 4.0 / 7.0);
}

/// A zero threshold preserves all nonzero finite inputs; complete omission
/// reconstructs a zero vector without an invalid empty selection.
#[test]
fn test_zero_and_all_discarded_selection_reconstruct_matched_inputs() {
    let input = [0.0, -0.0, -0.5, 1.0, 2.0];
    let zero = ActivationThreshold::new(-0.0).unwrap();
    assert_eq!(zero.magnitude().to_bits(), 0.0f32.to_bits());
    let selection = SparseInputSelection::select(&input, zero).unwrap();
    assert_eq!(selection.indices(), &[2, 3, 4]);
    assert_eq!(selection.reconstruct(), input);
    let empty =
        SparseInputSelection::select(&input, ActivationThreshold::new(2.0).unwrap()).unwrap();
    assert!(empty.indices().is_empty());
    assert!(empty.values().is_empty());
    assert_eq!(empty.sparsity(), 1.0);
    assert_eq!(empty.reconstruct(), vec![0.0; input.len()]);
}

/// Quantile fitting uses only supplied observations, leaves them unchanged,
/// and exposes the extra sparsity caused by tied magnitudes.
#[test]
fn test_calibration_measures_ties_instead_of_claiming_target_sparsity() {
    let observations = [0.0, 1.0, -1.0, 2.0, -3.0];
    let threshold = ActivationThreshold::calibrate(&observations, 0.4).unwrap();
    assert_eq!(threshold.magnitude(), 1.0);
    let selection = SparseInputSelection::select(&observations, threshold).unwrap();
    assert_eq!(selection.indices(), &[3, 4]);
    assert_eq!(selection.sparsity(), 0.6);
    assert_eq!(observations, [0.0, 1.0, -1.0, 2.0, -3.0]);
    assert_eq!(
        ActivationThreshold::calibrate(&observations, 0.0)
            .unwrap()
            .magnitude(),
        0.0
    );
}

/// Invalid thresholds, populations and requested fractions fail rather than
/// silently selecting a different approximation.
#[test]
fn test_invalid_thresholds_and_calibration_inputs_are_rejected() {
    for magnitude in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(ActivationThreshold::new(magnitude).is_err());
    }
    for target in [-0.01, 1.0, f64::NAN, f64::INFINITY] {
        assert!(ActivationThreshold::calibrate(&[1.0], target).is_err());
    }
    assert!(ActivationThreshold::calibrate(&[], 0.5).is_err());
    assert!(ActivationThreshold::calibrate(&[f32::NAN], 0.5).is_err());
    assert!(ActivationThreshold::calibrate(&[f32::INFINITY], 0.5).is_err());
    assert!(SparseInputSelection::select(&[], ActivationThreshold::new(0.0).unwrap()).is_err());
}

use anyhow::Result;
use m40_llm::infer::LayerMask;

/// Both the runtime default and an explicit empty mask preserve every original
/// layer ID, including single/two-layer models with no skippable interior.
#[test]
fn test_empty_layer_mask_preserves_all_layers() -> Result<()> {
    for count in [1, 2, 3, 36] {
        for mask in [LayerMask::default(), LayerMask::new(count, &[])?] {
            assert_eq!(
                mask.active_layers(count)?.collect::<Vec<_>>(),
                (0..count as usize).collect::<Vec<_>>()
            );
            assert_eq!(mask.active_layer_count(count)?, count as usize);
            assert!(!mask.has_skipped_layers());
        }
    }
    Ok(())
}

/// Skipping preserves original chronological IDs, canonicalizes input order,
/// and reports executed counts rather than renumbering the retained layers.
#[test]
fn test_interior_layer_masks_preserve_original_ids() -> Result<()> {
    let mask = LayerMask::new(7, &[4, 1, 3])?;
    assert_eq!(mask.skipped_layers(), &[1, 3, 4]);
    assert_eq!(mask.active_layers(7)?.collect::<Vec<_>>(), vec![0, 2, 5, 6]);
    assert_eq!(
        mask.active_layers(7)?.rev().collect::<Vec<_>>(),
        vec![6, 5, 2, 0]
    );
    assert_eq!(mask.active_layer_count(7)?, 4);
    assert!(mask.has_skipped_layers());
    assert_eq!(mask, LayerMask::new(7, &[1, 3, 4])?);
    Ok(())
}

/// Invalid, repeated, boundary, and cross-model IDs fail rather than silently
/// clipping a mask or allowing first/last layers to disappear.
#[test]
fn test_invalid_layer_masks_fail() -> Result<()> {
    for (count, ids) in [
        (0, vec![]),
        (3, vec![0]),
        (3, vec![2]),
        (3, vec![3]),
        (3, vec![1, 1]),
        (2, vec![1]),
        (36, vec![usize::MAX]),
    ] {
        assert!(
            LayerMask::new(count, &ids).is_err(),
            "count={count} ids={ids:?}"
        );
    }
    let mask = LayerMask::new(36, &[17, 18])?;
    assert!(mask.active_layers(35).is_err());
    assert!(mask.active_layer_count(37).is_err());
    assert!(LayerMask::default().validate(0).is_err());
    Ok(())
}

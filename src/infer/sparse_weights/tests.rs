use super::*;
use crate::infer::execution_weights::Q8MatrixShape;

fn fixture(k: usize, n: usize) -> (Q8SourceIdentity, Vec<u8>, Vec<u32>) {
    let shape = Q8MatrixShape::new(k, n).unwrap();
    let source =
        Q8SourceIdentity::from_column_blocks([7; 32], "synthetic".into(), 123, shape).unwrap();
    let mut dense = Vec::new();
    let mut masks = Vec::new();
    for index in 0..shape.block_count().unwrap() {
        let scale = [0.0, -0.25, 0.125][index % 3];
        dense.extend(f16::from_f32(scale).to_le_bytes());
        dense.extend(
            (0..32)
                .map(|position| (position as i8).wrapping_mul(13).wrapping_add(index as i8) as u8),
        );
        masks.push(0x5555_5555u32.rotate_left(index as u32));
    }
    (source, dense, masks)
}

/// Signed payload and scale bits must reconstruct the same pruned matrix;
/// invalid K-tail positions are zero even when the source padding is nonzero.
#[test]
fn test_sparse_q8_reconstructs_the_matched_zero_matrix() -> Result<()> {
    for (k, n) in [(1, 1), (17, 3), (32, 2), (35, 7), (67, 5), (2048, 17)] {
        let (source, dense, masks) = fixture(k, n);
        let host =
            HostSparseQ8Weight::pack(Q8SourceView::from_tensor_bytes(&source, &dense)?, &masks)?;
        assert_eq!(host.source(), &source);
        let restored = host.reconstruct_column_blocks();
        let decoded = host.decode_column_major();
        let mut retained = 0;
        for (index, (before, after)) in dense
            .chunks_exact(34)
            .zip(restored.chunks_exact(34))
            .enumerate()
        {
            assert_eq!(&before[..2], &after[..2]);
            let scale = f16::from_le_bytes([before[0], before[1]]).to_f32();
            let column = index / source.shape().blocks_per_column();
            let start = index % source.shape().blocks_per_column() * 32;
            for position in 0..32 {
                let is_retained = masks[index] & (1 << position) != 0 && start + position < k;
                let expected = if is_retained { before[2 + position] } else { 0 };
                assert_eq!(after[2 + position], expected);
                if start + position < k {
                    assert_eq!(
                        decoded[column * k + start + position],
                        scale * f32::from(expected as i8)
                    );
                    retained += usize::from(is_retained);
                }
            }
        }
        let storage = host.storage();
        assert_eq!(
            storage.allocated_bytes,
            storage.value_bytes
                + storage.mask_bytes
                + storage.scale_bytes
                + storage.alignment_bytes
        );
        assert_eq!(storage.allocated_bytes, storage.blocks * 24);
        assert_eq!(storage.ideal_bytes, storage.blocks * 22);
        assert_eq!(storage.dense_bytes, dense.len());
        assert_eq!(storage.logical_values, k * n);
        assert_eq!(storage.retained_logical_values, retained);
        if k % 32 == 0 {
            assert_eq!(retained * 2, k * n);
        }
    }
    Ok(())
}

/// Serialized mask population, payload length, finite scales, canonical
/// alignment and tail bytes must all be checked before device upload.
#[test]
fn test_sparse_q8_rejects_invalid_formats() -> Result<()> {
    let (source, dense, masks) = fixture(35, 2);
    let view = Q8SourceView::from_tensor_bytes(&source, &dense)?;
    assert!(HostSparseQ8Weight::pack(view, &masks[..1]).is_err());
    for mask in [0, 0x7fff, 0x1ffff, u32::MAX] {
        let mut invalid = masks.clone();
        invalid[0] = mask;
        assert!(HostSparseQ8Weight::pack(view, &invalid).is_err());
    }
    let host = HostSparseQ8Weight::pack(view, &masks)?;
    let packed = host.blocks();
    for length in [0, packed.len() - 1, packed.len() + 1] {
        let mut invalid = packed.to_vec();
        invalid.resize(length, 0);
        assert!(HostSparseQ8Weight::from_packed(source.clone(), invalid).is_err());
    }
    for (offset, value) in [(6, 1), (24 + 8 + 15, 1)] {
        let mut invalid = packed.to_vec();
        invalid[offset] = value;
        assert!(HostSparseQ8Weight::from_packed(source.clone(), invalid).is_err());
    }
    for scale in [f16::NAN, f16::INFINITY, f16::NEG_INFINITY] {
        let mut invalid = packed.to_vec();
        invalid[4..6].copy_from_slice(&scale.to_le_bytes());
        assert!(HostSparseQ8Weight::from_packed(source.clone(), invalid).is_err());
    }
    let restored = HostSparseQ8Weight::from_packed(source, packed.to_vec())?;
    assert_eq!(restored.decode_column_major(), host.decode_column_major());
    Ok(())
}

/// Masks containing both end bits and signed extreme values must preserve
/// retained-slot order, rather than accidentally reversing mask bit indices.
#[test]
fn test_sparse_q8_preserves_mask_order_and_signed_extremes() -> Result<()> {
    let shape = Q8MatrixShape::new(32, 1)?;
    let source = Q8SourceIdentity::from_column_blocks([0; 32], "extremes".into(), 0, shape)?;
    let mut dense = vec![0; 34];
    dense[..2].copy_from_slice(&f16::from_f32(-0.5).to_le_bytes());
    dense[2] = (-128i8) as u8;
    dense[33] = 127;
    let mask = 0x8000_7fff;
    let host =
        HostSparseQ8Weight::pack(Q8SourceView::from_tensor_bytes(&source, &dense)?, &[mask])?;
    let decoded = host.decode_column_major();
    assert_eq!(decoded[0], 64.0);
    assert_eq!(decoded[31], -63.5);
    assert!(decoded[1..31].iter().all(|&x| x == 0.0));
    assert_eq!(host.reconstruct_column_blocks(), dense);
    Ok(())
}

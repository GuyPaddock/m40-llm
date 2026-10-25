use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

pub const MASK_SEED: u64 = 0xC21_1632;

/// Uniform physical positions, independent of values or evaluation prompts.
/// This is a kernel workload generator, not a pruning calibration algorithm.
pub fn make_masks(blocks: usize) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(MASK_SEED);
    (0..blocks)
        .map(|_| {
            let mut positions = std::array::from_fn::<_, 32, _>(|i| i);
            positions.shuffle(&mut rng);
            positions[..16]
                .iter()
                .fold(0, |mask, position| mask | (1u32 << position))
        })
        .collect()
}

pub fn make_input(rows: usize, k: usize) -> Vec<f32> {
    (0..rows * k)
        .map(|i| ((i * 13 % 127) as f32 - 63.0) / 128.0)
        .collect()
}

pub fn hash_bytes(bytes: &[u8]) -> String {
    ring::digest::digest(&ring::digest::SHA256, bytes)
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Fixed seed must reproduce sixteen positions per block without restricting
/// support to particular source lanes or consulting model/evaluation values.
#[test]
fn test_masks_are_deterministic_and_cover_all_lanes() {
    let masks = make_masks(128);
    assert_eq!(masks, make_masks(128));
    assert!(masks.iter().all(|m| m.count_ones() == 16));
    assert_eq!(masks.iter().fold(0, |a, b| a | b), u32::MAX);
    assert_eq!(masks.iter().fold(u32::MAX, |a, b| a & b), 0);
    assert_ne!(masks[0], masks[1]);
}

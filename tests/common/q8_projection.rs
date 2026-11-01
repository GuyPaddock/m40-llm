use half::f16;

pub fn make_column_blocks(k: usize, n: usize) -> Vec<u8> {
    let blocks = k.div_ceil(32);
    let mut bytes = vec![0; n * blocks * 34];
    for column in 0..n {
        for block in 0..blocks {
            let base = (column * blocks + block) * 34;
            let scale = ((column * 7 + block * 13) % 31) as f32 * 0.0005 - 0.0075;
            bytes[base..base + 2].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            for lane in 0..32 {
                bytes[base + 2 + lane] = (column * 23 + block * 11 + lane * 17) as u8;
            }
        }
    }
    bytes
}

pub fn make_input(rows: usize, k: usize) -> Vec<f32> {
    (0..rows * k)
        .map(|i| ((i * 13 % 127) as f32 - 63.0) * 0.0078125)
        .collect()
}

pub fn project_cpu(input: &[f32], blocks: &[u8], row: usize, column: usize, k: usize) -> f64 {
    (0..k)
        .map(|index| {
            let base = (column * k.div_ceil(32) + index / 32) * 34;
            let scale =
                f16::from_bits(u16::from_le_bytes([blocks[base], blocks[base + 1]])).to_f64();
            f64::from(input[row * k + index])
                * scale
                * f64::from(blocks[base + 2 + index % 32] as i8)
        })
        .sum()
}

pub fn measure_error(actual: &[f32], expected: &[f32]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0_f64;
    let mut total = 0.0;
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && b.is_finite(),
            "non-finite projection result"
        );
        let difference = (f64::from(a) - f64::from(b)).abs();
        maximum = maximum.max(difference);
        total += difference;
    }
    (maximum, total / actual.len() as f64)
}

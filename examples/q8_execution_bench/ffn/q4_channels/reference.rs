use anyhow::{ensure, Result};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Selection {
    Contiguous,
    Scattered,
    Importance,
}

/// Compute all gate/up channels once in FP64, outside every GPU interval.
pub(super) fn hidden(input: &[f32], gate: &[f64], up: &[f64], k: usize) -> Vec<f64> {
    input
        .chunks_exact(k)
        .flat_map(|row| {
            gate.chunks_exact(k)
                .zip(up.chunks_exact(k))
                .map(move |(g, u)| {
                    let gate: f64 = row.iter().zip(g).map(|(&x, &w)| f64::from(x) * w).sum();
                    let up: f64 = row.iter().zip(u).map(|(&x, &w)| f64::from(x) * w).sum();
                    gate / (1.0 + (-gate).exp()) * up
                })
        })
        .collect()
}

/// Supplied indices are sorted for execution. Importance uses exact Q4 hidden
/// magnitude, not captured Q8 hidden values or a claimed inexpensive predictor.
pub(super) fn select(hidden: &[f64], h: usize, count: usize, mode: Selection) -> Result<Vec<u32>> {
    ensure!(
        h > 0 && count > 0 && count <= h && hidden.len().is_multiple_of(h),
        "invalid selection shape"
    );
    ensure!(
        hidden.iter().all(|x| x.is_finite()),
        "nonfinite selection scores"
    );
    let mut indices = Vec::with_capacity(hidden.len() / h * count);
    for (row, values) in hidden.chunks_exact(h).enumerate() {
        let mut selected: Vec<_> = (0..h as u32).collect();
        match mode {
            Selection::Contiguous => selected.rotate_left((row * 97) % h),
            Selection::Scattered => selected.shuffle(&mut StdRng::seed_from_u64(row as u64)),
            Selection::Importance => selected.sort_by(|&a, &b| {
                values[b as usize]
                    .abs()
                    .total_cmp(&values[a as usize].abs())
                    .then(a.cmp(&b))
            }),
        }
        selected.truncate(count);
        selected.sort_unstable();
        indices.extend(selected);
    }
    Ok(indices)
}

/// Dense masked reference, accumulating selected columns in FP64.
pub(super) fn down(
    hidden: &[f64],
    weights: &[f64],
    indices: &[u32],
    h: usize,
    count: usize,
) -> Vec<f64> {
    hidden
        .chunks_exact(h)
        .zip(indices.chunks_exact(count))
        .flat_map(|(row, selected)| {
            weights.chunks_exact(h).map(move |column| {
                selected
                    .iter()
                    .map(|&i| row[i as usize] * column[i as usize])
                    .sum()
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every selector supplies an exact budget, preserves ties, and handles full retention.
    #[test]
    fn test_selection_budgets_and_rows() -> Result<()> {
        let values = [1.0, -4.0, 4.0, 2.0, 7.0, 1.0, -2.0, 3.0];
        assert_eq!(select(&values, 4, 2, Selection::Importance)?, [1, 2, 0, 3]);
        assert_eq!(select(&values, 4, 2, Selection::Contiguous)?, [0, 1, 1, 2]);
        for mode in [
            Selection::Contiguous,
            Selection::Scattered,
            Selection::Importance,
        ] {
            assert_eq!(select(&values, 4, 4, mode)?, [0, 1, 2, 3, 0, 1, 2, 3]);
            let first = select(&values, 4, 3, mode)?;
            assert_eq!(first, select(&values, 4, 3, mode)?);
            assert!(first
                .chunks_exact(3)
                .all(|r| r.windows(2).all(|p| p[0] < p[1])));
        }
        assert!(select(&values, 4, 0, Selection::Importance).is_err());
        assert!(select(&[f64::NAN; 4], 4, 1, Selection::Importance).is_err());
        Ok(())
    }

    /// Selected down columns correspond to each row's own hidden values.
    #[test]
    fn test_masked_down() {
        let hidden = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0];
        let weights = [1.0, 10.0, 100.0, -1.0, -10.0, -100.0];
        assert_eq!(
            down(&hidden, &weights, &[0, 2, 0, 1], 3, 2),
            [502.0, -502.0, 117.0, -117.0]
        );
    }
}

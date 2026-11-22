use anyhow::{ensure, Result};
use m40_llm::evaluation::loss::{score_logits, TokenPrediction};
use serde::Serialize;
use std::collections::HashSet;

#[derive(Debug, Serialize)]
pub struct LogitComparison {
    pub reference: TokenPrediction,
    pub candidate: TokenPrediction,
    pub max_abs_logit_diff: f64,
    pub mean_abs_logit_diff: f64,
    pub kl_reference_candidate: f64,
    pub top10_overlap: usize,
}

pub fn compare_logits(
    reference: &[f32],
    candidate: &[f32],
    target: u32,
) -> Result<LogitComparison> {
    ensure!(
        reference.len() == candidate.len(),
        "different vocabulary sizes"
    );
    let a = score_logits(reference, target)?;
    let b = score_logits(candidate, target)?;
    let normalized = |values: &[f32]| {
        let maximum = values
            .iter()
            .map(|&x| f64::from(x))
            .fold(f64::NEG_INFINITY, f64::max);
        let log_sum = values
            .iter()
            .map(|&x| (f64::from(x) - maximum).exp())
            .sum::<f64>()
            .ln();
        values
            .iter()
            .map(|&x| (f64::from(x) - maximum) - log_sum)
            .collect::<Vec<_>>()
    };
    let log_a = normalized(reference);
    let log_b = normalized(candidate);
    let kl = log_a
        .iter()
        .zip(&log_b)
        .map(|(&a, &b)| a.exp() * (a - b))
        .sum::<f64>();
    ensure!(kl.is_finite() && kl >= -1e-10, "invalid KL divergence");
    let differences: Vec<_> = reference
        .iter()
        .zip(candidate)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
        .collect();
    Ok(LogitComparison {
        max_abs_logit_diff: differences.iter().copied().fold(0.0, f64::max),
        mean_abs_logit_diff: differences.iter().sum::<f64>() / differences.len() as f64,
        kl_reference_candidate: kl.max(0.0),
        top10_overlap: a
            .top_indices
            .iter()
            .filter(|id| b.top_indices.contains(id))
            .count(),
        reference: a,
        candidate: b,
    })
}

pub fn select_observation_steps(count: usize) -> Result<Vec<usize>> {
    ensure!(count > 0, "generation has no emitted tokens");
    let mut steps = vec![0, 1, 8, 32, 64, 127, count - 1];
    steps.retain(|&step| step < count);
    steps.sort_unstable();
    steps.dedup();
    Ok(steps)
}

#[derive(Debug, Serialize)]
pub struct Repetition {
    pub suffix_pattern_width: usize,
    pub suffix_pattern_copies: usize,
    pub repeated_fourgram_fraction: f64,
    pub analyzed_tokens: usize,
}

/// This diagnostic reports repetitions without declaring them a quality failure.
/// The caller excludes a terminal stop token so it cannot hide a repeated suffix.
pub fn measure_repetition(tokens: &[u32]) -> Repetition {
    let mut width = 0;
    let mut copies = 0;
    for candidate_width in 1..=16.min(tokens.len() / 2) {
        let pattern = &tokens[tokens.len() - candidate_width..];
        let count = tokens
            .rchunks_exact(candidate_width)
            .take_while(|chunk| *chunk == pattern)
            .count();
        if count >= 2 && candidate_width * count > width * copies {
            width = candidate_width;
            copies = count;
        }
    }
    let mut seen = HashSet::new();
    let windows = tokens.len().saturating_sub(3);
    let repeated = tokens
        .windows(4)
        .filter(|window| !seen.insert(window.to_vec()))
        .count();
    Repetition {
        suffix_pattern_width: width,
        suffix_pattern_copies: copies,
        repeated_fourgram_fraction: if windows == 0 {
            0.0
        } else {
            repeated as f64 / windows as f64
        },
        analyzed_tokens: tokens.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// KL is zero for an additive logit shift and matches the independent
    /// Bernoulli calculation for different two-token probabilities.
    #[test]
    fn test_logit_comparison_matches_probability_contract() -> Result<()> {
        let shifted = compare_logits(&[0.0, 1.0], &[100.0, 101.0], 1)?;
        assert!(shifted.kl_reference_candidate.abs() < 1e-12);
        assert_eq!(shifted.max_abs_logit_diff, 100.0);
        let changed = compare_logits(&[0.0, 0.0], &[0.0, 3.0_f32.ln()], 0)?;
        let expected = 0.5_f64 * (0.5_f64 / 0.25).ln() + 0.5_f64 * (0.5_f64 / 0.75).ln();
        assert!((changed.kl_reference_candidate - expected).abs() < 1e-7);
        assert_eq!(changed.candidate.target_rank, 2);
        assert!(compare_logits(&[f32::NAN], &[0.0], 0).is_err());
        assert!(compare_logits(&[0.0], &[0.0, 1.0], 0).is_err());
        assert!(compare_logits(&[0.0], &[0.0], 1).is_err());
        Ok(())
    }

    /// Sparse observations include the final emitted target exactly once and
    /// never request an ungenerated or future token.
    #[test]
    fn test_observation_steps_keep_natural_stops() -> Result<()> {
        assert_eq!(select_observation_steps(1)?, vec![0]);
        assert_eq!(select_observation_steps(9)?, vec![0, 1, 8]);
        assert_eq!(
            select_observation_steps(160)?,
            vec![0, 1, 8, 32, 64, 127, 159]
        );
        assert!(select_observation_steps(0).is_err());
        Ok(())
    }

    /// Distinct tokens have no repeated ngrams; repeated suffixes retain their
    /// whole span with the smallest winning pattern when spans tie.
    #[test]
    fn test_repetition_reports_suffix_and_reused_fourgrams() {
        let unique = measure_repetition(&[1, 2, 3, 4, 5]);
        assert_eq!(unique.suffix_pattern_copies, 0);
        assert_eq!(unique.repeated_fourgram_fraction, 0.0);
        let repeated = measure_repetition(&[9, 1, 2, 1, 2, 1, 2]);
        assert_eq!(
            (
                repeated.suffix_pattern_width,
                repeated.suffix_pattern_copies
            ),
            (2, 3)
        );
        assert_eq!(repeated.repeated_fourgram_fraction, 0.25);
        assert_eq!(measure_repetition(&[]).repeated_fourgram_fraction, 0.0);
    }
}

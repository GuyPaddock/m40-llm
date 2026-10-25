use anyhow::Result;
use m40_llm::evaluation::loss::{
    compare_teacher_forced, score_logits, score_teacher_forced, LossMetrics,
};

/// Stable log-sum-exp must preserve uniform loss even at huge equal logits,
/// agree with analytical probabilities, and reject invalid inputs explicitly.
#[test]
fn test_stable_token_loss_and_invalid_logits() -> Result<()> {
    let uniform = score_logits(&[f32::MAX; 4], 2)?;
    assert!((uniform.nll - 4.0_f64.ln()).abs() < 1e-12);
    assert_eq!(uniform.target_rank, 3);
    let analytic = score_logits(&[0.0, 3.0_f32.ln()], 1)?;
    assert!((analytic.nll - (-0.75_f64.ln())).abs() < 1e-7);
    let shifted = score_logits(&[1024.0, 1024.0 + 3.0_f32.ln()], 1)?;
    assert!((analytic.nll - shifted.nll).abs() < 2e-5);
    for logits in [
        &[][..],
        &[f32::NAN][..],
        &[f32::INFINITY][..],
        &[f32::NEG_INFINITY][..],
    ] {
        assert!(score_logits(logits, 0).is_err());
    }
    assert!(score_logits(&[0.0], 1).is_err());
    Ok(())
}

/// Loss aggregation is token-weighted; overflow is labeled rather than emitted
/// as an unmarked null, infinity, or silently dropped passage.
#[test]
fn test_weighted_perplexity_and_overflow_reporting() -> Result<()> {
    let first = LossMetrics::from_sum(2.0_f64.ln(), 1)?;
    let second = LossMetrics::from_sum(3.0 * 4.0_f64.ln(), 3)?;
    let aggregate = LossMetrics::combine(&[first, second])?;
    assert_eq!(aggregate.scored_tokens, 4);
    assert!(
        (aggregate.perplexity.unwrap() - 2.0_f64.powf(0.25) * 4.0_f64.powf(0.75)).abs() < 1e-12
    );
    let overflow = LossMetrics::from_sum(1000.0, 1)?;
    assert!(overflow.perplexity.is_none() && overflow.perplexity_overflow);
    assert!(LossMetrics::from_sum(0.0, 0).is_err());
    assert!(LossMetrics::combine(&[]).is_err());
    Ok(())
}

/// Predictors see strict reference prefixes, even when their top prediction
/// disagrees. Self-comparison must produce finite loss and zero full-vocab drift.
#[test]
fn test_teacher_forcing_uses_observed_tokens_not_samples() -> Result<()> {
    let ids = [0, 2, 3, 1];
    let mut reference_calls = Vec::new();
    let mut candidate_calls = Vec::new();
    let report = compare_teacher_forced(
        &ids,
        |prefix| {
            reference_calls.push(prefix.to_vec());
            Ok(vec![9.0, 0.0, 0.0, 0.0])
        },
        |prefix| {
            candidate_calls.push(prefix.to_vec());
            Ok(vec![9.0, 0.0, 0.0, 0.0])
        },
    )?;
    assert_eq!(reference_calls, vec![vec![0], vec![0, 2], vec![0, 2, 3]]);
    assert_eq!(candidate_calls, reference_calls);
    assert_eq!(report.reference, report.candidate);
    assert_eq!(report.max_abs_logit_diff, 0.0);
    assert_eq!(report.reference.scored_tokens, 3);
    assert_eq!(
        report
            .tokens
            .iter()
            .map(|token| token.target_id)
            .collect::<Vec<_>>(),
        vec![2, 3, 1]
    );
    Ok(())
}

/// Full-vocabulary drift includes low-ranked entries, and signed-zero ties use
/// vocabulary order rather than floating-point sign to choose the top ten.
#[test]
fn test_full_logit_drift_and_top10_ties() -> Result<()> {
    let reference = vec![0.0; 12];
    let mut candidate = reference.clone();
    candidate[11] = -2.0;
    let report = compare_teacher_forced(
        &[1, 2],
        |_| Ok(reference.clone()),
        |_| Ok(candidate.clone()),
    )?;
    assert_eq!(report.max_abs_logit_diff, 2.0);
    assert!((report.mean_abs_logit_diff - 2.0 / 12.0).abs() < 1e-12);
    assert_eq!(report.tokens[0].top10_overlap, 10);
    let signed = score_logits(&[-0.0, 0.0, -0.0], 2)?;
    assert_eq!(signed.top_indices, vec![0, 1, 2]);
    assert_eq!(signed.target_rank, 3);
    assert!(compare_teacher_forced(&[0], |_| unreachable!(), |_| unreachable!()).is_err());
    assert!(compare_teacher_forced(&[0, 0], |_| Ok(vec![0.0]), |_| Ok(vec![0.0, 0.0])).is_err());
    Ok(())
}

/// Single-model loss consumes the same observed prefixes/NLL as the paired
/// evaluator. Invalid shapes and vocabulary changes fail rather than truncate.
#[test]
fn test_single_model_loss_matches_paired_reference() -> Result<()> {
    let ids = [0, 2, 1, 3];
    let mut calls = Vec::new();
    let single = score_teacher_forced(&ids, |prefix| {
        calls.push(prefix.to_vec());
        Ok(vec![9.0, 0.0, 0.0, 0.0])
    })?;
    let paired = compare_teacher_forced(
        &ids,
        |_| Ok(vec![9.0, 0.0, 0.0, 0.0]),
        |_| Ok(vec![9.0, 0.0, 0.0, 0.0]),
    )?;
    assert_eq!(single, paired.reference);
    assert_eq!(calls, vec![vec![0], vec![0, 2], vec![0, 2, 1]]);
    assert!(score_teacher_forced(&[0], |_| unreachable!()).is_err());
    assert!(score_teacher_forced(&[0, 2], |_| Ok(vec![0.0])).is_err());
    assert!(score_teacher_forced(&[0, 1, 2], |prefix| Ok(vec![0.0; prefix.len() + 1])).is_err());
    Ok(())
}

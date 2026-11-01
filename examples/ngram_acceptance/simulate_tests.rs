use super::*;

const PROMPT: &[u32] = &[1, 2, 3, 7, 8, 9, 1, 2, 3];

/// Every rejection position must advance by the accepted prefix plus exactly
/// one correction, then draft again from committed history only.
#[test]
fn test_each_rejection_position_and_subsequent_history() -> Result<()> {
    let offered = propose(PROMPT, 4, &[])?.unwrap().tokens.to_vec();
    for accepted in 0..offered.len() {
        let mut future = offered[..accepted].to_vec();
        future.extend_from_slice(&[80, 81, 82, 83, 84]);
        let result = study(PROMPT, &future, &[], 4)?;
        assert_eq!(result.rounds[0].accepted, accepted);
        assert_eq!(result.rounds[0].committed, future[..accepted + 1]);
        for round in &result.rounds {
            let mut committed = PROMPT.to_vec();
            committed.extend_from_slice(&future[..round.generated_start]);
            let expected = propose(&committed, 4.min(future.len() - round.generated_start), &[])?;
            assert_eq!(
                round.offered,
                expected.map_or_else(Vec::new, |v| v.tokens.to_vec())
            );
        }
        assert_eq!(result.replayed_tokens, future);
    }
    Ok(())
}

/// Changing an uncommitted future cannot create a prompt lookup opportunity.
#[test]
fn test_future_is_not_available_to_the_drafter() -> Result<()> {
    for future in [&[1, 2, 3, 7, 1, 2, 3, 7][..], &[9, 8, 7, 6, 5][..]] {
        let result = study(&[100, 101, 102], future, &[], 4)?;
        assert!(result.rounds[0].offered.is_empty());
        assert_eq!(result.replayed_tokens, future);
    }
    Ok(())
}

/// A rejected suffix must not enter committed history or contaminate the next
/// lookup. The accepted prefix plus correction advances exactly two tokens.
#[test]
fn test_partial_rejection_commits_only_prefix_and_correction() -> Result<()> {
    let result = study(PROMPT, &[7, 80, 90], &[], 4)?;
    assert_eq!(result.rounds[0].accepted, 1);
    assert_eq!(result.rounds[0].committed, [7, 80]);
    assert_eq!(result.rounds[0].completion, Completion::Correction);
    assert_eq!(result.rounds[1].generated_start, 2);
    assert!(result.rounds[1].offered.is_empty());
    assert_eq!(result.replayed_tokens, [7, 80, 90]);
    assert_eq!(result.visited.partial_rejections, 1);
    Ok(())
}

/// Full acceptance emits one bonus token when available, but no invented
/// bonus at the output budget. Positions skipped by a round stay separate.
#[test]
fn test_full_acceptance_bonus_and_budget() -> Result<()> {
    let full = study(PROMPT, &[7, 8, 9, 42], &[], 2)?;
    assert_eq!(full.rounds[0].completion, Completion::Bonus);
    assert_eq!(full.rounds[0].committed, [7, 8, 9]);
    assert_eq!(full.rounds[1].generated_start, 3);
    assert_eq!(full.positions.steps, 4);
    assert_eq!(full.visited.steps, 2);
    assert_eq!(full.replayed_tokens, [7, 8, 9, 42]);
    let bounded = study(PROMPT, &[7, 8], &[], 16)?;
    assert_eq!(bounded.rounds.len(), 1);
    assert_eq!(bounded.rounds[0].completion, Completion::Budget);
    assert_eq!(bounded.rounds[0].committed, [7, 8]);
    Ok(())
}

/// Zero acceptance advances by the correction, while accepted or corrective
/// stop tokens terminate without a bonus. Post-stop references are invalid.
#[test]
fn test_zero_acceptance_and_stop_boundaries() -> Result<()> {
    let zero = study(PROMPT, &[4, 5, 6], &[], 4)?;
    assert_eq!(zero.rounds[0].accepted, 0);
    assert_eq!(zero.rounds[0].committed, [4]);
    assert_eq!(zero.visited.zero_rejections, 1);
    let prompt = [1, 2, 3, 7, 99, 8, 1, 2, 3];
    for future in [&[7, 99][..], &[99][..]] {
        let stopped = study(&prompt, future, &[99], 16)?;
        assert_eq!(stopped.rounds[0].completion, Completion::Stop);
        assert_eq!(stopped.replayed_tokens, future);
    }
    assert!(study(&prompt, &[7, 99, 8], &[99], 4).is_err());
    assert!(study(&prompt, &[], &[99], 4).is_err());
    Ok(())
}

/// No-match replay preserves the calibrated HTTP total for every verifier
/// assumption, while a more expensive verifier cannot improve an eligible run.
#[test]
fn test_cost_model_preserves_baseline_and_prices_verification() -> Result<()> {
    let baseline = Baseline {
        http_elapsed_ms: 100.0,
        prefill_elapsed_ms: 35.0,
        decode_elapsed_ms: 60.0,
        generated_tokens: 4,
    };
    let ordinary = study(&[100, 101, 102], &[1, 2, 3, 4], &[], 4)?;
    for factor in [1.0, 1.5, 2.0, 4.0] {
        assert_eq!(
            estimate_cost(&baseline, &ordinary.visited, factor)?.estimated_http_elapsed_ms,
            100.0
        );
    }
    let speculative = study(PROMPT, &[7, 8, 9, 42], &[], 2)?;
    let cheap = estimate_cost(&baseline, &speculative.visited, 1.0)?;
    let expensive = estimate_cost(&baseline, &speculative.visited, 4.0)?;
    assert!(cheap.estimated_http_elapsed_ms < 100.0);
    assert!(expensive.estimated_http_elapsed_ms > cheap.estimated_http_elapsed_ms);
    let invalid = Baseline {
        decode_elapsed_ms: f64::NAN,
        ..baseline
    };
    assert!(estimate_cost(&invalid, &ordinary.visited, 1.0).is_err());
    Ok(())
}

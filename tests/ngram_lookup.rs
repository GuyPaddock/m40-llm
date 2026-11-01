use anyhow::Result;
use m40_llm::decode::ngram::propose;

/// Missing/too-short history and an exhausted budget cannot fabricate a draft.
#[test]
fn test_no_match_and_bounded_budget() -> Result<()> {
    for history in [&[][..], &[1, 2, 3][..], &[1, 2, 3, 4, 5][..]] {
        assert!(propose(history, 16, &[])?.is_none());
    }
    assert!(propose(&[9; 8], 0, &[])?.is_none());
    assert!(propose(&[9; 8], 17, &[]).is_err());
    Ok(())
}

/// A longer match wins even if a shorter match occurs later in history.
#[test]
fn test_longest_suffix_has_priority_over_recency() -> Result<()> {
    let history = [
        1, 2, 3, 4, 5, 6, 7, 8, 90, 91, 6, 7, 8, 42, 1, 2, 3, 4, 5, 6, 7, 8,
    ];
    let draft = propose(&history, 2, &[])?.unwrap();
    assert_eq!(draft.suffix_len, 8);
    assert_eq!(draft.match_start, 0);
    assert_eq!(draft.tokens, &[90, 91]);
    Ok(())
}

/// Equal-length matches use the latest usable occurrence, not the terminal
/// suffix itself, and can return a shorter-than-requested known continuation.
#[test]
fn test_recency_and_known_history_limit() -> Result<()> {
    let history = [1, 2, 3, 10, 1, 2, 3, 20, 21, 1, 2, 3];
    let draft = propose(&history, 4, &[])?.unwrap();
    assert_eq!(draft.suffix_len, 3);
    assert_eq!(draft.match_start, 4);
    assert_eq!(draft.tokens, &[20, 21, 1, 2]);
    let repeated = [9; 4];
    assert_eq!(propose(&repeated, 16, &[])?.unwrap().tokens, &[9]);
    Ok(())
}

/// Drafts may include a stop token but never tokens beyond it, even when those
/// later tokens exist in prompt history. A tighter token budget still wins.
#[test]
fn test_stop_and_budget_truncate_known_continuation() -> Result<()> {
    let history = [1, 2, 3, 7, 99, 8, 1, 2, 3];
    assert_eq!(propose(&history, 16, &[99])?.unwrap().tokens, &[7, 99]);
    assert_eq!(propose(&history, 1, &[99])?.unwrap().tokens, &[7]);
    Ok(())
}

/// Enumerated short histories must select the maximal (width, start) pair
/// among valid matches and return only the corresponding bounded known slice.
#[test]
fn test_exhaustive_short_histories_match_independent_selection() -> Result<()> {
    for len in 0..=8_u32 {
        for mut encoded in 0..3_usize.pow(len) {
            let mut history = vec![0; len as usize];
            for token in &mut history {
                *token = (encoded % 3) as u32;
                encoded /= 3;
            }
            let mut candidates = Vec::new();
            for width in 3..=8.min(history.len()) {
                for start in 0..history.len().saturating_sub(width) {
                    if history[start..start + width] == history[history.len() - width..] {
                        candidates.push((width, start));
                    }
                }
            }
            let expected = candidates.into_iter().max();
            for limit in [2, 4, 8, 16] {
                let actual = propose(&history, limit, &[2])?;
                assert_eq!(
                    actual.as_ref().map(|v| (v.suffix_len, v.match_start)),
                    expected
                );
                if let Some(draft) = actual {
                    let start = draft.match_start + draft.suffix_len;
                    let end = (start + limit).min(history.len());
                    let source = &history[start..end];
                    let count = source.iter().take_while(|&&token| token != 2).count();
                    assert_eq!(draft.tokens, &source[..source.len().min(count + 1)]);
                }
            }
        }
    }
    Ok(())
}

//! Bounded prompt lookup. This does not invoke a model or enable speculation.

use anyhow::{ensure, Result};

pub const MAX_DRAFT_TOKENS: usize = 16;

/// A continuation borrowed exclusively from already committed history.
#[derive(Debug, PartialEq, Eq)]
pub struct NgramDraft<'h> {
    pub suffix_len: usize,
    pub match_start: usize,
    pub tokens: &'h [u32],
}

/// Prefer the longest suffix of length 3-8, then the most recent occurrence
/// with at least one known continuation token. Never extrapolate a repetition.
/// The returned draft includes at most the first stop token and the given budget.
pub fn propose<'h>(
    history: &'h [u32],
    max_tokens: usize,
    stop_ids: &[u32],
) -> Result<Option<NgramDraft<'h>>> {
    ensure!(
        max_tokens <= MAX_DRAFT_TOKENS,
        "ngram drafts are limited to 16 tokens"
    );
    if max_tokens == 0 {
        return Ok(None);
    }
    for suffix_len in (3..=8.min(history.len().saturating_sub(1))).rev() {
        let suffix = &history[history.len() - suffix_len..];
        for match_start in (0..history.len() - suffix_len).rev() {
            if history[match_start..match_start + suffix_len] != *suffix {
                continue;
            }
            let start = match_start + suffix_len;
            let end = start + max_tokens.min(history.len() - start);
            let tokens = &history[start..end];
            let count = tokens
                .iter()
                .position(|token| stop_ids.contains(token))
                .map_or(tokens.len(), |index| index + 1);
            return Ok(Some(NgramDraft {
                suffix_len,
                match_start,
                tokens: &tokens[..count],
            }));
        }
    }
    Ok(None)
}

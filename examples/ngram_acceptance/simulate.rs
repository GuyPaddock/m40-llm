use anyhow::{ensure, Result};
use m40_llm::decode::ngram::{propose, NgramDraft, MAX_DRAFT_TOKENS};
use serde::Serialize;
use serde_json::{json, Value};
use std::time::Instant;

#[derive(Debug, Default, Serialize)]
pub struct Statistics {
    pub steps: usize,
    pub eligible_steps: usize,
    pub no_match_steps: usize,
    pub offered_tokens: usize,
    pub accepted_tokens: usize,
    pub zero_rejections: usize,
    pub partial_rejections: usize,
    pub full_acceptances: usize,
    pub accepted_prefix_histogram: [usize; 17],
    pub offered_length_histogram: [usize; 17],
    pub drafting_elapsed_ns: u128,
}

impl Statistics {
    fn observe(&mut self, draft: Option<&NgramDraft<'_>>, accepted: usize) {
        self.steps += 1;
        let Some(draft) = draft else {
            self.no_match_steps += 1;
            return;
        };
        let offered = draft.tokens.len();
        self.eligible_steps += 1;
        self.offered_tokens += offered;
        self.accepted_tokens += accepted;
        self.accepted_prefix_histogram[accepted] += 1;
        self.offered_length_histogram[offered] += 1;
        if accepted == 0 {
            self.zero_rejections += 1;
        } else if accepted < offered {
            self.partial_rejections += 1;
        } else {
            self.full_acceptances += 1;
        }
    }

    pub fn report(&self) -> Value {
        let eligible = self.eligible_steps.max(1) as f64;
        json!({
            "counts":self,
            "eligibility_rate":self.eligible_steps as f64/self.steps.max(1) as f64,
            "mean_offered_draft_length":self.offered_tokens as f64/eligible,
            "mean_accepted_draft_length":self.accepted_tokens as f64/eligible,
            "zero_rejection_rate":self.zero_rejections as f64/eligible,
            "partial_rejection_rate":self.partial_rejections as f64/eligible,
            "full_acceptance_rate":self.full_acceptances as f64/eligible,
            "accepted_fraction_of_offered":self.accepted_tokens as f64/self.offered_tokens.max(1) as f64,
        })
    }
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Completion {
    Ordinary,
    Correction,
    Bonus,
    Budget,
    Stop,
}

#[derive(Debug, Serialize)]
pub struct Round {
    pub generated_start: usize,
    pub match_start: Option<usize>,
    pub suffix_len: Option<usize>,
    pub offered: Vec<u32>,
    pub accepted: usize,
    pub committed: Vec<u32>,
    pub completion: Completion,
}

pub struct Study {
    pub positions: Statistics,
    pub visited: Statistics,
    pub rounds: Vec<Round>,
    pub replayed_tokens: Vec<u32>,
}

fn count_accepted(draft: Option<&NgramDraft<'_>>, reference: &[u32]) -> usize {
    draft.map_or(0, |draft| {
        draft
            .tokens
            .iter()
            .zip(reference)
            .take_while(|(a, b)| a == b)
            .count()
    })
}

/// The reference future is visible only to scoring. Every proposal receives a
/// history rebuilt from the prompt and actual committed tokens, never a future slice.
pub fn study(
    prompt: &[u32],
    reference: &[u32],
    stop_ids: &[u32],
    draft_len: usize,
) -> Result<Study> {
    ensure!(!reference.is_empty(), "empty continuation");
    ensure!(
        (1..=MAX_DRAFT_TOKENS).contains(&draft_len),
        "invalid draft length"
    );
    ensure!(
        !reference[..reference.len() - 1]
            .iter()
            .any(|id| stop_ids.contains(id)),
        "reference contains tokens after a stop token"
    );
    let mut positions = Statistics::default();
    let mut history = prompt.to_vec();
    for position in 0..reference.len() {
        let start = Instant::now();
        let draft = propose(
            &history,
            draft_len.min(reference.len() - position),
            stop_ids,
        )?;
        positions.drafting_elapsed_ns += start.elapsed().as_nanos();
        positions.observe(
            draft.as_ref(),
            count_accepted(draft.as_ref(), &reference[position..]),
        );
        history.push(reference[position]);
    }
    let mut visited = Statistics::default();
    let mut rounds = Vec::new();
    let mut replayed_tokens = Vec::new();
    let mut history = prompt.to_vec();
    while replayed_tokens.len() < reference.len() {
        let position = replayed_tokens.len();
        let start = Instant::now();
        let draft = propose(
            &history,
            draft_len.min(reference.len() - position),
            stop_ids,
        )?;
        visited.drafting_elapsed_ns += start.elapsed().as_nanos();
        let accepted = count_accepted(draft.as_ref(), &reference[position..]);
        visited.observe(draft.as_ref(), accepted);
        let count = (accepted + 1).min(reference.len() - position);
        let committed = reference[position..position + count].to_vec();
        let completion = if stop_ids.contains(committed.last().unwrap()) {
            Completion::Stop
        } else if draft.is_none() {
            Completion::Ordinary
        } else if accepted < draft.as_ref().unwrap().tokens.len() {
            Completion::Correction
        } else if count == accepted {
            Completion::Budget
        } else {
            Completion::Bonus
        };
        rounds.push(Round {
            generated_start: position,
            match_start: draft.as_ref().map(|v| v.match_start),
            suffix_len: draft.as_ref().map(|v| v.suffix_len),
            offered: draft.map_or_else(Vec::new, |v| v.tokens.to_vec()),
            accepted,
            committed: committed.clone(),
            completion,
        });
        history.extend_from_slice(&committed);
        replayed_tokens.extend(committed);
    }
    Ok(Study {
        positions,
        visited,
        rounds,
        replayed_tokens,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct Baseline {
    pub http_elapsed_ms: f64,
    pub prefill_elapsed_ms: f64,
    pub decode_elapsed_ms: f64,
    pub generated_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct CostEstimate {
    pub verifier_cost_in_target_steps: f64,
    pub estimated_http_elapsed_ms: f64,
    pub estimated_throughput_gain: f64,
    pub target_step_ms: f64,
    pub fixed_http_residual_ms: f64,
}

/// Optimistic screening model, not measured verification. Preserve HTTP minus
/// decode as a fixed residual, charge one round per ordinary/verification call,
/// and credit the first prediction already included in prefill. Zero draft,
/// commit, rollback, transfer, and correction-processing overhead is assumed.
pub fn estimate_cost(
    baseline: &Baseline,
    stats: &Statistics,
    verifier_cost: f64,
) -> Result<CostEstimate> {
    ensure!(
        baseline.generated_tokens > 1,
        "cost model needs at least two generated tokens"
    );
    ensure!(
        baseline.http_elapsed_ms.is_finite()
            && baseline.decode_elapsed_ms.is_finite()
            && baseline.prefill_elapsed_ms.is_finite()
            && baseline.prefill_elapsed_ms >= 0.0
            && baseline.decode_elapsed_ms > 0.0
            && baseline.http_elapsed_ms > baseline.decode_elapsed_ms,
        "invalid measured timing baseline"
    );
    ensure!(
        verifier_cost.is_finite() && verifier_cost >= 1.0,
        "invalid verifier-cost assumption"
    );
    let target_step_ms = baseline.decode_elapsed_ms / (baseline.generated_tokens - 1) as f64;
    let fixed_http_residual_ms = baseline.http_elapsed_ms - baseline.decode_elapsed_ms;
    let weighted_rounds = stats.no_match_steps as f64 + stats.eligible_steps as f64 * verifier_cost;
    let estimated_http_elapsed_ms =
        fixed_http_residual_ms + (weighted_rounds - 1.0).max(0.0) * target_step_ms;
    Ok(CostEstimate {
        verifier_cost_in_target_steps: verifier_cost,
        estimated_http_elapsed_ms,
        estimated_throughput_gain: baseline.http_elapsed_ms / estimated_http_elapsed_ms - 1.0,
        target_step_ms,
        fixed_http_residual_ms,
    })
}

#[cfg(test)]
#[path = "simulate_tests.rs"]
mod tests;

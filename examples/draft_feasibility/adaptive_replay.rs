use super::{
    decision::{Controller, Decision, DecisionCosts},
    records::Trace,
    replay::{self, Replay, Round},
    source::{DraftSource, Proposal, SourceKind},
};
use anyhow::{ensure, Result};
use serde::Serialize;
use std::{hint::black_box, time::Instant};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    AlwaysAttempt,
    Adaptive,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DecisionRecord {
    pub input_row: usize,
    pub absolute_position: usize,
    pub remaining_budget: usize,
    pub proposal: Option<Proposal>,
    pub decision: Decision,
    pub observed_accepted_guesses: Option<usize>,
    pub next_input_row: usize,
}

#[derive(Serialize)]
pub struct AdaptiveReplay {
    pub replay: Replay,
    pub decisions: Vec<DecisionRecord>,
    pub teardown_ns: u128,
}

/// Only the scorer reads future IDs, after the policy has made its decision.
/// Timings cover source/controller construction, decisions, committed updates
/// and teardown, excluding future scoring and diagnostic-record serialization.
pub fn run(trace: &Trace, policy: Policy, table: &[DecisionCosts]) -> Result<AdaptiveReplay> {
    trace.validate()?;
    ensure!(
        table.len() == 505,
        "incomplete measured-position cost table"
    );
    let started = Instant::now();
    let mut source = DraftSource::new(
        SourceKind::PromptCopy,
        &trace.prompt_token_ids,
        &trace.stop_ids,
    );
    let mut controller = Controller::default();
    let construction_ns = started.elapsed().as_nanos();
    let mut result = AdaptiveReplay {
        replay: Replay {
            construction_ns,
            cpu_elapsed_ns: construction_ns,
            logical_payload_bytes: 0,
            rounds: Vec::new(),
        },
        decisions: Vec::new(),
        teardown_ns: 0,
    };
    let mut row = 0;
    while row + 1 < trace.generated_token_ids.len() {
        let started = Instant::now();
        let position = trace.prompt_token_ids.len() + row;
        let remaining = trace.max_tokens - 1 - row;
        let seed = trace.generated_token_ids[row];
        let proposal = if remaining >= 4 {
            source.propose(seed)
        } else {
            None
        };
        let costs = position.checked_sub(65).and_then(|index| table.get(index));
        let decision = match policy {
            Policy::Adaptive => controller.decide(position, remaining, proposal.is_some(), costs),
            Policy::AlwaysAttempt => Decision {
                should_attempt: proposal.is_some(),
                reason: if remaining < 4 {
                    "budget"
                } else if proposal.is_some() {
                    "always_attempt"
                } else {
                    "source_miss"
                },
                histogram: [0; 4],
                last_attempt_position: None,
                full_acceptance_gain: None,
                expected_gain: None,
            },
        };
        let attempted = if decision.should_attempt {
            proposal.clone()
        } else {
            None
        };
        black_box(
            attempted
                .as_ref()
                .map(|p| [seed, p.guesses[0], p.guesses[1], p.guesses[2]]),
        );
        let proposal_ns = started.elapsed().as_nanos();

        // This is offline target verification, never an input to `decide`.
        let accepted = replay::score(attempted.as_ref(), &trace.generated_token_ids[row + 1..]);
        let available = trace.generated_token_ids.len() - 1 - row;
        let progress = if attempted.is_none() {
            1
        } else if accepted == 3 {
            4
        } else {
            (accepted + 2).min(available)
        };
        ensure!(
            progress <= available,
            "replay crossed the final target input"
        );

        let started = Instant::now();
        source.commit(&trace.generated_token_ids[row..row + progress]);
        if policy == Policy::Adaptive && attempted.is_some() {
            controller.observe(position, accepted)?;
        }
        let maintenance_ns = started.elapsed().as_nanos();
        result.replay.cpu_elapsed_ns += proposal_ns + maintenance_ns;
        result.decisions.push(DecisionRecord {
            input_row: row,
            absolute_position: position,
            remaining_budget: remaining,
            proposal,
            decision,
            observed_accepted_guesses: attempted.as_ref().map(|_| accepted),
            next_input_row: row + progress,
        });
        result.replay.rounds.push(Round {
            input_row: row,
            absolute_position: position,
            known_seed: seed,
            guesses: attempted,
            accepted_guesses: accepted,
            progress,
            proposal_ns,
            maintenance_ns,
        });
        row += progress;
    }
    result.replay.logical_payload_bytes = source.logical_payload_bytes();
    let started = Instant::now();
    drop(source);
    result.teardown_ns = started.elapsed().as_nanos();
    result.replay.cpu_elapsed_ns += result.teardown_ns;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(ids: Vec<u32>, has_stop: bool) -> Trace {
        let mut prompt = vec![80; 56];
        prompt.extend([1, 2, 3, 4, 5, 6, 1, 2, 3]);
        Trace {
            id: "unit".into(),
            class: "unit".into(),
            split: "unit".into(),
            prompt_token_ids: prompt,
            generated_token_ids: ids,
            stop_ids: vec![99],
            stop_reason: if has_stop { "stop" } else { "length" }.into(),
            output: String::new(),
            serial_fixed_ms: None,
            max_tokens: 8,
        }
    }
    fn costs() -> Vec<DecisionCosts> {
        vec![
            DecisionCosts {
                serial_ms: [40., 60., 80., 80.],
                candidate_ms: [80., 80., 80., 60.]
            };
            505
        ]
    }

    /// Changed future tokens, including an early EOS, cannot alter the first
    /// proposal or policy decision. EOS suppresses only actual correction work.
    #[test]
    fn test_future_and_natural_stop_do_not_enter_policy() -> Result<()> {
        let a = run(
            &trace(vec![4, 5, 6, 1, 2, 3, 4, 5], false),
            Policy::Adaptive,
            &costs(),
        )?;
        let b = run(&trace(vec![4, 5, 99], true), Policy::Adaptive, &costs())?;
        assert_eq!(a.decisions[0].proposal, b.decisions[0].proposal);
        assert_eq!(a.decisions[0].decision, b.decisions[0].decision);
        assert!(b.decisions[0].decision.should_attempt);
        assert_eq!(b.replay.rounds[0].progress, 2);
        assert_eq!(b.replay.rounds.iter().map(|r| r.progress).sum::<usize>(), 2);
        Ok(())
    }

    /// Hardware rejection keeps all work serial and does not expose skipped
    /// acceptance labels; complete source costs and final teardown remain charged.
    #[test]
    fn test_skipped_proposals_do_not_teach_histogram() -> Result<()> {
        let table = vec![
            DecisionCosts {
                serial_ms: [40., 60., 80., 80.],
                candidate_ms: [120.; 4]
            };
            505
        ];
        let replay = run(
            &trace(vec![4, 5, 6, 1, 2, 3, 4, 5], false),
            Policy::Adaptive,
            &table,
        )?;
        assert!(replay.decisions[0].proposal.is_some());
        assert!(replay
            .decisions
            .iter()
            .all(|d| d.observed_accepted_guesses.is_none() && d.decision.histogram == [0; 4]));
        assert!(replay
            .replay
            .rounds
            .iter()
            .all(|r| r.guesses.is_none() && r.progress == 1));
        assert_eq!(replay.replay.rounds.len(), 7);
        assert_eq!(
            replay.replay.cpu_elapsed_ns,
            replay.replay.construction_ns
                + replay.teardown_ns
                + replay
                    .replay
                    .rounds
                    .iter()
                    .map(|r| r.proposal_ns + r.maintenance_ns)
                    .sum::<u128>()
        );
        Ok(())
    }

    /// The comparison policy preserves the existing prompt-copy visit path.
    #[test]
    fn test_always_attempt_reproduces_original_replay() -> Result<()> {
        let trace = trace(vec![4, 5, 6, 1, 2, 3, 4, 5], false);
        let original = replay::run(&trace, SourceKind::PromptCopy)?;
        let actual = run(&trace, Policy::AlwaysAttempt, &costs())?;
        assert_eq!(actual.replay.signature(), original.signature());
        Ok(())
    }
}

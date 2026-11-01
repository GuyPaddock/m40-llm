use super::{
    records::Trace,
    source::{DraftSource, Proposal, SourceKind},
};
use anyhow::{ensure, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::{hint::black_box, time::Instant};

#[derive(Serialize)]
pub struct Round {
    pub input_row: usize,
    pub absolute_position: usize,
    pub known_seed: u32,
    pub guesses: Option<Proposal>,
    pub accepted_guesses: usize,
    pub progress: usize,
    pub proposal_ns: u128,
    pub maintenance_ns: u128,
}
#[derive(Serialize)]
pub struct Replay {
    pub construction_ns: u128,
    pub cpu_elapsed_ns: u128,
    pub logical_payload_bytes: usize,
    pub rounds: Vec<Round>,
}
impl Replay {
    pub fn signature(&self) -> Vec<(usize, Option<[u32; 3]>, usize, usize)> {
        self.rounds
            .iter()
            .map(|r| {
                (
                    r.input_row,
                    r.guesses.as_ref().map(|g| g.guesses),
                    r.accepted_guesses,
                    r.progress,
                )
            })
            .collect()
    }
    pub fn statistics(&self) -> Value {
        statistics(&self.rounds)
    }
}
pub fn statistics(rounds: &[Round]) -> Value {
    let mut reached = [0; 3];
    let mut matched = [0; 3];
    let mut rejects = [0; 4];
    let mut eligible = 0;
    for r in rounds.iter().filter(|r| r.guesses.is_some()) {
        eligible += 1;
        for j in 0..3 {
            if r.accepted_guesses >= j {
                reached[j] += 1;
            }
            if r.accepted_guesses > j {
                matched[j] += 1;
            }
        }
        rejects[r.accepted_guesses] += 1;
    }
    let conditional: Vec<_> = (0..3)
        .map(|j| {
            if reached[j] == 0 {
                None
            } else {
                Some(matched[j] as f64 / reached[j] as f64)
            }
        })
        .collect();
    let mut bins = Vec::new();
    for (lo, hi) in [
        (0, 128),
        (128, 256),
        (256, 384),
        (384, 576),
        (576, usize::MAX),
    ] {
        let selected: Vec<_> = rounds
            .iter()
            .filter(|r| r.absolute_position >= lo && r.absolute_position < hi)
            .collect();
        let count = selected.iter().filter(|r| r.guesses.is_some()).count();
        let full = selected
            .iter()
            .filter(|r| r.guesses.is_some() && r.accepted_guesses == 3)
            .count();
        let total_ns: u128 = selected
            .iter()
            .map(|r| r.proposal_ns + r.maintenance_ns)
            .sum();
        bins.push(json!({"absolute_start":lo,"absolute_end":hi,"decisions":selected.len(),"eligible":count,"full_acceptances":full,"cpu_ns":total_ns}));
    }
    json!({"decisions":rounds.len(),"eligible":eligible,"eligible_coverage":eligible as f64/rounds.len().max(1) as f64,
        "conditional_reached":reached,"conditional_matched":matched,"conditional_acceptance":conditional,
        "full_batch_acceptance":if eligible>0 {Some(rejects[3] as f64/eligible as f64)} else {None},
        "rejected_guess_or_full_histogram":rejects,"context_bins":bins})
}

/// Future reference access is confined to this scorer, after proposal returns.
pub fn score(proposal: Option<&Proposal>, future: &[u32]) -> usize {
    proposal.map_or(0, |p| {
        p.guesses
            .iter()
            .zip(future)
            .take_while(|(a, b)| a == b)
            .count()
    })
}
pub fn run(trace: &Trace, kind: SourceKind) -> Result<Replay> {
    trace.validate()?;
    let start = Instant::now();
    let mut source = DraftSource::new(kind, &trace.prompt_token_ids, &trace.stop_ids);
    let construction_ns = start.elapsed().as_nanos();
    let mut result = Replay {
        construction_ns,
        cpu_elapsed_ns: construction_ns,
        logical_payload_bytes: 0,
        rounds: Vec::new(),
    };
    let mut row = 0;
    while row + 1 < trace.generated_token_ids.len() {
        let available = trace.generated_token_ids.len() - 1 - row;
        let began = Instant::now();
        let proposal = if trace.max_tokens - 1 - row >= 4 {
            source.propose(trace.generated_token_ids[row])
        } else {
            None
        };
        let _candidates = black_box(proposal.as_ref().map(|p| {
            [
                trace.generated_token_ids[row],
                p.guesses[0],
                p.guesses[1],
                p.guesses[2],
            ]
        }));
        let proposal_ns = began.elapsed().as_nanos();
        let accepted = score(proposal.as_ref(), &trace.generated_token_ids[row + 1..]);
        let progress = if proposal.is_none() {
            1
        } else if accepted == 3 {
            4
        } else {
            (accepted + 2).min(available)
        };
        ensure!(progress <= available, "scorer crossed final target row");
        let began = Instant::now();
        source.commit(&trace.generated_token_ids[row..row + progress]);
        let maintenance_ns = began.elapsed().as_nanos();
        result.cpu_elapsed_ns += proposal_ns + maintenance_ns;
        result.rounds.push(Round {
            input_row: row,
            absolute_position: trace.prompt_token_ids.len() + row,
            known_seed: trace.generated_token_ids[row],
            guesses: proposal,
            accepted_guesses: accepted,
            progress,
            proposal_ns,
            maintenance_ns,
        });
        row += progress;
    }
    result.logical_payload_bytes = source.logical_payload_bytes();
    Ok(result)
}

/// Observe every possible position independently; these are not visited cycles.
pub fn all_positions(trace: &Trace, kind: SourceKind) -> Result<Value> {
    let mut source = DraftSource::new(kind, &trace.prompt_token_ids, &trace.stop_ids);
    let mut rounds = Vec::new();
    for row in 0..trace.generated_token_ids.len().saturating_sub(1) {
        let proposal = if trace.max_tokens - 1 - row >= 4 {
            source.propose(trace.generated_token_ids[row])
        } else {
            None
        };
        let accepted = score(proposal.as_ref(), &trace.generated_token_ids[row + 1..]);
        rounds.push(Round {
            input_row: row,
            absolute_position: trace.prompt_token_ids.len() + row,
            known_seed: trace.generated_token_ids[row],
            guesses: proposal,
            accepted_guesses: accepted,
            progress: 1,
            proposal_ns: 0,
            maintenance_ns: 0,
        });
        source.commit(&trace.generated_token_ids[row..row + 1]);
    }
    Ok(statistics(&rounds))
}

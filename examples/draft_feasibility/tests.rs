use super::{
    records::Trace,
    replay,
    source::{DraftSource, Proposal, SourceKind},
};

fn trace(future: Vec<u32>) -> Trace {
    let max_tokens = future.len();
    Trace {
        max_tokens,
        id: "unit".into(),
        class: "unit".into(),
        split: "unit".into(),
        prompt_token_ids: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3],
        generated_token_ids: future,
        stop_ids: vec![99],
        stop_reason: "length".into(),
        output: String::new(),
        serial_fixed_ms: None,
    }
}

/// Changing a future continuation cannot affect the first causal proposal.
#[test]
fn test_unseen_future_does_not_change_proposal() {
    for kind in SourceKind::ALL {
        let a = replay::run(&trace(vec![4, 5, 6, 7, 8, 9, 10, 11]), kind).unwrap();
        let b = replay::run(&trace(vec![4, 70, 71, 72, 73, 74, 75, 76]), kind).unwrap();
        assert_eq!(a.rounds[0].guesses, b.rounds[0].guesses);
    }
}
/// Repeated hypothetical proposals must not train counts or commit guesses.
#[test]
fn test_proposals_do_not_mutate_committed_state() {
    for kind in SourceKind::ALL {
        let mut a = DraftSource::new(kind, &[1, 2, 3, 4, 5, 6, 1, 2, 3], &[]);
        let mut b = DraftSource::new(kind, &[1, 2, 3, 4, 5, 6, 1, 2, 3], &[]);
        let first = a.propose(4);
        assert_eq!(a.propose(4), first);
        a.commit(&[42, 43]);
        b.commit(&[42, 43]);
        assert_eq!(a.propose(1), b.propose(1));
    }
}
/// The known seed is excluded from guess acceptance and each mismatch stops it.
#[test]
fn test_conditional_guess_scoring() {
    let p = Proposal {
        guesses: [2, 3, 4],
        match_start: None,
        suffix_len: None,
    };
    for (future, count) in [
        (vec![9, 3, 4], 0),
        (vec![2, 9, 4], 1),
        (vec![2, 3, 9], 2),
        (vec![2, 3, 4], 3),
    ] {
        assert_eq!(replay::score(Some(&p), &future), count);
    }
    assert_eq!(replay::score(None, &[2, 3, 4]), 0);
}
/// The source copies only three tokens already present inside the prompt.
#[test]
fn test_prompt_copy_provenance_and_bounds() {
    let mut source = DraftSource::new(
        SourceKind::PromptCopy,
        &[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3],
        &[],
    );
    let p = source.propose(4).unwrap();
    assert_eq!(p.guesses, [5, 6, 7]);
    assert_eq!(p.match_start, Some(0));
    let mut short = DraftSource::new(SourceKind::PromptCopy, &[1, 2, 3], &[]);
    assert!(short.propose(4).is_none());
}
/// Sparse-history misses and final tails remain ordinary rows with charged costs.
#[test]
fn test_no_match_replay_covers_every_input_once() {
    let t = trace(vec![81, 82, 83, 84, 85, 86, 87, 88]);
    let r = replay::run(&t, SourceKind::PromptCopy).unwrap();
    assert_eq!(r.rounds.iter().map(|r| r.progress).sum::<usize>(), 7);
    assert!(r.rounds.iter().all(|r| r.guesses.is_none()));
    assert_eq!(
        r.cpu_elapsed_ns,
        r.construction_ns
            + r.rounds
                .iter()
                .map(|r| r.proposal_ns + r.maintenance_ns)
                .sum::<u128>()
    );
}
/// A trace ending naturally cannot hide tokens after an earlier stop.
#[test]
fn test_rejects_post_stop_reference() {
    let t = trace(vec![1, 99, 2]);
    assert!(t.validate().is_err());
}

/// A reference EOS cannot make a still-legal four-row proposal disappear.
/// Only its verified stop may suppress correction, never the proposal itself.
#[test]
fn test_future_stop_is_not_a_drafting_budget() {
    let mut short = trace(vec![4, 5, 99]);
    short.max_tokens = 8;
    short.stop_reason = "stop".into();
    let long = trace(vec![4, 5, 6, 7, 8, 9, 10, 11]);
    for kind in SourceKind::ALL {
        let a = replay::run(&short, kind).unwrap();
        let b = replay::run(&long, kind).unwrap();
        assert_eq!(a.rounds[0].guesses, b.rounds[0].guesses);
        assert!(a.rounds[0].guesses.is_some());
        assert_eq!(a.rounds[0].progress, 2);
    }
}

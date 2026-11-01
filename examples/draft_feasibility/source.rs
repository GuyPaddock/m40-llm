use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum SourceKind {
    HistoryLookup,
    PromptCopy,
    OnlineBackoff,
}
impl SourceKind {
    pub const ALL: [Self; 3] = [Self::HistoryLookup, Self::PromptCopy, Self::OnlineBackoff];
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Proposal {
    pub guesses: [u32; 3],
    pub match_start: Option<usize>,
    pub suffix_len: Option<usize>,
}

/// Owns only the input prompt and tokens already committed by the target.
/// The target's known seed is context for proposal, never a training example.
pub struct DraftSource {
    kind: SourceKind,
    prompt_len: usize,
    history: Vec<u32>,
    stops: Vec<u32>,
    counts: HashMap<Vec<u32>, BTreeMap<u32, u64>>,
}

impl DraftSource {
    pub fn new(kind: SourceKind, prompt: &[u32], stops: &[u32]) -> Self {
        let mut source = Self {
            kind,
            prompt_len: prompt.len(),
            history: Vec::new(),
            stops: stops.to_vec(),
            counts: HashMap::new(),
        };
        source.commit(prompt);
        source
    }

    pub fn commit(&mut self, tokens: &[u32]) {
        for &token in tokens {
            if self.kind == SourceKind::OnlineBackoff {
                for order in 1..=4.min(self.history.len()) {
                    let key = self.history[self.history.len() - order..].to_vec();
                    *self
                        .counts
                        .entry(key)
                        .or_default()
                        .entry(token)
                        .or_default() += 1;
                }
            }
            self.history.push(token);
        }
    }

    pub fn propose(&mut self, known_seed: u32) -> Option<Proposal> {
        self.history.push(known_seed);
        let proposal = match self.kind {
            SourceKind::HistoryLookup => {
                m40_llm::decode::ngram::propose(&self.history, 3, &self.stops)
                    .expect("fixed legal draft size")
                    .filter(|d| d.tokens.len() == 3)
                    .map(|d| Proposal {
                        guesses: d.tokens.try_into().unwrap(),
                        match_start: Some(d.match_start),
                        suffix_len: Some(d.suffix_len),
                    })
            }
            SourceKind::PromptCopy => self.copy_prompt(),
            SourceKind::OnlineBackoff => self.predict_backoff(),
        };
        self.history.pop();
        proposal.filter(|p| !p.guesses.iter().any(|t| self.stops.contains(t)))
    }

    fn copy_prompt(&self) -> Option<Proposal> {
        for len in (4..=16.min(self.history.len())).rev() {
            let suffix = &self.history[self.history.len() - len..];
            for start in (0..=self.prompt_len.saturating_sub(len + 3)).rev() {
                if start + len + 3 <= self.prompt_len && &self.history[start..start + len] == suffix
                {
                    return Some(Proposal {
                        guesses: self.history[start + len..start + len + 3]
                            .try_into()
                            .unwrap(),
                        match_start: Some(start),
                        suffix_len: Some(len),
                    });
                }
            }
        }
        None
    }

    fn predict_backoff(&self) -> Option<Proposal> {
        let mut context = self.history[self.history.len().saturating_sub(4)..].to_vec();
        let mut guesses = [0; 3];
        for guess in &mut guesses {
            let next = (1..=4.min(context.len())).rev().find_map(|order| {
                self.counts
                    .get(&context[context.len() - order..])
                    .and_then(|counts| {
                        counts
                            .iter()
                            .max_by(|(ta, ca), (tb, cb)| ca.cmp(cb).then_with(|| tb.cmp(ta)))
                            .map(|(&t, _)| t)
                    })
            })?;
            *guess = next;
            context.push(next);
        }
        Some(Proposal {
            guesses,
            match_start: None,
            suffix_len: None,
        })
    }

    /// Logical owned payload only; standard-library hash/tree overhead is excluded.
    pub fn logical_payload_bytes(&self) -> usize {
        self.history.capacity() * 4
            + self.stops.capacity() * 4
            + self
                .counts
                .iter()
                .map(|(key, counts)| key.capacity() * 4 + counts.len() * 12)
                .sum::<usize>()
    }
}

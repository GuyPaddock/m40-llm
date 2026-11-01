use anyhow::{ensure, Context, Result};
use std::collections::VecDeque;

pub const SKETCH_WIDTH: usize = 16;
pub const HASH_SEED: u64 = 0x4d34_305f_5345_4c31;
pub const BANK_CAPACITY: usize = 8;

/// Fixed SplitMix64 finalizer; no coefficient is fitted to captured vectors.
fn hash(value: u64) -> u64 {
    let mut value = value.wrapping_add(HASH_SEED);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub fn select_top_k(scores: &[f64], count: usize) -> Result<Vec<usize>> {
    ensure!(count > 0 && count <= scores.len(), "invalid selection size");
    ensure!(
        scores.iter().all(|value| value.is_finite()),
        "nonfinite score"
    );
    let mut ranked: Vec<_> = (0..scores.len()).collect();
    ranked.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    ranked.truncate(count);
    ranked.sort_unstable();
    Ok(ranked)
}

pub fn random_selection(width: usize, count: usize, row_identity: u64) -> Result<Vec<usize>> {
    ensure!(count > 0 && count <= width, "invalid random selection size");
    let mut ranked: Vec<_> = (0..width).collect();
    ranked.sort_unstable_by_key(|&channel| (hash(channel as u64 ^ hash(row_identity)), channel));
    ranked.truncate(count);
    ranked.sort_unstable();
    Ok(ranked)
}

/// Signed additions form a CountSketch, normalized for cosine retrieval.
/// Zero inputs remain zero and use the latest refresh as a documented fallback.
pub fn sketch(input: &[f32]) -> Result<[f64; SKETCH_WIDTH]> {
    ensure!(
        !input.is_empty() && input.iter().all(|value| value.is_finite()),
        "invalid sketch input"
    );
    let mut output = [0.0; SKETCH_WIDTH];
    for (index, &value) in input.iter().enumerate() {
        let code = hash(index as u64);
        let sign = if code >> 63 == 0 { 1.0 } else { -1.0 };
        output[code as usize % SKETCH_WIDTH] += sign * f64::from(value);
    }
    let norm = output.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm > 0.0 {
        for value in &mut output {
            *value /= norm;
        }
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug)]
pub enum Signal {
    Latest,
    Frequency,
    CurrentSketch,
    PreviousSketch,
}

pub struct Prediction {
    pub channels: Vec<usize>,
    pub source_rows: Vec<usize>,
}

struct Refresh {
    row: usize,
    channels: Vec<usize>,
    current: [f64; SKETCH_WIDTH],
    previous: Option<[f64; SKETCH_WIDTH]>,
}

/// Each task/layer/retention gets a fresh owner. Only scheduled dense refreshes
/// may supply labels; prediction has no hidden-vector or oracle argument.
pub struct History {
    interval: usize,
    count: usize,
    frequency: Vec<f64>,
    source_rows: Vec<usize>,
    bank: VecDeque<Refresh>,
}

impl History {
    pub fn new(width: usize, count: usize, interval: usize) -> Result<Self> {
        ensure!(
            width > 0 && count > 0 && count <= width && matches!(interval, 8 | 16),
            "invalid history geometry"
        );
        Ok(Self {
            interval,
            count,
            frequency: vec![0.0; width],
            source_rows: Vec::new(),
            bank: VecDeque::new(),
        })
    }

    pub fn should_refresh(&self, row: usize) -> bool {
        row.is_multiple_of(self.interval)
    }

    pub fn observe_refresh(
        &mut self,
        row: usize,
        channels: &[usize],
        current: [f64; SKETCH_WIDTH],
        previous: Option<[f64; SKETCH_WIDTH]>,
    ) -> Result<()> {
        ensure!(
            self.should_refresh(row)
                && self
                    .source_rows
                    .last()
                    .map_or(row == 0, |&last| row == last + self.interval),
            "out-of-order or unscheduled refresh"
        );
        ensure!(
            channels.len() == self.count
                && channels.iter().all(|&index| index < self.frequency.len())
                && channels.windows(2).all(|pair| pair[0] < pair[1]),
            "invalid refresh selection"
        );
        ensure!(
            current
                .iter()
                .chain(previous.iter().flatten())
                .all(|value| value.is_finite()),
            "nonfinite refresh feature"
        );
        for &channel in channels {
            self.frequency[channel] += 1.0;
        }
        self.source_rows.push(row);
        if self.bank.len() == BANK_CAPACITY {
            self.bank.pop_front();
        }
        self.bank.push_back(Refresh {
            row,
            channels: channels.to_vec(),
            current,
            previous,
        });
        Ok(())
    }

    pub fn predict(
        &self,
        row: usize,
        signal: Signal,
        feature: Option<[f64; SKETCH_WIDTH]>,
    ) -> Result<Prediction> {
        let latest = self
            .bank
            .back()
            .context("prediction needs an earlier dense refresh")?;
        ensure!(
            row > latest.row && row < latest.row + self.interval,
            "prediction crosses a required dense refresh"
        );
        if matches!(signal, Signal::Frequency) {
            return Ok(Prediction {
                channels: select_top_k(&self.frequency, self.count)?,
                source_rows: self.source_rows.clone(),
            });
        }
        let selected = if matches!(signal, Signal::Latest) {
            latest
        } else {
            let query = feature.context("sketch predictor needs a query feature")?;
            ensure!(
                query.iter().all(|value| value.is_finite()),
                "nonfinite query feature"
            );
            if query.iter().all(|&value| value == 0.0) {
                latest
            } else {
                let mut best = None;
                // Newest first preserves the specified recency tie break.
                for candidate in self.bank.iter().rev() {
                    let reference = if matches!(signal, Signal::PreviousSketch) {
                        candidate
                            .previous
                            .as_ref()
                            .context("no predecessor feature")?
                    } else {
                        &candidate.current
                    };
                    let score: f64 = query.iter().zip(reference).map(|(a, b)| a * b).sum();
                    if best.as_ref().is_none_or(|(_, maximum)| score > *maximum) {
                        best = Some((candidate, score));
                    }
                }
                best.context("empty sketch bank")?.0
            }
        };
        Ok(Prediction {
            channels: selected.channels.clone(),
            source_rows: vec![selected.row],
        })
    }
}

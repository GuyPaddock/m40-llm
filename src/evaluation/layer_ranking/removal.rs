use super::LayerInfluence;
use crate::evaluation::loss::LossMetrics;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerRemovalTrial {
    pub layer_id: usize,
    /// Loss on the same frozen calibration passage tokens, never held-out data.
    pub calibration_loss: LossMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerRemovalStage {
    pub winner_layer_id: usize,
    pub trials: Vec<LayerRemovalTrial>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerRemovalRanking {
    pub schema_version: u32,
    pub influence: LayerInfluence,
    pub stages: Vec<LayerRemovalStage>,
}

fn candidates(layer_count: usize, selected: &[usize]) -> Vec<usize> {
    (1..layer_count - 1)
        .filter(|layer| !selected.contains(layer))
        .collect()
}

fn choose_winner(
    influence: &LayerInfluence,
    selected: &[usize],
    trials: &[LayerRemovalTrial],
) -> Result<usize> {
    let expected = candidates(influence.layers.len(), selected);
    ensure!(
        trials.iter().map(|trial| trial.layer_id).eq(expected),
        "rank every remaining original interior layer exactly once in canonical order"
    );
    let rows = influence.plan.observed_rows()?;
    for trial in trials {
        let loss = &trial.calibration_loss;
        let checked = LossMetrics::from_sum(loss.nll_sum, loss.scored_tokens)?;
        ensure!(
            loss.scored_tokens == rows && (loss.mean_nll - checked.mean_nll).abs() <= 1e-12,
            "removal loss must cover identical calibration targets"
        );
        ensure!(
            loss.perplexity_overflow == checked.perplexity_overflow
                && match (loss.perplexity, checked.perplexity) {
                    (Some(a), Some(b)) => a.is_finite() && (a - b).abs() <= b * 1e-12,
                    (None, None) => true,
                    _ => false,
                },
            "inconsistent removal loss metrics"
        );
    }
    let order = influence.ranked_layers()?;
    trials
        .iter()
        .min_by(|a, b| {
            a.calibration_loss
                .mean_nll
                .total_cmp(&b.calibration_loss.mean_nll)
                .then_with(|| {
                    order
                        .iter()
                        .position(|&id| id == a.layer_id)
                        .cmp(&order.iter().position(|&id| id == b.layer_id))
                })
        })
        .map(|trial| trial.layer_id)
        .context("no interior layer remains")
}

impl LayerRemovalRanking {
    pub fn parse(json: &str) -> Result<Self> {
        let ranking: Self = serde_json::from_str(json)?;
        ranking.validate()?;
        Ok(ranking)
    }

    pub fn validate(&self) -> Result<()> {
        self.influence.validate()?;
        ensure!(
            self.schema_version == 1
                && (1..=3.min(self.influence.layers.len() - 2)).contains(&self.stages.len()),
            "layer ranking must freeze one to three interior removals"
        );
        let mut selected = Vec::new();
        for stage in &self.stages {
            let winner = choose_winner(&self.influence, &selected, &stage.trials)?;
            ensure!(
                winner == stage.winner_layer_id,
                "ranking winner differs from calibration evidence"
            );
            selected.push(winner);
        }
        Ok(())
    }

    pub fn selected_layers(&self, count: usize) -> Result<Vec<usize>> {
        self.validate()?;
        ensure!(
            (1..=self.stages.len()).contains(&count),
            "removal count outside frozen ranking"
        );
        let mut layers: Vec<_> = self.stages[..count]
            .iter()
            .map(|stage| stage.winner_layer_id)
            .collect();
        layers.sort_unstable();
        Ok(layers)
    }
}

/// Re-evaluate every remaining layer after each removal. BI resolves exact loss
/// ties; no held-out score or task answer is accepted by this state machine.
pub struct LayerRemovalRanker {
    ranking: LayerRemovalRanking,
    requested_removals: usize,
}

impl LayerRemovalRanker {
    pub fn new(influence: LayerInfluence, requested_removals: usize) -> Result<Self> {
        influence.validate()?;
        ensure!(
            (1..=3.min(influence.layers.len() - 2)).contains(&requested_removals),
            "request one to three protected-interior removals"
        );
        Ok(Self {
            ranking: LayerRemovalRanking {
                schema_version: 1,
                influence,
                stages: Vec::new(),
            },
            requested_removals,
        })
    }

    pub fn selected_layers(&self) -> Vec<usize> {
        let mut layers: Vec<_> = self
            .ranking
            .stages
            .iter()
            .map(|stage| stage.winner_layer_id)
            .collect();
        layers.sort_unstable();
        layers
    }

    pub fn candidate_layers(&self) -> Vec<usize> {
        candidates(self.ranking.influence.layers.len(), &self.selected_layers())
    }

    pub fn record_stage(&mut self, mut trials: Vec<LayerRemovalTrial>) -> Result<usize> {
        ensure!(
            self.ranking.stages.len() < self.requested_removals,
            "all removal stages are already frozen"
        );
        trials.sort_by_key(|trial| trial.layer_id);
        let winner_layer_id =
            choose_winner(&self.ranking.influence, &self.selected_layers(), &trials)?;
        self.ranking.stages.push(LayerRemovalStage {
            winner_layer_id,
            trials,
        });
        Ok(winner_layer_id)
    }

    pub fn finish(self) -> Result<LayerRemovalRanking> {
        ensure!(
            self.ranking.stages.len() == self.requested_removals,
            "incomplete removal ranking"
        );
        self.ranking.validate()?;
        Ok(self.ranking)
    }
}

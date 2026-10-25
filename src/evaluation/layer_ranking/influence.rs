use crate::evaluation::corpus::{Corpus, Split};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LayerRankingPassage {
    pub id: String,
    pub token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerRankingPlan {
    pub model_sha256: String,
    pub corpus_sha256: String,
    pub execution_sha256: String,
    pub split: Split,
    pub layer_count: u32,
    pub hidden_dim: usize,
    pub passages: Vec<LayerRankingPassage>,
}

impl LayerRankingPlan {
    pub fn new(
        model_sha256: String,
        corpus_sha256: String,
        execution_sha256: String,
        layer_count: u32,
        hidden_dim: usize,
        corpus: &Corpus,
        passages: Vec<LayerRankingPassage>,
    ) -> Result<Self> {
        let plan = Self {
            model_sha256,
            corpus_sha256,
            execution_sha256,
            split: Split::Calibration,
            layer_count,
            hidden_dim,
            passages,
        };
        plan.validate()?;
        plan.validate_corpus(corpus)?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<()> {
        for hash in [
            &self.model_sha256,
            &self.corpus_sha256,
            &self.execution_sha256,
        ] {
            ensure!(
                hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
                "layer ranking requires SHA-256 identities"
            );
        }
        ensure!(
            self.split == Split::Calibration,
            "held-out layer fitting is forbidden"
        );
        ensure!(
            self.layer_count >= 3 && self.hidden_dim > 0 && !self.passages.is_empty(),
            "layer influence needs hidden vectors, calibration passages and interior layers"
        );
        let mut ids = HashSet::new();
        for passage in &self.passages {
            ensure!(
                !passage.id.is_empty()
                    && ids.insert(&passage.id)
                    && (2..=u32::MAX as usize).contains(&passage.token_ids.len()),
                "invalid layer-ranking passage identity/length"
            );
        }
        self.observed_rows()?;
        Ok(())
    }

    pub fn validate_corpus(&self, corpus: &Corpus) -> Result<()> {
        corpus.validate()?;
        ensure!(
            self.passages.iter().map(|passage| &passage.id).eq(corpus
                .calibration
                .passages
                .iter()
                .map(|passage| &passage.id)),
            "rank only the frozen calibration passages in their original order"
        );
        Ok(())
    }

    pub fn observed_rows(&self) -> Result<usize> {
        self.passages
            .iter()
            .try_fold(0usize, |count, passage| {
                count.checked_add(passage.token_ids.len().checked_sub(1)?)
            })
            .context("layer influence row count overflow")
    }
}

/// ShortGPT-style residual block change, accumulated on the host in FP64.
/// Undefined zero-vector cosine and nonfinite inputs are errors, not zero BI.
pub fn measure_block_influence(input: &[f32], output: &[f32]) -> Result<f64> {
    ensure!(
        !input.is_empty() && input.len() == output.len(),
        "block influence needs equal nonempty hidden vectors"
    );
    let (mut dot, mut input_norm, mut output_norm) = (0.0, 0.0, 0.0);
    for (&a, &b) in input.iter().zip(output) {
        ensure!(
            a.is_finite() && b.is_finite(),
            "nonfinite block hidden state"
        );
        let (a, b) = (f64::from(a), f64::from(b));
        dot += a * b;
        input_norm += a * a;
        output_norm += b * b;
    }
    ensure!(
        input_norm > 0.0 && output_norm > 0.0,
        "zero hidden-state norm"
    );
    Ok(1.0 - (dot / (input_norm.sqrt() * output_norm.sqrt())).clamp(-1.0, 1.0))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfluenceValues {
    pub layer_id: usize,
    /// Flattened passage order, positions 0..token_count-1, excluding the last target.
    pub block_influence: Vec<f64>,
}

impl LayerInfluenceValues {
    pub fn mean(&self) -> f64 {
        self.block_influence.iter().sum::<f64>() / self.block_influence.len() as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfluence {
    pub schema_version: u32,
    pub plan: LayerRankingPlan,
    pub layers: Vec<LayerInfluenceValues>,
}

impl LayerInfluence {
    pub fn validate(&self) -> Result<()> {
        self.plan.validate()?;
        let count = self.plan.observed_rows()?;
        ensure!(
            self.schema_version == 1 && self.layers.len() == self.plan.layer_count as usize,
            "invalid layer influence schema/layer count"
        );
        for (id, layer) in self.layers.iter().enumerate() {
            ensure!(
                layer.layer_id == id
                    && layer.block_influence.len() == count
                    && layer
                        .block_influence
                        .iter()
                        .all(|v| v.is_finite() && (0.0..=2.0).contains(v)),
                "incomplete/out-of-order/nonfinite block influence"
            );
        }
        Ok(())
    }

    /// Only interior IDs can become removal candidates; ties use original IDs.
    pub fn ranked_layers(&self) -> Result<Vec<usize>> {
        self.validate()?;
        let mut layers: Vec<_> = (1..self.layers.len() - 1).collect();
        layers.sort_by(|&a, &b| {
            self.layers[a]
                .mean()
                .total_cmp(&self.layers[b].mean())
                .then(a.cmp(&b))
        });
        Ok(layers)
    }
}

struct ActivePassage {
    counts: Vec<usize>,
}

/// Reserve bounded per-token scalar observations. Hidden vectors need not be
/// retained; every layer must observe all declared teacher-forced positions.
pub struct LayerInfluenceCollector {
    plan: LayerRankingPlan,
    layers: Vec<LayerInfluenceValues>,
    completed_passages: usize,
    active: Option<ActivePassage>,
}

impl LayerInfluenceCollector {
    pub fn new(plan: LayerRankingPlan, budget_bytes: usize) -> Result<Self> {
        plan.validate()?;
        let rows = plan.observed_rows()?;
        let bytes = rows
            .checked_mul(plan.layer_count as usize)
            .and_then(|n| n.checked_mul(8))
            .context("layer influence capacity overflow")?;
        ensure!(
            bytes <= budget_bytes,
            "layer influence observations exceed byte budget"
        );
        let mut layers = Vec::new();
        for layer_id in 0..plan.layer_count as usize {
            let mut block_influence = Vec::new();
            block_influence.try_reserve_exact(rows)?;
            layers.push(LayerInfluenceValues {
                layer_id,
                block_influence,
            });
        }
        Ok(Self {
            plan,
            layers,
            completed_passages: 0,
            active: None,
        })
    }

    pub fn begin_passage(&mut self, id: &str) -> Result<()> {
        ensure!(
            self.active.is_none(),
            "finish the active influence passage first"
        );
        ensure!(
            self.plan
                .passages
                .get(self.completed_passages)
                .is_some_and(|passage| passage.id == id),
            "observe only the next frozen calibration passage"
        );
        self.active = Some(ActivePassage {
            counts: vec![0; self.layers.len()],
        });
        Ok(())
    }

    pub fn has_active_passage(&self) -> bool {
        self.active.is_some()
    }

    pub fn observe(
        &mut self,
        layer: usize,
        position: u32,
        input: &[f32],
        output: &[f32],
    ) -> Result<()> {
        let active = self
            .active
            .as_mut()
            .context("begin a calibration passage first")?;
        let count = active
            .counts
            .get_mut(layer)
            .context("influence layer outside model")?;
        let expected = self.plan.passages[self.completed_passages].token_ids.len() - 1;
        ensure!(
            position as usize == *count && *count < expected,
            "observe contiguous absolute positions once per layer, excluding the final target"
        );
        ensure!(
            input.len() == self.plan.hidden_dim,
            "influence hidden dimension mismatch"
        );
        let value = measure_block_influence(input, output)?;
        self.layers[layer].block_influence.push(value);
        *count += 1;
        Ok(())
    }

    pub fn finish_passage(&mut self) -> Result<()> {
        let active = self
            .active
            .as_ref()
            .context("no active influence passage")?;
        let expected = self.plan.passages[self.completed_passages].token_ids.len() - 1;
        ensure!(
            active.counts.iter().all(|&count| count == expected),
            "incomplete influence passage/layers"
        );
        self.active = None;
        self.completed_passages += 1;
        Ok(())
    }

    pub fn validate_complete(&self) -> Result<()> {
        ensure!(
            self.active.is_none() && self.completed_passages == self.plan.passages.len(),
            "finish all calibration influence passages"
        );
        Ok(())
    }

    pub fn finish(self) -> Result<LayerInfluence> {
        self.validate_complete()?;
        let artifact = LayerInfluence {
            schema_version: 1,
            plan: self.plan,
            layers: self.layers,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

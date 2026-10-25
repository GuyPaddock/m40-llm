use super::ActivationThreshold;
use crate::evaluation::corpus::{Corpus, Split};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const TARGETS: [f64; 5] = [0.0, 0.25, 0.4, 0.5, 0.6];

fn validate_hash(hash: &str) -> Result<()> {
    ensure!(
        hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "calibration identity requires SHA-256 hex strings"
    );
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationCalibrationPlan {
    pub model_sha256: String,
    pub corpus_sha256: String,
    pub execution_sha256: String,
    pub split: Split,
    pub calibration_task_ids: Vec<String>,
    pub max_generated_rows_per_task: usize,
}

impl ActivationCalibrationPlan {
    /// Only the frozen calibration task IDs are eligible. Caller hashes identify
    /// original files and the exact execution configuration, not a model label.
    pub fn new(
        model_sha256: String,
        corpus_sha256: String,
        execution_sha256: String,
        corpus: &Corpus,
        max_generated_rows_per_task: usize,
    ) -> Result<Self> {
        corpus.validate()?;
        let plan = Self {
            model_sha256,
            corpus_sha256,
            execution_sha256,
            split: Split::Calibration,
            calibration_task_ids: corpus
                .calibration
                .tasks
                .iter()
                .map(|task| task.id.clone())
                .collect(),
            max_generated_rows_per_task,
        };
        plan.validate()?;
        Ok(plan)
    }

    fn validate(&self) -> Result<()> {
        for hash in [
            &self.model_sha256,
            &self.corpus_sha256,
            &self.execution_sha256,
        ] {
            validate_hash(hash)?;
        }
        ensure!(
            self.split == Split::Calibration,
            "held-out activation fitting is forbidden"
        );
        ensure!(
            (1..=256).contains(&self.max_generated_rows_per_task),
            "calibration row cap must be in 1..=256"
        );
        let ids: HashSet<_> = self.calibration_task_ids.iter().collect();
        ensure!(
            !ids.is_empty()
                && ids.len() == self.calibration_task_ids.len()
                && ids.iter().all(|id| !id.is_empty()),
            "invalid calibration task identities"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationLayer {
    pub layer_id: usize,
    pub tensor_name: String,
    pub byte_offset: u64,
    pub input_dim: usize,
    pub output_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationTaskObservation {
    pub id: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub captured_absolute_positions: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerActivationThreshold {
    pub target_sparsity: f64,
    pub magnitude: f32,
    pub achieved_sparsity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerActivationCalibration {
    pub source: CalibrationLayer,
    pub observed_rows: usize,
    pub thresholds: Vec<LayerActivationThreshold>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationCalibration {
    pub schema_version: u32,
    pub plan: ActivationCalibrationPlan,
    pub source_weights_sha256: String,
    pub tasks: Vec<CalibrationTaskObservation>,
    pub layers: Vec<LayerActivationCalibration>,
    pub observation_payload_bytes: usize,
    pub observation_capacity_bytes: usize,
}

impl ActivationCalibration {
    pub fn parse(json: &str) -> Result<Self> {
        let artifact: Self = serde_json::from_str(json)?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1,
            "unsupported activation calibration schema"
        );
        self.plan.validate()?;
        validate_hash(&self.source_weights_sha256)?;
        validate_layers(
            &self
                .layers
                .iter()
                .map(|layer| layer.source.clone())
                .collect::<Vec<_>>(),
        )?;
        ensure!(
            self.tasks.len() == self.plan.calibration_task_ids.len(),
            "incomplete calibration tasks"
        );
        let mut rows = 0usize;
        for (task, expected) in self.tasks.iter().zip(&self.plan.calibration_task_ids) {
            ensure!(
                &task.id == expected && (1..=u32::MAX as usize).contains(&task.prompt_tokens),
                "calibration task order/identity mismatch"
            );
            let count = task
                .generated_tokens
                .saturating_sub(1)
                .min(self.plan.max_generated_rows_per_task);
            let end = task
                .prompt_tokens
                .checked_add(count)
                .context("capture position overflow")?;
            let expected_positions = (task.prompt_tokens..end)
                .map(u32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            ensure!(
                task.captured_absolute_positions == expected_positions,
                "calibration must capture generated-token inputs, not prompt or future tokens"
            );
            rows = rows
                .checked_add(count)
                .context("calibration row overflow")?;
        }
        ensure!(rows > 0, "calibration has no generated-token observations");
        let mut payload = 0usize;
        for layer in &self.layers {
            ensure!(
                layer.observed_rows == rows && layer.thresholds.len() == TARGETS.len(),
                "incomplete layer calibration"
            );
            for (threshold, &target) in layer.thresholds.iter().zip(&TARGETS) {
                ensure!(
                    threshold.target_sparsity == target
                        && threshold.achieved_sparsity.is_finite()
                        && (0.0..=1.0).contains(&threshold.achieved_sparsity),
                    "invalid activation sparsity statistics"
                );
                ActivationThreshold::new(threshold.magnitude)?;
                ensure!(
                    target != 0.0 || threshold.magnitude == 0.0,
                    "zero-threshold control changed"
                );
            }
            ensure!(
                layer
                    .thresholds
                    .windows(2)
                    .all(|pair| pair[0].magnitude <= pair[1].magnitude
                        && pair[0].achieved_sparsity <= pair[1].achieved_sparsity),
                "calibrated quantiles must be monotonic"
            );
            payload = payload
                .checked_add(
                    rows.checked_mul(layer.source.input_dim)
                        .and_then(|count| count.checked_mul(4))
                        .context("observation byte overflow")?,
                )
                .context("observation byte overflow")?;
        }
        ensure!(
            self.observation_payload_bytes == payload && self.observation_capacity_bytes >= payload,
            "invalid observation memory accounting"
        );
        Ok(())
    }

    pub fn thresholds(&self, target: f64) -> Result<Vec<ActivationThreshold>> {
        self.validate()?;
        ensure!(
            TARGETS.contains(&target),
            "activation target must be 0, 0.25, 0.4, 0.5 or 0.6"
        );
        self.layers
            .iter()
            .map(|layer| {
                ActivationThreshold::new(
                    layer
                        .thresholds
                        .iter()
                        .find(|value| value.target_sparsity == target)
                        .context("missing threshold")?
                        .magnitude,
                )
            })
            .collect()
    }
}

fn validate_layers(layers: &[CalibrationLayer]) -> Result<()> {
    let mut names = HashSet::new();
    ensure!(
        !layers.is_empty(),
        "activation calibration needs model layers"
    );
    for (id, layer) in layers.iter().enumerate() {
        ensure!(
            layer.layer_id == id
                && !layer.tensor_name.is_empty()
                && names.insert(&layer.tensor_name),
            "invalid or duplicate calibration layer identity"
        );
        super::super::execution_weights::Q8MatrixShape::new(layer.input_dim, layer.output_dim)?;
    }
    Ok(())
}

struct ActiveTask {
    id: String,
    prompt_tokens: usize,
    positions: Vec<Vec<u32>>,
}

/// Bounded exact observations, not an adaptive sampler. Reserve before capture
/// and fail on nonfinite/missing/out-of-order generated rows rather than fitting
/// a threshold from an accidentally different prefix distribution.
pub struct ActivationCalibrationCollector {
    plan: ActivationCalibrationPlan,
    source_weights_sha256: String,
    layers: Vec<CalibrationLayer>,
    observations: Vec<Vec<f32>>,
    tasks: Vec<CalibrationTaskObservation>,
    active: Option<ActiveTask>,
}

impl ActivationCalibrationCollector {
    pub fn new(
        plan: ActivationCalibrationPlan,
        source_weights_sha256: String,
        layers: Vec<CalibrationLayer>,
        budget_bytes: usize,
    ) -> Result<Self> {
        plan.validate()?;
        validate_hash(&source_weights_sha256)?;
        validate_layers(&layers)?;
        let rows = plan
            .calibration_task_ids
            .len()
            .checked_mul(plan.max_generated_rows_per_task)
            .context("calibration capacity overflow")?;
        let capacities = layers
            .iter()
            .map(|layer| {
                rows.checked_mul(layer.input_dim)
                    .context("calibration capacity overflow")
            })
            .collect::<Result<Vec<_>>>()?;
        let bytes = capacities
            .iter()
            .try_fold(0usize, |sum, &count| sum.checked_add(count.checked_mul(4)?))
            .context("calibration byte overflow")?;
        ensure!(
            bytes <= budget_bytes,
            "activation observations need {bytes} bytes, budget is {budget_bytes}"
        );
        let mut observations = Vec::new();
        for capacity in capacities {
            let mut values = Vec::new();
            values.try_reserve_exact(capacity)?;
            observations.push(values);
        }
        Ok(Self {
            plan,
            source_weights_sha256,
            layers,
            observations,
            tasks: Vec::new(),
            active: None,
        })
    }

    pub fn begin_task(&mut self, id: &str, prompt_tokens: usize) -> Result<()> {
        ensure!(
            self.active.is_none() && prompt_tokens > 0 && prompt_tokens <= u32::MAX as usize,
            "finish prior calibration task and provide valid prompt tokens"
        );
        ensure!(
            prompt_tokens
                .checked_add(self.plan.max_generated_rows_per_task)
                .is_some_and(|end| end <= u32::MAX as usize),
            "capture positions exceed u32 indexing"
        );
        ensure!(
            self.plan
                .calibration_task_ids
                .get(self.tasks.len())
                .map(String::as_str)
                == Some(id),
            "capture only the next frozen calibration task, never held-out tasks"
        );
        self.active = Some(ActiveTask {
            id: id.to_owned(),
            prompt_tokens,
            positions: vec![Vec::new(); self.layers.len()],
        });
        Ok(())
    }

    pub fn should_observe(&self, position: u32) -> bool {
        self.active.as_ref().is_some_and(|task| {
            (position as usize) >= task.prompt_tokens
                && (position as usize) - task.prompt_tokens < self.plan.max_generated_rows_per_task
        })
    }

    pub fn observe(&mut self, layer: usize, position: u32, values: &[f32]) -> Result<()> {
        ensure!(
            self.should_observe(position),
            "activation observation is outside generated-token capture range"
        );
        let source = self
            .layers
            .get(layer)
            .context("activation layer outside model")?;
        ensure!(
            values.len() == source.input_dim && values.iter().all(|value| value.is_finite()),
            "activation observation shape/nonfinite failure"
        );
        let task = self
            .active
            .as_mut()
            .context("begin a calibration task before observing")?;
        ensure!(
            position as usize == task.prompt_tokens + task.positions[layer].len(),
            "activation positions must be contiguous and increasing per layer"
        );
        self.observations[layer].extend_from_slice(values);
        task.positions[layer].push(position);
        Ok(())
    }

    pub fn finish_task(&mut self, generated_tokens: usize) -> Result<()> {
        let task = self.active.as_ref().context("no active calibration task")?;
        let count = generated_tokens
            .saturating_sub(1)
            .min(self.plan.max_generated_rows_per_task);
        ensure!(
            task.positions
                .iter()
                .all(|positions| positions.len() == count),
            "generated-token capture is incomplete or differs across layers"
        );
        let task = self.active.take().expect("validated active task");
        self.tasks.push(CalibrationTaskObservation {
            id: task.id,
            prompt_tokens: task.prompt_tokens,
            generated_tokens,
            captured_absolute_positions: task.positions[0].clone(),
        });
        Ok(())
    }

    pub fn finish(self) -> Result<ActivationCalibration> {
        ensure!(
            self.active.is_none() && self.tasks.len() == self.plan.calibration_task_ids.len(),
            "finish every calibration task before fitting thresholds"
        );
        let capacity = self
            .observations
            .iter()
            .map(|values| values.capacity() * 4)
            .sum();
        let payload = self
            .observations
            .iter()
            .map(|values| values.len() * 4)
            .sum();
        let mut layers = Vec::new();
        for (source, values) in self.layers.into_iter().zip(self.observations) {
            let mut thresholds = Vec::new();
            for target in TARGETS {
                let threshold = ActivationThreshold::calibrate(&values, target)?;
                let omitted = values
                    .iter()
                    .filter(|&&value| !threshold.retains(value))
                    .count();
                thresholds.push(LayerActivationThreshold {
                    target_sparsity: target,
                    magnitude: threshold.magnitude(),
                    achieved_sparsity: omitted as f64 / values.len() as f64,
                });
            }
            layers.push(LayerActivationCalibration {
                observed_rows: values.len() / source.input_dim,
                source,
                thresholds,
            });
        }
        let artifact = ActivationCalibration {
            schema_version: 1,
            plan: self.plan,
            source_weights_sha256: self.source_weights_sha256,
            tasks: self.tasks,
            layers,
            observation_payload_bytes: payload,
            observation_capacity_bytes: capacity,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

#[cfg(test)]
#[path = "calibration_tests.rs"]
mod tests;

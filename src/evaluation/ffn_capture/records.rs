use crate::{
    evaluation::corpus::{Corpus, Split},
    gguf::GgmlDType,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const MAX_JSON_BYTES: usize = 64 * 1024 * 1024;

pub(super) fn validate_digest(digest: &str) -> Result<()> {
    ensure!(
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "FFN capture identities require SHA-256 hex strings"
    );
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfnCapturePlan {
    pub model_sha256: String,
    pub corpus_sha256: String,
    pub execution_sha256: String,
    pub split: Split,
    pub calibration_task_ids: Vec<String>,
    pub source_layer_count: usize,
    pub selected_layer_ids: Vec<usize>,
    pub d_model: usize,
    pub hidden_dim: usize,
    pub max_generated_rows_per_task: usize,
}

impl FfnCapturePlan {
    /// Freeze the eligible calibration tasks and original layer IDs before any
    /// observations. Neither browser nor held-out inputs belong in this plan.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_sha256: String,
        corpus_sha256: String,
        execution_sha256: String,
        corpus: &Corpus,
        source_layer_count: usize,
        selected_layer_ids: Vec<usize>,
        d_model: usize,
        hidden_dim: usize,
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
            source_layer_count,
            selected_layer_ids,
            d_model,
            hidden_dim,
            max_generated_rows_per_task,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<()> {
        for digest in [
            &self.model_sha256,
            &self.corpus_sha256,
            &self.execution_sha256,
        ] {
            validate_digest(digest)?;
        }
        ensure!(
            self.split == Split::Calibration,
            "FFN capture is calibration-only"
        );
        ensure!(
            self.source_layer_count > 0
                && !self.selected_layer_ids.is_empty()
                && self
                    .selected_layer_ids
                    .iter()
                    .all(|&id| id < self.source_layer_count)
                && self
                    .selected_layer_ids
                    .windows(2)
                    .all(|ids| ids[0] < ids[1]),
            "FFN capture needs unique increasing original layer IDs"
        );
        ensure!(
            self.d_model > 0
                && self.d_model.is_multiple_of(32)
                && self.hidden_dim > 0
                && self.hidden_dim.is_multiple_of(32),
            "FFN capture requires complete Q8_0 input blocks"
        );
        ensure!(
            (1..=32).contains(&self.max_generated_rows_per_task),
            "FFN capture accepts 1..32 generated input rows per task"
        );
        let ids: HashSet<_> = self.calibration_task_ids.iter().collect();
        ensure!(
            !ids.is_empty()
                && ids.len() == self.calibration_task_ids.len()
                && ids.iter().all(|id| !id.is_empty()),
            "invalid FFN calibration task identities"
        );
        Ok(())
    }

    /// Parsing alone cannot establish provenance. Consumers also bind the plan
    /// to the actual frozen corpus file/hash they are authorized to use.
    pub fn validate_corpus(&self, corpus: &Corpus, corpus_sha256: &str) -> Result<()> {
        self.validate()?;
        corpus.validate()?;
        ensure!(
            self.corpus_sha256 == corpus_sha256
                && self.calibration_task_ids
                    == corpus
                        .calibration
                        .tasks
                        .iter()
                        .map(|task| task.id.clone())
                        .collect::<Vec<_>>(),
            "FFN capture does not match the frozen calibration corpus"
        );
        Ok(())
    }

    pub(super) fn max_rows(&self) -> Result<usize> {
        self.calibration_task_ids
            .len()
            .checked_mul(self.max_generated_rows_per_task)
            .context("FFN row capacity overflow")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FfnTensorSource {
    pub name: String,
    pub byte_offset: u64,
    pub dtype: GgmlDType,
    /// GGUF column blocks: [input dimension, output dimension].
    pub shape: [usize; 2],
    pub source_sha256: String,
}

impl FfnTensorSource {
    pub fn byte_len(&self) -> Result<usize> {
        ensure!(
            self.dtype == GgmlDType::Q8_0
                && self.shape[0] > 0
                && self.shape[0].is_multiple_of(32)
                && self.shape[1] > 0,
            "FFN source must be a complete Q8_0 matrix"
        );
        (self.shape[0] / 32)
            .checked_mul(self.shape[1])
            .and_then(|blocks| blocks.checked_mul(34))
            .context("FFN source byte overflow")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FfnLayerSource {
    pub layer_id: usize,
    pub gate: FfnTensorSource,
    pub up: FfnTensorSource,
    pub down: FfnTensorSource,
}

pub(super) fn validate_sources(plan: &FfnCapturePlan, sources: &[FfnLayerSource]) -> Result<()> {
    ensure!(
        sources.len() == plan.selected_layer_ids.len(),
        "FFN source layer count mismatch"
    );
    let mut names = HashSet::new();
    let mut ranges = Vec::new();
    for (source, &id) in sources.iter().zip(&plan.selected_layer_ids) {
        ensure!(
            source.layer_id == id,
            "FFN source original layer ID mismatch"
        );
        for (tensor, shape) in [
            (&source.gate, [plan.d_model, plan.hidden_dim]),
            (&source.up, [plan.d_model, plan.hidden_dim]),
            (&source.down, [plan.hidden_dim, plan.d_model]),
        ] {
            validate_digest(&tensor.source_sha256)?;
            ensure!(
                !tensor.name.is_empty() && names.insert(&tensor.name) && tensor.shape == shape,
                "FFN source name/shape mismatch"
            );
            let bytes = u64::try_from(tensor.byte_len()?)?;
            let end = tensor
                .byte_offset
                .checked_add(bytes)
                .context("FFN source range overflow")?;
            ranges.push((tensor.byte_offset, end));
        }
    }
    ranges.sort_unstable();
    ensure!(
        ranges.windows(2).all(|pair| pair[0].1 <= pair[1].0),
        "aliased FFN source tensors"
    );
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FfnRowPosition {
    pub task_id: String,
    pub absolute_position: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfnTaskObservation {
    pub id: String,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
}

impl FfnTaskObservation {
    pub(super) fn positions(&self, max_rows: usize) -> Result<Vec<FfnRowPosition>> {
        ensure!(
            !self.prompt_token_ids.is_empty()
                && self.prompt_token_ids.len() <= u32::MAX as usize
                && (1..=512).contains(&self.generated_token_ids.len()),
            "FFN task needs a prompt and 1..512 actual generated tokens"
        );
        let count = self
            .generated_token_ids
            .len()
            .saturating_sub(1)
            .min(max_rows);
        (0..count)
            .map(|index| {
                let position = self
                    .prompt_token_ids
                    .len()
                    .checked_add(index)
                    .context("FFN position overflow")?;
                Ok(FfnRowPosition {
                    task_id: self.id.clone(),
                    absolute_position: u32::try_from(position)?,
                })
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfnLayerSamples {
    pub source: FfnLayerSource,
    pub rows: Vec<FfnRowPosition>,
    /// Row-major FP32 inputs to gate/up, after the model's weighted RMSNorm.
    pub normalized_inputs: Vec<f32>,
    /// Row-major FP32 inputs to down, after the model's gate/up and SwiGLU.
    pub swiglu_inputs: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfnCaptureArtifact {
    pub schema_version: u32,
    pub plan: FfnCapturePlan,
    pub source_weights_sha256: String,
    pub tasks: Vec<FfnTaskObservation>,
    pub layers: Vec<FfnLayerSamples>,
    /// Only FP32 observation vectors, not JSON/metadata, allocator or RSS bytes.
    pub observation_payload_bytes: usize,
    pub observation_capacity_bytes: usize,
}

impl FfnCaptureArtifact {
    pub fn parse(json: &str) -> Result<Self> {
        ensure!(
            json.len() <= MAX_JSON_BYTES,
            "FFN capture JSON exceeds 64 MiB"
        );
        let artifact: Self = serde_json::from_str(json)?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema_version == 1, "unsupported FFN capture schema");
        self.plan.validate()?;
        validate_digest(&self.source_weights_sha256)?;
        validate_sources(
            &self.plan,
            &self
                .layers
                .iter()
                .map(|layer| layer.source.clone())
                .collect::<Vec<_>>(),
        )?;
        ensure!(
            self.tasks.len() == self.plan.calibration_task_ids.len(),
            "incomplete FFN capture tasks"
        );
        let mut rows = Vec::new();
        for (task, id) in self.tasks.iter().zip(&self.plan.calibration_task_ids) {
            ensure!(&task.id == id, "FFN capture task order/identity mismatch");
            rows.extend(task.positions(self.plan.max_generated_rows_per_task)?);
        }
        ensure!(!rows.is_empty(), "FFN capture has no generated input rows");
        let mut payload = 0usize;
        for layer in &self.layers {
            ensure!(
                layer.rows == rows,
                "FFN capture rows differ across layers/tasks"
            );
            for (values, dimension) in [
                (&layer.normalized_inputs, self.plan.d_model),
                (&layer.swiglu_inputs, self.plan.hidden_dim),
            ] {
                let count = rows
                    .len()
                    .checked_mul(dimension)
                    .context("FFN observation size overflow")?;
                ensure!(
                    values.len() == count && values.iter().all(|value| value.is_finite()),
                    "FFN observation shape/nonfinite failure"
                );
                payload = payload
                    .checked_add(count.checked_mul(4).context("FFN payload overflow")?)
                    .context("FFN payload overflow")?;
            }
        }
        ensure!(
            self.observation_payload_bytes == payload && self.observation_capacity_bytes >= payload,
            "invalid FFN observation memory accounting"
        );
        Ok(())
    }
}

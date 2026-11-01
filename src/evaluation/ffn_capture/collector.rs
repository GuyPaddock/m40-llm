use super::{
    records::{validate_digest, validate_sources},
    FfnCaptureArtifact, FfnCapturePlan, FfnLayerSamples, FfnLayerSource, FfnRowPosition,
    FfnTaskObservation,
};
use anyhow::{ensure, Context, Result};

struct ActiveTask {
    id: String,
    prompt_token_ids: Vec<u32>,
    layer_rows: Vec<usize>,
}

/// Observes paired vectors without fitting thresholds or changing model math.
/// All FP32 storage is reserved before capture; invalid rows publish no values.
pub struct FfnCaptureCollector {
    plan: FfnCapturePlan,
    source_weights_sha256: String,
    layers: Vec<FfnLayerSamples>,
    tasks: Vec<FfnTaskObservation>,
    active: Option<ActiveTask>,
    capacity_bytes: usize,
}

impl FfnCaptureCollector {
    pub fn new(
        plan: FfnCapturePlan,
        source_weights_sha256: String,
        sources: Vec<FfnLayerSource>,
        budget_bytes: usize,
    ) -> Result<Self> {
        plan.validate()?;
        validate_digest(&source_weights_sha256)?;
        validate_sources(&plan, &sources)?;
        let rows = plan.max_rows()?;
        let normalized_values = rows
            .checked_mul(plan.d_model)
            .context("FFN capacity overflow")?;
        let swiglu_values = rows
            .checked_mul(plan.hidden_dim)
            .context("FFN capacity overflow")?;
        let requested = normalized_values
            .checked_add(swiglu_values)
            .and_then(|count| count.checked_mul(sources.len()))
            .and_then(|count| count.checked_mul(4))
            .context("FFN observation budget overflow")?;
        ensure!(
            requested <= budget_bytes,
            "FFN observations need {requested} bytes, budget is {budget_bytes}"
        );
        let mut layers = Vec::new();
        let mut capacity_bytes = 0usize;
        for source in sources {
            let mut normalized_inputs = Vec::new();
            let mut swiglu_inputs = Vec::new();
            let mut positions = Vec::new();
            normalized_inputs.try_reserve_exact(normalized_values)?;
            swiglu_inputs.try_reserve_exact(swiglu_values)?;
            positions.try_reserve_exact(rows)?;
            let bytes = normalized_inputs
                .capacity()
                .checked_add(swiglu_inputs.capacity())
                .and_then(|count| count.checked_mul(4))
                .context("FFN capacity overflow")?;
            capacity_bytes = capacity_bytes
                .checked_add(bytes)
                .context("FFN capacity overflow")?;
            ensure!(
                capacity_bytes <= budget_bytes,
                "actual FFN observation capacity exceeds budget"
            );
            layers.push(FfnLayerSamples {
                source,
                rows: positions,
                normalized_inputs,
                swiglu_inputs,
            });
        }
        Ok(Self {
            plan,
            source_weights_sha256,
            layers,
            tasks: Vec::new(),
            active: None,
            capacity_bytes,
        })
    }

    pub fn begin_task(&mut self, id: &str, prompt_token_ids: &[u32]) -> Result<()> {
        ensure!(self.active.is_none(), "finish the active FFN task first");
        ensure!(
            self.plan
                .calibration_task_ids
                .get(self.tasks.len())
                .map(String::as_str)
                == Some(id),
            "capture only the next frozen calibration task, never held-out tasks"
        );
        ensure!(
            !prompt_token_ids.is_empty()
                && prompt_token_ids
                    .len()
                    .checked_add(self.plan.max_generated_rows_per_task)
                    .is_some_and(|end| end <= u32::MAX as usize),
            "FFN capture needs a nonempty prompt within u32 positions"
        );
        self.active = Some(ActiveTask {
            id: id.to_owned(),
            prompt_token_ids: prompt_token_ids.to_vec(),
            layer_rows: vec![0; self.layers.len()],
        });
        Ok(())
    }

    pub fn should_observe(&self, position: u32) -> bool {
        self.active.as_ref().is_some_and(|task| {
            (position as usize) >= task.prompt_token_ids.len()
                && (position as usize) - task.prompt_token_ids.len()
                    < self.plan.max_generated_rows_per_task
        })
    }

    pub fn observe(
        &mut self,
        layer_id: usize,
        position: u32,
        normalized: &[f32],
        swiglu: &[f32],
    ) -> Result<()> {
        ensure!(
            self.should_observe(position),
            "FFN observation is outside generated input range"
        );
        let index = self
            .plan
            .selected_layer_ids
            .binary_search(&layer_id)
            .map_err(|_| anyhow::anyhow!("FFN layer is not selected"))?;
        ensure!(
            normalized.len() == self.plan.d_model
                && swiglu.len() == self.plan.hidden_dim
                && normalized
                    .iter()
                    .chain(swiglu)
                    .all(|value| value.is_finite()),
            "FFN observation shape/nonfinite failure"
        );
        let task = self
            .active
            .as_mut()
            .context("begin an FFN task before observing")?;
        ensure!(
            position as usize == task.prompt_token_ids.len() + task.layer_rows[index],
            "FFN positions must be contiguous and increasing per original layer"
        );
        let layer = &mut self.layers[index];
        layer.normalized_inputs.extend_from_slice(normalized);
        layer.swiglu_inputs.extend_from_slice(swiglu);
        layer.rows.push(FfnRowPosition {
            task_id: task.id.clone(),
            absolute_position: position,
        });
        task.layer_rows[index] += 1;
        Ok(())
    }

    pub fn finish_task(&mut self, generated_token_ids: &[u32]) -> Result<()> {
        let task = self.active.as_ref().context("no active FFN task")?;
        ensure!(
            (1..=512).contains(&generated_token_ids.len()),
            "FFN task needs 1..512 actual generated tokens"
        );
        let count = generated_token_ids
            .len()
            .saturating_sub(1)
            .min(self.plan.max_generated_rows_per_task);
        ensure!(
            task.layer_rows.iter().all(|&rows| rows == count),
            "FFN generated input capture incomplete or differs across layers"
        );
        let task = self.active.take().expect("validated FFN task");
        self.tasks.push(FfnTaskObservation {
            id: task.id,
            prompt_token_ids: task.prompt_token_ids,
            generated_token_ids: generated_token_ids.to_vec(),
        });
        Ok(())
    }

    pub fn validate_complete(&self) -> Result<()> {
        ensure!(
            self.active.is_none() && self.tasks.len() == self.plan.calibration_task_ids.len(),
            "finish every FFN calibration task before export"
        );
        ensure!(
            self.layers.iter().all(|layer| !layer.rows.is_empty()),
            "FFN capture has no generated input rows"
        );
        Ok(())
    }

    pub fn finish(self) -> Result<FfnCaptureArtifact> {
        self.validate_complete()?;
        let payload = self.layers.iter().try_fold(0usize, |sum, layer| {
            layer
                .normalized_inputs
                .len()
                .checked_add(layer.swiglu_inputs.len())
                .and_then(|count| count.checked_mul(4))
                .and_then(|bytes| sum.checked_add(bytes))
                .context("FFN payload overflow")
        })?;
        let artifact = FfnCaptureArtifact {
            schema_version: 1,
            plan: self.plan,
            source_weights_sha256: self.source_weights_sha256,
            tasks: self.tasks,
            layers: self.layers,
            observation_payload_bytes: payload,
            observation_capacity_bytes: self.capacity_bytes,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

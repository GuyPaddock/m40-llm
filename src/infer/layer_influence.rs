use super::LoadedModel;
use crate::{
    cuda::CudaStream,
    evaluation::layer_ranking::{LayerInfluence, LayerInfluenceCollector, LayerRankingPlan},
};
use anyhow::{ensure, Context, Result};
use std::{ffi::c_void, sync::Mutex};

struct CaptureState {
    collector: LayerInfluenceCollector,
    input: Vec<f32>,
    output: Vec<f32>,
}

/// Host-only diagnostics owned by one model, never a process-global observer.
pub struct LayerInfluenceCapture {
    state: Mutex<CaptureState>,
}

impl LoadedModel {
    pub fn begin_layer_influence_capture(
        &mut self,
        plan: LayerRankingPlan,
        budget_bytes: usize,
    ) -> Result<()> {
        ensure!(
            self.q8_ffn.is_native() && self.q8_prefill.is_native(),
            "layer influence capture requires native Q8 FFN and prefill backends"
        );
        plan.validate()?;
        ensure!(
            self.layer_influence.is_none()
                && self.activation_research.is_none()
                && self.ffn_capture.is_none()
                && !self.layer_mask.has_skipped_layers()
                && self.kv_cache.is_none()
                && self.forward_workspace.lock().map_err(|_| anyhow::anyhow!("poisoned workspace"))?.is_none(),
            "configure influence capture before KV/workspace/session creation, without other model transforms"
        );
        ensure!(
            plan.layer_count == self.model_config.block_count
                && plan.hidden_dim == self.model_config.embedding_length as usize,
            "influence plan dimensions differ from the model"
        );
        let props = self.cuda.current_device_props()?;
        ensure!(
            props.major == 5 && props.minor == 2,
            "layer influence study requires sm_52"
        );
        let scratch_bytes = plan
            .hidden_dim
            .checked_mul(8)
            .context("influence scratch overflow")?;
        let observation_budget = budget_bytes
            .checked_sub(scratch_bytes)
            .context("influence scratch exceeds host budget")?;
        let dimension = plan.hidden_dim;
        let collector = LayerInfluenceCollector::new(plan, observation_budget)?;
        let mut input = Vec::new();
        let mut output = Vec::new();
        input.try_reserve_exact(dimension)?;
        output.try_reserve_exact(dimension)?;
        input.resize(dimension, 0.0);
        output.resize(dimension, 0.0);
        self.layer_influence = Some(LayerInfluenceCapture {
            state: Mutex::new(CaptureState {
                collector,
                input,
                output,
            }),
        });
        Ok(())
    }

    pub fn begin_layer_influence_passage(&mut self, id: &str) -> Result<()> {
        self.layer_influence
            .as_ref()
            .context("enable layer influence capture first")?
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned influence capture"))?
            .collector
            .begin_passage(id)
    }

    pub fn finish_layer_influence_passage(&mut self) -> Result<()> {
        self.layer_influence
            .as_ref()
            .context("enable layer influence capture first")?
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned influence capture"))?
            .collector
            .finish_passage()
    }

    pub fn finish_layer_influence_capture(&mut self) -> Result<LayerInfluence> {
        self.layer_influence
            .as_ref()
            .context("enable layer influence capture first")?
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned influence capture"))?
            .collector
            .validate_complete()?;
        self.layer_influence
            .take()
            .expect("validated influence capture")
            .state
            .into_inner()
            .map_err(|_| anyhow::anyhow!("poisoned influence capture"))?
            .collector
            .finish()
    }

    /// Both hidden vectors stay alive in the ordinary forward workspace. Drain
    /// their producers before reading, without modifying either device buffer.
    pub(crate) unsafe fn observe_layer_influence(
        &self,
        layer: usize,
        position: u32,
        input_pointer: *const c_void,
        output_pointer: *const c_void,
    ) -> Result<()> {
        let Some(capture) = &self.layer_influence else {
            return Ok(());
        };
        let mut state = capture
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned influence capture"))?;
        if !state.collector.has_active_passage() {
            return Ok(());
        }
        self.cuda.synchronize_stream(CudaStream::Decode)?;
        self.cuda.synchronize_stream(CudaStream::Prefill)?;
        let CaptureState {
            collector,
            input,
            output,
        } = &mut *state;
        let bytes = input.len() * 4;
        unsafe {
            self.cuda
                .memcpy_d2h(input.as_mut_ptr().cast(), input_pointer, bytes)?;
            self.cuda
                .memcpy_d2h(output.as_mut_ptr().cast(), output_pointer, bytes)?;
        }
        collector.observe(layer, position, input, output)
    }
}

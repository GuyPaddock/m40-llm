use super::{DeviceTensorView, LoadedModel};
use crate::{
    cuda::CudaStream,
    evaluation::ffn_capture::{
        FfnCaptureArtifact, FfnCaptureCollector, FfnCapturePlan, FfnLayerSource, FfnTensorSource,
    },
    gguf::GgmlDType,
};
use anyhow::{ensure, Context, Result};
use std::{collections::HashMap, ffi::c_void, sync::Mutex};

fn digest(bytes: &[u8]) -> String {
    ring::digest::digest(&ring::digest::SHA256, bytes)
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

struct CaptureState {
    collector: FfnCaptureCollector,
    normalized: Vec<f32>,
    swiglu: Vec<f32>,
}

/// Read-only observations owned by a model, not an execution-weight policy or
/// a process-global callback. No device allocation belongs to this observer.
pub struct FfnCapture {
    layers_by_gate: HashMap<usize, usize>,
    d_model: usize,
    hidden_dim: usize,
    state: Mutex<CaptureState>,
}

impl LoadedModel {
    fn ffn_capture_source(&self, view: &DeviceTensorView) -> Result<FfnTensorSource> {
        ensure!(
            view.dtype == GgmlDType::Q8_0 && view.shape.len() == 2,
            "FFN capture requires Q8_0 projection weights"
        );
        self.tensor_device_ptr("FFN capture source", view)?;
        let mut matching = self.gguf.tensors.iter().filter(|tensor| {
            tensor.offset == view.byte_offset
                && tensor.dtype == view.dtype
                && tensor.shape == view.shape
        });
        let tensor = matching.next().context("missing FFN source identity")?;
        ensure!(matching.next().is_none(), "ambiguous FFN source identity");
        let mut source = FfnTensorSource {
            name: tensor.name.clone(),
            byte_offset: tensor.offset,
            dtype: tensor.dtype,
            shape: [
                usize::try_from(tensor.shape[0])?,
                usize::try_from(tensor.shape[1])?,
            ],
            source_sha256: String::new(),
        };
        let begin = usize::try_from(source.byte_offset)?;
        let end = begin
            .checked_add(source.byte_len()?)
            .context("FFN source range overflow")?;
        source.source_sha256 = digest(
            self.host_weights
                .get(begin..end)
                .context("FFN source outside uploaded weight bytes")?,
        );
        Ok(source)
    }

    /// Configure before KV/workspace creation. Unlike sparsity calibration,
    /// this read-only observer permits the qualified prepared down/vocabulary
    /// control. Caller binds the original model/corpus/execution hashes.
    pub fn begin_ffn_capture(&mut self, plan: FfnCapturePlan, budget_bytes: usize) -> Result<()> {
        ensure!(
            self.q8_ffn.is_native() && self.q8_prefill.is_native(),
            "frozen FFN capture requires native Q8 FFN and prefill backends"
        );
        plan.validate()?;
        ensure!(
            self.ffn_capture.is_none()
                && self.activation_research.is_none()
                && self.layer_influence.is_none()
                && !self.layer_mask.has_skipped_layers()
                && self.kv_cache.is_none()
                && self.forward_workspace.lock().map_err(|_| anyhow::anyhow!("poisoned workspace"))?.is_none(),
            "configure FFN capture before KV/workspace/session creation, without other model transforms"
        );
        ensure!(
            plan.source_layer_count == self.model_config.block_count as usize
                && plan.d_model == self.model_config.embedding_length as usize
                && plan.hidden_dim == self.model_config.feed_forward_length as usize,
            "FFN capture plan dimensions differ from the model"
        );
        ensure!(
            super::gemm::q8_decode_kernel_decode_stream_enabled(),
            "FFN capture requires M40LLM_Q8_DECODE_STREAM=decode"
        );
        let props = self.cuda.current_device_props()?;
        ensure!(
            props.major == 5 && props.minor == 2,
            "FFN capture requires sm_52"
        );
        let mut sources = Vec::new();
        let mut layers_by_gate = HashMap::new();
        for &layer_id in &plan.selected_layer_ids {
            let mapped = self.map_standard_layer(layer_id)?;
            ensure!(
                mapped.d_model == plan.d_model && mapped.hidden_dim == plan.hidden_dim,
                "FFN capture layer dimensions differ"
            );
            let pointer = self.tensor_device_ptr("FFN gate", &mapped.w_gate)? as usize;
            ensure!(
                layers_by_gate.insert(pointer, layer_id).is_none(),
                "aliased FFN gate pointers"
            );
            sources.push(FfnLayerSource {
                layer_id,
                gate: self.ffn_capture_source(&mapped.w_gate)?,
                up: self.ffn_capture_source(&mapped.w_up)?,
                down: self.ffn_capture_source(&mapped.w_down)?,
            });
        }
        let scratch_bytes = plan
            .d_model
            .checked_add(plan.hidden_dim)
            .and_then(|values| values.checked_mul(4))
            .context("FFN capture scratch overflow")?;
        let observation_budget = budget_bytes
            .checked_sub(scratch_bytes)
            .context("FFN read scratch exceeds host budget")?;
        let mut normalized = Vec::new();
        let mut swiglu = Vec::new();
        normalized.try_reserve_exact(plan.d_model)?;
        swiglu.try_reserve_exact(plan.hidden_dim)?;
        normalized.resize(plan.d_model, 0.0);
        swiglu.resize(plan.hidden_dim, 0.0);
        ensure!(
            normalized.capacity() == plan.d_model && swiglu.capacity() == plan.hidden_dim,
            "FFN read scratch capacity exceeds budgeted dimensions"
        );
        let d_model = plan.d_model;
        let hidden_dim = plan.hidden_dim;
        let collector = FfnCaptureCollector::new(
            plan,
            digest(&self.host_weights),
            sources,
            observation_budget,
        )?;
        self.ffn_capture = Some(FfnCapture {
            layers_by_gate,
            d_model,
            hidden_dim,
            state: Mutex::new(CaptureState {
                collector,
                normalized,
                swiglu,
            }),
        });
        Ok(())
    }

    pub fn begin_ffn_capture_task(&mut self, id: &str, prompt_token_ids: &[u32]) -> Result<()> {
        self.ffn_capture
            .as_mut()
            .context("enable FFN capture first")?
            .state
            .get_mut()
            .map_err(|_| anyhow::anyhow!("poisoned FFN capture"))?
            .collector
            .begin_task(id, prompt_token_ids)
    }

    pub fn finish_ffn_capture_task(&mut self, generated_token_ids: &[u32]) -> Result<()> {
        self.ffn_capture
            .as_mut()
            .context("enable FFN capture first")?
            .state
            .get_mut()
            .map_err(|_| anyhow::anyhow!("poisoned FFN capture"))?
            .collector
            .finish_task(generated_token_ids)
    }

    pub fn finish_ffn_capture(&mut self) -> Result<FfnCaptureArtifact> {
        self.ffn_capture
            .as_ref()
            .context("enable FFN capture first")?
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned FFN capture"))?
            .collector
            .validate_complete()?;
        self.cuda.synchronize_stream(CudaStream::Decode)?;
        self.cuda.synchronize_stream(CudaStream::Prefill)?;
        self.ffn_capture
            .take()
            .expect("validated FFN capture")
            .state
            .into_inner()
            .map_err(|_| anyhow::anyhow!("poisoned FFN capture"))?
            .collector
            .finish()
    }

    pub(crate) fn validate_ffn_capture_forward(
        &self,
        gate: *const c_void,
        slot: u32,
        d_model: i32,
        hidden_dim: i32,
    ) -> Result<()> {
        let Some(capture) = &self.ffn_capture else {
            return Ok(());
        };
        ensure!(
            d_model as usize == capture.d_model && hidden_dim as usize == capture.hidden_dim,
            "FFN capture forward dimensions differ from the plan"
        );
        let kv = self
            .kv_cache
            .as_ref()
            .context("FFN capture requires allocated dense KV")?;
        ensure!(
            !kv.is_compressed() && slot < self.model_config.block_count,
            "FFN capture requires explicit dense KV and sequence 0"
        );
        // Sequence-major addressing maps sequence zero's original layers to
        // their own physical slots. Reject direct low-level calls to other slots.
        if let Some(&layer) = capture.layers_by_gate.get(&(gate as usize)) {
            ensure!(
                slot == layer as u32,
                "FFN capture requires sequence 0 and matching original layer slot"
            );
        }
        Ok(())
    }

    /// Both buffers are still live immediately before down projection. The
    /// decode stream produces normalized input and completed SwiGLU; drain once
    /// before two read-only copies. Never run this inside a timed benchmark.
    pub(crate) unsafe fn observe_ffn_inputs(
        &self,
        gate: *const c_void,
        position: u32,
        normalized_pointer: *const c_void,
        swiglu_pointer: *const c_void,
    ) -> Result<()> {
        let Some(capture) = &self.ffn_capture else {
            return Ok(());
        };
        let Some(&layer) = capture.layers_by_gate.get(&(gate as usize)) else {
            return Ok(());
        };
        let mut state = capture
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned FFN capture"))?;
        if !state.collector.should_observe(position) {
            return Ok(());
        }
        self.cuda.synchronize_stream(CudaStream::Decode)?;
        let CaptureState {
            collector,
            normalized,
            swiglu,
        } = &mut *state;
        unsafe {
            self.cuda.memcpy_d2h(
                normalized.as_mut_ptr().cast(),
                normalized_pointer,
                normalized.len() * 4,
            )?;
            self.cuda
                .memcpy_d2h(swiglu.as_mut_ptr().cast(), swiglu_pointer, swiglu.len() * 4)?;
        }
        collector.observe(layer, position, normalized, swiglu)
    }
}

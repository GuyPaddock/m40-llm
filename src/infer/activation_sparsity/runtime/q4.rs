use super::{digest, ActivationResearch, ActivationResearchStats, Mode};
use crate::{
    cuda::{DeviceQ4RepackedDown, Q4RepackedWorkspace},
    evaluation::ffn_capture::FfnTensorSource,
    gguf::GgmlDType,
    infer::{
        activation_sparsity::ActivationThreshold,
        execution_weights::Q8ExecutionMode,
        q4_channels::{
            Q4ChannelWeights, Q4ResearchLayout, Q4_REPACKED_INPUT_DIM, Q4_REPACKED_OUTPUT_DIM,
        },
        LoadedModel,
    },
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::{collections::HashMap, ffi::c_void, sync::Mutex, time::Instant};

const LAYERS: [usize; 3] = [0, 17, 35];
const SHAPE: [usize; 2] = [Q4_REPACKED_INPUT_DIM, Q4_REPACKED_OUTPUT_DIM];
const DOWN_BYTES: usize = Q4_REPACKED_INPUT_DIM / 32 * Q4_REPACKED_OUTPUT_DIM * 18;
const WORKSPACE_BYTES: usize = Q4_REPACKED_INPUT_DIM * 8 + 4 + Q4_REPACKED_OUTPUT_DIM * 32 * 4;

/// One provenance-bound layer of the frozen Q4 quality experiment.
pub struct Q4ResearchCutoff {
    pub layer_id: usize,
    pub down: FfnTensorSource,
    pub threshold: ActivationThreshold,
}

#[derive(Serialize)]
pub struct Q4ResearchStats {
    layout: Q4ResearchLayout,
    layers: [usize; 3],
    thresholds: [f32; 3],
    projection_calls: [u64; 3],
    repack_host_payload_bytes: usize,
    preparation_ms: f64,
}

pub(super) struct Q4Research {
    layout: Q4ResearchLayout,
    targets: Vec<Q4Target>,
    workspace: Q4RepackedWorkspace,
    preparation_ms: f64,
}

struct Q4Target {
    original_pointer: usize,
    threshold: ActivationThreshold,
    weights: DeviceQ4RepackedDown,
    projection_calls: u64,
}

impl LoadedModel {
    /// Configure the three admitted Q4 down projections before creating KV or
    /// forward scratch. Dense and masked controls use the same selected layout.
    pub fn configure_q4_research(
        &mut self,
        layout: Q4ResearchLayout,
        source_weights_sha256: &str,
        cutoffs: &[Q4ResearchCutoff; 3],
        budget_bytes: usize,
    ) -> Result<()> {
        let began = Instant::now();
        ensure!(
            self.activation_research.is_none()
                && self.ffn_capture.is_none()
                && self.layer_influence.is_none()
                && self.kv_cache.is_none()
                && !self.layer_mask.has_skipped_layers()
                && self
                    .forward_workspace
                    .lock()
                    .map_err(|_| anyhow::anyhow!("poisoned workspace"))?
                    .is_none(),
            "configure sparse Q4 before KV or workspaces with no other research transforms"
        );
        ensure!(
            cfg!(nvcc)
                && self.q8_ffn.is_native()
                && self.q8_prefill.is_native()
                && self.q8_execution_stats().mode == Q8ExecutionMode::Off
                && super::super::super::gemm::q8_decode_kernel_decode_stream_enabled(),
            "sparse Q4 requires native projections, decode stream, and a real CUDA build"
        );
        let props = self.cuda.current_device_props()?;
        ensure!(
            props.major == 5 && props.minor == 2,
            "sparse Q4 requires sm_52"
        );
        ensure!(
            self.model_config.block_count == 36
                && source_weights_sha256 == digest(&self.host_weights)
                && cutoffs.each_ref().map(|cutoff| cutoff.layer_id) == LAYERS,
            "sparse Q4 source or layer set differs from the frozen capture"
        );
        let is_thresholded = cutoffs[0].threshold.magnitude() > 0.0;
        ensure!(
            cutoffs
                .iter()
                .all(|cutoff| (cutoff.threshold.magnitude() > 0.0) == is_thresholded),
            "sparse Q4 requires one selection mode across the three projections"
        );
        ensure!(
            budget_bytes >= LAYERS.len() * DOWN_BYTES + WORKSPACE_BYTES,
            "sparse Q4 device budget is too small"
        );
        let mut original = Vec::with_capacity(LAYERS.len());
        for cutoff in cutoffs {
            let mapped = self.map_standard_layer(cutoff.layer_id)?;
            let view = &mapped.w_down;
            ensure!(
                view.dtype == GgmlDType::Q4_0
                    && cutoff.down.shape == SHAPE
                    && cutoff.down == self.ffn_capture_source(view)?,
                "sparse Q4 down tensor differs from the frozen capture at layer {}",
                cutoff.layer_id
            );
            let pointer = self.tensor_device_ptr("sparse Q4 down", view)? as usize;
            ensure!(
                !original.contains(&pointer),
                "aliased sparse Q4 down tensor"
            );
            original.push(pointer);
        }
        let mut targets = Vec::with_capacity(LAYERS.len());
        for (cutoff, original_pointer) in cutoffs.iter().zip(original) {
            let begin = usize::try_from(cutoff.down.byte_offset)?;
            let end = begin
                .checked_add(DOWN_BYTES)
                .context("Q4 source range overflow")?;
            let bytes = self
                .host_weights
                .get(begin..end)
                .context("sparse Q4 source outside model")?;
            let host = Q4ChannelWeights::pack(bytes, SHAPE[0], SHAPE[1])?;
            let weights = DeviceQ4RepackedDown::upload(&self.cuda, &host, layout)?;
            ensure!(
                weights.bytes() == DOWN_BYTES,
                "unexpected repacked Q4 payload"
            );
            targets.push(Q4Target {
                original_pointer,
                threshold: cutoff.threshold,
                weights,
                projection_calls: 0,
            });
        }
        let workspace = Q4RepackedWorkspace::new(&self.cuda, is_thresholded)?;
        self.activation_research = Some(ActivationResearch {
            sources: Vec::new(),
            pointers: HashMap::new(),
            mode: Mutex::new(Mode::ApplyQ4(Q4Research {
                layout,
                targets,
                workspace,
                preparation_ms: began.elapsed().as_secs_f64() * 1000.0,
            })),
        });
        Ok(())
    }

    pub fn q4_research_stats(&self) -> Result<Option<Q4ResearchStats>> {
        let Some(research) = &self.activation_research else {
            return Ok(None);
        };
        let state = research
            .mode
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned activation state"))?;
        let Mode::ApplyQ4(state) = &*state else {
            return Ok(None);
        };
        Ok(Some(Q4ResearchStats {
            layout: state.layout,
            layers: LAYERS,
            thresholds: std::array::from_fn(|index| state.targets[index].threshold.magnitude()),
            projection_calls: std::array::from_fn(|index| state.targets[index].projection_calls),
            repack_host_payload_bytes: DOWN_BYTES
                * if matches!(state.layout, Q4ResearchLayout::Zigzag) {
                    2
                } else {
                    1
                },
            preparation_ms: state.preparation_ms,
        }))
    }
}

impl Q4Research {
    pub(super) fn memory_stats(&self) -> ActivationResearchStats {
        ActivationResearchStats {
            is_enabled: true,
            thresholds: self
                .targets
                .iter()
                .map(|target| target.threshold.magnitude())
                .collect(),
            extra_weight_bytes: self
                .targets
                .iter()
                .map(|target| target.weights.bytes())
                .sum(),
            selection_workspace_bytes: self.workspace.bytes(),
            preparation_elapsed_ms: self.preparation_ms,
            ..Default::default()
        }
    }

    pub(super) unsafe fn try_project(
        &mut self,
        source: *const c_void,
        input: *const c_void,
        output: *mut c_void,
        shape: [usize; 2],
    ) -> Result<bool> {
        let Some(target) = self
            .targets
            .iter_mut()
            .find(|target| target.original_pointer == source as usize)
        else {
            return Ok(false);
        };
        ensure!(shape == SHAPE, "sparse Q4 down shape changed");
        self.workspace
            .enqueue(&target.weights, input, output, target.threshold)?;
        target.projection_calls = target.projection_calls.saturating_add(1);
        Ok(true)
    }
}

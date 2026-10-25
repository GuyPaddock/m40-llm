use super::{
    ActivationCalibration, ActivationCalibrationCollector, ActivationCalibrationPlan,
    ActivationThreshold, CalibrationLayer, DeviceSparseInputWorkspace,
};
use crate::{
    cuda::{CudaStream, DeviceBuffer},
    gguf::GgmlDType,
    infer::{
        execution_weights::{
            DeviceQ8ExecutionWeight, HostQ8ExecutionWeight, Q8ExecutionMode, Q8SourceIdentity,
            Q8SourceView, Q8TiledLaunch,
        },
        LoadedModel,
    },
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::{collections::HashMap, ffi::c_void, sync::Mutex, time::Instant};

fn digest(bytes: &[u8]) -> String {
    ring::digest::digest(&ring::digest::SHA256, bytes)
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn parse_digest(hex: &str) -> Result<[u8; 32]> {
    ensure!(hex.len() == 64, "invalid activation model hash");
    let mut hash = [0u8; 32];
    for (index, value) in hash.iter_mut().enumerate() {
        *value = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16)?;
    }
    Ok(hash)
}

#[derive(Default, Debug, Serialize)]
pub struct ActivationLayerStats {
    pub layer_id: usize,
    pub retained_values: u64,
    pub input_values: u64,
    pub achieved_sparsity: Option<f64>,
}

#[derive(Default, Debug, Serialize)]
pub struct ActivationResearchStats {
    pub is_enabled: bool,
    pub is_capturing: bool,
    pub target_sparsity: Option<f64>,
    pub thresholds: Vec<f32>,
    pub native_bypass_layer_ids: Vec<usize>,
    pub extra_weight_bytes: usize,
    pub selection_workspace_bytes: usize,
    pub counter_bytes: usize,
    pub preparation_elapsed_ms: f64,
    pub layers: Vec<ActivationLayerStats>,
}

enum Mode {
    Capture {
        collector: ActivationCalibrationCollector,
        row: Vec<f32>,
    },
    Apply {
        weights: Vec<Option<DeviceQ8ExecutionWeight>>,
        thresholds: Vec<ActivationThreshold>,
        scratch: Option<DeviceSparseInputWorkspace>,
        totals: Option<DeviceBuffer>,
        target: f64,
        preparation_ms: f64,
    },
}

/// Configure through LoadedModel before KV/workspace creation. No process-global
/// policy, callback or threshold state participates in forward dispatch.
pub struct ActivationResearch {
    sources: Vec<CalibrationLayer>,
    pointers: HashMap<usize, usize>,
    mode: Mutex<Mode>,
}

impl LoadedModel {
    fn activation_sources(&self) -> Result<(Vec<CalibrationLayer>, HashMap<usize, usize>)> {
        ensure!(
            self.activation_research.is_none()
                && self.kv_cache.is_none()
                && self
                    .forward_workspace
                    .lock()
                    .map_err(|_| anyhow::anyhow!("poisoned workspace"))?
                    .is_none(),
            "configure activation research before KV/workspace/session creation"
        );
        ensure!(
            !self.layer_mask.has_skipped_layers()
                && self.q8_execution_stats().mode == Q8ExecutionMode::Off,
            "activation research cannot combine layer masks or other prepared-weight policies"
        );
        ensure!(
            super::super::gemm::q8_decode_kernel_decode_stream_enabled(),
            "activation research requires M40LLM_Q8_DECODE_STREAM=decode"
        );
        let props = self.cuda.current_device_props()?;
        ensure!(
            props.major == 5 && props.minor == 2,
            "activation research requires sm_52"
        );
        let mut sources = Vec::new();
        let mut pointers = HashMap::new();
        for layer in 0..self.model_config.block_count as usize {
            let mapped = self.map_standard_layer(layer)?;
            let down = &mapped.w_down;
            ensure!(
                down.dtype == GgmlDType::Q8_0 && mapped.hidden_dim.is_multiple_of(32),
                "activation research requires Q8_0 down projections with complete input blocks"
            );
            let pointer = self.tensor_device_ptr("activation down", down)? as usize;
            let mut matching = self.gguf.tensors.iter().filter(|tensor| {
                tensor.offset == down.byte_offset
                    && tensor.shape == down.shape
                    && tensor.dtype == down.dtype
            });
            let tensor = matching
                .next()
                .context("missing activation down source identity")?;
            ensure!(
                matching.next().is_none() && pointers.insert(pointer, layer).is_none(),
                "aliased activation down tensors are unsupported"
            );
            sources.push(CalibrationLayer {
                layer_id: layer,
                tensor_name: tensor.name.clone(),
                byte_offset: tensor.offset,
                input_dim: mapped.hidden_dim,
                output_dim: mapped.d_model,
            });
        }
        Ok((sources, pointers))
    }

    pub fn begin_activation_calibration(
        &mut self,
        plan: ActivationCalibrationPlan,
        budget_bytes: usize,
    ) -> Result<()> {
        let (sources, pointers) = self.activation_sources()?;
        let row = vec![0.0; self.model_config.feed_forward_length as usize];
        let collector = ActivationCalibrationCollector::new(
            plan,
            digest(&self.host_weights),
            sources.clone(),
            budget_bytes,
        )?;
        self.activation_research = Some(ActivationResearch {
            sources,
            pointers,
            mode: Mutex::new(Mode::Capture { collector, row }),
        });
        Ok(())
    }

    pub fn begin_activation_calibration_task(
        &mut self,
        id: &str,
        prompt_tokens: usize,
    ) -> Result<()> {
        let research = self
            .activation_research
            .as_mut()
            .context("enable activation capture first")?;
        match research
            .mode
            .get_mut()
            .map_err(|_| anyhow::anyhow!("poisoned activation state"))?
        {
            Mode::Capture { collector, .. } => collector.begin_task(id, prompt_tokens),
            Mode::Apply { .. } => anyhow::bail!("cannot calibrate an applied activation variant"),
        }
    }

    pub fn finish_activation_calibration_task(&mut self, generated_tokens: usize) -> Result<()> {
        let research = self
            .activation_research
            .as_mut()
            .context("enable activation capture first")?;
        match research
            .mode
            .get_mut()
            .map_err(|_| anyhow::anyhow!("poisoned activation state"))?
        {
            Mode::Capture { collector, .. } => collector.finish_task(generated_tokens),
            Mode::Apply { .. } => anyhow::bail!("cannot calibrate an applied activation variant"),
        }
    }

    pub fn finish_activation_calibration(&mut self) -> Result<ActivationCalibration> {
        let research = self
            .activation_research
            .as_mut()
            .context("enable activation capture first")?;
        ensure!(
            matches!(
                research
                    .mode
                    .get_mut()
                    .map_err(|_| anyhow::anyhow!("poisoned activation state"))?,
                Mode::Capture { .. }
            ),
            "cannot export applied activation state as calibration"
        );
        self.cuda.synchronize_stream(CudaStream::Decode)?;
        self.cuda.synchronize_stream(CudaStream::Prefill)?;
        let state = self
            .activation_research
            .take()
            .context("enable activation capture first")?;
        match state
            .mode
            .into_inner()
            .map_err(|_| anyhow::anyhow!("poisoned activation state"))?
        {
            Mode::Capture { collector, .. } => collector.finish(),
            Mode::Apply { .. } => {
                anyhow::bail!("cannot export applied activation state as calibration")
            }
        }
    }

    /// Freeze the checked artifact before session creation. Caller verifies the
    /// original GGUF file hash; uploaded-source hash and every down identity are
    /// independently checked here. The budget counts duplicate device payloads.
    pub fn apply_activation_calibration(
        &mut self,
        artifact: &ActivationCalibration,
        target: f64,
        budget_bytes: usize,
    ) -> Result<()> {
        let began = Instant::now();
        let thresholds = artifact.thresholds(target)?;
        let (sources, pointers) = self.activation_sources()?;
        ensure!(
            artifact.source_weights_sha256 == digest(&self.host_weights)
                && sources
                    == artifact
                        .layers
                        .iter()
                        .map(|layer| layer.source.clone())
                        .collect::<Vec<_>>(),
            "activation artifact does not match uploaded model weights/layers"
        );
        let model_hash = parse_digest(&artifact.plan.model_sha256)?;
        let mut identities = Vec::new();
        let should_prepare = thresholds
            .iter()
            .any(|threshold| threshold.magnitude() > 0.0);
        let mut estimated = if should_prepare {
            self.model_config.feed_forward_length as usize * 8 + 4 + sources.len() * 16
        } else {
            0
        };
        for (source, threshold) in sources.iter().zip(&thresholds) {
            if threshold.magnitude() == 0.0 {
                identities.push(None);
                continue;
            }
            let tensor = self
                .gguf
                .tensors
                .iter()
                .find(|tensor| tensor.name == source.tensor_name)
                .context("activation tensor disappeared")?;
            let identity = Q8SourceIdentity::from_gguf_tensor(model_hash, tensor)?;
            let plan =
                super::super::execution_weights::Q8ExecutionLayout::InputTiled { columns: 16 }
                    .allocation_plan(identity.shape())?;
            estimated = estimated
                .checked_add(plan.resident_bytes())
                .context("activation cache overflow")?;
            ensure!(
                estimated <= budget_bytes,
                "activation prepared payload exceeds {budget_bytes} byte budget"
            );
            identities.push(Some(identity));
        }
        let mut weights = Vec::new();
        for identity in &identities {
            let Some(identity) = identity else {
                weights.push(None);
                continue;
            };
            let view = Q8SourceView::new(identity, &self.host_weights)?;
            let host = HostQ8ExecutionWeight::tile_inputs(view, 16)?;
            weights.push(Some(DeviceQ8ExecutionWeight::upload(&self.cuda, &host)?));
        }
        let (scratch, totals) = if should_prepare {
            let scratch = DeviceSparseInputWorkspace::new(
                &self.cuda,
                self.model_config.feed_forward_length as usize,
            )?;
            let totals =
                DeviceBuffer::new_tagged(&self.cuda, sources.len() * 16, "activation:totals")?;
            let zeros = vec![0u64; sources.len() * 2];
            unsafe {
                self.cuda
                    .memcpy_h2d(totals.as_mut_ptr(), zeros.as_ptr().cast(), totals.bytes())?;
            }
            (Some(scratch), Some(totals))
        } else {
            (None, None)
        };
        self.activation_research = Some(ActivationResearch {
            sources,
            pointers,
            mode: Mutex::new(Mode::Apply {
                weights,
                thresholds,
                scratch,
                totals,
                target,
                preparation_ms: began.elapsed().as_secs_f64() * 1000.0,
            }),
        });
        Ok(())
    }

    pub fn reset_activation_statistics(&self) -> Result<()> {
        let Some(research) = &self.activation_research else {
            return Ok(());
        };
        let state = research
            .mode
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned activation state"))?;
        if let Mode::Apply {
            totals: Some(totals),
            ..
        } = &*state
        {
            self.cuda.synchronize_stream(CudaStream::Decode)?;
            let zeros = vec![0u64; totals.bytes() / 8];
            unsafe {
                self.cuda
                    .memcpy_h2d(totals.as_mut_ptr(), zeros.as_ptr().cast(), totals.bytes())?;
            }
        }
        Ok(())
    }

    /// Cold/report-only counters; not called within measured generation enqueue.
    pub fn activation_research_stats(&self) -> Result<ActivationResearchStats> {
        let Some(research) = &self.activation_research else {
            return Ok(ActivationResearchStats::default());
        };
        let state = research
            .mode
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned activation state"))?;
        match &*state {
            Mode::Capture { .. } => Ok(ActivationResearchStats {
                is_enabled: true,
                is_capturing: true,
                ..Default::default()
            }),
            Mode::Apply {
                weights,
                thresholds,
                scratch,
                totals,
                target,
                preparation_ms,
            } => {
                let mut counts = vec![0u64; research.sources.len() * 2];
                if let Some(totals) = totals {
                    self.cuda.synchronize_stream(CudaStream::Decode)?;
                    unsafe {
                        self.cuda.memcpy_d2h(
                            counts.as_mut_ptr().cast(),
                            totals.as_ptr(),
                            totals.bytes(),
                        )?;
                    }
                }
                Ok(ActivationResearchStats {
                    is_enabled: true,
                    is_capturing: false,
                    target_sparsity: Some(*target),
                    thresholds: thresholds
                        .iter()
                        .map(|threshold| threshold.magnitude())
                        .collect(),
                    native_bypass_layer_ids: thresholds
                        .iter()
                        .enumerate()
                        .filter_map(|(layer, threshold)| {
                            (threshold.magnitude() == 0.0).then_some(layer)
                        })
                        .collect(),
                    extra_weight_bytes: weights
                        .iter()
                        .flatten()
                        .map(|weight| weight.plan().resident_bytes())
                        .sum(),
                    selection_workspace_bytes: scratch
                        .as_ref()
                        .map_or(0, DeviceSparseInputWorkspace::bytes),
                    counter_bytes: totals.as_ref().map_or(0, DeviceBuffer::bytes),
                    preparation_elapsed_ms: *preparation_ms,
                    layers: counts
                        .chunks_exact(2)
                        .enumerate()
                        .map(|(layer, counts)| ActivationLayerStats {
                            layer_id: layer,
                            retained_values: counts[0],
                            input_values: counts[1],
                            achieved_sparsity: (counts[1] != 0)
                                .then(|| 1.0 - counts[0] as f64 / counts[1] as f64),
                        })
                        .collect(),
                })
            }
        }
    }

    pub(crate) unsafe fn try_activation_down(
        &self,
        source: *const c_void,
        input: *const c_void,
        output: *mut c_void,
        h: usize,
        n: usize,
        position: u32,
    ) -> Result<bool> {
        let Some(research) = &self.activation_research else {
            return Ok(false);
        };
        ensure!(
            self.kv_cache
                .as_ref()
                .is_some_and(|cache| !cache.is_compressed()),
            "activation research requires explicit dense reference KV"
        );
        let layer = *research
            .pointers
            .get(&(source as usize))
            .context("unrecognized activation down pointer")?;
        ensure!(
            research.sources[layer].input_dim == h && research.sources[layer].output_dim == n,
            "activation down dimensions differ from configured model"
        );
        let mut state = research
            .mode
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned activation state"))?;
        match &mut *state {
            Mode::Capture { collector, row } => {
                if collector.should_observe(position) {
                    ensure!(row.len() == h, "capture row dimension mismatch");
                    self.cuda.synchronize_stream(CudaStream::Decode)?;
                    self.cuda
                        .memcpy_d2h(row.as_mut_ptr().cast(), input, h * 4)?;
                    collector.observe(layer, position, row)?;
                }
                Ok(false)
            }
            Mode::Apply {
                weights,
                thresholds,
                scratch,
                totals,
                ..
            } => {
                if thresholds[layer].magnitude() == 0.0 {
                    return Ok(false);
                }
                let totals = totals.as_ref().context("missing activation counters")?;
                let scratch = scratch.as_mut().context("missing activation scratch")?;
                let counts = (totals.as_mut_ptr() as *mut u8).add(layer * 16).cast();
                scratch.select_raw_async(input, thresholds[layer], counts)?;
                scratch.project_raw_async(
                    weights[layer]
                        .as_ref()
                        .context("missing prepared down weight")?,
                    output,
                    n * 4,
                    Q8TiledLaunch::new(16, 128)?,
                )?;
                Ok(true)
            }
        }
    }
}

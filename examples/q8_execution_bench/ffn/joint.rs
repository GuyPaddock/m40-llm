//! Per-case paired preparation, retained alongside the control for both arms.
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaContext,
    evaluation::ffn_capture::FfnLayerSamples,
    infer::{
        execution_weights::{DeviceQ8FfnPair, HostQ8FfnPair, Q8SourceIdentity, Q8SourceView},
        LoadedModel,
    },
};
use serde_json::{json, Value};
use std::time::Instant;

pub(super) struct JointFfnWeights {
    pub device: DeviceQ8FfnPair,
    host: HostQ8FfnPair,
    ctx: CudaContext,
    pub report: Value,
    pub bytes: usize,
}

impl JointFfnWeights {
    pub fn new(model: &LoadedModel, layer: &FfnLayerSamples, control_bytes: usize) -> Result<Self> {
        let expected = layer
            .source
            .gate
            .byte_len()?
            .checked_add(layer.source.up.byte_len()?)
            .context("joint FFN preparation overflow")?;
        let combined = expected
            .checked_add(control_bytes)
            .context("joint/control preparation overflow")?;
        ensure!(
            combined <= 128 * 1024 * 1024,
            "joint plus control preparation exceeds 128 MiB"
        );
        let stats = model.q8_execution_stats();
        let digest = *stats
            .tensors
            .first()
            .context("missing qualified prepared weights")?
            .key
            .source
            .model_sha256();
        let identity = |name: &str| -> Result<_> {
            Q8SourceIdentity::from_gguf_tensor(
                digest,
                model
                    .gguf
                    .tensors
                    .iter()
                    .find(|t| t.name == name)
                    .context("missing joint FFN source")?,
            )
        };
        let gate = identity(&layer.source.gate.name)?;
        let up = identity(&layer.source.up.name)?;
        let began = Instant::now();
        let host = HostQ8FfnPair::pack(
            Q8SourceView::from_tensor_bytes(
                &gate,
                super::fixture::source_bytes(model, &layer.source.gate)?,
            )?,
            Q8SourceView::from_tensor_bytes(
                &up,
                super::fixture::source_bytes(model, &layer.source.up)?,
            )?,
        )?;
        let packing_ms = began.elapsed().as_secs_f64() * 1000.0;
        let bytes = host.bytes();
        ensure!(bytes == expected, "joint FFN packing changed payload size");
        let began = Instant::now();
        let before = CudaContext::total_device_bytes();
        let device = DeviceQ8FfnPair::upload(&model.cuda, &host)?;
        let upload_ms = began.elapsed().as_secs_f64() * 1000.0;
        let after = CudaContext::total_device_bytes();
        ensure!(
            after.checked_sub(before) == Some(bytes),
            "joint FFN allocation accounting mismatch"
        );
        let report = json!({"layout":"JointGateUp4","key":host.key(),
            "paired_scale_bytes":host.planes()[0].len(),"paired_payload_bytes":host.planes()[1].len(),
            "additional_prepared_bytes":bytes,"retained_host_prepared_bytes":bytes,
            "control_prepared_bytes":control_bytes,"combined_prepared_bytes":combined,
            "packing_ms":packing_ms,"allocation_upload_ms":upload_ms,
            "tracked_bytes_before":before,"tracked_bytes_after":after,
            "scope":"benchmark-only joint pair; both arms retain joint and control pairs; packing and upload excluded from primary timing"});
        Ok(Self {
            device,
            host,
            ctx: model.cuda.clone(),
            report,
            bytes,
        })
    }

    pub fn verify(&self) -> Result<()> {
        ensure!(
            self.device.key() == self.host.key(),
            "joint FFN source identity changed"
        );
        for (buffer, expected) in self.device.planes().into_iter().zip(self.host.planes()) {
            let mut actual = vec![0u8; buffer.bytes()];
            unsafe {
                self.ctx
                    .memcpy_d2h(actual.as_mut_ptr().cast(), buffer.as_ptr(), actual.len())?;
            }
            ensure!(actual == expected, "joint FFN immutable plane changed");
        }
        Ok(())
    }
}

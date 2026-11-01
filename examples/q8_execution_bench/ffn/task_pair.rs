//! Three-plane task-pair preparation, retained in both timing arms.
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaContext,
    evaluation::ffn_capture::FfnLayerSamples,
    infer::{
        execution_weights::{
            DeviceQ8FfnTaskPair, HostQ8FfnTaskPair, Q8SourceIdentity, Q8SourceView,
        },
        LoadedModel,
    },
};
use serde_json::{json, Value};
use std::time::Instant;

pub(super) struct PreparedFfnTaskPair {
    pub device: DeviceQ8FfnTaskPair,
    host: HostQ8FfnTaskPair,
    ctx: CudaContext,
    pub report: Value,
    pub bytes: usize,
}
impl PreparedFfnTaskPair {
    pub fn new(model: &LoadedModel, layer: &FfnLayerSamples, control_bytes: usize) -> Result<Self> {
        let expected = layer
            .source
            .gate
            .byte_len()?
            .checked_add(layer.source.up.byte_len()?)
            .context("task-pair size overflow")?;
        ensure!(
            expected
                .checked_add(control_bytes)
                .is_some_and(|v| v <= 128 * 1024 * 1024),
            "task pair plus control exceed 128 MiB"
        );
        let stats = model.q8_execution_stats();
        let digest = *stats
            .tensors
            .first()
            .context("missing original prepared cache")?
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
                    .context("missing task-pair source")?,
            )
        };
        let g = identity(&layer.source.gate.name)?;
        let u = identity(&layer.source.up.name)?;
        let raw_g = super::fixture::source_bytes(model, &layer.source.gate)?;
        let raw_u = super::fixture::source_bytes(model, &layer.source.up)?;
        let began = Instant::now();
        let host = HostQ8FfnTaskPair::pack(
            Q8SourceView::from_tensor_bytes(&g, raw_g)?,
            Q8SourceView::from_tensor_bytes(&u, raw_u)?,
        )?;
        let inverse = host.unpack_column_blocks()?;
        ensure!(
            inverse[0] == raw_g && inverse[1] == raw_u,
            "task-pair source roundtrip changed bits"
        );
        drop(inverse);
        let packing_and_roundtrip_ms = began.elapsed().as_secs_f64() * 1000.0;
        let bytes = host.bytes();
        ensure!(bytes == expected, "task-pair payload size changed");
        let before = CudaContext::total_device_bytes();
        let began = Instant::now();
        let device = DeviceQ8FfnTaskPair::upload(&model.cuda, &host)?;
        let upload_ms = began.elapsed().as_secs_f64() * 1000.0;
        let after = CudaContext::total_device_bytes();
        ensure!(
            after.checked_sub(before) == Some(bytes),
            "task-pair allocation accounting mismatch"
        );
        let report = json!({"layout":"ThreadOwnedTaskPairsV1","key":host.key(),"additional_prepared_bytes":bytes,
            "control_prepared_bytes":control_bytes,"combined_prepared_bytes":bytes+control_bytes,
            "retained_host_prepared_bytes":bytes,"inverse_temporary_host_bytes":bytes,
            "planes":host.planes().map(|p|p.len()),"source_roundtrip_bits_equal":true,
            "packing_and_roundtrip_ms":packing_and_roundtrip_ms,"allocation_upload_ms":upload_ms,
            "tracked_bytes_before":before,"tracked_bytes_after":after,
            "scope":"both arms retain original and task-pair planes; setup/inverse excluded from primary timing"});
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
            "task-pair identity changed"
        );
        for (device, original) in self.device.planes().into_iter().zip(self.host.planes()) {
            let mut actual = vec![0u8; device.bytes()];
            unsafe {
                self.ctx
                    .memcpy_d2h(actual.as_mut_ptr().cast(), device.as_ptr(), actual.len())?;
            }
            ensure!(actual == original, "immutable task-pair plane changed");
        }
        Ok(())
    }
}

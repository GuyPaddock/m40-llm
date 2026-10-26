//! Per-case packet preparation, held identically in both timing arms.
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaContext,
    evaluation::identity::sha256,
    infer::{
        execution_weights::{DeviceQ8WarpPacket, HostQ8WarpPacket, Q8SourceView},
        LoadedModel,
    },
};
use serde_json::{json, Value};
use std::{ffi::c_void, time::Instant};

pub(super) struct PreparedWarpPacket {
    pub device: DeviceQ8WarpPacket,
    host: HostQ8WarpPacket,
    ctx: CudaContext,
    pub report: Value,
}

impl PreparedWarpPacket {
    pub fn new(
        model: &LoadedModel,
        source_pointer: *const c_void,
        extra_budget: usize,
    ) -> Result<Self> {
        let weight = model
            .q8_execution
            .get_separate_weight(source_pointer)
            .context("missing original prepared source")?;
        let identity = &weight.key().source;
        let source = Q8SourceView::new(identity, &model.host_weights)?;
        ensure!(
            source.blocks().len() <= extra_budget,
            "warp packet extra preparation exceeds its fixed case budget"
        );
        let began = Instant::now();
        let host = HostQ8WarpPacket::pack(source)?;
        ensure!(
            host.unpack_column_blocks()? == source.blocks(),
            "warp packet original-source roundtrip changed bits"
        );
        let packing_and_roundtrip_ms = began.elapsed().as_secs_f64() * 1000.0;
        let before = CudaContext::total_device_bytes();
        let began = Instant::now();
        let device = DeviceQ8WarpPacket::upload(&model.cuda, &host)?;
        let upload_ms = began.elapsed().as_secs_f64() * 1000.0;
        let after = CudaContext::total_device_bytes();
        ensure!(
            after.checked_sub(before) == Some(host.bytes()),
            "warp packet allocation accounting mismatch"
        );
        let report = json!({"layout":"WarpTaskPacket4x8V1","key":host.key(),"original_source_sha256":sha256(source.blocks())?,
            "additional_prepared_bytes":host.bytes(),"retained_host_prepared_bytes":host.bytes(),"source_bytes":source.blocks().len(),
            "inverse_roundtrip_temporary_host_bytes":host.bytes(),
            "scale_bytes":host.planes()[0].len(),"payload_bytes":host.planes()[1].len(),"extra_budget_bytes":extra_budget,
            "packing_and_roundtrip_ms":packing_and_roundtrip_ms,"allocation_upload_ms":upload_ms,
            "source_roundtrip_bits_equal":true,"tracked_bytes_before":before,"tracked_bytes_after":after,
            "scope":"both arms retain original and candidate planes; setup and CPU inverse excluded from primary timing"});
        Ok(Self {
            device,
            host,
            ctx: model.cuda.clone(),
            report,
        })
    }

    pub fn verify(&self) -> Result<()> {
        ensure!(
            self.device.key() == self.host.key(),
            "warp packet identity changed"
        );
        for (device, host) in self.device.planes().into_iter().zip(self.host.planes()) {
            let mut actual = vec![0u8; device.bytes()];
            unsafe {
                self.ctx
                    .memcpy_d2h(actual.as_mut_ptr().cast(), device.as_ptr(), actual.len())?;
            }
            ensure!(actual == host, "immutable warp packet plane changed");
        }
        Ok(())
    }
}

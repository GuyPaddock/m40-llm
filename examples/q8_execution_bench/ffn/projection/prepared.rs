//! Per-case split QKV/output ownership, outside timed projection enqueue.
use super::super::fixture::source_bytes;
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, CudaStream},
    evaluation::ffn_capture::FfnTensorSource,
    infer::{
        execution_weights::{
            DeviceQ8ExecutionView, DeviceQ8ExecutionWeight, HostQ8ExecutionWeight,
            Q8SourceIdentity, Q8SourceView,
        },
        LoadedModel,
    },
};
use serde_json::{json, Value};
use std::{ffi::c_void, time::Instant};

pub(super) struct PreparedProjectionWeights {
    ctx: CudaContext,
    host: Vec<HostQ8ExecutionWeight>,
    device: Vec<DeviceQ8ExecutionWeight>,
    pub report: Value,
    pub bytes: usize,
}

impl PreparedProjectionWeights {
    pub fn new(model: &LoadedModel, sources: &[FfnTensorSource]) -> Result<Self> {
        ensure!(
            matches!(sources.len(), 1 | 3),
            "expected output or QKV weights"
        );
        let expected = sources.iter().try_fold(0usize, |sum, source| {
            sum.checked_add(source.byte_len()?)
                .context("projection preparation overflow")
        })?;
        ensure!(
            expected <= 128 * 1024 * 1024,
            "projection preparation exceeds 128 MiB"
        );
        let stats = model.q8_execution_stats();
        let digest = *stats
            .tensors
            .first()
            .context("missing prepared control identity")?
            .key
            .source
            .model_sha256();
        let began = Instant::now();
        let host = sources
            .iter()
            .map(|source| {
                let tensor = model
                    .gguf
                    .tensors
                    .iter()
                    .find(|t| t.name == source.name)
                    .context("missing projection source")?;
                let identity = Q8SourceIdentity::from_gguf_tensor(digest, tensor)?;
                HostQ8ExecutionWeight::separate_scales_values(Q8SourceView::from_tensor_bytes(
                    &identity,
                    source_bytes(model, source)?,
                )?)
            })
            .collect::<Result<Vec<_>>>()?;
        let packing_ms = began.elapsed().as_secs_f64() * 1000.0;
        let bytes = host
            .iter()
            .map(|w| w.plan().resident_bytes())
            .sum::<usize>();
        ensure!(bytes == expected, "split projection changed payload size");
        let before = CudaContext::total_device_bytes();
        let began = Instant::now();
        let device = host
            .iter()
            .map(|w| DeviceQ8ExecutionWeight::upload(&model.cuda, w))
            .collect::<Result<Vec<_>>>()?;
        let allocation_upload_ms = began.elapsed().as_secs_f64() * 1000.0;
        let after = CudaContext::total_device_bytes();
        ensure!(
            after.checked_sub(before) == Some(bytes),
            "split projection allocation mismatch"
        );
        let report = json!({"layout":"SeparateScalesValues",
            "keys":host.iter().map(|w|w.key()).collect::<Vec<_>>(),
            "planes":host.iter().map(|w|w.plan()).collect::<Vec<_>>(),
            "additional_prepared_bytes":bytes,"retained_host_prepared_bytes":bytes,
            "packing_ms":packing_ms,"allocation_upload_ms":allocation_upload_ms,
            "tracked_bytes_before":before,"tracked_bytes_after":after,
            "estimated_all_layers_extra_bytes":bytes.checked_mul(model.model_config.block_count as usize),
            "scope":"one layer projection family, retained in both arms; setup excluded; full-model estimate is not observed residency"});
        Ok(Self {
            ctx: model.cuda.clone(),
            host,
            device,
            report,
            bytes,
        })
    }

    fn planes(&self, index: usize) -> Result<(*const c_void, *const c_void)> {
        let weight = self
            .device
            .get(index)
            .context("missing prepared projection")?;
        match weight.view() {
            DeviceQ8ExecutionView::SeparateScalesValues { scales, values, .. } => {
                Ok((scales.as_ptr(), values.as_ptr()))
            }
            _ => anyhow::bail!("projection requires typed separate planes"),
        }
    }

    /// # Safety
    /// Raw input/bias/output spans obey the CUDA consumer's alignment, shape,
    /// disjointness and producer-order contract and survive stream completion.
    pub unsafe fn enqueue_qkv(
        &self,
        input: *const c_void,
        biases: [Option<*const c_void>; 3],
        outputs: [*mut c_void; 3],
        stream: CudaStream,
    ) -> Result<()> {
        ensure!(
            self.device.len() == 3,
            "QKV requires three owned split weights"
        );
        let planes = [self.planes(0)?, self.planes(1)?, self.planes(2)?];
        let mut widths = [0; 3];
        for (width, weight) in widths.iter_mut().zip(&self.device) {
            *width = i32::try_from(weight.key().source.shape().output_dim())?;
        }
        let k = i32::try_from(self.device[0].key().source.shape().input_dim())?;
        self.ctx.qkv_f32xq8_separate_packed_warp_async(
            input,
            planes.map(|p| p.0),
            planes.map(|p| p.1),
            biases,
            outputs,
            widths,
            k,
            stream,
        )
    }

    /// # Safety
    /// The raw spans match this weight and the split output consumer's contract.
    pub unsafe fn enqueue_output(
        &self,
        input: *const c_void,
        output: *mut c_void,
        stream: CudaStream,
    ) -> Result<()> {
        ensure!(
            self.device.len() == 1,
            "output requires one owned split weight"
        );
        let shape = self.device[0].key().source.shape();
        let (scales, values) = self.planes(0)?;
        self.ctx.gemm_f32xq8_separate_coltile4_packed_warp_async(
            input,
            scales,
            values,
            output,
            i32::try_from(shape.output_dim())?,
            i32::try_from(shape.input_dim())?,
            stream,
        )
    }

    pub fn verify(&self) -> Result<()> {
        for (device, host) in self.device.iter().zip(&self.host) {
            let DeviceQ8ExecutionView::SeparateScalesValues { scales, values, .. } = device.view()
            else {
                anyhow::bail!("projection requires typed separate planes");
            };
            for (buffer, expected) in [scales, values].into_iter().zip(host.planes()) {
                let mut actual = vec![0; buffer.bytes()];
                unsafe {
                    self.ctx.memcpy_d2h(
                        actual.as_mut_ptr().cast(),
                        buffer.as_ptr(),
                        actual.len(),
                    )?;
                }
                ensure!(
                    actual == *expected,
                    "split projection changed a prepared plane"
                );
            }
        }
        Ok(())
    }
}

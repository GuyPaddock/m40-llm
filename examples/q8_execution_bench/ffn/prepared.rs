//! Per-case lossless gate/up preparation, never part of timed enqueue.
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, DeviceBuffer},
    evaluation::ffn_capture::FfnLayerSamples,
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

pub(super) struct PreparedFfnWeights {
    ctx: CudaContext,
    host: [HostQ8ExecutionWeight; 2],
    device: [DeviceQ8ExecutionWeight; 2],
    pub report: Value,
    pub bytes: usize,
}

fn split_planes(weight: &DeviceQ8ExecutionWeight) -> Result<(&DeviceBuffer, &DeviceBuffer)> {
    match weight.view() {
        DeviceQ8ExecutionView::SeparateScalesValues { scales, values, .. } => Ok((scales, values)),
        _ => anyhow::bail!("FFN split experiment requires separate scale/value ownership"),
    }
}

impl PreparedFfnWeights {
    pub fn prepare_column_gap(&self) -> Result<super::column_gap::GappedPayloads> {
        super::column_gap::GappedPayloads::new(&self.ctx, &self.host, self.bytes)
    }

    /// # Safety
    /// Alternate payloads use 128-byte column gaps and remain live through
    /// decode completion; output is a disjoint rows*hidden*2 u32 span.
    pub unsafe fn enqueue_checksum_pitched(
        &self,
        alternate: Option<[*const c_void; 2]>,
        output: *mut c_void,
        rows: usize,
    ) -> Result<()> {
        let (gs, gv) = split_planes(&self.device[0])?;
        let (us, uv) = split_planes(&self.device[1])?;
        let shape = self.device[0].key().source.shape();
        self.ctx.ffn_q8_weight_checksum_pitched_async(
            [gs.as_ptr(), us.as_ptr()],
            alternate.unwrap_or([gv.as_ptr(), uv.as_ptr()]),
            output,
            i32::try_from(rows)?,
            i32::try_from(shape.output_dim())?,
            i32::try_from(shape.input_dim())?,
            if alternate.is_some() { 128 } else { 0 },
        )
    }

    /// # Safety
    /// Output is a live, disjoint rows*hidden*2 u32 span until decode drains.
    pub unsafe fn enqueue_checksum(&self, output: *mut c_void, rows: usize) -> Result<()> {
        let (gs, gv) = split_planes(&self.device[0])?;
        let (us, uv) = split_planes(&self.device[1])?;
        let shape = self.device[0].key().source.shape();
        self.ctx.ffn_q8_weight_checksum_async(
            [gs.as_ptr(), us.as_ptr()],
            [gv.as_ptr(), uv.as_ptr()],
            output,
            i32::try_from(rows)?,
            i32::try_from(shape.output_dim())?,
            i32::try_from(shape.input_dim())?,
        )
    }

    pub fn new(model: &LoadedModel, layer: &FfnLayerSamples) -> Result<Self> {
        let expected_bytes = layer
            .source
            .gate
            .byte_len()?
            .checked_add(layer.source.up.byte_len()?)
            .context("FFN preparation byte overflow")?;
        ensure!(
            expected_bytes <= 128 * 1024 * 1024,
            "FFN candidate preparation exceeds the per-case 128 MiB bound"
        );
        let stats = model.q8_execution_stats();
        let digest = *stats
            .tensors
            .first()
            .context("missing qualified prepared weights")?
            .key
            .source
            .model_sha256();
        let pack = |source: &m40_llm::evaluation::ffn_capture::FfnTensorSource| -> Result<_> {
            let tensor = model
                .gguf
                .tensors
                .iter()
                .find(|t| t.name == source.name)
                .context("missing gate/up source")?;
            let identity = Q8SourceIdentity::from_gguf_tensor(digest, tensor)?;
            HostQ8ExecutionWeight::separate_scales_values(Q8SourceView::from_tensor_bytes(
                &identity,
                super::fixture::source_bytes(model, source)?,
            )?)
        };
        let began = Instant::now();
        let host = [pack(&layer.source.gate)?, pack(&layer.source.up)?];
        let packing_ms = began.elapsed().as_secs_f64() * 1000.0;
        let bytes = host
            .iter()
            .map(|w| w.plan().resident_bytes())
            .sum::<usize>();
        ensure!(
            bytes == expected_bytes,
            "split packing changed weight payload size"
        );
        let began = Instant::now();
        let before = CudaContext::total_device_bytes();
        let device = [
            DeviceQ8ExecutionWeight::upload(&model.cuda, &host[0])?,
            DeviceQ8ExecutionWeight::upload(&model.cuda, &host[1])?,
        ];
        let upload_ms = began.elapsed().as_secs_f64() * 1000.0;
        let after = CudaContext::total_device_bytes();
        ensure!(
            after.checked_sub(before) == Some(bytes),
            "unexpected split FFN allocation accounting"
        );
        let report = json!({"layout":"SeparateScalesValues","keys":host.iter().map(|w|w.key()).collect::<Vec<_>>(),
            "planes":host.iter().map(|w|w.plan()).collect::<Vec<_>>(),
            "additional_prepared_bytes":bytes,"retained_host_prepared_bytes":bytes,
            "packing_ms":packing_ms,"allocation_upload_ms":upload_ms,"tracked_bytes_before":before,"tracked_bytes_after":after,
            "estimated_all_layers_extra_bytes":bytes.checked_mul(model.model_config.block_count as usize),
            "scope":"one layer gate/up pair; both timing arms retain the same prepared pair; setup excluded from primary timings; full-model extra bytes are an estimate, not observed runtime residency"});
        Ok(Self {
            ctx: model.cuda.clone(),
            host,
            device,
            report,
            bytes,
        })
    }

    /// # Safety
    /// Caller keeps every raw activation/output span on this context live until
    /// decode drains, with the spans/alignment required by the CUDA wrapper.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue(
        &self,
        input: *const c_void,
        output: *mut c_void,
        rows: i32,
        hidden: i32,
        k: i32,
        intermediates: Option<(*mut c_void, *mut c_void)>,
        kernel: super::pipeline::FfnKernel,
    ) -> Result<()> {
        let mut scales = [std::ptr::null(); 2];
        let mut values = [std::ptr::null(); 2];
        for (index, weight) in self.device.iter().enumerate() {
            let shape = weight.key().source.shape();
            ensure!(
                shape.input_dim() == k as usize && shape.output_dim() == hidden as usize,
                "split FFN prepared shape mismatch"
            );
            let (s, v) = split_planes(weight)?;
            scales[index] = s.as_ptr();
            values[index] = v.as_ptr();
        }
        let enqueue = match kernel {
            super::pipeline::FfnKernel::NativeWarp => {
                CudaContext::ffn_q8_split_native_warp_f32_async
            }
            super::pipeline::FfnKernel::SplitF32x8 => CudaContext::ffn_q8_split_f32x8_async,
            super::pipeline::FfnKernel::SplitF32x4Cohorts => {
                CudaContext::ffn_q8_split_cohorts_f32_async
            }
            super::pipeline::FfnKernel::SplitF32x4 => CudaContext::ffn_q8_split_f32_async,
            super::pipeline::FfnKernel::SplitF32x4VirtualWarp => {
                CudaContext::ffn_q8_split_virtual_warp_f32_async
            }
            super::pipeline::FfnKernel::SplitF32x4PersistentTiles => {
                CudaContext::ffn_q8_split_persistent_tiles_f32_async
            }
            super::pipeline::FfnKernel::SplitF32x4TwoColumns => {
                CudaContext::ffn_q8_split_f32_two_columns_async
            }
            super::pipeline::FfnKernel::SplitF32x4K2048 => {
                CudaContext::ffn_q8_split_f32_k2048_async
            }
            super::pipeline::FfnKernel::SplitF32x4Unfused => {
                CudaContext::ffn_q8_split_unfused_f32_async
            }
            super::pipeline::FfnKernel::SplitF32x4WeightCg => {
                CudaContext::ffn_q8_split_weight_cg_f32_async
            }
            super::pipeline::FfnKernel::SplitF32x4BytePermute => {
                CudaContext::ffn_q8_split_byte_permute_f32_async
            }
            _ => anyhow::bail!("split FFN consumer requires a separate-plane kernel"),
        };
        enqueue(
            &self.ctx,
            input,
            scales,
            values,
            output,
            rows,
            hidden,
            k,
            intermediates,
        )
    }

    pub fn verify(&self) -> Result<()> {
        for (device, host) in self.device.iter().zip(&self.host) {
            let (scales, values) = split_planes(device)?;
            for (buffer, expected) in [scales, values].into_iter().zip(host.planes()) {
                let mut actual = vec![0u8; buffer.bytes()];
                unsafe {
                    self.ctx.memcpy_d2h(
                        actual.as_mut_ptr().cast(),
                        buffer.as_ptr(),
                        actual.len(),
                    )?;
                }
                ensure!(actual == *expected, "split FFN immutable plane changed");
            }
        }
        Ok(())
    }
}

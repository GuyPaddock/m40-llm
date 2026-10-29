//! Existing single-stream CUDA operations, with persistent scratch and weights.
use super::reference::{Parameters, D, HEADS, R, V};
use anyhow::{ensure, Result};
use m40_llm::cuda::{CudaContext, CudaEvent, CudaStream, DeviceBuffer};
use serde_json::{json, Value};
use std::{ffi::c_void, time::Instant};

fn upload(ctx: &CudaContext, values: &[f32]) -> Result<DeviceBuffer> {
    let b = DeviceBuffer::new_tagged(ctx, values.len() * 4, "compact_draft:immutable")?;
    unsafe {
        ctx.memcpy_h2d(b.as_mut_ptr(), values.as_ptr().cast(), b.bytes())?;
    }
    Ok(b)
}
fn read(ctx: &CudaContext, b: &DeviceBuffer) -> Result<Vec<f32>> {
    let mut values = vec![0f32; b.bytes() / 4];
    unsafe {
        ctx.memcpy_d2h(values.as_mut_ptr().cast(), b.as_ptr(), b.bytes())?;
    }
    Ok(values)
}
pub(super) struct Proposal {
    ctx: CudaContext,
    maps: DeviceBuffer,
    bias: DeviceBuffer,
    vocabulary: DeviceBuffer,
    feature_source: DeviceBuffer,
    feature: DeviceBuffer,
    ones: DeviceBuffer,
    affine: DeviceBuffer,
    biased: DeviceBuffer,
    hidden: DeviceBuffer,
    logits: DeviceBuffer,
    ids: DeviceBuffer,
    events: Vec<CudaEvent>,
    pub setup: Value,
}
impl Proposal {
    pub fn new(ctx: &CudaContext, p: &Parameters) -> Result<Self> {
        let start = Instant::now();
        let initial = CudaContext::total_device_bytes();
        let maps = upload(ctx, &p.maps)?;
        let bias = upload(ctx, &p.bias)?;
        let vocabulary = upload(ctx, &p.vocabulary)?;
        let weights_upload_ms = start.elapsed().as_secs_f64() * 1000.;
        let feature_source = upload(ctx, &p.feature)?;
        let ones = upload(ctx, &vec![1.; HEADS * R])?;
        let alloc = |bytes| DeviceBuffer::new_tagged(ctx, bytes, "compact_draft:scratch");
        let feature = alloc(D * 4)?;
        let affine = alloc(HEADS * R * 4)?;
        let biased = alloc(HEADS * R * 4)?;
        let hidden = alloc(HEADS * R * 4)?;
        let logits = alloc(HEADS * V * 4)?;
        let ids = alloc(HEADS * 4)?;
        let events = (0..7)
            .map(|_| ctx.create_event())
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            ctx: ctx.clone(),
            maps,
            bias,
            vocabulary,
            feature_source,
            feature,
            ones,
            affine,
            biased,
            hidden,
            logits,
            ids,
            events,
            setup: json!({"weights_upload_ms":weights_upload_ms,"total_allocation_upload_ms":start.elapsed().as_secs_f64()*1000.,
                "tracked_device_bytes":CudaContext::total_device_bytes()-initial,"weights_bytes":p.bytes(),
                "host_parameter_payload_bytes":p.bytes(),"new_kernels":0}),
        })
    }
    fn enqueue_stage(&self, stage: usize) -> Result<()> {
        let ctx = &self.ctx;
        unsafe {
            match stage {
                0 => ctx.memcpy_d2d_async(
                    self.feature.as_mut_ptr(),
                    self.feature_source.as_ptr(),
                    D * 4,
                    CudaStream::Decode,
                ),
                1 => ctx.gemm_f32xf32_f32_stream_async(
                    self.feature.as_ptr(),
                    self.maps.as_ptr(),
                    self.affine.as_mut_ptr(),
                    1,
                    (HEADS * R) as i32,
                    D as i32,
                    CudaStream::Decode,
                ),
                2 => ctx.residual_add_f32_async(
                    self.affine.as_ptr(),
                    self.bias.as_ptr(),
                    self.biased.as_mut_ptr(),
                    HEADS * R,
                ),
                3 => ctx.swiglu_f32_async(
                    self.biased.as_ptr(),
                    self.ones.as_ptr(),
                    self.hidden.as_mut_ptr(),
                    HEADS * R,
                ),
                4 => ctx.gemm_f32xf32_f32_stream_async(
                    self.hidden.as_ptr(),
                    self.vocabulary.as_ptr(),
                    self.logits.as_mut_ptr(),
                    HEADS as i32,
                    V as i32,
                    R as i32,
                    CudaStream::Decode,
                ),
                5 => {
                    for h in 0..HEADS {
                        ctx.argmax_f32_async(
                            self.logits.as_ptr().cast::<f32>().add(h * V).cast(),
                            V as u32,
                            self.ids.as_mut_ptr().cast::<u32>().add(h).cast(),
                            CudaStream::Decode,
                        )?;
                    }
                    Ok(())
                }
                _ => anyhow::bail!("invalid fixed proposal stage"),
            }
        }
    }
    fn complete(&self) -> Result<[u32; 3]> {
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        let mut ids = [0u32; 3];
        unsafe {
            self.ctx
                .memcpy_d2h(ids.as_mut_ptr().cast::<c_void>(), self.ids.as_ptr(), 12)?;
        }
        ensure!(ids.iter().all(|v| *v < V as u32), "invalid proposal token");
        Ok(ids)
    }
    pub fn propose(&mut self) -> Result<[u32; 3]> {
        for stage in 0..6 {
            self.enqueue_stage(stage)?;
        }
        self.complete()
    }
    pub fn read_hidden(&self) -> Result<Vec<f32>> {
        read(&self.ctx, &self.hidden)
    }
    pub fn read_logits(&self) -> Result<Vec<f32>> {
        read(&self.ctx, &self.logits)
    }
    pub fn diagnostic(&self) -> Result<Value> {
        self.events[0].record(CudaStream::Decode)?;
        for stage in 0..6 {
            self.enqueue_stage(stage)?;
            self.events[stage + 1].record(CudaStream::Decode)?;
        }
        let ids = self.complete()?;
        let mut times = Vec::new();
        for stage in 0..6 {
            times.push(
                self.events[stage].elapsed_sync(&self.events[stage + 1], "compact_draft_stage")?,
            );
        }
        Ok(
            json!({"order":["feature_snapshot","affine","bias","silu","vocabulary","three_argmax"],"gpu_ms":times,"ids":ids,
            "scope":"instrumented diagnostic outside primary trials; extra events/syncs"}),
        )
    }
    pub fn verify_owners(&self, p: &Parameters) -> Result<()> {
        for (buffer, original) in [
            (&self.maps, &p.maps),
            (&self.bias, &p.bias),
            (&self.vocabulary, &p.vocabulary),
            (&self.feature_source, &p.feature),
        ] {
            let actual = read(&self.ctx, buffer)?;
            ensure!(
                actual
                    .iter()
                    .zip(original)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "proposal immutable owner changed"
            );
        }
        ensure!(
            read(&self.ctx, &self.ones)?.iter().all(|v| *v == 1.),
            "SiLU multiplier changed"
        );
        Ok(())
    }
}
impl Drop for Proposal {
    fn drop(&mut self) {
        if let Err(e) = self.ctx.synchronize_stream(CudaStream::Decode) {
            eprintln!("[compact-draft] drain failed: {e:#}");
        }
    }
}

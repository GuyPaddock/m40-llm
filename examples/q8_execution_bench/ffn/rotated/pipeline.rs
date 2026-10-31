use super::super::pipeline::GuardedF32;
use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{
        CudaContext, CudaEvent, CudaStream, DeviceBuffer, DeviceRotatedActivation, Q8FfnQ81Launch,
        RotatedPreparation,
    },
    infer::q8_activation::Q8ActivationShape,
    profile,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub(super) const CHANNELS: usize = 256;
pub(super) const ARMS: [RotatedPreparation; 2] =
    [RotatedPreparation::Staged, RotatedPreparation::Fused];

#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Stage {
    Prepare,
    Complete,
}

pub(super) struct Pipeline<'a> {
    pub ctx: &'a CudaContext,
    pub input: GuardedF32,
    pub producer: DeviceRotatedActivation,
    weights: &'a [DeviceBuffer; 2],
    output: GuardedF32,
    shape: Q8ActivationShape,
    start: CudaEvent,
    stop: CudaEvent,
}

impl<'a> Pipeline<'a> {
    pub fn new(
        ctx: &'a CudaContext,
        weights: &'a [DeviceBuffer; 2],
        input: &[f32],
        signs: &[i8],
    ) -> Result<Self> {
        ensure!(
            !signs.is_empty() && input.len().is_multiple_of(signs.len()),
            "invalid producer input shape"
        );
        let rows = input.len() / signs.len();
        Ok(Self {
            ctx,
            weights,
            input: GuardedF32::new(ctx, input, "rotated:input")?,
            producer: DeviceRotatedActivation::upload(ctx, rows, signs)?,
            output: GuardedF32::new(ctx, &vec![0.0; rows * CHANNELS], "rotated:output")?,
            shape: Q8ActivationShape::new(1, signs.len(), signs.len())?,
            start: ctx.create_event()?,
            stop: ctx.create_event()?,
        })
    }

    pub fn scratch_bytes(&self) -> usize {
        self.input.bytes() + self.output.bytes() + self.producer.workspace_bytes()
    }

    pub fn drain(&self) -> Result<()> {
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        self.ctx.synchronize_stream(CudaStream::Prefill)
    }

    pub fn enqueue(&self, arm: RotatedPreparation, stage: Stage) -> Result<()> {
        unsafe {
            self.producer
                .enqueue_async(self.input.pointer(), arm, CudaStream::Decode)?;
            if stage == Stage::Complete {
                for row in 0..self.producer.rows() {
                    self.ctx.ffn_q8_1_async(
                        self.producer
                            .get_quantized_pointer()
                            .cast::<u8>()
                            .add(row * self.shape.bytes())
                            .cast(),
                        self.weights[0].as_ptr(),
                        self.weights[1].as_ptr(),
                        self.output
                            .pointer()
                            .cast::<f32>()
                            .add(row * CHANNELS)
                            .cast(),
                        None,
                        CHANNELS as i32,
                        self.shape,
                        Q8FfnQ81Launch::Threads128,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub fn read(&self) -> Result<(Vec<f32>, Vec<u8>, Vec<f32>)> {
        let (rotated, encoded) = self.producer.read_preparation()?;
        self.input.read(self.ctx)?;
        Ok((rotated, encoded, self.output.read(self.ctx)?))
    }

    pub fn time(&self, arm: RotatedPreparation, stage: Stage, iterations: usize) -> Result<Value> {
        ensure!(iterations > 0, "empty rotated timing interval");
        self.drain()?;
        let before = profile::snapshot();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let began = Instant::now();
        self.start.record(CudaStream::Decode)?;
        for _ in 0..iterations {
            self.enqueue(arm, stage)?;
        }
        self.stop.record(CudaStream::Decode)?;
        let enqueue_ms = began.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        let counters = profile::counter_delta(&before, &profile::snapshot());
        let producer_launches = if arm == RotatedPreparation::Staged {
            3
        } else {
            1
        };
        let consumer_launches = if stage == Stage::Complete {
            self.producer.rows()
        } else {
            0
        };
        ensure!(
            counters.launches == ((producer_launches + consumer_launches) * iterations) as u64,
            "rotated dispatch count mismatch"
        );
        ensure!(
            counters.device_allocs
                + counters.device_frees
                + counters.h2d_copies
                + counters.d2h_copies
                + counters.stream_syncs
                + counters.stream_waits
                + counters.cublas_calls
                == 0,
            "rotated timing allocated, copied or synchronized"
        );
        let gpu_ms =
            f64::from(self.start.elapsed_sync(&self.stop, "rotated:timed")?) / iterations as f64;
        ensure!(gpu_ms.is_finite() && gpu_ms > 0.0, "invalid rotated timing");
        Ok(
            json!({"arm":arm,"stage":stage,"gpu_ms":gpu_ms,"host_enqueue_ms":enqueue_ms,
            "host_enqueue_and_completion_ms":began.elapsed().as_secs_f64()*1000.0/iterations as f64,
            "iterations":iterations,"started_unix_us":timestamp,"enqueue_counters":counters}),
        )
    }
}

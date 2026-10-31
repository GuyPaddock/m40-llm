use super::super::pipeline::GuardedF32;
use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{CudaContext, CudaEvent, CudaStream, DeviceBuffer},
    profile,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Arm {
    SerialDown,
    SerialGateBatchedDown,
    UnfusedBatched,
    FusedRows,
}

impl Arm {
    pub const ALL: [Self; 4] = [
        Self::SerialDown,
        Self::SerialGateBatchedDown,
        Self::UnfusedBatched,
        Self::FusedRows,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Stage {
    GateUp,
    Complete,
}

pub(super) struct Pipeline<'a> {
    pub ctx: &'a CudaContext,
    weights: &'a [DeviceBuffer; 3],
    pub input: GuardedF32,
    output: GuardedF32,
    gate: GuardedF32,
    up: GuardedF32,
    hidden: GuardedF32,
    start: CudaEvent,
    stop: CudaEvent,
    rows: usize,
    k: usize,
    h: usize,
}

impl<'a> Pipeline<'a> {
    pub fn new(
        ctx: &'a CudaContext,
        weights: &'a [DeviceBuffer; 3],
        input: &[f32],
        k: usize,
        h: usize,
    ) -> Result<Self> {
        let rows = input.len() / k;
        Ok(Self {
            ctx,
            weights,
            rows,
            k,
            h,
            input: GuardedF32::new(ctx, input, "q4_fusion:input")?,
            output: GuardedF32::new(ctx, &vec![0.0; rows * k], "q4_fusion:output")?,
            gate: GuardedF32::new(ctx, &vec![0.0; rows * h], "q4_fusion:gate")?,
            up: GuardedF32::new(ctx, &vec![0.0; rows * h], "q4_fusion:up")?,
            hidden: GuardedF32::new(ctx, &vec![0.0; rows * h], "q4_fusion:hidden")?,
            start: ctx.create_event()?,
            stop: ctx.create_event()?,
        })
    }

    pub fn scratch_bytes(&self) -> usize {
        [
            &self.input,
            &self.output,
            &self.gate,
            &self.up,
            &self.hidden,
        ]
        .iter()
        .map(|v| v.bytes())
        .sum()
    }

    pub fn drain(&self) -> Result<()> {
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        self.ctx.synchronize_stream(CudaStream::Prefill)
    }

    pub fn enqueue(&self, arm: Arm, stage: Stage) -> Result<()> {
        let [gate, up, down] = self.weights;
        unsafe {
            match arm {
                Arm::SerialDown | Arm::SerialGateBatchedDown => {
                    for row in 0..self.rows {
                        let input = self.input.pointer().cast::<f32>().add(row * self.k).cast();
                        let hidden = self.hidden.pointer().cast::<f32>().add(row * self.h).cast();
                        self.ctx.mlp_gate_up_swiglu_f32xq4_0_gguf_decode_async(
                            input,
                            gate.as_ptr(),
                            up.as_ptr(),
                            hidden,
                            self.h as i32,
                            self.k as i32,
                        )?;
                        if arm == Arm::SerialDown && stage == Stage::Complete {
                            let output =
                                self.output.pointer().cast::<f32>().add(row * self.k).cast();
                            self.ctx.gemm_f32xq4_0_gguf_f32_decode_async(
                                hidden,
                                down.as_ptr(),
                                output,
                                1,
                                self.k as i32,
                                self.h as i32,
                            )?;
                        }
                    }
                }
                Arm::UnfusedBatched => {
                    self.ctx.stream_wait_for_stream(
                        CudaStream::Prefill,
                        CudaStream::Decode,
                        "q4_fusion:before_gate",
                    )?;
                    for (weight, output) in [(gate, &self.gate), (up, &self.up)] {
                        self.ctx.gemm_f32xq4_0_gguf_f32_blockloop_async(
                            self.input.pointer(),
                            weight.as_ptr(),
                            output.pointer(),
                            self.rows as i32,
                            self.h as i32,
                            self.k as i32,
                        )?;
                    }
                    self.ctx.stream_wait_for_stream(
                        CudaStream::Decode,
                        CudaStream::Prefill,
                        "q4_fusion:before_swiglu",
                    )?;
                    self.ctx.swiglu_f32_async(
                        self.gate.pointer(),
                        self.up.pointer(),
                        self.hidden.pointer(),
                        self.rows * self.h,
                    )?;
                }
                Arm::FusedRows => {
                    self.ctx.q4_fused_rows_async(
                        self.input.pointer(),
                        gate.as_ptr(),
                        up.as_ptr(),
                        self.hidden.pointer(),
                        self.rows,
                        self.h,
                        self.k,
                        CudaStream::Decode,
                    )?;
                }
            }
            if stage == Stage::Complete && arm != Arm::SerialDown {
                self.ctx.stream_wait_for_stream(
                    CudaStream::Prefill,
                    CudaStream::Decode,
                    "q4_fusion:before_down",
                )?;
                self.ctx.gemm_f32xq4_0_gguf_f32_blockloop_async(
                    self.hidden.pointer(),
                    down.as_ptr(),
                    self.output.pointer(),
                    self.rows as i32,
                    self.k as i32,
                    self.h as i32,
                )?;
                self.ctx.stream_wait_for_stream(
                    CudaStream::Decode,
                    CudaStream::Prefill,
                    "q4_fusion:after_down",
                )?;
            }
        }
        Ok(())
    }

    pub fn read(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        self.drain()?;
        for buffer in [&self.input, &self.gate, &self.up] {
            buffer.read(self.ctx)?;
        }
        Ok((self.hidden.read(self.ctx)?, self.output.read(self.ctx)?))
    }

    pub fn time(&self, arm: Arm, stage: Stage, iterations: usize) -> Result<Value> {
        ensure!(iterations > 0, "empty fusion timing interval");
        self.drain()?;
        let before = profile::snapshot();
        let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let began = Instant::now();
        self.start.record(CudaStream::Decode)?;
        for _ in 0..iterations {
            self.enqueue(arm, stage)?;
        }
        self.stop.record(CudaStream::Decode)?;
        let enqueue_ms = began.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        let counters = profile::counter_delta(&before, &profile::snapshot());
        let gate_launches = match arm {
            Arm::SerialDown | Arm::SerialGateBatchedDown => self.rows,
            Arm::UnfusedBatched => 3,
            Arm::FusedRows => 1,
        };
        let down_launches = if stage == Stage::GateUp {
            0
        } else if arm == Arm::SerialDown {
            self.rows
        } else {
            1
        };
        ensure!(
            counters.launches == ((gate_launches + down_launches) * iterations) as u64,
            "fusion dispatch count mismatch"
        );
        ensure!(
            counters.device_allocs
                + counters.device_frees
                + counters.h2d_copies
                + counters.d2h_copies
                + counters.stream_syncs
                + counters.cublas_calls
                == 0,
            "fusion timing allocated/copied/synchronized"
        );
        let gpu_ms = f64::from(self.start.elapsed_sync(&self.stop, "q4_fusion:complete")?)
            / iterations as f64;
        ensure!(gpu_ms.is_finite() && gpu_ms > 0.0, "invalid fusion timing");
        Ok(
            json!({"arm":arm,"stage":stage,"gpu_ms":gpu_ms,"host_enqueue_ms":enqueue_ms,
            "host_enqueue_and_completion_ms":began.elapsed().as_secs_f64()*1000.0/iterations as f64,
            "iterations":iterations,"started_unix_us":started,"enqueue_counters":counters}),
        )
    }
}

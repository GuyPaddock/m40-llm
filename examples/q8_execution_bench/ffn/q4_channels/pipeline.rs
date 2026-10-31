use super::super::pipeline::GuardedF32;
use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{CudaContext, CudaEvent, CudaStream, DeviceQ4SelectedFfn, Q4DownLayout},
    profile,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Arm {
    DenseSerial,
    DenseBatched,
    NativeSelected,
    PackedSelected,
}

pub(super) struct Pipeline<'a> {
    pub ctx: &'a CudaContext,
    pub selected: DeviceQ4SelectedFfn,
    pub input: GuardedF32,
    pub output: GuardedF32,
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
        selected: DeviceQ4SelectedFfn,
        input: &[f32],
        k: usize,
        h: usize,
    ) -> Result<Self> {
        let rows = input.len() / k;
        Ok(Self {
            ctx,
            selected,
            rows,
            k,
            h,
            input: GuardedF32::new(ctx, input, "q4_probe:input")?,
            output: GuardedF32::new(ctx, &vec![0.0; rows * k], "q4_probe:output")?,
            gate: GuardedF32::new(ctx, &vec![0.0; rows * h], "q4_probe:gate")?,
            up: GuardedF32::new(ctx, &vec![0.0; rows * h], "q4_probe:up")?,
            hidden: GuardedF32::new(ctx, &vec![0.0; rows * h], "q4_probe:hidden")?,
            start: ctx.create_event()?,
            stop: ctx.create_event()?,
        })
    }

    pub fn scratch_bytes(&self) -> usize {
        self.selected.workspace_bytes()
            + [
                &self.input,
                &self.output,
                &self.gate,
                &self.up,
                &self.hidden,
            ]
            .iter()
            .map(|v| v.bytes())
            .sum::<usize>()
    }

    pub fn drain(&self) -> Result<()> {
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        self.ctx.synchronize_stream(CudaStream::Prefill)
    }

    /// All pointers belong to this invocation; both dense paths use the native
    /// weights in the selected owner, with identical residency in every arm.
    pub fn enqueue(&self, arm: Arm) -> Result<()> {
        let [gate, up, down, _, _, _] = self.selected.immutable_buffers();
        unsafe {
            match arm {
                Arm::NativeSelected | Arm::PackedSelected => {
                    let layout = if matches!(arm, Arm::NativeSelected) {
                        Q4DownLayout::Native
                    } else {
                        Q4DownLayout::ChannelMajor
                    };
                    self.selected.enqueue_async(
                        self.input.pointer(),
                        self.output.pointer(),
                        layout,
                        CudaStream::Decode,
                    )?;
                }
                Arm::DenseSerial => {
                    for row in 0..self.rows {
                        let input = self.input.pointer().cast::<f32>().add(row * self.k).cast();
                        let hidden = self.hidden.pointer().cast::<f32>().add(row * self.h).cast();
                        let output = self.output.pointer().cast::<f32>().add(row * self.k).cast();
                        self.ctx.mlp_gate_up_swiglu_f32xq4_0_gguf_decode_async(
                            input,
                            gate.as_ptr(),
                            up.as_ptr(),
                            hidden,
                            self.h as i32,
                            self.k as i32,
                        )?;
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
                Arm::DenseBatched => {
                    // The Q4 block-loop projections use prefill, SwiGLU uses decode.
                    self.ctx.stream_wait_for_stream(
                        CudaStream::Prefill,
                        CudaStream::Decode,
                        "q4_probe:before_gate",
                    )?;
                    for (weight, out) in [(gate, &self.gate), (up, &self.up)] {
                        self.ctx.gemm_f32xq4_0_gguf_f32_blockloop_async(
                            self.input.pointer(),
                            weight.as_ptr(),
                            out.pointer(),
                            self.rows as i32,
                            self.h as i32,
                            self.k as i32,
                        )?;
                    }
                    self.ctx.stream_wait_for_stream(
                        CudaStream::Decode,
                        CudaStream::Prefill,
                        "q4_probe:before_swiglu",
                    )?;
                    self.ctx.swiglu_f32_async(
                        self.gate.pointer(),
                        self.up.pointer(),
                        self.hidden.pointer(),
                        self.rows * self.h,
                    )?;
                    self.ctx.stream_wait_for_stream(
                        CudaStream::Prefill,
                        CudaStream::Decode,
                        "q4_probe:before_down",
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
                        "q4_probe:after_down",
                    )?;
                }
            }
        }
        Ok(())
    }

    pub fn read(&self) -> Result<Vec<f32>> {
        self.drain()?;
        for guard in [&self.input, &self.gate, &self.up, &self.hidden] {
            guard.read(self.ctx)?;
        }
        self.output.read(self.ctx)
    }

    /// Follow the existing FFN timing boundary: warm allocations, CUDA events,
    /// host enqueue plus one completion wait, and explicit counter deltas.
    pub fn time(&self, arm: Arm, iterations: usize) -> Result<Value> {
        ensure!(iterations > 0, "empty Q4 timing interval");
        self.drain()?;
        let before = profile::snapshot();
        let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let began = Instant::now();
        self.start.record(CudaStream::Decode)?;
        for _ in 0..iterations {
            self.enqueue(arm)?;
        }
        self.stop.record(CudaStream::Decode)?;
        let enqueue_ms = began.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        let counters = profile::counter_delta(&before, &profile::snapshot());
        let expected = match arm {
            Arm::DenseSerial => 2 * self.rows,
            Arm::DenseBatched => 4,
            _ => 3,
        };
        ensure!(
            counters.launches == (iterations * expected) as u64,
            "Q4 dispatch count mismatch"
        );
        ensure!(
            counters.device_allocs == 0
                && counters.device_frees == 0
                && counters.h2d_copies == 0
                && counters.d2h_copies == 0
                && counters.stream_syncs == 0,
            "Q4 timed enqueue allocated/copied/synchronized"
        );
        let gpu_ms = f64::from(self.start.elapsed_sync(&self.stop, "q4_probe:complete")?)
            / iterations as f64;
        ensure!(gpu_ms.is_finite() && gpu_ms > 0.0, "invalid Q4 timing");
        Ok(
            json!({"arm":arm,"gpu_ms":gpu_ms,"host_enqueue_ms":enqueue_ms,
            "host_enqueue_and_completion_ms":began.elapsed().as_secs_f64()*1000.0/iterations as f64,
            "iterations":iterations,"started_unix_us":started,"enqueue_counters":counters}),
        )
    }
}

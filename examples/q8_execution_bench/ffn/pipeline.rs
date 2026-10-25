use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, CudaEvent, CudaStream, DeviceBuffer, Q8FfnFp32Variant, Q8FfnQ81Launch},
    evaluation::ffn_capture::FfnLayerSamples,
    infer::{q8_activation::Q8ActivationShape, LoadedModel},
    profile,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    ffi::c_void,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const GUARD_VALUES: usize = 8;
const GUARD_BITS: u32 = 0x4b12_3456;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub(super) enum FfnKernel {
    Native,
    Fp32(Q8FfnFp32Variant),
    Q81(Q8FfnQ81Launch),
}

impl FfnKernel {
    pub fn has_quantization(self) -> bool {
        matches!(self, Self::Q81(_))
    }

    pub fn threads(self) -> u32 {
        match self {
            Self::Native => 256,
            Self::Fp32(variant) => variant.threads(),
            Self::Q81(launch) => launch.threads() as u32,
        }
    }

    pub fn has_packed_loads(self) -> bool {
        match self {
            Self::Native => false,
            Self::Fp32(variant) => variant.has_packed_loads(),
            Self::Q81(_) => true,
        }
    }

    pub fn has_warp_reduction(self) -> bool {
        match self {
            Self::Native => false,
            Self::Fp32(variant) => variant.has_warp_reduction(),
            Self::Q81(_) => true,
        }
    }
}

pub(super) struct GuardedQ8Activation {
    storage: DeviceBuffer,
    pub shape: Q8ActivationShape,
}

impl GuardedQ8Activation {
    fn new(ctx: &CudaContext, rows: usize, columns: usize) -> Result<Self> {
        let shape = Q8ActivationShape::new(rows, columns, columns)?;
        let storage = DeviceBuffer::new_tagged(ctx, shape.bytes() + 32, "ffn_bench:q8-activation")?;
        let guards = vec![0xa5u8; storage.bytes()];
        unsafe {
            ctx.memcpy_h2d(storage.as_mut_ptr(), guards.as_ptr().cast(), guards.len())?;
        }
        Ok(Self { storage, shape })
    }

    pub fn pointer(&self) -> *mut c_void {
        unsafe { self.storage.as_mut_ptr().cast::<u8>().add(16).cast() }
    }

    pub fn read(&self, ctx: &CudaContext) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; self.storage.bytes()];
        unsafe {
            ctx.memcpy_d2h(
                bytes.as_mut_ptr().cast(),
                self.storage.as_ptr(),
                bytes.len(),
            )?;
        }
        ensure!(
            bytes[..16]
                .iter()
                .chain(&bytes[16 + self.shape.bytes()..])
                .all(|&v| v == 0xa5),
            "FFN quantized activation guard overwritten"
        );
        Ok(bytes[16..16 + self.shape.bytes()].to_vec())
    }
}

pub(super) struct GuardedF32 {
    storage: DeviceBuffer,
    values: usize,
}

impl GuardedF32 {
    pub fn new(ctx: &CudaContext, values: &[f32], tag: &'static str) -> Result<Self> {
        let count = values
            .len()
            .checked_add(2 * GUARD_VALUES)
            .context("FFN buffer overflow")?;
        let bytes = count.checked_mul(4).context("FFN byte overflow")?;
        let storage = DeviceBuffer::new_tagged(ctx, bytes, tag)?;
        let mut host = vec![f32::from_bits(GUARD_BITS); count];
        host[GUARD_VALUES..GUARD_VALUES + values.len()].copy_from_slice(values);
        unsafe {
            ctx.memcpy_h2d(storage.as_mut_ptr(), host.as_ptr().cast(), bytes)?;
        }
        Ok(Self {
            storage,
            values: values.len(),
        })
    }

    pub fn pointer(&self) -> *mut c_void {
        // Eight f32 guards preserve 32-byte alignment of every matrix row zero.
        unsafe {
            self.storage
                .as_mut_ptr()
                .cast::<f32>()
                .add(GUARD_VALUES)
                .cast()
        }
    }

    pub fn read(&self, ctx: &CudaContext) -> Result<Vec<f32>> {
        let all = super::super::read_output(ctx, &self.storage)?;
        ensure!(
            all[..GUARD_VALUES]
                .iter()
                .chain(&all[GUARD_VALUES + self.values..])
                .all(|value| value.to_bits() == GUARD_BITS),
            "FFN output/input guard overwritten"
        );
        Ok(all[GUARD_VALUES..GUARD_VALUES + self.values].to_vec())
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Stage {
    Quantize,
    GateUpSwiglu,
    IsolatedDown,
    Complete,
}

pub(super) struct FfnPipeline<'a> {
    pub model: &'a LoadedModel,
    pub rows: usize,
    pub d_model: usize,
    pub hidden_dim: usize,
    gate_weight: *const c_void,
    up_weight: *const c_void,
    down_weight: *const c_void,
    pub input: GuardedF32,
    pub gate: GuardedF32,
    pub up: GuardedF32,
    pub hidden: GuardedF32,
    pub captured_hidden: GuardedF32,
    pub output: GuardedF32,
    pub isolated_output: GuardedF32,
    start: CudaEvent,
    stop: CudaEvent,
    pub kernel: FfnKernel,
    pub quantized: Option<GuardedQ8Activation>,
}

impl<'a> FfnPipeline<'a> {
    pub fn new(model: &'a LoadedModel, layer: &FfnLayerSamples, rows: usize) -> Result<Self> {
        ensure!(
            matches!(rows, 1 | 4) && layer.rows.len() >= rows,
            "FFN pipeline requires 1 or 4 actual input rows"
        );
        let mapped = model.map_standard_layer(layer.source.layer_id)?;
        let (d_model, hidden_dim) = (mapped.d_model, mapped.hidden_dim);
        ensure!(
            d_model <= i32::MAX as usize
                && hidden_dim <= i32::MAX as usize
                && layer.normalized_inputs.len() >= rows * d_model
                && layer.swiglu_inputs.len() >= rows * hidden_dim,
            "FFN input dimensions/ranges differ from model"
        );
        let ctx = &model.cuda;
        Ok(Self {
            model,
            rows,
            d_model,
            hidden_dim,
            gate_weight: mapped.w_gate.dptr.cast_const(),
            up_weight: mapped.w_up.dptr.cast_const(),
            down_weight: mapped.w_down.dptr.cast_const(),
            input: GuardedF32::new(
                ctx,
                &layer.normalized_inputs[..rows * d_model],
                "ffn_bench:input",
            )?,
            gate: GuardedF32::new(ctx, &vec![0.0; rows * hidden_dim], "ffn_bench:gate")?,
            up: GuardedF32::new(ctx, &vec![0.0; rows * hidden_dim], "ffn_bench:up")?,
            hidden: GuardedF32::new(ctx, &vec![0.0; rows * hidden_dim], "ffn_bench:hidden")?,
            captured_hidden: GuardedF32::new(
                ctx,
                &layer.swiglu_inputs[..rows * hidden_dim],
                "ffn_bench:captured-hidden",
            )?,
            output: GuardedF32::new(ctx, &vec![0.0; rows * d_model], "ffn_bench:output")?,
            isolated_output: GuardedF32::new(
                ctx,
                &vec![0.0; rows * d_model],
                "ffn_bench:isolated-output",
            )?,
            start: ctx.create_event()?,
            stop: ctx.create_event()?,
            kernel: FfnKernel::Native,
            quantized: None,
        })
    }

    pub fn drain(&self) -> Result<()> {
        self.model.cuda.synchronize_stream(CudaStream::Decode)?;
        self.model.cuda.synchronize_stream(CudaStream::Prefill)
    }

    /// Reuse the same scratch only after its previous producer has finished.
    pub fn select_kernel(&mut self, kernel: FfnKernel) -> Result<()> {
        ensure!(
            !kernel.has_quantization() || self.quantized.is_some(),
            "reserve quantized FFN scratch before selecting Q8_1"
        );
        self.drain()?;
        self.kernel = kernel;
        Ok(())
    }

    pub fn reserve_quantized_scratch(&mut self) -> Result<()> {
        ensure!(
            self.quantized.is_none(),
            "Q8_1 FFN scratch already reserved"
        );
        self.drain()?;
        self.quantized = Some(GuardedQ8Activation::new(
            &self.model.cuda,
            self.rows,
            self.d_model,
        )?);
        Ok(())
    }

    fn projection_stream(&self) -> CudaStream {
        if self.rows == 1 {
            CudaStream::Decode
        } else {
            CudaStream::Prefill
        }
    }

    pub fn scratch_bytes(&self) -> usize {
        [
            &self.input,
            &self.gate,
            &self.up,
            &self.hidden,
            &self.captured_hidden,
            &self.output,
            &self.isolated_output,
        ]
        .iter()
        .map(|buffer| buffer.storage.bytes())
        .sum::<usize>()
            + self.quantized.as_ref().map_or(0, |q| q.storage.bytes())
    }

    pub fn enqueue_gate_up_projections(&self) -> Result<()> {
        if self.kernel.has_quantization() {
            return self.enqueue_q81(true);
        }
        if self.rows > 1 {
            // The previous iteration's SwiGLU may still be reading gate/up.
            self.model.cuda.stream_wait_for_stream(
                CudaStream::Prefill,
                CudaStream::Decode,
                "ffn_bench_gate_reuse",
            )?;
        }
        unsafe {
            self.model.mlp_gates_f32xf16_gguf_f32_async(
                self.input.pointer(),
                self.rows as i32,
                self.d_model as i32,
                self.gate_weight,
                self.up_weight,
                self.hidden_dim as i32,
                self.gate.pointer(),
                self.up.pointer(),
            )?;
        }
        Ok(())
    }

    fn enqueue_gate_up_swiglu(&self) -> Result<()> {
        if self.kernel.has_quantization() {
            return self.enqueue_q81(false);
        }
        if let FfnKernel::Fp32(variant) = self.kernel {
            if self.rows > 1 {
                // Previous down work on prefill may still consume hidden.
                self.model.cuda.stream_wait_for_stream(
                    CudaStream::Decode,
                    CudaStream::Prefill,
                    "ffn_bench_fused_hidden_reuse",
                )?;
            }
            unsafe {
                self.model.cuda.ffn_q8_f32_ablation_async(
                    self.input.pointer(),
                    self.gate_weight,
                    self.up_weight,
                    self.hidden.pointer(),
                    self.rows as i32,
                    self.hidden_dim as i32,
                    self.d_model as i32,
                    variant,
                )?;
            }
        } else if self.rows == 1 {
            let fused = unsafe {
                self.model
                    .try_mlp_gate_up_swiglu_f32xf16_gguf_decode_async(
                        self.input.pointer(),
                        1,
                        self.d_model as i32,
                        self.gate_weight,
                        self.up_weight,
                        self.hidden_dim as i32,
                        self.hidden.pointer(),
                    )?
            };
            ensure!(
                fused,
                "FFN baseline requires native fused Q8 gate/up/SwiGLU; no silent fallback"
            );
        } else {
            self.enqueue_gate_up_projections()?;
            self.model.cuda.stream_wait_for_stream(
                CudaStream::Decode,
                CudaStream::Prefill,
                "ffn_bench_gate_to_swiglu",
            )?;
            unsafe {
                self.model.cuda.swiglu_f32_async(
                    self.gate.pointer(),
                    self.up.pointer(),
                    self.hidden.pointer(),
                    self.rows * self.hidden_dim,
                )?;
            }
        }
        Ok(())
    }

    fn enqueue_quantization(&self) -> Result<()> {
        ensure!(
            self.kernel.has_quantization(),
            "quantization requires the explicit Q8_1 kernel"
        );
        let quantized = self
            .quantized
            .as_ref()
            .context("missing Q8_1 FFN scratch")?;
        if self.rows > 1 {
            self.model.cuda.stream_wait_for_stream(
                CudaStream::Decode,
                CudaStream::Prefill,
                "ffn_bench_quantized_input_reuse",
            )?;
        }
        unsafe {
            self.model.cuda.quantize_q8_1_async(
                self.input.pointer(),
                quantized.pointer(),
                quantized.shape,
            )?;
        }
        Ok(())
    }

    fn enqueue_q81(&self, has_intermediates: bool) -> Result<()> {
        let FfnKernel::Q81(launch) = self.kernel else {
            anyhow::bail!("Q8_1 consumer was not selected")
        };
        let quantized = self
            .quantized
            .as_ref()
            .context("missing Q8_1 FFN scratch")?;
        self.enqueue_quantization()?;
        unsafe {
            self.model.cuda.ffn_q8_1_async(
                quantized.pointer(),
                self.gate_weight,
                self.up_weight,
                self.hidden.pointer(),
                has_intermediates.then_some((self.gate.pointer(), self.up.pointer())),
                self.hidden_dim as i32,
                quantized.shape,
                launch,
            )?;
        }
        Ok(())
    }

    fn enqueue_down(&self, captured: bool) -> Result<()> {
        if self.rows > 1 {
            self.model.cuda.stream_wait_for_stream(
                CudaStream::Prefill,
                CudaStream::Decode,
                "ffn_bench_swiglu_to_down",
            )?;
        }
        let (input, output) = if captured {
            (&self.captured_hidden, &self.isolated_output)
        } else {
            (&self.hidden, &self.output)
        };
        unsafe {
            self.model.mlp_down_proj_f32xf16_gguf_f32_async(
                input.pointer(),
                self.rows as i32,
                self.hidden_dim as i32,
                self.down_weight,
                self.d_model as i32,
                output.pointer(),
            )?;
        }
        Ok(())
    }

    pub fn enqueue(&self, stage: Stage) -> Result<()> {
        match stage {
            Stage::Quantize => self.enqueue_quantization(),
            Stage::GateUpSwiglu => self.enqueue_gate_up_swiglu(),
            Stage::IsolatedDown => self.enqueue_down(true),
            Stage::Complete => {
                self.enqueue_gate_up_swiglu()?;
                self.enqueue_down(false)
            }
        }
    }

    pub fn time(&self, stage: Stage, iterations: usize) -> Result<Value> {
        ensure!(iterations > 0, "FFN timing needs at least one iteration");
        ensure!(
            !matches!(stage, Stage::Quantize) || self.kernel.has_quantization(),
            "native FFN has no quantization stage"
        );
        self.drain()?;
        let start_stream =
            if self.kernel != FfnKernel::Native && !matches!(stage, Stage::IsolatedDown) {
                CudaStream::Decode
            } else {
                self.projection_stream()
            };
        let stop_stream = if matches!(stage, Stage::GateUpSwiglu | Stage::Quantize) {
            CudaStream::Decode
        } else {
            self.projection_stream()
        };
        let before = profile::snapshot();
        let started_unix_us = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let began = Instant::now();
        self.start.record(start_stream)?;
        for _ in 0..iterations {
            self.enqueue(stage)?;
        }
        self.stop.record(stop_stream)?;
        let enqueued = profile::counter_delta(&before, &profile::snapshot());
        ensure!(
            enqueued.device_allocs == 0
                && enqueued.device_frees == 0
                && enqueued.h2d_copies == 0
                && enqueued.d2h_copies == 0
                && enqueued.stream_syncs == 0,
            "FFN timed enqueue allocated, copied or synchronized"
        );
        let gpu_ms = f64::from(self.start.elapsed_sync(&self.stop, "ffn_benchmark_stage")?)
            / iterations as f64;
        let wall_ms = began.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        Ok(
            json!({"stage":stage,"gpu_ms":gpu_ms,"host_enqueue_and_completion_ms":wall_ms,"iterations":iterations,
            "started_unix_us":started_unix_us,"completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros(),
            "enqueue_counters":enqueued,"completion":"one event synchronization after the repeated stage; excluded from enqueue counters"}),
        )
    }

    /// Run a fixed minimum of measured complete-FFN work, not a search for a
    /// favorable latency. A finite batch cap fails explicitly if it cannot finish.
    pub fn warm_up(&self, minimum_gpu_ms: f64, maximum_batches: usize) -> Result<Value> {
        ensure!(
            minimum_gpu_ms.is_finite()
                && minimum_gpu_ms > 0.0
                && (1..=256).contains(&maximum_batches),
            "invalid bounded FFN warmup policy"
        );
        let began = Instant::now();
        let mut elapsed_gpu_ms = 0.0;
        for batch in 1..=maximum_batches {
            let timing = self.time(Stage::Complete, 32)?;
            elapsed_gpu_ms += timing["gpu_ms"]
                .as_f64()
                .context("missing warmup GPU time")?
                * 32.0;
            if elapsed_gpu_ms >= minimum_gpu_ms {
                return Ok(
                    json!({"minimum_gpu_ms":minimum_gpu_ms,"elapsed_gpu_ms":elapsed_gpu_ms,
                    "batches":batch,"iterations_per_batch":32,"maximum_batches":maximum_batches,
                    "wall_ms":began.elapsed().as_secs_f64()*1000.0,"scope":"untimed warmup; not pooled with primary trials; no clock settings modified"}),
                );
            }
        }
        anyhow::bail!("bounded FFN warmup exhausted {maximum_batches} batches after {elapsed_gpu_ms} ms; required {minimum_gpu_ms} ms")
    }

    pub fn verify_guards_and_inputs(&self, layer: &FfnLayerSamples) -> Result<()> {
        self.drain()?;
        if let Some(quantized) = &self.quantized {
            quantized.read(&self.model.cuda)?;
        }
        for buffer in [
            &self.gate,
            &self.up,
            &self.hidden,
            &self.output,
            &self.isolated_output,
        ] {
            buffer.read(&self.model.cuda)?;
        }
        for (buffer, expected) in [
            (
                &self.input,
                &layer.normalized_inputs[..self.rows * self.d_model],
            ),
            (
                &self.captured_hidden,
                &layer.swiglu_inputs[..self.rows * self.hidden_dim],
            ),
        ] {
            let actual = buffer.read(&self.model.cuda)?;
            ensure!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "FFN benchmark modified immutable input values"
            );
        }
        Ok(())
    }
}

impl Drop for FfnPipeline<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.drain() {
            eprintln!("[q8_execution_bench] FFN scratch drain failed: {error:#}");
        }
    }
}

use super::super::{fixture::source_bytes, pipeline::GuardedF32, validation};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaEvent, CudaStream},
    evaluation::{ffn_capture::FfnTensorSource, identity::sha256},
    gguf::GgmlDType,
    infer::{DeviceTensorView, LoadedModel},
    profile,
};
use serde_json::{json, Value};
use std::{
    ffi::c_void,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub(super) struct QkvProbe<'a> {
    model: &'a LoadedModel,
    pub rows: usize,
    k: usize,
    widths: [usize; 3],
    weights: [DeviceTensorView; 3],
    biases: [Option<DeviceTensorView>; 3],
    pub sources: [FfnTensorSource; 3],
    input: GuardedF32,
    outputs: [GuardedF32; 3],
    original_input: Vec<f32>,
    start: CudaEvent,
    stop: CudaEvent,
}

impl<'a> QkvProbe<'a> {
    pub fn new(model: &'a LoadedModel, layer: usize, rows: usize, input: &[f32]) -> Result<Self> {
        let mapped = model.map_standard_layer(layer)?;
        let k = mapped.d_model;
        ensure!(
            matches!(rows, 1 | 4) && input.len() == rows * k,
            "QKV probe requires distinct 1/4 input rows"
        );
        let weights = [mapped.wq, mapped.wk, mapped.wv];
        let biases = [mapped.bq, mapped.bk, mapped.bv];
        ensure!(
            k <= i32::MAX as usize
                && weights
                    .iter()
                    .all(|w| w.shape.len() == 2 && w.shape[1] > 0 && w.shape[1] <= i32::MAX as u64),
            "QKV probe dimensions exceed CUDA interface"
        );
        let widths = std::array::from_fn(|i| weights[i].shape[1] as usize);
        let sources = weights
            .iter()
            .map(|view| -> Result<FfnTensorSource> {
                ensure!(
                    view.dtype == GgmlDType::Q8_0
                        && view.shape.len() == 2
                        && view.shape[0] == k as u64,
                    "QKV probe requires raw Q8_0 [K,N] weights"
                );
                let tensor = model
                    .gguf
                    .tensors
                    .iter()
                    .find(|t| t.offset == view.byte_offset && t.shape == view.shape)
                    .context("missing QKV source tensor")?;
                let mut source = FfnTensorSource {
                    name: tensor.name.clone(),
                    byte_offset: view.byte_offset,
                    dtype: view.dtype,
                    shape: [k, view.shape[1] as usize],
                    source_sha256: String::new(),
                };
                source.source_sha256 = sha256(source_bytes(model, &source)?)?;
                Ok(source)
            })
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| anyhow::anyhow!("missing QKV source"))?;
        for (i, bias) in biases.iter().enumerate() {
            if let Some(bias) = bias {
                ensure!(
                    bias.dtype == GgmlDType::F32 && bias.nbytes == widths[i] * 4,
                    "QKV probe requires FP32 bias vectors"
                );
            }
        }
        let outputs = widths
            .map(|n| GuardedF32::new(&model.cuda, &vec![0.0; rows * n], "qkv_probe:output"))
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| anyhow::anyhow!("missing QKV output"))?;
        Ok(Self {
            model,
            rows,
            k,
            widths,
            weights,
            biases,
            sources,
            input: GuardedF32::new(&model.cuda, input, "qkv_probe:input")?,
            outputs,
            original_input: input.to_vec(),
            start: model.cuda.create_event()?,
            stop: model.cuda.create_event()?,
        })
    }

    pub fn bytes(&self) -> usize {
        self.input.bytes() + self.outputs.iter().map(GuardedF32::bytes).sum::<usize>()
    }

    pub fn enqueue(&self, is_warp: bool) -> Result<()> {
        let weights = self.weights.each_ref().map(|v| v.dptr.cast_const());
        let biases = self
            .biases
            .each_ref()
            .map(|v| v.as_ref().map(|v| v.dptr.cast_const()));
        for row in 0..self.rows {
            unsafe {
                let input = self.input.pointer().cast::<f32>().add(row * self.k).cast();
                let outputs = std::array::from_fn(|i| {
                    self.outputs[i]
                        .pointer()
                        .cast::<f32>()
                        .add(row * self.widths[i])
                        .cast()
                });
                if is_warp {
                    self.model.cuda.qkv_f32xq8_0_gguf_warp_async(
                        input,
                        weights,
                        biases,
                        outputs,
                        self.widths.map(|n| n as i32),
                        self.k as i32,
                        CudaStream::Decode,
                    )?;
                } else {
                    self.model.cuda.qkv_f32xq8_0_gguf_decode_async(
                        input,
                        weights[0],
                        weights[1],
                        weights[2],
                        biases[0],
                        biases[1],
                        biases[2],
                        outputs[0],
                        outputs[1],
                        outputs[2],
                        self.widths[0] as i32,
                        self.widths[1] as i32,
                        self.widths[2] as i32,
                        self.k as i32,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn read(&self) -> Result<[Vec<f32>; 3]> {
        self.model.cuda.synchronize_stream(CudaStream::Decode)?;
        self.outputs
            .each_ref()
            .map(|b| b.read(&self.model.cuda))
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| anyhow::anyhow!("missing QKV results"))
    }

    pub fn verify(&self) -> Result<()> {
        self.model.cuda.synchronize_stream(CudaStream::Decode)?;
        ensure!(
            self.input
                .read(&self.model.cuda)?
                .iter()
                .zip(&self.original_input)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "QKV probe changed input"
        );
        for buffer in &self.outputs {
            buffer.read(&self.model.cuda)?;
        }
        for i in 0..3 {
            let source = source_bytes(self.model, &self.sources[i])?;
            self.verify_span(self.weights[i].dptr, source)?;
            if let Some(bias) = &self.biases[i] {
                let start = bias.byte_offset as usize;
                self.verify_span(
                    bias.dptr,
                    &self.model.host_weights[start..start + bias.nbytes],
                )?;
            }
        }
        Ok(())
    }

    fn verify_span(&self, ptr: *const c_void, expected: &[u8]) -> Result<()> {
        let mut bytes = vec![0u8; expected.len()];
        unsafe {
            self.model
                .cuda
                .memcpy_d2h(bytes.as_mut_ptr().cast(), ptr, bytes.len())?;
        }
        ensure!(bytes == expected, "QKV probe changed a weight/bias span");
        Ok(())
    }

    pub fn validate(&self) -> Result<Value> {
        self.enqueue(false)?;
        let control = self.read()?;
        self.enqueue(true)?;
        let candidate = self.read()?;
        let mut errors = Vec::new();
        for i in 0..3 {
            ensure!(
                candidate[i]
                    .iter()
                    .zip(&control[i])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "QKV warp changed original reduction bits"
            );
            let mut cpu = validation::project(
                &self.original_input,
                source_bytes(self.model, &self.sources[i])?,
                &self.sources[i],
                self.rows,
            )?;
            if let Some(bias) = &self.biases[i] {
                let start = bias.byte_offset as usize;
                let biases: Vec<f32> = self.model.host_weights[start..start + bias.nbytes]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                for row in cpu.chunks_exact_mut(self.widths[i]) {
                    for (v, b) in row.iter_mut().zip(&biases) {
                        *v += f64::from(*b);
                    }
                }
            }
            errors.push(
                json!({"source":self.sources[i],"cpu":validation::compare(&candidate[i],&cpu)?}),
            );
        }
        self.verify()?;
        Ok(
            json!({"control_bits_equal":true,"guards_and_immutable_read_spans":true,"outputs":errors,
            "biases":self.biases.each_ref().map(|b|b.as_ref().map(|b|json!({"offset":b.byte_offset,"bytes":b.nbytes}))),
            "cpu_reference":"existing independent scalar Q8_0/FP64 projection plus original FP32 bias"}),
        )
    }

    pub fn time(&self, is_warp: bool, iterations: usize) -> Result<Value> {
        ensure!(iterations > 0, "QKV timing requires iterations");
        let expected_launches = u64::try_from(
            iterations
                .checked_mul(self.rows)
                .context("QKV launch count overflow")?,
        )?;
        self.model.cuda.synchronize_stream(CudaStream::Decode)?;
        let before = profile::snapshot();
        let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let began = Instant::now();
        self.start.record(CudaStream::Decode)?;
        for _ in 0..iterations {
            self.enqueue(is_warp)?;
        }
        self.stop.record(CudaStream::Decode)?;
        let after = profile::snapshot();
        let counters = profile::counter_delta(&before, &after);
        ensure!(
            counters.launches == expected_launches
                && counters.device_allocs == 0
                && counters.device_frees == 0
                && counters.h2d_copies == 0
                && counters.d2h_copies == 0
                && counters.stream_syncs == 0
                && counters.stream_waits == 0
                && counters.cublas_calls == 0,
            "unexpected QKV timed work"
        );
        let op = if is_warp {
            "qkv_f32xq8_0_gguf_warp"
        } else {
            "qkv_f32xq8_0_gguf_decode"
        };
        let launches = after.by_op.get(op).map_or(0, |v| v.launches)
            - before.by_op.get(op).map_or(0, |v| v.launches);
        ensure!(
            launches == expected_launches,
            "QKV timing dispatched wrong operation"
        );
        let gpu_ms =
            f64::from(self.start.elapsed_sync(&self.stop, "qkv_probe_timing")?) / iterations as f64;
        Ok(
            json!({"is_warp":is_warp,"gpu_ms":gpu_ms,"gpu_ms_per_request":gpu_ms/self.rows as f64,
            "host_enqueue_and_completion_ms":began.elapsed().as_secs_f64()*1000.0/iterations as f64,
            "iterations":iterations,"enqueue_counters":counters,"operation":op,"operation_launches":launches,
            "started_unix_us":started,"completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros()}),
        )
    }

    pub fn warm(&self, is_warp: bool, minimum_ms: f64) -> Result<Value> {
        let mut elapsed = 0.0;
        for batch in 1..=256 {
            elapsed += self.time(is_warp, 128)?["gpu_ms"]
                .as_f64()
                .context("missing QKV warmup")?
                * 128.0;
            if elapsed >= minimum_ms {
                return Ok(json!({"gpu_ms":elapsed,"batches":batch,"iterations_per_batch":128}));
            }
        }
        anyhow::bail!("QKV bounded warmup exhausted")
    }
}

impl Drop for QkvProbe<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.model.cuda.synchronize_stream(CudaStream::Decode) {
            eprintln!("[qkv_probe] drain failed: {error:#}");
        }
    }
}

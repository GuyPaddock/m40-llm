use super::super::{fixture::source_bytes, pipeline::GuardedF32, validation};
use super::prepared::PreparedProjectionWeights;
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, CudaEvent, CudaStream},
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProjectionKind {
    Qkv,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProjectionComparison {
    WarpReduction,
    PackedLoads,
    SplitPlanes,
}

impl ProjectionKind {
    pub fn operation(self, is_warp: bool) -> &'static str {
        match (self, is_warp) {
            (Self::Qkv, false) => "qkv_f32xq8_0_gguf_decode",
            (Self::Qkv, true) => "qkv_f32xq8_0_gguf_warp",
            (Self::Output, false) => "gemm_f32xq8_0_gguf_f32_decode",
            (Self::Output, true) => "gemm_f32xq8_0_gguf_coltile4_warp",
        }
    }
}

pub(super) struct ProjectionProbe<'a> {
    model: &'a LoadedModel,
    kind: ProjectionKind,
    comparison: ProjectionComparison,
    pub rows: usize,
    k: usize,
    widths: Vec<usize>,
    weights: Vec<DeviceTensorView>,
    biases: Vec<Option<DeviceTensorView>>,
    pub sources: Vec<FfnTensorSource>,
    input: GuardedF32,
    outputs: Vec<GuardedF32>,
    original_input: Vec<f32>,
    start: CudaEvent,
    stop: CudaEvent,
    prepared: Option<PreparedProjectionWeights>,
}

impl<'a> ProjectionProbe<'a> {
    pub fn new(
        model: &'a LoadedModel,
        layer: usize,
        rows: usize,
        input: &[f32],
        kind: ProjectionKind,
        comparison: ProjectionComparison,
    ) -> Result<Self> {
        let mapped = model.map_standard_layer(layer)?;
        let k = mapped.d_model;
        ensure!(
            matches!(rows, 1 | 4) && input.len() == rows * k,
            "projection probe requires distinct 1/4 input rows"
        );
        let (weights, biases) = match kind {
            ProjectionKind::Qkv => (
                vec![mapped.wq, mapped.wk, mapped.wv],
                vec![mapped.bq, mapped.bk, mapped.bv],
            ),
            ProjectionKind::Output => (vec![mapped.wo], vec![None]),
        };
        ensure!(
            k <= i32::MAX as usize
                && weights
                    .iter()
                    .all(|w| w.shape.len() == 2 && w.shape[1] > 0 && w.shape[1] <= i32::MAX as u64),
            "projection probe dimensions exceed CUDA interface"
        );
        let widths: Vec<_> = weights.iter().map(|w| w.shape[1] as usize).collect();
        let sources = weights
            .iter()
            .map(|view| -> Result<FfnTensorSource> {
                ensure!(
                    view.dtype == GgmlDType::Q8_0
                        && view.shape.len() == 2
                        && view.shape[0] == k as u64,
                    "projection probe requires raw Q8_0 [K,N] weights"
                );
                let tensor = model
                    .gguf
                    .tensors
                    .iter()
                    .find(|t| t.offset == view.byte_offset && t.shape == view.shape)
                    .context("missing projection source tensor")?;
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
            .collect::<Result<Vec<_>>>()?;
        for (i, bias) in biases.iter().enumerate() {
            if let Some(bias) = bias {
                ensure!(
                    bias.dtype == GgmlDType::F32 && bias.nbytes == widths[i] * 4,
                    "projection probe requires FP32 bias vectors"
                );
            }
        }
        let outputs = widths
            .iter()
            .map(|n| GuardedF32::new(&model.cuda, &vec![0.0; rows * n], "projection_probe:output"))
            .collect::<Result<Vec<_>>>()?;
        let input_buffer = GuardedF32::new(&model.cuda, input, "projection_probe:input")?;
        let prepared = (comparison == ProjectionComparison::SplitPlanes)
            .then(|| PreparedProjectionWeights::new(model, &sources))
            .transpose()?;
        Ok(Self {
            model,
            kind,
            comparison,
            rows,
            k,
            widths,
            weights,
            biases,
            sources,
            input: input_buffer,
            outputs,
            original_input: input.to_vec(),
            start: model.cuda.create_event()?,
            stop: model.cuda.create_event()?,
            prepared,
        })
    }

    pub fn bytes(&self) -> usize {
        self.scratch_bytes() + self.prepared.as_ref().map_or(0, |p| p.bytes)
    }

    pub fn scratch_bytes(&self) -> usize {
        self.input.bytes() + self.outputs.iter().map(GuardedF32::bytes).sum::<usize>()
    }

    pub fn preparation(&self) -> Option<&Value> {
        self.prepared.as_ref().map(|p| &p.report)
    }

    fn has_packed_loads(&self, is_candidate: bool) -> bool {
        self.comparison == ProjectionComparison::SplitPlanes
            || (self.comparison == ProjectionComparison::PackedLoads && is_candidate)
    }

    pub fn operation(&self, is_candidate: bool) -> &'static str {
        if self.comparison == ProjectionComparison::SplitPlanes && is_candidate {
            match self.kind {
                ProjectionKind::Qkv => "qkv_f32xq8_separate_packed_warp",
                ProjectionKind::Output => "gemm_f32xq8_separate_coltile4_packed_warp",
            }
        } else if self.has_packed_loads(is_candidate) {
            match self.kind {
                ProjectionKind::Qkv => "qkv_f32xq8_0_gguf_packed_warp",
                ProjectionKind::Output => "gemm_f32xq8_0_gguf_coltile4_packed_warp",
            }
        } else {
            self.kind
                .operation(is_candidate || self.comparison == ProjectionComparison::PackedLoads)
        }
    }

    pub fn enqueue(&self, is_candidate: bool) -> Result<()> {
        let has_packed_loads = self.has_packed_loads(is_candidate);
        let has_warp_reduction =
            is_candidate || self.comparison != ProjectionComparison::WarpReduction;
        for row in 0..self.rows {
            unsafe {
                let input = self.input.pointer().cast::<f32>().add(row * self.k).cast();
                if self.kind == ProjectionKind::Output {
                    let output = self.outputs[0]
                        .pointer()
                        .cast::<f32>()
                        .add(row * self.widths[0])
                        .cast();
                    let weight = self.weights[0].dptr.cast_const();
                    if self.comparison == ProjectionComparison::SplitPlanes && is_candidate {
                        self.prepared
                            .as_ref()
                            .context("missing split projection owner")?
                            .enqueue_output(input, output, CudaStream::Decode)?;
                    } else if has_warp_reduction {
                        let enqueue = if has_packed_loads {
                            CudaContext::gemm_f32xq8_0_gguf_coltile4_packed_warp_async
                        } else {
                            CudaContext::gemm_f32xq8_0_gguf_coltile4_warp_async
                        };
                        enqueue(
                            &self.model.cuda,
                            input,
                            weight,
                            output,
                            self.widths[0] as i32,
                            self.k as i32,
                            CudaStream::Decode,
                        )?;
                    } else {
                        self.model.cuda.gemm_f32xq8_0_gguf_f32_decode_async(
                            input,
                            weight,
                            output,
                            1,
                            self.widths[0] as i32,
                            self.k as i32,
                        )?;
                    }
                    continue;
                }
                let weights = std::array::from_fn(|i| self.weights[i].dptr.cast_const());
                let biases =
                    std::array::from_fn(|i| self.biases[i].as_ref().map(|v| v.dptr.cast_const()));
                let outputs = std::array::from_fn(|i| {
                    self.outputs[i]
                        .pointer()
                        .cast::<f32>()
                        .add(row * self.widths[i])
                        .cast()
                });
                if self.comparison == ProjectionComparison::SplitPlanes && is_candidate {
                    self.prepared
                        .as_ref()
                        .context("missing split projection owner")?
                        .enqueue_qkv(input, biases, outputs, CudaStream::Decode)?;
                } else if has_warp_reduction {
                    let enqueue = if has_packed_loads {
                        CudaContext::qkv_f32xq8_0_gguf_packed_warp_async
                    } else {
                        CudaContext::qkv_f32xq8_0_gguf_warp_async
                    };
                    enqueue(
                        &self.model.cuda,
                        input,
                        weights,
                        biases,
                        outputs,
                        std::array::from_fn(|i| self.widths[i] as i32),
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

    fn read(&self) -> Result<Vec<Vec<f32>>> {
        self.model.cuda.synchronize_stream(CudaStream::Decode)?;
        self.outputs
            .iter()
            .map(|b| b.read(&self.model.cuda))
            .collect()
    }

    pub fn verify(&self) -> Result<()> {
        self.model.cuda.synchronize_stream(CudaStream::Decode)?;
        if let Some(prepared) = &self.prepared {
            prepared.verify()?;
        }
        ensure!(
            self.input
                .read(&self.model.cuda)?
                .iter()
                .zip(&self.original_input)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "projection probe changed input"
        );
        for buffer in &self.outputs {
            buffer.read(&self.model.cuda)?;
        }
        for i in 0..self.weights.len() {
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
        ensure!(
            bytes == expected,
            "projection probe changed a weight/bias span"
        );
        Ok(())
    }

    pub fn validate(&self) -> Result<Value> {
        self.enqueue(false)?;
        let control = self.read()?;
        self.enqueue(true)?;
        let candidate = self.read()?;
        let mut errors = Vec::new();
        for i in 0..self.weights.len() {
            ensure!(
                candidate[i]
                    .iter()
                    .zip(&control[i])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "projection candidate changed timing-control bits"
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
            "biases":self.biases.iter().map(|b|b.as_ref().map(|b|json!({"offset":b.byte_offset,"bytes":b.nbytes}))).collect::<Vec<_>>(),
            "cpu_reference":"existing independent scalar Q8_0/FP64 projection plus original FP32 bias"}),
        )
    }

    pub fn time(&self, is_candidate: bool, iterations: usize) -> Result<Value> {
        ensure!(iterations > 0, "projection timing requires iterations");
        let expected_launches = u64::try_from(
            iterations
                .checked_mul(self.rows)
                .context("projection launch count overflow")?,
        )?;
        self.model.cuda.synchronize_stream(CudaStream::Decode)?;
        let before = profile::snapshot();
        let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let began = Instant::now();
        self.start.record(CudaStream::Decode)?;
        for _ in 0..iterations {
            self.enqueue(is_candidate)?;
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
            "unexpected projection timed work"
        );
        let op = self.operation(is_candidate);
        let launches = after.by_op.get(op).map_or(0, |v| v.launches)
            - before.by_op.get(op).map_or(0, |v| v.launches);
        ensure!(
            launches == expected_launches,
            "projection timing dispatched wrong operation"
        );
        let gpu_ms = f64::from(
            self.start
                .elapsed_sync(&self.stop, "projection_probe_timing")?,
        ) / iterations as f64;
        Ok(json!({"is_candidate":is_candidate,
            "is_warp":is_candidate || self.comparison!=ProjectionComparison::WarpReduction,
            "has_packed_loads":self.has_packed_loads(is_candidate),
            "split_projection_dispatch":if self.comparison==ProjectionComparison::SplitPlanes && is_candidate {
                json!({"operation":op,"launches":launches,"threads":256,
                    "columns":if self.kind==ProjectionKind::Qkv {1} else {4},
                    "shared_bytes":if self.kind==ProjectionKind::Qkv {1024} else {4096},
                    "payload_load_bytes":8,"additional_prepared_bytes":self.prepared.as_ref().unwrap().bytes})
            } else { Value::Null },
            "gpu_ms":gpu_ms,"gpu_ms_per_request":gpu_ms/self.rows as f64,
            "host_enqueue_and_completion_ms":began.elapsed().as_secs_f64()*1000.0/iterations as f64,
            "iterations":iterations,"enqueue_counters":counters,"operation":op,"operation_launches":launches,
            "started_unix_us":started,"completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros()}))
    }

    pub fn warm(&self, is_candidate: bool, minimum_ms: f64) -> Result<Value> {
        super::warmup::measure(minimum_ms, |iterations| {
            self.time(is_candidate, iterations)?["gpu_ms"]
                .as_f64()
                .context("missing projection warmup")
        })
    }
}

impl Drop for ProjectionProbe<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.model.cuda.synchronize_stream(CudaStream::Decode) {
            eprintln!("[projection_probe] drain failed: {error:#}");
        }
    }
}

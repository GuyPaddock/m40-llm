use super::{
    super::{pipeline::GuardedF32, validation, warmup},
    data::{hash_f32, GuardedWords},
};
use anyhow::{ensure, Context, Result};
use half::f16;
use m40_llm::{
    cuda::{CudaContext, CudaEvent, CudaStream},
    profile,
};
use serde_json::{json, Value};
use std::{
    ffi::c_void,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub(super) struct NormSource {
    pub pointer: *const c_void,
    pub bytes: Vec<u8>,
    pub dtype: u32,
    pub eps: f32,
}

enum Operation {
    Norm {
        source: NormSource,
        output: GuardedF32,
    },
    Argmax {
        output: GuardedWords,
        scratch: GuardedWords,
        capacity: usize,
    },
}

pub(super) struct ReductionProbe<'a> {
    ctx: &'a CudaContext,
    operation: Operation,
    input: GuardedF32,
    original: Vec<f32>,
    rows: usize,
    dim: usize,
    start: CudaEvent,
    stop: CudaEvent,
}

impl<'a> ReductionProbe<'a> {
    /// # Safety
    /// Source is a matching FP16/FP32 device weight span on ctx, remains
    /// immutable and alive until this probe is dropped, and has a completed upload.
    pub unsafe fn new_norm(
        ctx: &'a CudaContext,
        input: &[f32],
        rows: usize,
        source: NormSource,
    ) -> Result<Self> {
        ensure!(
            source.dtype <= 1 && source.eps.is_finite() && source.eps >= 0.0,
            "invalid norm source"
        );
        let width = if source.dtype == 0 { 2 } else { 4 };
        ensure!(
            !source.pointer.is_null() && source.bytes.len().is_multiple_of(width),
            "invalid norm weight span"
        );
        let dim = source.bytes.len() / width;
        let output = GuardedF32::new(ctx, &vec![0.0; input.len()], "reduction_probe:norm_output")?;
        Self::new(ctx, input, rows, dim, Operation::Norm { source, output })
    }

    pub fn new_argmax(
        ctx: &'a CudaContext,
        input: &[f32],
        rows: usize,
        dim: usize,
    ) -> Result<Self> {
        let capacity = CudaContext::argmax_f32_partitioned_scratch_bytes(u32::try_from(dim)?)?;
        let output = GuardedWords::new(ctx, rows)?;
        let scratch = GuardedWords::new(ctx, capacity / 4)?;
        Self::new(
            ctx,
            input,
            rows,
            dim,
            Operation::Argmax {
                output,
                scratch,
                capacity,
            },
        )
    }

    fn new(
        ctx: &'a CudaContext,
        input: &[f32],
        rows: usize,
        dim: usize,
        operation: Operation,
    ) -> Result<Self> {
        ensure!(
            matches!(rows, 1 | 4)
                && dim > 0
                && dim <= i32::MAX as usize
                && rows.checked_mul(dim) == Some(input.len())
                && input.iter().all(|v| v.is_finite()),
            "invalid finite reduction inputs"
        );
        for row in 1..rows {
            ensure!(
                input[..dim]
                    .iter()
                    .zip(&input[row * dim..(row + 1) * dim])
                    .any(|(a, b)| a.to_bits() != b.to_bits()),
                "reduction rows must be distinct"
            );
        }
        let input_buffer = GuardedF32::new(ctx, input, "reduction_probe:input")?;
        Ok(Self {
            ctx,
            operation,
            input: input_buffer,
            original: input.to_vec(),
            rows,
            dim,
            start: ctx.create_event()?,
            stop: ctx.create_event()?,
        })
    }

    pub fn bytes(&self) -> usize {
        self.input.bytes()
            + match &self.operation {
                Operation::Norm { output, .. } => output.bytes(),
                Operation::Argmax {
                    output, scratch, ..
                } => output.bytes() + scratch.bytes(),
            }
    }

    pub fn describe(&self) -> Result<Value> {
        Ok(
            json!({"input_sha256":hash_f32(&self.original)?, "rows":self.rows,"dim":self.dim,
            "scratch_capacity_bytes":self.bytes(),"candidate_partial_payload_bytes":match &self.operation { Operation::Argmax {capacity,..}=>*capacity, _=>0 },
            "threads":256,"control_grid":if matches!(self.operation,Operation::Norm{..}) {self.rows} else {1},
            "candidate_grid":if matches!(self.operation,Operation::Norm{..}) {json!([self.rows])} else {json!([self.dim.div_ceil(256).min(48),1])},
            "shared_bytes_per_block":if matches!(self.operation,Operation::Norm{..}) {1024} else {2048},
            "rows_scope":if matches!(self.operation,Operation::Norm{..}) {"one multi-row norm launch, one block per distinct row"} else {"independent requests, each control one launch or candidate two launches, one scratch reused serially"}}),
        )
    }

    pub fn operation_counts(&self, is_candidate: bool) -> Vec<(&'static str, u64)> {
        match (&self.operation, is_candidate) {
            (Operation::Norm { .. }, false) => vec![("rms_norm_f32_weighted", 1)],
            (Operation::Norm { .. }, true) => vec![("rms_norm_f32_weighted_warp", 1)],
            (Operation::Argmax { .. }, false) => vec![("argmax_f32", self.rows as u64)],
            (Operation::Argmax { .. }, true) => vec![
                ("argmax_f32_partition_partials", self.rows as u64),
                ("argmax_f32_partition_reduce", self.rows as u64),
            ],
        }
    }

    fn enqueue(&self, is_candidate: bool) -> Result<()> {
        unsafe {
            match &self.operation {
                Operation::Norm { source, output } => {
                    if is_candidate {
                        self.ctx.rms_norm_f32_weighted_warp_async(
                            self.input.pointer(),
                            source.pointer,
                            output.pointer(),
                            self.rows as u32,
                            self.dim as u32,
                            source.eps,
                            source.dtype,
                            CudaStream::Decode,
                        )
                    } else {
                        self.ctx.rms_norm_f32_weighted_async(
                            self.input.pointer(),
                            source.pointer,
                            output.pointer(),
                            self.rows as u32,
                            self.dim as u32,
                            source.eps,
                            source.dtype,
                        )
                    }
                }
                Operation::Argmax {
                    output,
                    scratch,
                    capacity,
                } => {
                    for row in 0..self.rows {
                        let values = self
                            .input
                            .pointer()
                            .cast::<f32>()
                            .add(row * self.dim)
                            .cast();
                        let out = output.pointer().cast::<u32>().add(row).cast();
                        if is_candidate {
                            self.ctx.argmax_f32_partitioned_async(
                                values,
                                self.dim as u32,
                                scratch.pointer(),
                                *capacity,
                                out,
                                CudaStream::Decode,
                            )?;
                        } else {
                            self.ctx.argmax_f32_async(
                                values,
                                self.dim as u32,
                                out,
                                CudaStream::Decode,
                            )?;
                        }
                    }
                    Ok(())
                }
            }
        }
    }

    fn read_bits(&self) -> Result<Vec<u32>> {
        self.ctx.synchronize_stream(CudaStream::Decode)?;
        match &self.operation {
            Operation::Norm { output, .. } => {
                Ok(output.read(self.ctx)?.iter().map(|v| v.to_bits()).collect())
            }
            Operation::Argmax {
                output, scratch, ..
            } => {
                scratch.read(self.ctx)?;
                output.read(self.ctx)
            }
        }
    }

    pub fn verify(&self) -> Result<()> {
        self.read_bits()?;
        ensure!(
            self.input
                .read(self.ctx)?
                .iter()
                .zip(&self.original)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "reduction input mutated"
        );
        if let Operation::Norm { source, .. } = &self.operation {
            let mut actual = vec![0u8; source.bytes.len()];
            unsafe {
                self.ctx
                    .memcpy_d2h(actual.as_mut_ptr().cast(), source.pointer, actual.len())?;
            }
            ensure!(
                actual == source.bytes,
                "norm weight mutated or source mismatch"
            );
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<Value> {
        self.enqueue(false)?;
        let control = self.read_bits()?;
        self.enqueue(true)?;
        let candidate = self.read_bits()?;
        ensure!(
            control == candidate,
            "reduction candidate changed control bits/indices"
        );
        let cpu = match &self.operation {
            Operation::Norm { source, .. } => {
                let weights: Vec<f64> = if source.dtype == 0 {
                    source
                        .bytes
                        .chunks_exact(2)
                        .map(|b| f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f64())
                        .collect()
                } else {
                    source
                        .bytes
                        .chunks_exact(4)
                        .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
                        .collect()
                };
                let mut expected = Vec::with_capacity(self.original.len());
                for row in self.original.chunks_exact(self.dim) {
                    let sum = row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
                    let scale = (sum / self.dim as f64 + f64::from(source.eps))
                        .sqrt()
                        .recip();
                    expected.extend(
                        row.iter()
                            .zip(&weights)
                            .map(|(&v, &w)| f64::from(v) * scale * w),
                    );
                }
                let actual: Vec<_> = candidate.iter().copied().map(f32::from_bits).collect();
                json!(validation::compare(&actual, &expected)?)
            }
            Operation::Argmax { .. } => {
                let expected: Vec<u32> = self
                    .original
                    .chunks_exact(self.dim)
                    .map(|row| {
                        let mut best = 0;
                        for index in 1..row.len() {
                            if row[index] > row[best] {
                                best = index;
                            }
                        }
                        best as u32
                    })
                    .collect();
                ensure!(
                    candidate == expected,
                    "argmax failed independent finite CPU reference"
                );
                json!({"expected_indices":expected,"actual_indices":candidate})
            }
        };
        self.verify()?;
        Ok(json!({"control_bits_equal":true,"guards_and_immutable_read_spans":true,"cpu":cpu}))
    }

    pub fn time(&self, is_candidate: bool, iterations: usize) -> Result<Value> {
        ensure!(
            iterations > 0 && iterations <= u64::MAX as usize / 8,
            "invalid reduction iteration count"
        );
        let expected = self.operation_counts(is_candidate);
        let launches = expected.iter().map(|(_, count)| count).sum::<u64>() * iterations as u64;
        self.ctx.synchronize_stream(CudaStream::Decode)?;
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
            counters.launches == launches
                && counters.device_allocs == 0
                && counters.device_frees == 0
                && counters.h2d_copies == 0
                && counters.d2h_copies == 0
                && counters.stream_syncs == 0
                && counters.stream_waits == 0
                && counters.cublas_calls == 0,
            "unexpected timed reduction side effects"
        );
        let mut operations = Vec::new();
        for (op, count) in expected {
            let actual = after.by_op.get(op).map_or(0, |v| v.launches)
                - before.by_op.get(op).map_or(0, |v| v.launches);
            ensure!(
                actual == count * iterations as u64,
                "wrong reduction operation dispatched"
            );
            operations.push(json!({"name":op,"launches":actual}));
        }
        let gpu_ms = f64::from(
            self.start
                .elapsed_sync(&self.stop, "reduction_probe_timing")?,
        ) / iterations as f64;
        ensure!(
            gpu_ms.is_finite() && gpu_ms > 0.0,
            "invalid reduction event duration"
        );
        Ok(
            json!({"is_candidate":is_candidate,"gpu_ms":gpu_ms,"gpu_ms_per_request":gpu_ms/self.rows as f64,
            "host_enqueue_and_completion_ms":began.elapsed().as_secs_f64()*1000.0/iterations as f64,
            "iterations":iterations,"operations":operations,"enqueue_counters":counters,
            "started_unix_us":started,"completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros()}),
        )
    }

    pub fn warm(&self, is_candidate: bool, minimum_ms: f64) -> Result<Value> {
        warmup::measure_with_budget(minimum_ms, 4096, 256, |iterations| {
            self.time(is_candidate, iterations)?["gpu_ms"]
                .as_f64()
                .context("missing reduction warmup")
        })
    }
}

impl Drop for ReductionProbe<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.ctx.synchronize_stream(CudaStream::Decode) {
            eprintln!("[reduction_probe] drain failed: {error:#}");
        }
    }
}

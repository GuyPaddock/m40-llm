use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{ensure, Result};
use half::f16;
use m40_llm::{
    cuda::{CudaContext, CudaStream, DeviceBuffer},
    infer::{
        execution_weights::Q8MatrixShape,
        sparse_weights::{DeviceSparseQ8Weight, SparseQ8Launch},
    },
};
use serde_json::{json, Value};

pub struct Projection<'a> {
    pub ctx: &'a CudaContext,
    pub sparse: &'a DeviceSparseQ8Weight,
    pub dense: &'a DeviceBuffer,
    pub input: DeviceBuffer,
    pub output: DeviceBuffer,
    pub reference: DeviceBuffer,
    pub shape: Q8MatrixShape,
    pub rows: usize,
}

impl Projection<'_> {
    pub fn stream(&self, launch: Option<SparseQ8Launch>) -> CudaStream {
        if launch.is_some() || self.rows == 1 {
            CudaStream::Decode
        } else {
            CudaStream::Prefill
        }
    }

    pub fn enqueue(&self, launch: Option<SparseQ8Launch>) -> Result<()> {
        unsafe {
            if let Some(launch) = launch {
                self.sparse
                    .project_async(&self.input, &self.output, self.rows, launch)
            } else {
                self.ctx.gemm_f32xq8_0_gguf_f32_async(
                    self.input.as_ptr(),
                    self.dense.as_ptr(),
                    self.reference.as_mut_ptr(),
                    self.rows as i32,
                    self.shape.output_dim() as i32,
                    self.shape.input_dim() as i32,
                )
            }
        }
    }

    pub fn time(&self, launch: Option<SparseQ8Launch>, iterations: usize) -> Result<Value> {
        let start = self.ctx.create_event()?;
        let stop = self.ctx.create_event()?;
        let began_us = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let began = Instant::now();
        start.record(self.stream(launch))?;
        for _ in 0..iterations {
            self.enqueue(launch)?;
        }
        stop.record(self.stream(launch))?;
        let gpu_ms = f64::from(start.elapsed_sync(&stop, "q8_sparse_bench")?) / iterations as f64;
        Ok(
            json!({"gpu_ms":gpu_ms,"wall_ms":began.elapsed().as_secs_f64()*1000.0/iterations as f64,
            "started_unix_us":began_us,"completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros()}),
        )
    }

    pub fn check(&self, launch: SparseQ8Launch, dense: &[u8], input: &[f32]) -> Result<Value> {
        self.enqueue(None)?;
        self.ctx.synchronize_stream(self.stream(None))?;
        self.enqueue(Some(launch))?;
        self.ctx.synchronize_stream(self.stream(Some(launch)))?;
        let expected = self.read(&self.reference)?;
        let actual = self.read(&self.output)?;
        let mut maximum = 0.0_f64;
        let mut total = 0.0;
        for (&a, &b) in actual.iter().zip(&expected) {
            ensure!(
                a.is_finite() && b.is_finite(),
                "non-finite sparse projection"
            );
            let diff = (f64::from(a) - f64::from(b)).abs();
            maximum = maximum.max(diff);
            total += diff;
            ensure!(
                diff < 1e-3 + f64::from(b.abs()) * 2e-5,
                "sparse/dense mismatch: {diff}"
            );
        }
        let k = self.shape.input_dim();
        let n = self.shape.output_dim();
        let mut cpu_max = 0.0_f64;
        for row in [0, self.rows - 1] {
            for sample in 0..8 {
                let column = sample * (n - 1) / 7;
                let expected: f64 = (0..k)
                    .map(|index| {
                        let base = (column * k.div_ceil(32) + index / 32) * 34;
                        let scale = f16::from_le_bytes([dense[base], dense[base + 1]]).to_f64();
                        f64::from(input[row * k + index])
                            * scale
                            * f64::from(dense[base + 2 + index % 32] as i8)
                    })
                    .sum();
                let difference = (f64::from(actual[row * n + column]) - expected).abs();
                cpu_max = cpu_max.max(difference);
                ensure!(
                    difference < 1e-3 + expected.abs() * 2e-5,
                    "sparse/scalar mismatch: {difference}"
                );
            }
        }
        Ok(
            json!({"max_abs_diff":maximum,"mean_abs_diff":total/actual.len() as f64,"cpu_sample_max_abs_diff":cpu_max}),
        )
    }

    fn read(&self, buffer: &DeviceBuffer) -> Result<Vec<f32>> {
        let mut result = vec![0.0; buffer.bytes() / 4];
        unsafe {
            self.ctx
                .memcpy_d2h(result.as_mut_ptr().cast(), buffer.as_ptr(), buffer.bytes())?;
        }
        Ok(result)
    }
}

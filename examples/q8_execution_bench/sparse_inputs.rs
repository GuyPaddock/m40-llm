use super::{
    configurations, inspect_gpu, read_output, write_record, Args, Projection, ProjectionLaunch,
};
use anyhow::{ensure, Result};
use half::f16;
use m40_llm::{
    cuda::{CudaStream, DeviceBuffer},
    gguf::GgufTensor,
    infer::{
        activation_sparsity::{
            ActivationThreshold, DeviceSparseInputWorkspace, SparseInputSelection,
        },
        execution_weights::{DeviceQ8ExecutionView, Q8TiledLaunch},
    },
};
use serde_json::{json, Value};
use std::{
    fs::File,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy)]
enum Operation {
    NativeMatchedZero,
    SelectAndProject,
    SelectOnly,
    ProjectOnly,
}

struct SparseProjection<'a, 'b> {
    projection: &'a Projection<'b>,
    matched: DeviceBuffer,
    scratch: DeviceSparseInputWorkspace,
    threshold: ActivationThreshold,
    config: Q8TiledLaunch,
}

impl SparseProjection<'_, '_> {
    unsafe fn enqueue(&mut self, operation: Operation) -> Result<()> {
        let projection = self.projection;
        match operation {
            Operation::NativeMatchedZero => {
                let DeviceQ8ExecutionView::ColumnBlocks { key, blocks } =
                    projection.original.view()
                else {
                    unreachable!()
                };
                let shape = key.source.shape();
                projection.ctx.gemm_f32xq8_0_gguf_f32_async(
                    self.matched.as_ptr(),
                    blocks.as_ptr(),
                    projection.reference.as_mut_ptr(),
                    1,
                    shape.output_dim() as i32,
                    shape.input_dim() as i32,
                )
            }
            Operation::SelectAndProject => {
                self.enqueue(Operation::SelectOnly)?;
                self.enqueue(Operation::ProjectOnly)
            }
            Operation::SelectOnly => self.scratch.select_async(&projection.input, self.threshold),
            Operation::ProjectOnly => self.scratch.project_tiled_async(
                projection.execution,
                &projection.output,
                self.config,
            ),
        }
    }

    fn time(&mut self, operation: Operation, iterations: usize) -> Result<Value> {
        let ctx = self.projection.ctx.clone();
        let start = ctx.create_event()?;
        let stop = ctx.create_event()?;
        let started_unix_us = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let began = Instant::now();
        start.record(CudaStream::Decode)?;
        for _ in 0..iterations {
            unsafe {
                self.enqueue(operation)?;
            }
        }
        stop.record(CudaStream::Decode)?;
        let gpu_ms =
            f64::from(start.elapsed_sync(&stop, "sparse_input_screen")?) / iterations as f64;
        Ok(
            json!({"gpu_ms":gpu_ms,"wall_ms":began.elapsed().as_secs_f64()*1000.0/iterations as f64,
            "started_unix_us":started_unix_us,
            "completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros()}),
        )
    }

    fn check(&mut self, bytes: &[u8], selected: &SparseInputSelection) -> Result<Value> {
        unsafe {
            self.enqueue(Operation::NativeMatchedZero)?;
            self.enqueue(Operation::SelectAndProject)?;
        }
        let actual_selection = self.scratch.read_selection_sync()?;
        ensure!(
            actual_selection.indices() == selected.indices()
                && actual_selection.values() == selected.values(),
            "GPU selection differs from CPU reference"
        );
        let projection = self.projection;
        let actual = read_output(&projection.ctx, &projection.output)?;
        let reference = read_output(&projection.ctx, &projection.reference)?;
        let mut maximum = 0.0f64;
        let mut total = 0.0f64;
        for (&a, &b) in actual.iter().zip(&reference) {
            ensure!(
                a.is_finite() && b.is_finite(),
                "nonfinite sparse projection"
            );
            let difference = (f64::from(a) - f64::from(b)).abs();
            maximum = maximum.max(difference);
            total += difference;
            ensure!(
                difference < 1e-3 + f64::from(b.abs()) * 2e-5,
                "sparse matched-zero drift {difference}"
            );
        }
        let shape = projection.original.key().source.shape();
        let (k, n) = (shape.input_dim(), shape.output_dim());
        let matched = selected.reconstruct();
        let mut cpu_maximum = 0.0f64;
        for sample in 0..16 {
            let column = sample * (n - 1) / 15;
            let expected: f64 = (0..k)
                .map(|index| {
                    let base = (column * k.div_ceil(32) + index / 32) * 34;
                    let scale =
                        f16::from_bits(u16::from_le_bytes([bytes[base], bytes[base + 1]])).to_f64();
                    f64::from(matched[index])
                        * scale
                        * f64::from(bytes[base + 2 + index % 32] as i8)
                })
                .sum();
            let difference = (f64::from(actual[column]) - expected).abs();
            ensure!(
                difference < 1e-3 + expected.abs() * 2e-5,
                "sparse CPU reference drift {difference}"
            );
            cpu_maximum = cpu_maximum.max(difference);
        }
        Ok(
            json!({"max_abs_diff":maximum,"mean_abs_diff":total/n as f64,
            "cpu_sample_max_abs_diff":cpu_maximum,"selected_indices_match":true,"selected_values_match":true}),
        )
    }
}

pub(super) fn measure(
    projection: &Projection<'_>,
    file: &mut File,
    args: &Args,
    bytes: &[u8],
    input: &[f32],
    tensor: &GgufTensor,
) -> Result<()> {
    let shape = projection.original.key().source.shape();
    let k = shape.input_dim();
    let matched = DeviceBuffer::new_tagged(&projection.ctx, k * 4, "sparse_screen:matched")?;
    let scratch = DeviceSparseInputWorkspace::new(&projection.ctx, k)?;
    let mut sparse = SparseProjection {
        projection,
        matched,
        scratch,
        threshold: ActivationThreshold::new(0.0)?,
        config: Q8TiledLaunch::new(8, 128)?,
    };
    for config in configurations(projection.execution.key().layout)? {
        let ProjectionLaunch::Tiled(config) = config else {
            unreachable!()
        };
        sparse.config = config;
        for target in [0.0, 0.25, 0.4, 0.5, 0.6] {
            // These synthetic observations screen kernel geometry, not model calibration.
            sparse.threshold = ActivationThreshold::calibrate(input, target)?;
            let selected = SparseInputSelection::select(input, sparse.threshold)?;
            let matched = selected.reconstruct();
            unsafe {
                projection.ctx.memcpy_h2d(
                    sparse.matched.as_mut_ptr(),
                    matched.as_ptr().cast(),
                    k * 4,
                )?;
            }
            let errors = sparse.check(bytes, &selected)?;
            let mut qblocks = std::collections::BTreeSet::new();
            for &index in selected.indices() {
                qblocks.insert(index / 32);
            }
            let columns = config.columns() as usize;
            let padded_n = shape.output_dim().div_ceil(columns) * columns;
            let value_load_bytes = selected.indices().len() * padded_n;
            let memory = json!({"source_weight_bytes":bytes.len(),
                "extra_tiled_weight_bytes":projection.execution.plan().resident_bytes(),
                "workspace_capacity_bytes":sparse.scratch.bytes(),
                "selected_payload_bytes":selected.indices().len()*8+4,
                "requested_value_load_bytes":value_load_bytes,
                "requested_scale_load_bytes":value_load_bytes*2,
                "unique_scale_payload_bytes":qblocks.len()*padded_n*2,
                "requested_index_value_load_bytes":selected.indices().len()*8*shape.output_dim().div_ceil(columns),
                "traffic_scope":"source-level requested bytes, NOT measured DRAM; scale/index cache reuse and transaction granularity unknown"});
            for _ in 0..3 {
                sparse.time(Operation::NativeMatchedZero, 1)?;
                sparse.time(Operation::SelectAndProject, 1)?;
            }
            for trial in 0..args.trials {
                let gpu_before = inspect_gpu()?;
                let (baseline, candidate) = if trial % 2 == 0 {
                    (
                        sparse.time(Operation::NativeMatchedZero, args.iterations)?,
                        sparse.time(Operation::SelectAndProject, args.iterations)?,
                    )
                } else {
                    let candidate = sparse.time(Operation::SelectAndProject, args.iterations)?;
                    (
                        sparse.time(Operation::NativeMatchedZero, args.iterations)?,
                        candidate,
                    )
                };
                let selection_only = sparse.time(Operation::SelectOnly, args.iterations)?;
                let projection_only = sparse.time(Operation::ProjectOnly, args.iterations)?;
                write_record(
                    file,
                    json!({"kind":"sparse_input_timing","tensor":tensor.name,"shape":shape,
                    "rows":1,"config":config,"target_sparsity":target,"achieved_sparsity":selected.sparsity(),
                    "threshold":sparse.threshold,"retained_inputs":selected.indices().len(),
                    "input_pattern":"((i*13%127)-63)/128, synthetic completed-input proxy, not captured model SwiGLU",
                    "trial":trial,"iterations":args.iterations,
                    "baseline":baseline,"candidate":candidate,"selection_only":selection_only,"projection_only":projection_only,
                    "errors":errors,"memory":memory,"gpu_before":gpu_before,"gpu_after":inspect_gpu()?}),
                )?;
            }
            eprintln!(
                "[sparse-input] {} {config:?} target={target} achieved={:.4} parity={errors}",
                tensor.name,
                selected.sparsity()
            );
        }
    }
    projection.ctx.synchronize_stream(CudaStream::Decode)?;
    Ok(())
}

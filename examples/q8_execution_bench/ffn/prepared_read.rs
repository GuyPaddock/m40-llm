//! Independent cost diagnostic for the qualified prepared projection mapping.
use super::{
    fixture::source_bytes,
    pipeline::{FfnKernel, FfnPipeline, GuardedF32, Stage},
    validation, warmup,
    weight_read::checksum_columns,
};
use crate::run::{inspect_gpu, write_record, Args};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, CudaEvent, CudaStream},
    evaluation::{ffn_capture::FfnLayerSamples, identity::sha256},
    infer::{
        execution_weights::{DeviceQ8ExecutionView, DeviceQ8ExecutionWeight},
        LoadedModel,
    },
    profile,
};
use serde_json::{json, Value};
use std::{
    fs::File,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

struct PreparedReadProbe<'a> {
    model: &'a LoadedModel,
    weight: &'a DeviceQ8ExecutionWeight,
    output: GuardedF32,
    start: CudaEvent,
    stop: CudaEvent,
    rows: usize,
    n: usize,
    k: usize,
    threads: i32,
    stream: CudaStream,
}

impl<'a> PreparedReadProbe<'a> {
    fn new(
        model: &'a LoadedModel,
        weight: &'a DeviceQ8ExecutionWeight,
        rows: usize,
        threads: i32,
    ) -> Result<Self> {
        let shape = weight.key().source.shape();
        ensure!(
            matches!(rows, 1 | 4) && shape.input_dim().is_multiple_of(32),
            "unsupported prepared read shape"
        );
        Ok(Self {
            model,
            weight,
            rows,
            n: shape.output_dim(),
            k: shape.input_dim(),
            threads,
            stream: if rows == 1 {
                CudaStream::Decode
            } else {
                CudaStream::Prefill
            },
            output: GuardedF32::new(
                &model.cuda,
                &vec![0.0; rows * shape.output_dim()],
                "prepared_read:checksums",
            )?,
            start: model.cuda.create_event()?,
            stop: model.cuda.create_event()?,
        })
    }

    fn enqueue(&self) -> Result<()> {
        let DeviceQ8ExecutionView::SeparateScalesValues { scales, values, .. } = self.weight.view()
        else {
            anyhow::bail!("prepared read requires owned separate planes")
        };
        unsafe {
            self.model.cuda.projection_q8_weight_checksum_async(
                scales.as_ptr(),
                values.as_ptr(),
                self.output.pointer(),
                self.rows as i32,
                self.n as i32,
                self.k as i32,
                self.threads,
                self.stream,
            )
        }
    }

    fn time(
        &self,
        iterations: usize,
        stage: &str,
        operation: &str,
        mut enqueue: impl FnMut() -> Result<()>,
    ) -> Result<Value> {
        ensure!(iterations > 0, "prepared read timing needs iterations");
        self.model.cuda.synchronize_stream(self.stream)?;
        let before = profile::snapshot();
        let began = Instant::now();
        let started_unix_us = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        self.start.record(self.stream)?;
        for _ in 0..iterations {
            enqueue()?;
        }
        self.stop.record(self.stream)?;
        let after = profile::snapshot();
        let counters = profile::counter_delta(&before, &after);
        let launches = after.by_op.get(operation).map_or(0, |v| v.launches)
            - before.by_op.get(operation).map_or(0, |v| v.launches);
        ensure!(
            launches == iterations as u64
                && counters.launches == iterations as u64
                && counters.device_allocs
                    + counters.device_frees
                    + counters.h2d_copies
                    + counters.d2h_copies
                    + counters.cublas_calls
                    + counters.stream_waits
                    + counters.stream_syncs
                    == 0,
            "unexpected prepared diagnostic/control enqueue work"
        );
        let gpu_ms = f64::from(
            self.start
                .elapsed_sync(&self.stop, "prepared_read:completion")?,
        ) / iterations as f64;
        Ok(
            json!({"stage":stage,"operation":operation,"gpu_ms":gpu_ms,"iterations":iterations,
            "started_unix_us":started_unix_us,"completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros(),
            "host_enqueue_and_completion_ms":began.elapsed().as_secs_f64()*1000.0/iterations as f64,
            "enqueue_counters":counters,"logical_unique_weight_bytes_per_call":self.weight.plan().resident_bytes()*self.rows,
            "has_inference_output":stage!="weight_read_checksum","has_measured_dram_traffic":false,
            "geometry":{"columns":4,"values_per_task":8,"threads":self.threads,"blocks_per_row":self.n.div_ceil(4),"shared_bytes":self.threads*16}}),
        )
    }

    fn time_read(&self, iterations: usize) -> Result<Value> {
        self.time(
            iterations,
            "weight_read_checksum",
            "projection_q8_weight_checksum",
            || self.enqueue(),
        )
    }

    fn verify(&self, expected_columns: &[u32]) -> Result<Value> {
        ensure!(
            expected_columns.len() == self.n,
            "checksum reference column mismatch"
        );
        self.model.cuda.synchronize_stream(self.stream)?;
        let actual: Vec<_> = self
            .output
            .read(&self.model.cuda)?
            .into_iter()
            .map(f32::to_bits)
            .collect();
        ensure!(
            actual
                .iter()
                .copied()
                .eq((0..self.rows).flat_map(|_| expected_columns.iter().copied())),
            "prepared GPU checksums differ from original GGUF blocks"
        );
        let bytes: Vec<_> = actual.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(
            json!({"has_matching_cpu_checksums":true,"has_intact_output_guards":true,"checksum_count":actual.len(),
            "checksum_sha256":sha256(&bytes)?,"checksum_storage_bytes":self.output.bytes(),"scale_reads_per_block":4,
            "reference":"original interleaved GGUF bytes; eight u32 payload words plus four u16 scale-bit contributions per Q8 block modulo 2^32"}),
        )
    }

    fn describe(&self) -> Value {
        json!({"key":self.weight.key(),"planes":self.weight.plan(),"rows":self.rows,"n":self.n,"k":self.k,
            "columns":4,"values_per_task":8,"threads":self.threads,"shared_bytes":self.threads*16,
            "dynamic_barriers_per_block":if self.threads==256 {9} else {8},"checksum_storage_bytes":self.output.bytes(),
            "stream":if self.rows==1 {"decode"} else {"prefill"}})
    }
}

impl Drop for PreparedReadProbe<'_> {
    fn drop(&mut self) {
        let _ = self.model.cuda.synchronize_stream(self.stream);
    }
}

fn hash_planes(weight: &DeviceQ8ExecutionWeight) -> Result<Vec<String>> {
    unsafe { weight.read_planes()? }
        .iter()
        .map(|plane| sha256(plane))
        .collect()
}

fn time_down_control(pipeline: &FfnPipeline<'_>, stage: Stage, iterations: usize) -> Result<Value> {
    let before = profile::snapshot();
    let mut timing = pipeline.time(stage, iterations)?;
    let after = profile::snapshot();
    let op = "gemm_f32xq8_separate_f32";
    let launches = after.by_op.get(op).map_or(0, |v| v.launches)
        - before.by_op.get(op).map_or(0, |v| v.launches);
    ensure!(
        launches
            == if matches!(stage, Stage::GateUpSwiglu) {
                0
            } else {
                iterations as u64
            },
        "wrong qualified prepared down dispatch"
    );
    let op_gate = "ffn_q8_split_f32x4_128";
    let gate_launches = after.by_op.get(op_gate).map_or(0, |v| v.launches)
        - before.by_op.get(op_gate).map_or(0, |v| v.launches);
    ensure!(
        gate_launches
            == if matches!(stage, Stage::IsolatedDown) {
                0
            } else {
                iterations as u64
            },
        "wrong qualified gate/up dispatch"
    );
    timing["control_dispatch"] = json!({"projection_operation":op,"projection_launches":launches,"gate_up_operation":op_gate,"gate_up_launches":gate_launches});
    Ok(timing)
}

pub(super) fn benchmark_down(
    file: &mut File,
    model: &LoadedModel,
    layer: &FfnLayerSamples,
    rows: usize,
    args: &Args,
) -> Result<()> {
    let mut pipeline = FfnPipeline::new(model, layer, rows)?;
    pipeline.prepare_split_weights(layer)?;
    pipeline.select_kernel(FfnKernel::SplitF32x4)?;
    let down = model.map_standard_layer(layer.source.layer_id)?.w_down;
    let weight = model
        .q8_execution
        .get_separate_weight(down.dptr.cast_const())
        .context("missing prepared down owner")?;
    let probe = PreparedReadProbe::new(model, weight, rows, if rows == 1 { 256 } else { 128 })?;
    let original_planes = hash_planes(weight)?;
    let expected = checksum_columns(
        source_bytes(model, &layer.source.down)?,
        probe.k,
        probe.n,
        4,
    )?;
    probe.enqueue()?;
    let checks = probe.verify(&expected)?;
    let resident = CudaContext::total_device_bytes();
    write_record(
        file,
        json!({"kind":"prepared_read_validation","family":"down","layer_id":layer.source.layer_id,"rows":rows,
        "checks":checks,"real_ffn_checks":validation::validate(&pipeline,layer)?,"configuration":probe.describe(),"plane_sha256":original_planes,
        "gate_up_preparation":pipeline.prepared.as_ref().unwrap().report,"real_scratch_bytes":pipeline.scratch_bytes(),"resident_bytes":resident}),
    )?;
    let real = pipeline.warm_up(1000.0, 256)?;
    let read = warmup::measure(100.0, |iterations| {
        Ok(probe.time_read(iterations)?["gpu_ms"].as_f64().unwrap())
    })?;
    write_record(
        file,
        json!({"kind":"prepared_read_warmup","family":"down","layer_id":layer.source.layer_id,"rows":rows,
        "real_ffn":real,"read_checksum":read,"gpu_after":inspect_gpu()?}),
    )?;
    for trial in 0..args.trials {
        let gpu_before = inspect_gpu()?;
        let mut measurements = Vec::new();
        for stage in if trial % 2 == 0 {
            [0, 1, 2, 3]
        } else {
            [3, 2, 1, 0]
        } {
            measurements.push(match stage {
                0 => probe.time_read(args.iterations)?,
                1 => time_down_control(&pipeline, Stage::GateUpSwiglu, args.iterations)?,
                2 => time_down_control(&pipeline, Stage::IsolatedDown, args.iterations)?,
                _ => time_down_control(&pipeline, Stage::Complete, args.iterations)?,
            });
        }
        ensure!(
            CudaContext::total_device_bytes() == resident,
            "prepared read changed allocations"
        );
        write_record(
            file,
            json!({"kind":"prepared_read_pair","family":"down","layer_id":layer.source.layer_id,"rows":rows,
            "trial":trial,"measurements":measurements,"gpu_before":gpu_before,"gpu_after":inspect_gpu()?,"resident_bytes":resident}),
        )?;
    }
    ensure!(
        probe.verify(&expected)? == checks,
        "prepared down checksum changed after reuse"
    );
    pipeline.verify_guards_and_inputs(layer)?;
    ensure!(
        hash_planes(weight)? == original_planes,
        "prepared down planes changed"
    );
    write_record(
        file,
        json!({"kind":"prepared_read_case_complete","family":"down","layer_id":layer.source.layer_id,"rows":rows,
        "checks":checks,"has_intact_planes_and_inputs":true,"runtime_integration_allowed":false}),
    )?;
    eprintln!(
        "[prepared_read] down layer={} rows={rows}: {} serial pairs",
        layer.source.layer_id, args.trials
    );
    Ok(())
}

/// The vocabulary control is unchanged; sample real outputs independently
/// against original blocks, rather than using checksums as inference values.
fn validate_vocabulary(
    source: &[u8],
    input: &[f32],
    output: &[f32],
    rows: usize,
    n: usize,
    k: usize,
) -> Result<f64> {
    let mut max_error = 0.0f64;
    for row in 0..rows {
        for col in [0, n / 7, n / 2, n - 1] {
            let cpu: f64 = (0..k)
                .map(|i| {
                    let block = (col * (k / 32) + i / 32) * 34;
                    let scale = half::f16::from_bits(u16::from_le_bytes([
                        source[block],
                        source[block + 1],
                    ]))
                    .to_f64();
                    f64::from(input[row * k + i])
                        * scale
                        * f64::from(source[block + 2 + i % 32] as i8)
                })
                .sum();
            let diff = (f64::from(output[row * n + col]) - cpu).abs();
            ensure!(
                diff.is_finite() && diff < 1e-3 + 2e-5 * cpu.abs(),
                "vocabulary control CPU drift"
            );
            max_error = max_error.max(diff);
        }
    }
    Ok(max_error)
}

pub(super) fn benchmark_vocabulary(
    file: &mut File,
    model: &LoadedModel,
    layer: &FfnLayerSamples,
    rows: usize,
    args: &Args,
) -> Result<()> {
    let (name, view, k, n, _) = model.map_lm_head()?;
    let weight = model
        .q8_execution
        .get_separate_weight(view.dptr.cast_const())
        .context("missing prepared vocabulary owner")?;
    let offset = usize::try_from(weight.key().source.byte_offset())?;
    let source = model
        .host_weights
        .get(
            offset
                ..offset
                    .checked_add(weight.plan().source_bytes())
                    .context("vocabulary span overflow")?,
        )
        .context("vocabulary source span")?;
    let expected = checksum_columns(source, k, n, 4)?;
    let probe = PreparedReadProbe::new(model, weight, rows, 128)?;
    let input_values = &layer.normalized_inputs[..rows * k];
    let input = GuardedF32::new(&model.cuda, input_values, "prepared_read:real-input")?;
    let output = GuardedF32::new(
        &model.cuda,
        &vec![0.0; rows * n],
        "prepared_read:real-output",
    )?;
    let enqueue_real = || -> Result<()> {
        unsafe {
            model.matmul_f32xf16_gguf_f32_async(
                input.pointer(),
                view.dptr.cast_const(),
                output.pointer(),
                rows as i32,
                n as i32,
                k as i32,
            )
        }
    };
    enqueue_real()?;
    model.cuda.synchronize_stream(probe.stream)?;
    let real_output = output.read(&model.cuda)?;
    let cpu_error = validate_vocabulary(source, input_values, &real_output, rows, n, k)?;
    let original_planes = hash_planes(weight)?;
    probe.enqueue()?;
    let checks = probe.verify(&expected)?;
    let resident = CudaContext::total_device_bytes();
    write_record(
        file,
        json!({"kind":"prepared_read_validation","family":"vocabulary","rows":rows,"tensor":name,
        "checks":checks,"cpu_sample_max_abs_diff":cpu_error,"plane_sha256":original_planes,"configuration":probe.describe(),
        "input_scope":"distinct frozen normalized FFN layer-0 surrogates, not captured output-norm hidden vectors",
        "real_scratch_bytes":input.bytes()+output.bytes(),"resident_bytes":resident}),
    )?;
    let time_real = |count| {
        probe.time(
            count,
            "vocabulary",
            "gemm_f32xq8_separate_f32",
            enqueue_real,
        )
    };
    let real = warmup::measure(1000.0, |count| {
        Ok(time_real(count)?["gpu_ms"].as_f64().unwrap())
    })?;
    let read = warmup::measure(100.0, |count| {
        Ok(probe.time_read(count)?["gpu_ms"].as_f64().unwrap())
    })?;
    write_record(
        file,
        json!({"kind":"prepared_read_warmup","family":"vocabulary","rows":rows,"real_projection":real,"read_checksum":read,"gpu_after":inspect_gpu()?}),
    )?;
    for trial in 0..args.trials {
        let gpu_before = inspect_gpu()?;
        let mut measurements = Vec::new();
        for is_read in if trial % 2 == 0 {
            [true, false]
        } else {
            [false, true]
        } {
            measurements.push(if is_read {
                probe.time_read(args.iterations)?
            } else {
                time_real(args.iterations)?
            });
        }
        ensure!(
            CudaContext::total_device_bytes() == resident,
            "vocabulary read changed allocations"
        );
        write_record(
            file,
            json!({"kind":"prepared_read_pair","family":"vocabulary","rows":rows,"trial":trial,
            "measurements":measurements,"gpu_before":gpu_before,"gpu_after":inspect_gpu()?,"resident_bytes":resident}),
        )?;
    }
    ensure!(
        probe.verify(&expected)? == checks,
        "vocabulary checksum changed after reuse"
    );
    let actual = output.read(&model.cuda)?;
    ensure!(
        actual
            .iter()
            .zip(&real_output)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "vocabulary control output bits changed"
    );
    ensure!(
        input
            .read(&model.cuda)?
            .iter()
            .zip(input_values)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "vocabulary input changed"
    );
    ensure!(
        hash_planes(weight)? == original_planes,
        "prepared vocabulary planes changed"
    );
    write_record(
        file,
        json!({"kind":"prepared_read_case_complete","family":"vocabulary","rows":rows,"checks":checks,
        "has_intact_planes_and_inputs":true,"has_matching_real_control_bits":true,"runtime_integration_allowed":false}),
    )?;
    eprintln!(
        "[prepared_read] vocabulary rows={rows}: {} serial pairs",
        args.trials
    );
    Ok(())
}

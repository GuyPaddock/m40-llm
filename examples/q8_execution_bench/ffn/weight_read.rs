//! Actual-plane read/checksum cost, never an inference candidate.
use super::{
    fixture::source_bytes,
    pipeline::{FfnKernel, FfnPipeline, GuardedF32, Stage},
    validation, warmup,
};
use crate::run::{inspect_gpu, write_record, Args};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, CudaEvent, CudaStream},
    evaluation::{ffn_capture::FfnLayerSamples, identity::sha256},
    profile,
};
use serde_json::{json, Value};
use std::{
    fs::File,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

/// Derive modular sums directly from original interleaved GGUF blocks.
fn checksum_columns(source: &[u8], k: usize, hidden: usize) -> Result<Vec<u32>> {
    ensure!(
        k > 0 && k.is_multiple_of(32) && hidden > 0,
        "invalid checksum shape"
    );
    let column_bytes = (k / 32)
        .checked_mul(34)
        .context("checksum column overflow")?;
    ensure!(
        Some(source.len()) == column_bytes.checked_mul(hidden),
        "checksum source span mismatch"
    );
    Ok(source
        .chunks_exact(column_bytes)
        .map(|column| {
            column.chunks_exact(34).fold(0u32, |sum, block| {
                let scale = u32::from(u16::from_le_bytes([block[0], block[1]]));
                block[2..].chunks_exact(4).fold(sum, |acc, word| {
                    acc.wrapping_add(u32::from_le_bytes(word.try_into().unwrap()))
                        .wrapping_add(scale)
                })
            })
        })
        .collect())
}

struct ReadProbe {
    output: GuardedF32,
    start: CudaEvent,
    stop: CudaEvent,
}

fn time_control(pipeline: &FfnPipeline<'_>, stage: Stage, iterations: usize) -> Result<Value> {
    let before = profile::snapshot();
    let mut measurement = pipeline.time(stage, iterations)?;
    let after = profile::snapshot();
    let operation = "ffn_q8_split_f32x4_128";
    let launches = after.by_op.get(operation).map_or(0, |v| v.launches)
        - before.by_op.get(operation).map_or(0, |v| v.launches);
    ensure!(
        launches
            == if matches!(stage, Stage::IsolatedDown) {
                0
            } else {
                iterations as u64
            },
        "wrong qualified gate/up control dispatch"
    );
    measurement["control_dispatch"] = json!({"operation":operation,"launches":launches});
    Ok(measurement)
}

impl ReadProbe {
    fn new(pipeline: &FfnPipeline<'_>) -> Result<Self> {
        let ctx = &pipeline.model.cuda;
        Ok(Self {
            output: GuardedF32::new(
                ctx,
                &vec![0.0; pipeline.rows * pipeline.hidden_dim * 2],
                "ffn_read:checksums",
            )?,
            start: ctx.create_event()?,
            stop: ctx.create_event()?,
        })
    }

    fn enqueue(&self, pipeline: &FfnPipeline<'_>) -> Result<()> {
        unsafe {
            pipeline
                .prepared
                .as_ref()
                .context("missing read planes")?
                .enqueue_checksum(self.output.pointer(), pipeline.rows)
        }
    }

    fn time(&self, pipeline: &FfnPipeline<'_>, iterations: usize) -> Result<Value> {
        ensure!(iterations > 0, "read timing needs iterations");
        pipeline.drain()?;
        let before = profile::snapshot();
        let started_unix_us = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let began = Instant::now();
        self.start.record(CudaStream::Decode)?;
        for _ in 0..iterations {
            self.enqueue(pipeline)?;
        }
        self.stop.record(CudaStream::Decode)?;
        let after = profile::snapshot();
        let counters = profile::counter_delta(&before, &after);
        let op = "ffn_q8_weight_checksum_128";
        let launches = after.by_op.get(op).map_or(0, |v| v.launches)
            - before.by_op.get(op).map_or(0, |v| v.launches);
        ensure!(
            launches == iterations as u64
                && counters.launches == iterations as u64
                && counters.device_allocs == 0
                && counters.device_frees == 0
                && counters.h2d_copies == 0
                && counters.d2h_copies == 0
                && counters.stream_syncs == 0
                && counters.cublas_calls == 0
                && counters.stream_waits == 0,
            "unexpected read diagnostic enqueue work"
        );
        let gpu_ms = f64::from(self.start.elapsed_sync(&self.stop, "ffn_read_benchmark")?)
            / iterations as f64;
        Ok(
            json!({"stage":"weight_read_checksum","gpu_ms":gpu_ms,"iterations":iterations,
            "started_unix_us":started_unix_us,"completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros(),
            "host_enqueue_and_completion_ms":began.elapsed().as_secs_f64()*1000.0/iterations as f64,
            "enqueue_counters":counters,"operation":op,"blocks_per_row":pipeline.hidden_dim,"threads":128,"shared_bytes":1024,
            "logical_unique_weight_bytes_per_call":pipeline.prepared.as_ref().unwrap().bytes * pipeline.rows,
            "has_inference_output":false,"has_measured_dram_traffic":false}),
        )
    }

    fn verify(&self, pipeline: &FfnPipeline<'_>, layer: &FfnLayerSamples) -> Result<Value> {
        pipeline.drain()?;
        let gate = checksum_columns(
            source_bytes(pipeline.model, &layer.source.gate)?,
            pipeline.d_model,
            pipeline.hidden_dim,
        )?;
        let up = checksum_columns(
            source_bytes(pipeline.model, &layer.source.up)?,
            pipeline.d_model,
            pipeline.hidden_dim,
        )?;
        let expected: Vec<_> = (0..pipeline.rows)
            .flat_map(|_| gate.iter().zip(&up).flat_map(|(&g, &u)| [g, u]))
            .collect();
        let actual: Vec<_> = self
            .output
            .read(&pipeline.model.cuda)?
            .into_iter()
            .map(f32::to_bits)
            .collect();
        ensure!(
            actual == expected,
            "GPU weight checksums differ from original GGUF blocks"
        );
        let bytes: Vec<_> = actual.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(
            json!({"has_matching_cpu_checksums":true,"has_intact_output_guards":true,
            "checksum_count":actual.len(),"checksum_sha256":sha256(&bytes)?,"checksum_storage_bytes":self.output.bytes(),
            "reference":"original interleaved GGUF blocks; u32 payload words plus u16 scale eight times per block modulo 2^32"}),
        )
    }
}

pub(super) fn benchmark_case(
    output: &mut File,
    pipeline: &mut FfnPipeline<'_>,
    layer: &FfnLayerSamples,
    args: &Args,
) -> Result<()> {
    pipeline.select_kernel(FfnKernel::SplitF32x4)?;
    let probe = ReadProbe::new(pipeline)?;
    let resident = CudaContext::total_device_bytes();
    let prepared_before = serde_json::to_value(pipeline.model.q8_execution_stats())?;
    probe.enqueue(pipeline)?;
    let checks = probe.verify(pipeline, layer)?;
    write_record(
        output,
        json!({"kind":"ffn_read_validation","layer_id":layer.source.layer_id,"rows":pipeline.rows,
        "checks":checks,"real_ffn_checks":validation::validate(pipeline, layer)?,
        "preparation":pipeline.prepared.as_ref().unwrap().report,"real_scratch_bytes":pipeline.scratch_bytes(),
        "tracked_device_bytes":resident,"runtime_integration_allowed":false}),
    )?;
    let control_warmup = pipeline.warm_up(1000.0, 256)?;
    let read_warmup = warmup::measure(100.0, |count| {
        Ok(probe.time(pipeline, count)?["gpu_ms"].as_f64().unwrap())
    })?;
    write_record(
        output,
        json!({"kind":"ffn_read_warmup","layer_id":layer.source.layer_id,"rows":pipeline.rows,
        "real_ffn":control_warmup,"read_checksum":read_warmup,"gpu_after":inspect_gpu()?}),
    )?;
    for trial in 0..args.trials {
        let gpu_before = inspect_gpu()?;
        let mut measurements = Vec::new();
        for arm in if trial % 2 == 0 {
            [0, 1, 2, 3]
        } else {
            [3, 2, 1, 0]
        } {
            measurements.push(match arm {
                0 => probe.time(pipeline, args.iterations)?,
                1 => time_control(pipeline, Stage::GateUpSwiglu, args.iterations)?,
                2 => time_control(pipeline, Stage::IsolatedDown, args.iterations)?,
                _ => time_control(pipeline, Stage::Complete, args.iterations)?,
            });
        }
        ensure!(
            CudaContext::total_device_bytes() == resident,
            "read timing changed allocation accounting"
        );
        write_record(
            output,
            json!({"kind":"ffn_read_pair","layer_id":layer.source.layer_id,"rows":pipeline.rows,"trial":trial,
            "measurements":measurements,"gpu_before":gpu_before,"gpu_after":inspect_gpu()?,"tracked_device_bytes":resident,
            "interpretation":"standalone read/checksum versus real gate/up, down and complete FFN; never inference speedup or measured bandwidth"}),
        )?;
    }
    ensure!(
        probe.verify(pipeline, layer)? == checks,
        "read checksum changed after scratch reuse"
    );
    pipeline.verify_guards_and_inputs(layer)?;
    ensure!(
        serde_json::to_value(pipeline.model.q8_execution_stats())? == prepared_before,
        "read diagnostic changed the qualified down/vocabulary cache"
    );
    write_record(
        output,
        json!({"kind":"ffn_read_case_complete","layer_id":layer.source.layer_id,"rows":pipeline.rows,
        "checks":checks,"has_intact_planes_and_inputs":true,"runtime_integration_allowed":false}),
    )?;
    eprintln!(
        "[ffn_read] layer={} rows={}: {} serial pairs",
        layer.source.layer_id, pipeline.rows, args.trials
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every original word and scale bit contributes with defined wrapping;
    /// malformed spans cannot silently truncate the checksum reference.
    #[test]
    fn test_original_block_checksums_wrap_and_reject_wrong_spans() -> Result<()> {
        let mut block = vec![255u8; 34];
        block[..2].copy_from_slice(&0xabcd_u16.to_le_bytes());
        assert_eq!(
            checksum_columns(&block, 32, 1)?,
            vec![0xabcd_u32.wrapping_sub(1).wrapping_mul(8)]
        );
        let mut columns = block.clone();
        columns.extend(vec![0; 34]);
        assert_eq!(
            checksum_columns(&columns, 32, 2)?,
            vec![0xabcd_u32.wrapping_sub(1).wrapping_mul(8), 0]
        );
        for (k, hidden) in [(0, 1), (31, 1), (32, 0), (64, 1), (32, 2)] {
            assert!(checksum_columns(&block, k, hidden).is_err());
        }
        block[2] ^= 1;
        assert_eq!(
            checksum_columns(&block, 32, 1)?[0],
            0xabcd_u32.wrapping_sub(1).wrapping_mul(8).wrapping_sub(1)
        );
        Ok(())
    }
}

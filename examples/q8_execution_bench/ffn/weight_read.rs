//! Actual-plane read/checksum cost, never an inference candidate.
use super::{
    fixture::source_bytes,
    pipeline::{FfnKernel, FfnPipeline, GuardedF32, Stage},
    validation, warmup,
};
use crate::run::{inspect_gpu, write_record, Args};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, CudaEvent, CudaStream, DeviceBuffer},
    evaluation::{ffn_capture::FfnLayerSamples, identity::sha256},
    profile,
};
use serde_json::{json, Value};
use std::{
    fs::File,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

/// Derive modular sums directly from original interleaved GGUF blocks.
pub(super) fn checksum_columns(
    source: &[u8],
    k: usize,
    hidden: usize,
    scale_reads_per_block: u32,
) -> Result<Vec<u32>> {
    ensure!(
        k > 0 && k.is_multiple_of(32) && hidden > 0,
        "invalid checksum shape"
    );
    ensure!(
        matches!(scale_reads_per_block, 4 | 8),
        "unsupported checksum scale multiplicity"
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
                block[2..].chunks_exact(4).fold(
                    sum.wrapping_add(scale.wrapping_mul(scale_reads_per_block)),
                    |acc, word| acc.wrapping_add(u32::from_le_bytes(word.try_into().unwrap())),
                )
            })
        })
        .collect())
}

struct ReadProbe {
    output: GuardedF32,
    cub_scratch: Option<DeviceBuffer>,
    start: CudaEvent,
    stop: CudaEvent,
}

#[derive(Clone, Copy)]
enum ReadLayout<'a> {
    Original,
    Cub,
    Ordered(bool),
    Pitched,
    ColumnGap(&'a super::column_gap::GappedPayloads),
}

fn calculate_dispatch_owner(column: usize, hidden: usize, is_transposed: bool) -> usize {
    if is_transposed {
        let width = hidden / 32;
        (column % width) * 32 + column / width
    } else {
        column
    }
}

fn summarize_cub_pairs(pairs: &[(f64, f64)]) -> Result<Value> {
    let mut result = super::ablation::summarize(pairs)?;
    result["control_median_ms"] = result
        .as_object_mut()
        .unwrap()
        .remove("native_median_ms")
        .unwrap();
    result
        .as_object_mut()
        .unwrap()
        .remove("passes_micro_screen");
    result["passes_read_screen"] = json!(
        result["throughput_gain"].as_f64().unwrap() >= 0.10
            && result["bootstrap_gain_ci95"][0].as_f64().unwrap() > 0.0
    );
    result["interpretation"] =
        json!("library checksum reference only; not FFN or measured DRAM cost");
    Ok(result)
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
    fn new(pipeline: &FfnPipeline<'_>, should_use_cub: bool) -> Result<Self> {
        let ctx = &pipeline.model.cuda;
        Ok(Self {
            cub_scratch: if should_use_cub {
                Some(DeviceBuffer::new_tagged(
                    ctx,
                    ctx.ffn_q8_cub_checksum_scratch_bytes()?,
                    "ffn_read:cub_scratch",
                )?)
            } else {
                None
            },
            output: GuardedF32::new(
                ctx,
                &vec![0.0; pipeline.rows * pipeline.hidden_dim * 2],
                "ffn_read:checksums",
            )?,
            start: ctx.create_event()?,
            stop: ctx.create_event()?,
        })
    }

    fn enqueue(&self, pipeline: &FfnPipeline<'_>, layout: ReadLayout<'_>) -> Result<()> {
        unsafe {
            let prepared = pipeline.prepared.as_ref().context("missing read planes")?;
            match layout {
                ReadLayout::Cub => prepared.enqueue_checksum_cub(
                    self.output.pointer(),
                    pipeline.rows,
                    self.cub_scratch
                        .as_ref()
                        .context("missing CUB checksum scratch")?,
                ),
                ReadLayout::Ordered(is_transposed) => prepared.enqueue_checksum_ordered(
                    self.output.pointer(),
                    pipeline.rows,
                    is_transposed,
                ),
                ReadLayout::Original => {
                    prepared.enqueue_checksum(self.output.pointer(), pipeline.rows)
                }
                ReadLayout::Pitched => {
                    prepared.enqueue_checksum_pitched(None, self.output.pointer(), pipeline.rows)
                }
                ReadLayout::ColumnGap(payloads) => prepared.enqueue_checksum_pitched(
                    Some(payloads.pointers()),
                    self.output.pointer(),
                    pipeline.rows,
                ),
            }
        }
    }

    fn time(
        &self,
        pipeline: &FfnPipeline<'_>,
        iterations: usize,
        layout: ReadLayout<'_>,
    ) -> Result<Value> {
        ensure!(iterations > 0, "read timing needs iterations");
        pipeline.drain()?;
        let before = profile::snapshot();
        let started_unix_us = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let began = Instant::now();
        self.start.record(CudaStream::Decode)?;
        for _ in 0..iterations {
            self.enqueue(pipeline, layout)?;
        }
        self.stop.record(CudaStream::Decode)?;
        let after = profile::snapshot();
        let counters = profile::counter_delta(&before, &after);
        let op = match layout {
            ReadLayout::Cub => "ffn_q8_cub_checksum",
            ReadLayout::Original => "ffn_q8_weight_checksum_128",
            ReadLayout::Ordered(_) => "ffn_q8_weight_checksum_ordered_128",
            _ => "ffn_q8_weight_checksum_pitched_128",
        };
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
            json!({"stage":match layout {ReadLayout::Original=>"weight_read_checksum",ReadLayout::Cub=>"cub_segmented_read_checksum",ReadLayout::Ordered(false)=>"ordered_read_sequential",ReadLayout::Ordered(true)=>"ordered_read_transpose32",ReadLayout::Pitched=>"pitched_read_zero_gap",ReadLayout::ColumnGap(_)=>"pitched_read_column_gap"},"gpu_ms":gpu_ms,"iterations":iterations,
            "has_cta_index_tag":matches!(layout, ReadLayout::Ordered(_)),
            "column_gap_bytes":if matches!(layout,ReadLayout::ColumnGap(_)) {128} else {0},
            "started_unix_us":started_unix_us,"completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros(),
            "host_enqueue_and_completion_ms":began.elapsed().as_secs_f64()*1000.0/iterations as f64,
            "enqueue_counters":counters,"operation":op,"blocks_per_row":pipeline.hidden_dim,"threads":if matches!(layout,ReadLayout::Cub) {256} else {128},"shared_bytes":if matches!(layout,ReadLayout::Cub) {0} else {1024},
            "shared_bytes_kind":"requested_dynamic; static library storage reported separately",
            "cub_scratch_bytes":self.cub_scratch.as_ref().map_or(0,DeviceBuffer::bytes),
            "logical_unique_weight_bytes_per_call":pipeline.prepared.as_ref().unwrap().bytes * pipeline.rows,
            "has_inference_output":false,"has_measured_dram_traffic":false}),
        )
    }

    fn verify(
        &self,
        pipeline: &FfnPipeline<'_>,
        layer: &FfnLayerSamples,
        layout: ReadLayout<'_>,
    ) -> Result<Value> {
        pipeline.drain()?;
        let gate = checksum_columns(
            source_bytes(pipeline.model, &layer.source.gate)?,
            pipeline.d_model,
            pipeline.hidden_dim,
            8,
        )?;
        let up = checksum_columns(
            source_bytes(pipeline.model, &layer.source.up)?,
            pipeline.d_model,
            pipeline.hidden_dim,
            8,
        )?;
        let expected: Vec<_> = (0..pipeline.rows)
            .flat_map(|_| gate.iter().zip(&up).flat_map(|(&g, &u)| [g, u]))
            .collect();
        let mut actual: Vec<_> = self
            .output
            .read(&pipeline.model.cuda)?
            .into_iter()
            .map(f32::to_bits)
            .collect();
        if let ReadLayout::Ordered(is_transposed) = layout {
            for (index, word) in actual.iter_mut().enumerate() {
                let column = (index / 2) % pipeline.hidden_dim;
                let owner = calculate_dispatch_owner(column, pipeline.hidden_dim, is_transposed);
                *word = word.wrapping_sub(owner as u32);
            }
        }
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
    let probe = ReadProbe::new(pipeline, args.ffn_weight_read_cub)?;
    let gap = if args.ffn_weight_read_gap {
        Some(
            pipeline
                .prepared
                .as_ref()
                .context("missing gap control")?
                .prepare_column_gap()?,
        )
    } else {
        None
    };
    let resident = CudaContext::total_device_bytes();
    let prepared_before = serde_json::to_value(pipeline.model.q8_execution_stats())?;
    probe.enqueue(pipeline, ReadLayout::Original)?;
    let checks = probe.verify(pipeline, layer, ReadLayout::Original)?;
    if args.ffn_weight_read_cub {
        probe.enqueue(pipeline, ReadLayout::Cub)?;
        ensure!(
            probe.verify(pipeline, layer, ReadLayout::Cub)? == checks,
            "CUB checksum differs from original/CPU"
        );
    }
    if let Some(gap) = &gap {
        for layout in [ReadLayout::Pitched, ReadLayout::ColumnGap(gap)] {
            probe.enqueue(pipeline, layout)?;
            ensure!(
                probe.verify(pipeline, layer, layout)? == checks,
                "pitched checksum differs from original/CPU"
            );
        }
        gap.verify()?;
    }
    if args.ffn_weight_read_column_order {
        for layout in [ReadLayout::Ordered(false), ReadLayout::Ordered(true)] {
            probe.enqueue(pipeline, layout)?;
            ensure!(
                probe.verify(pipeline, layer, layout)? == checks,
                "ordered checksum/tag differs from original/CPU"
            );
        }
    }
    write_record(
        output,
        json!({"kind":"ffn_read_validation","layer_id":layer.source.layer_id,"rows":pipeline.rows,
        "checks":checks,"real_ffn_checks":validation::validate(pipeline, layer)?,
        "preparation":pipeline.prepared.as_ref().unwrap().report,"real_scratch_bytes":pipeline.scratch_bytes(),
        "column_gap_preparation":gap.as_ref().map(|g|&g.report),
        "has_verified_dispatch_owner_tags":args.ffn_weight_read_column_order,
        "has_verified_cub_checksums":args.ffn_weight_read_cub,
        "cub_scratch_bytes":probe.cub_scratch.as_ref().map_or(0,DeviceBuffer::bytes),
        "tracked_device_bytes":resident,"runtime_integration_allowed":false}),
    )?;
    let control_warmup = pipeline.warm_up(1000.0, 256)?;
    let read_warmup = warmup::measure(100.0, |count| {
        Ok(probe.time(pipeline, count, ReadLayout::Original)?["gpu_ms"]
            .as_f64()
            .unwrap())
    })?;
    let mut gap_warmups = Vec::new();
    let mut order_warmups = Vec::new();
    let cub_warmup = if args.ffn_weight_read_cub {
        Some(warmup::measure(100.0, |count| {
            Ok(probe.time(pipeline, count, ReadLayout::Cub)?["gpu_ms"]
                .as_f64()
                .unwrap())
        })?)
    } else {
        None
    };
    if args.ffn_weight_read_column_order {
        for layout in [ReadLayout::Ordered(false), ReadLayout::Ordered(true)] {
            order_warmups.push(warmup::measure(100.0, |count| {
                Ok(probe.time(pipeline, count, layout)?["gpu_ms"]
                    .as_f64()
                    .unwrap())
            })?);
        }
    }
    if let Some(gap) = &gap {
        for layout in [ReadLayout::Pitched, ReadLayout::ColumnGap(gap)] {
            gap_warmups.push(warmup::measure(100.0, |count| {
                Ok(probe.time(pipeline, count, layout)?["gpu_ms"]
                    .as_f64()
                    .unwrap())
            })?);
        }
    }
    write_record(
        output,
        json!({"kind":"ffn_read_warmup","layer_id":layer.source.layer_id,"rows":pipeline.rows,
        "real_ffn":control_warmup,"read_checksum":read_warmup,"cub_read_warmup":cub_warmup,"pitched_read_warmups":gap_warmups,"ordered_read_warmups":order_warmups,"gpu_after":inspect_gpu()?}),
    )?;
    let mut gap_pairs = Vec::new();
    let mut order_pairs = Vec::new();
    let mut original_order_pairs = Vec::new();
    let mut cub_pairs = Vec::new();
    for trial in 0..args.trials {
        let gpu_before = inspect_gpu()?;
        let mut measurements = Vec::new();
        let mut arms = if gap.is_some() {
            vec![0, 4, 5, 1, 2, 3]
        } else if args.ffn_weight_read_column_order {
            vec![0, 6, 7, 1, 2, 3]
        } else if args.ffn_weight_read_cub {
            vec![0, 8, 1, 2, 3]
        } else {
            vec![0, 1, 2, 3]
        };
        if trial % 2 == 1 {
            arms.reverse();
        }
        for arm in arms {
            measurements.push(match arm {
                0 => probe.time(pipeline, args.iterations, ReadLayout::Original)?,
                8 => probe.time(pipeline, args.iterations, ReadLayout::Cub)?,
                1 => time_control(pipeline, Stage::GateUpSwiglu, args.iterations)?,
                2 => time_control(pipeline, Stage::IsolatedDown, args.iterations)?,
                6 => probe.time(pipeline, args.iterations, ReadLayout::Ordered(false))?,
                7 => probe.time(pipeline, args.iterations, ReadLayout::Ordered(true))?,
                4 => probe.time(pipeline, args.iterations, ReadLayout::Pitched)?,
                5 => probe.time(
                    pipeline,
                    args.iterations,
                    ReadLayout::ColumnGap(gap.as_ref().unwrap()),
                )?,
                _ => time_control(pipeline, Stage::Complete, args.iterations)?,
            });
        }
        if args.ffn_weight_read_cub {
            let get = |stage| -> Result<f64> {
                measurements
                    .iter()
                    .find(|m| m["stage"] == stage)
                    .and_then(|m| m["gpu_ms"].as_f64())
                    .context("missing CUB reference timing")
            };
            cub_pairs.push((
                get("weight_read_checksum")?,
                get("cub_segmented_read_checksum")?,
            ));
        }
        if args.ffn_weight_read_column_order {
            let get = |stage| -> Result<f64> {
                measurements
                    .iter()
                    .find(|m| m["stage"] == stage)
                    .and_then(|m| m["gpu_ms"].as_f64())
                    .context("missing ordered read timing")
            };
            order_pairs.push((
                get("ordered_read_sequential")?,
                get("ordered_read_transpose32")?,
            ));
            original_order_pairs.push((
                get("weight_read_checksum")?,
                get("ordered_read_transpose32")?,
            ));
        }
        if gap.is_some() {
            let get = |stage| -> Result<f64> {
                measurements
                    .iter()
                    .find(|m| m["stage"] == stage)
                    .and_then(|m| m["gpu_ms"].as_f64())
                    .context("missing pitched read timing")
            };
            gap_pairs.push((
                get("pitched_read_zero_gap")?,
                get("pitched_read_column_gap")?,
            ));
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
    if args.ffn_weight_read_cub {
        let result = summarize_cub_pairs(&cub_pairs)?;
        write_record(
            output,
            json!({"kind":"ffn_read_cub_screen","layer_id":layer.source.layer_id,
            "rows":pipeline.rows,"result":result,"one_row_read_gain_gate_percent":10,
            "runtime_integration_allowed":false}),
        )?;
        probe.enqueue(pipeline, ReadLayout::Cub)?;
        ensure!(
            probe.verify(pipeline, layer, ReadLayout::Cub)? == checks,
            "CUB checksum changed after timed scratch reuse"
        );
        probe.enqueue(pipeline, ReadLayout::Original)?;
    }
    if gap.is_some() {
        let mut result = super::ablation::summarize(&gap_pairs)?;
        result["control_median_ms"] = result
            .as_object_mut()
            .unwrap()
            .remove("native_median_ms")
            .unwrap();
        result["interpretation"] = json!("read-cost diagnostic only; positive gate may justify a separate inference experiment, never runtime integration");
        write_record(
            output,
            json!({"kind":"ffn_read_gap_screen","layer_id":layer.source.layer_id,"rows":pipeline.rows,
            "result":result,"runtime_integration_allowed":false}),
        )?;
    }
    if args.ffn_weight_read_column_order {
        let mut comparisons = Vec::new();
        for (control, pairs) in [
            ("same_kernel_sequential", &order_pairs),
            ("original_reader", &original_order_pairs),
        ] {
            let mut result = super::ablation::summarize(pairs)?;
            result["control_median_ms"] = result
                .as_object_mut()
                .unwrap()
                .remove("native_median_ms")
                .unwrap();
            comparisons.push(json!({"control":control,"result":result}));
        }
        write_record(
            output,
            json!({"kind":"ffn_read_order_screen","layer_id":layer.source.layer_id,"rows":pipeline.rows,
            "comparisons":comparisons,"runtime_integration_allowed":false}),
        )?;
        for layout in [ReadLayout::Ordered(false), ReadLayout::Ordered(true)] {
            probe.enqueue(pipeline, layout)?;
            ensure!(
                probe.verify(pipeline, layer, layout)? == checks,
                "ordered checksum/tag changed after timed scratch reuse"
            );
        }
        probe.enqueue(pipeline, ReadLayout::Original)?;
    }
    ensure!(
        probe.verify(pipeline, layer, ReadLayout::Original)? == checks,
        "read checksum changed after scratch reuse"
    );
    pipeline.verify_guards_and_inputs(layer)?;
    if let Some(gap) = &gap {
        gap.verify()?;
    }
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

    /// Read-reference qualification uses its prespecified 10% threshold,
    /// not the shared inference screen's 5%, and still requires valid pairs.
    #[test]
    fn test_cub_summary_requires_ten_percent_read_gain() -> Result<()> {
        for (ratio, should_pass) in [(1.0, false), (1.07, false), (1.11, true)] {
            let pairs = vec![(ratio, 1.0); 5];
            let result = summarize_cub_pairs(&pairs)?;
            assert_eq!(result["passes_read_screen"], should_pass);
            assert_eq!(result["control_median_ms"], ratio);
            assert!(result.get("passes_micro_screen").is_none());
        }
        assert!(summarize_cub_pairs(&[(2.0, 1.0); 4]).is_err());
        assert!(summarize_cub_pairs(&[(f64::NAN, 1.0); 5]).is_err());
        Ok(())
    }

    /// The diagnostic transpose covers every column once, and inverse tags
    /// distinguish it from an ignored selector for nontrivial output widths.
    #[test]
    fn test_column_order_inverse_covers_each_physical_owner() {
        for hidden in [32, 64, 96, 11008] {
            let columns: Vec<_> = (0..hidden)
                .map(|b| (b % 32) * (hidden / 32) + b / 32)
                .collect();
            let mut sorted = columns.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..hidden).collect::<Vec<_>>());
            for (b, &column) in columns.iter().enumerate() {
                assert_eq!(calculate_dispatch_owner(column, hidden, true), b);
                assert_eq!(calculate_dispatch_owner(column, hidden, false), column);
            }
            assert_eq!(columns != sorted, hidden != 32);
        }
    }

    /// Every original word and scale bit contributes with defined wrapping;
    /// malformed spans cannot silently truncate the checksum reference.
    #[test]
    fn test_original_block_checksums_wrap_and_reject_wrong_spans() -> Result<()> {
        let mut block = vec![255u8; 34];
        block[..2].copy_from_slice(&0xabcd_u16.to_le_bytes());
        assert_eq!(
            checksum_columns(&block, 32, 1, 8)?,
            vec![0xabcd_u32.wrapping_sub(1).wrapping_mul(8)]
        );
        assert_eq!(
            checksum_columns(&block, 32, 1, 4)?,
            vec![0xabcd_u32.wrapping_mul(4).wrapping_sub(8)]
        );
        for repeats in [0, 1, 3, 16, u32::MAX] {
            assert!(checksum_columns(&block, 32, 1, repeats).is_err());
        }
        let mut columns = block.clone();
        columns.extend(vec![0; 34]);
        assert_eq!(
            checksum_columns(&columns, 32, 2, 8)?,
            vec![0xabcd_u32.wrapping_sub(1).wrapping_mul(8), 0]
        );
        for (k, hidden) in [(0, 1), (31, 1), (32, 0), (64, 1), (32, 2)] {
            assert!(checksum_columns(&block, k, hidden, 8).is_err());
        }
        block[2] ^= 1;
        assert_eq!(
            checksum_columns(&block, 32, 1, 8)?[0],
            0xabcd_u32.wrapping_sub(1).wrapping_mul(8).wrapping_sub(1)
        );
        Ok(())
    }
}

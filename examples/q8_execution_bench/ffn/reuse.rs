//! Same-work temporal-order diagnostic; no model or kernel changes.
use super::{
    fixture::FfnInputs,
    pipeline::{FfnKernel, FfnPipeline, Stage},
    validation, warmup,
};
use crate::run::{inspect_gpu, write_record, Args};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, CudaEvent, CudaStream},
    infer::LoadedModel,
    profile,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    fs::File,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const STAGES: [Stage; 3] = [Stage::GateUpSwiglu, Stage::IsolatedDown, Stage::Complete];

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum LayerOrder {
    Grouped,
    Alternating,
}

pub(super) fn validate_args(args: &Args) -> Result<()> {
    ensure!(
        args.rows == [1],
        "layer reuse diagnostic requires --rows 1; four-row work has different stream dependencies"
    );
    ensure!(
        matches!(args.trials, 5 | 10 | 20) && args.iterations == 10,
        "layer reuse diagnostic requires 5/10/20 fresh trials and 10 iterations"
    );
    Ok(())
}

fn visit_layers(
    order: LayerOrder,
    iterations: usize,
    mut visit: impl FnMut(usize) -> Result<()>,
) -> Result<()> {
    ensure!(
        (1..=4096).contains(&iterations),
        "invalid bounded reuse iteration count"
    );
    match order {
        LayerOrder::Grouped => {
            for layer in 0..2 {
                for _ in 0..iterations {
                    visit(layer)?;
                }
            }
        }
        LayerOrder::Alternating => {
            for _ in 0..iterations {
                for layer in 0..2 {
                    visit(layer)?;
                }
            }
        }
    }
    Ok(())
}

pub(super) fn read_outputs(pipeline: &FfnPipeline<'_>, stage: Stage) -> Result<Vec<u32>> {
    let buffers = match stage {
        Stage::GateUpSwiglu => vec![&pipeline.hidden],
        Stage::IsolatedDown => vec![&pipeline.isolated_output],
        Stage::Complete => vec![&pipeline.hidden, &pipeline.output],
        Stage::Quantize => anyhow::bail!("no activation quantization in reuse diagnostic"),
    };
    let mut bits = Vec::new();
    for buffer in buffers {
        bits.extend(
            buffer
                .read(&pipeline.model.cuda)?
                .into_iter()
                .map(f32::to_bits),
        );
    }
    Ok(bits)
}

pub(super) struct ReuseTimer {
    start: CudaEvent,
    stop: CudaEvent,
}

impl ReuseTimer {
    pub(super) fn new(ctx: &CudaContext) -> Result<Self> {
        Ok(Self {
            start: ctx.create_event()?,
            stop: ctx.create_event()?,
        })
    }

    pub(super) fn measure(
        &self,
        pipelines: [&FfnPipeline<'_>; 2],
        stage: Stage,
        order: LayerOrder,
        iterations: usize,
    ) -> Result<Value> {
        ensure!(
            (1..=4096).contains(&iterations) && !matches!(stage, Stage::Quantize),
            "invalid reuse interval"
        );
        ensure!(
            std::ptr::eq(pipelines[0].model, pipelines[1].model)
                && pipelines
                    .iter()
                    .all(|p| p.rows == 1 && p.kernel == FfnKernel::SplitF32x4),
            "reuse diagnostic requires one model and one-row qualified kernels"
        );
        pipelines[0].drain()?;
        let before = profile::snapshot();
        let started_unix_us = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let started = Instant::now();
        self.start.record(CudaStream::Decode)?;
        if let Err(error) = visit_layers(order, iterations, |index| pipelines[index].enqueue(stage))
        {
            pipelines[0]
                .drain()
                .context("drain failed reuse interval")?;
            return Err(error);
        }
        self.stop.record(CudaStream::Decode)?;
        let after = profile::snapshot();
        let counters = profile::counter_delta(&before, &after);
        let calls = (iterations * 2) as u64;
        let gate_calls = if matches!(stage, Stage::IsolatedDown) {
            0
        } else {
            calls
        };
        let down_calls = if matches!(stage, Stage::GateUpSwiglu) {
            0
        } else {
            calls
        };
        let count = |name| {
            after.by_op.get(name).map_or(0, |v| v.launches)
                - before.by_op.get(name).map_or(0, |v| v.launches)
        };
        ensure!(
            counters.launches == gate_calls + down_calls
                && count("ffn_q8_split_f32x4_128") == gate_calls
                && count("gemm_f32xq8_separate_f32") == down_calls
                && counters.device_allocs
                    + counters.device_frees
                    + counters.h2d_copies
                    + counters.d2h_copies
                    + counters.stream_syncs
                    + counters.stream_waits
                    + counters.cublas_calls
                    == 0,
            "reuse interval has wrong work or enqueue side effects"
        );
        let gpu_interval_ms = f64::from(self.start.elapsed_sync(&self.stop, "ffn_reuse_interval")?);
        Ok(
            json!({"stage":stage,"order":order,"iterations_per_layer":iterations,
            "layer_operations":calls,"gate_up_calls":gate_calls,"down_calls":down_calls,
            "gpu_interval_ms":gpu_interval_ms,"gpu_ms_per_layer":gpu_interval_ms/calls as f64,
            "host_enqueue_and_completion_ms":started.elapsed().as_secs_f64()*1000.0,
            "started_unix_us":started_unix_us,"completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros(),
            "enqueue_counters":counters,"has_measured_dram_traffic":false}),
        )
    }
}

fn summarize(pairs: &[(f64, f64)]) -> Result<Value> {
    let mut result = super::ablation::summarize(pairs)?;
    let gain = result["throughput_gain"].as_f64().unwrap();
    let interval = &result["bootstrap_gain_ci95"];
    let change = 1.0 / (1.0 + gain) - 1.0;
    let ci = [
        1.0 / (1.0 + interval[1].as_f64().unwrap()) - 1.0,
        1.0 / (1.0 + interval[0].as_f64().unwrap()) - 1.0,
    ];
    let object = result.as_object_mut().unwrap();
    let grouped = object.remove("native_median_ms").unwrap();
    let alternating = object.remove("candidate_median_ms").unwrap();
    for name in [
        "throughput_gain",
        "bootstrap_gain_ci95",
        "passes_micro_screen",
    ] {
        object.remove(name);
    }
    result["grouped_median_ms"] = grouped;
    result["alternating_median_ms"] = alternating;
    result["alternating_latency_change"] = json!(change);
    result["latency_change_ci95"] = json!(ci);
    result["has_material_order_sensitivity"] =
        json!(change.abs() >= 0.03 && (ci[0] > 0.0 || ci[1] < 0.0));
    result["has_bounded_small_order_effect"] = json!(ci[0] > -0.03 && ci[1] < 0.03);
    result["interpretation"] = json!(
        "temporal-order diagnostic, not measured cache cost, inference gain or HTTP qualification"
    );
    Ok(result)
}

pub(super) fn benchmark(
    output: &mut File,
    model: &LoadedModel,
    inputs: &FfnInputs,
    args: &Args,
) -> Result<()> {
    for (a, b) in [(0, 1), (1, 2), (2, 0)] {
        let layers = [&inputs.artifact.layers[a], &inputs.artifact.layers[b]];
        let ids = [layers[0].source.layer_id, layers[1].source.layer_id];
        let before_case = CudaContext::total_device_bytes();
        let mut left = FfnPipeline::new(model, layers[0], 1)?;
        let mut right = FfnPipeline::new(model, layers[1], 1)?;
        left.prepare_split_weights(layers[0])?;
        right.prepare_split_weights(layers[1])?;
        left.select_kernel(FfnKernel::SplitF32x4)?;
        right.select_kernel(FfnKernel::SplitF32x4)?;
        let pipelines = [&left, &right];
        let preparation_bytes = pipelines
            .iter()
            .map(|p| p.prepared.as_ref().unwrap().bytes)
            .sum::<usize>();
        ensure!(
            preparation_bytes <= 128 * 1024 * 1024,
            "reuse preparation exceeds 128 MiB"
        );
        let resident = CudaContext::total_device_bytes();
        let cache = serde_json::to_value(model.q8_execution_stats())?;
        let checks = pipelines
            .iter()
            .zip(layers)
            .map(|(p, layer)| validation::validate(p, layer))
            .collect::<Result<Vec<_>>>()?;
        let mut expected = Vec::new();
        for stage in STAGES {
            for p in pipelines {
                p.enqueue(stage)?;
            }
            left.drain()?;
            expected.push([read_outputs(&left, stage)?, read_outputs(&right, stage)?]);
        }
        let timer = ReuseTimer::new(&model.cuda)?;
        write_record(
            output,
            json!({"kind":"ffn_reuse_validation","layer_ids":ids,"rows":1,"checks":checks,
            "preparation":pipelines.iter().map(|p|&p.prepared.as_ref().unwrap().report).collect::<Vec<_>>(),
            "additional_prepared_bytes":preparation_bytes,"scratch_bytes":left.scratch_bytes()+right.scratch_bytes(),
            "tracked_device_bytes":resident,"runtime_integration_allowed":false}),
        )?;
        let mut warmups = Vec::new();
        for order in [LayerOrder::Grouped, LayerOrder::Alternating] {
            let result = warmup::measure(1000.0, |n| {
                Ok(
                    timer.measure(pipelines, Stage::Complete, order, n)?["gpu_ms_per_layer"]
                        .as_f64()
                        .unwrap()
                        * 2.0,
                )
            })?;
            warmups.push(json!({"order":order,"result":result}));
        }
        write_record(
            output,
            json!({"kind":"ffn_reuse_warmup","layer_ids":ids,"warmups":warmups}),
        )?;
        let mut pairs: [Vec<(f64, f64)>; 3] = std::array::from_fn(|_| Vec::new());
        for trial in 0..args.trials {
            let gpu_before = inspect_gpu()?;
            let mut measurements = Vec::new();
            let mut orders = [LayerOrder::Grouped, LayerOrder::Alternating];
            if trial % 2 == 1 {
                orders.reverse();
            }
            for (index, stage) in STAGES.into_iter().enumerate() {
                let mut times = [0.0; 2];
                for order in orders {
                    let timing = timer.measure(pipelines, stage, order, args.iterations)?;
                    times[match order {
                        LayerOrder::Grouped => 0,
                        LayerOrder::Alternating => 1,
                    }] = timing["gpu_ms_per_layer"]
                        .as_f64()
                        .context("missing reuse time")?;
                    for (which, p) in pipelines.into_iter().enumerate() {
                        ensure!(
                            read_outputs(p, stage)? == expected[index][which],
                            "reuse order changes output bits"
                        );
                    }
                    measurements.push(timing);
                }
                pairs[index].push((times[0], times[1]));
            }
            ensure!(
                CudaContext::total_device_bytes() == resident,
                "reuse interval changes device allocation accounting"
            );
            write_record(
                output,
                json!({"kind":"ffn_reuse_pair","layer_ids":ids,"trial":trial,"rows":1,
                "measurements":measurements,"has_matching_output_bits":true,"gpu_before":gpu_before,
                "gpu_after":inspect_gpu()?,"tracked_device_bytes":resident}),
            )?;
        }
        let results = STAGES
            .into_iter()
            .zip(pairs)
            .map(|(stage, pairs)| Ok(json!({"stage":stage,"result":summarize(&pairs)?})))
            .collect::<Result<Vec<_>>>()?;
        for (p, layer) in pipelines.into_iter().zip(layers) {
            p.verify_guards_and_inputs(layer)?;
        }
        ensure!(
            serde_json::to_value(model.q8_execution_stats())? == cache,
            "reuse diagnostic changes qualified cache"
        );
        drop((timer, left, right));
        ensure!(
            CudaContext::total_device_bytes() == before_case,
            "reuse pair leaks device payloads"
        );
        write_record(
            output,
            json!({"kind":"ffn_reuse_case_complete","layer_ids":ids,"results":results,
            "has_intact_inputs_and_guards":true,"has_matching_output_bits":true,
            "tracked_bytes_before":before_case,"tracked_bytes_after":CudaContext::total_device_bytes(),
            "runtime_integration_allowed":false}),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Reuse diagnostics require captured one-row work and cannot combine
    /// with another arithmetic/layout experiment or silently accept batching.
    #[test]
    fn test_layer_reuse_requires_exclusive_one_row_inputs() {
        let base = ["bench", "--model", "model.gguf", "--output", "out.jsonl"];
        let parse = |extra: &[&str]| Args::try_parse_from([base.as_slice(), extra].concat());
        assert!(!parse(&[]).unwrap().ffn_layer_reuse);
        assert!(parse(&["--ffn-layer-reuse"]).is_err());
        let valid = [
            "--ffn-inputs",
            "inputs.json",
            "--ffn-capture-report",
            "capture.jsonl",
            "--ffn-layer-reuse",
        ];
        let one = parse(&[valid.as_slice(), &["--rows", "1"]].concat()).unwrap();
        assert!(validate_args(&one).is_ok());
        for rows in ["4", "1,4", "0"] {
            let args = parse(&[valid.as_slice(), &["--rows", rows]].concat()).unwrap();
            assert!(validate_args(&args).is_err());
        }
        for other in ["--ffn-weight-read", "--ffn-q81", "--ffn-f32-shared-scales"] {
            assert!(parse(&[valid.as_slice(), &[other]].concat()).is_err());
        }
        for extra in [["--trials", "7"], ["--iterations", "9"]] {
            let args = parse(&[valid.as_slice(), &["--rows", "1"], &extra].concat()).unwrap();
            assert!(validate_args(&args).is_err());
        }
    }

    /// Both orders visit each independent layer equally; only reuse distance
    /// changes, and invalid counts or callback errors cannot add hidden work.
    #[test]
    fn test_visit_orders_preserve_work_and_propagate_failures() -> Result<()> {
        for n in [1, 3, 10] {
            for order in [LayerOrder::Grouped, LayerOrder::Alternating] {
                let mut visits = Vec::new();
                visit_layers(order, n, |index| {
                    visits.push(index);
                    Ok(())
                })?;
                let expected = match order {
                    LayerOrder::Grouped => (0..2)
                        .flat_map(|i| std::iter::repeat_n(i, n))
                        .collect::<Vec<_>>(),
                    LayerOrder::Alternating => (0..n).flat_map(|_| [0, 1]).collect(),
                };
                assert_eq!(visits, expected);
                let mut count = 0;
                assert!(visit_layers(order, n, |_| {
                    count += 1;
                    anyhow::bail!("failure")
                })
                .is_err());
                assert_eq!(count, 1);
            }
        }
        for n in [0, 4097, usize::MAX] {
            let mut count = 0;
            assert!(visit_layers(LayerOrder::Grouped, n, |_| {
                count += 1;
                Ok(())
            })
            .is_err());
            assert_eq!(count, 0);
        }
        Ok(())
    }

    /// Order sensitivity is not an optimization gate; test both latency signs
    /// and bounded small effects rather than inheriting the FFN 5% decision.
    #[test]
    fn test_summary_distinguishes_small_and_material_order_effects() -> Result<()> {
        for (time, material) in [(1.01, false), (1.05, true), (0.95, true)] {
            let result = summarize(&[(1.0, time); 5])?;
            assert!(
                (result["alternating_latency_change"].as_f64().unwrap() - (time - 1.0)).abs()
                    < 1e-12
            );
            assert_eq!(result["has_material_order_sensitivity"], material);
            assert_eq!(result["has_bounded_small_order_effect"], !material);
            assert!(result.get("passes_micro_screen").is_none());
        }
        Ok(())
    }
}

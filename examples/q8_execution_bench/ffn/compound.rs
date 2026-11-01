//! One fixed four-arm comparison. Preparation remains resident in every arm.
use super::{
    ablation,
    pipeline::{FfnKernel, FfnPipeline, Stage},
    validation,
};
use crate::run::{inspect_gpu, write_record, Args};
use anyhow::{ensure, Context, Result};
use m40_llm::{cuda::CudaContext, evaluation::ffn_capture::FfnLayerSamples};
use serde_json::{json, Value};
use std::fs::File;

pub(super) const ARMS: [FfnKernel; 4] = [
    FfnKernel::SplitF32x4,
    FfnKernel::JointF32x4,
    FfnKernel::SharedScales,
    FfnKernel::JointSharedScales,
];

fn arm_order(trial: usize) -> [usize; 4] {
    let mut order = [0, 1, 2, 3];
    order.rotate_left(trial % 4);
    if trial % 2 == 1 {
        order.reverse();
    }
    order
}

fn complete_ms(stages: &[Value]) -> Result<f64> {
    stages
        .iter()
        .find(|s| s["stage"] == "complete")
        .and_then(|s| s["gpu_ms"].as_f64())
        .context("missing compound complete FFN time")
}

pub(super) fn benchmark_case(
    output: &mut File,
    pipeline: &mut FfnPipeline<'_>,
    layer: &FfnLayerSamples,
    args: &Args,
) -> Result<()> {
    let rows = pipeline.rows;
    let mut reference: Option<[Vec<f32>; 4]> = None;
    for arm in ARMS {
        pipeline.select_kernel(arm)?;
        let checks = validation::validate(pipeline, layer)?;
        let values = [
            pipeline.gate.read(&pipeline.model.cuda)?,
            pipeline.up.read(&pipeline.model.cuda)?,
            pipeline.hidden.read(&pipeline.model.cuda)?,
            pipeline.output.read(&pipeline.model.cuda)?,
        ];
        if let Some(expected) = &reference {
            for (actual, expected) in values.iter().zip(expected) {
                ensure!(
                    actual.len() == expected.len()
                        && actual
                            .iter()
                            .zip(expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "compound arm changed gate/up/SwiGLU/down bits"
                );
            }
        } else {
            reference = Some(values);
        }
        write_record(
            output,
            json!({"kind":"ffn_compound_validation","layer_id":layer.source.layer_id,
            "rows":rows,"variant":arm,"checks":checks,"has_control_bits":true,
            "scratch_capacity_bytes":pipeline.scratch_bytes(),"source":layer.source}),
        )?;
    }
    for arm in ARMS {
        pipeline.select_kernel(arm)?;
        let warmup = pipeline.warm_up(1000.0, 256)?;
        write_record(
            output,
            json!({"kind":"ffn_compound_warmup","layer_id":layer.source.layer_id,
            "rows":rows,"variant":arm,"warmup":warmup,"gpu_after":inspect_gpu()?}),
        )?;
    }
    let mut timings = Vec::new();
    for trial in 0..args.trials {
        let gpu_before = inspect_gpu()?;
        let order = arm_order(trial);
        let mut measurements = Vec::new();
        let mut complete = [0.0; 4];
        for index in order {
            let arm = ARMS[index];
            pipeline.select_kernel(arm)?;
            let mut stages = [Stage::GateUpSwiglu, Stage::IsolatedDown, Stage::Complete];
            if trial % 2 == 1 {
                stages.reverse();
            }
            let stages = stages
                .into_iter()
                .map(|s| pipeline.time(s, args.iterations))
                .collect::<Result<Vec<_>>>()?;
            complete[index] = complete_ms(&stages)?;
            measurements.push(json!({"variant":arm,"stages":stages}));
        }
        timings.push(complete);
        write_record(
            output,
            json!({"kind":"ffn_compound_trial","layer_id":layer.source.layer_id,
            "rows":rows,"trial":trial,"order":order,"measurements":measurements,
            "gpu_before":gpu_before,"gpu_after":inspect_gpu()?,
            "scratch_capacity_bytes":pipeline.scratch_bytes(),
            "split_prepared_bytes":pipeline.prepared.as_ref().context("missing split")?.bytes,
            "joint_prepared_bytes":pipeline.joint.as_ref().context("missing joint")?.bytes,
            "tracked_device_bytes":CudaContext::total_device_bytes()}),
        )?;
    }
    let compare = |a: usize, b: usize| {
        ablation::summarize(&timings.iter().map(|t| (t[a], t[b])).collect::<Vec<_>>())
    };
    write_record(
        output,
        json!({"kind":"ffn_compound_summary","layer_id":layer.source.layer_id,"rows":rows,
        "compound_vs_split":compare(0,3)?,"compound_vs_joint":compare(1,3)?,
        "joint_vs_split":compare(0,1)?,"shared_vs_split":compare(0,2)?,
        "acceptance":"compound complete one-row FFN >=5% gain versus qualified split with positive paired bounds at all three layers; no runtime/E2E qualification implied"}),
    )?;
    pipeline.verify_guards_and_inputs(layer)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Each contemporaneous trial measures every arm exactly once with
    /// rotating/reversed traversal rather than independent historical runs.
    #[test]
    fn test_four_arm_order_visits_each_arm_once() {
        for trial in 0..20 {
            let mut order = arm_order(trial);
            order.sort();
            assert_eq!(order, [0, 1, 2, 3]);
        }
        assert_ne!(arm_order(0), arm_order(1));
    }

    /// Compound selection is exclusive and requires captured real inputs.
    #[test]
    fn test_compound_cli_rejects_combined_experiments() {
        let base = [
            "bench",
            "--model",
            "missing.gguf",
            "--output",
            "unused.jsonl",
            "--ffn-inputs",
            "missing.json",
            "--ffn-capture-report",
            "missing.jsonl",
            "--ffn-f32-compound",
        ];
        assert!(Args::try_parse_from(base).is_ok());
        for other in [
            "--ffn-f32-joint",
            "--ffn-f32-shared-scales",
            "--ffn-q81",
            "--ffn-layer-reuse",
        ] {
            assert!(Args::try_parse_from(base.into_iter().chain([other])).is_err());
        }
        assert!(Args::try_parse_from([
            "bench",
            "--model",
            "m",
            "--output",
            "o",
            "--ffn-f32-compound"
        ])
        .is_err());
    }
}

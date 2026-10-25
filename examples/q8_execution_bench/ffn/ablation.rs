use super::{
    pipeline::{FfnKernel, FfnPipeline, Stage},
    validation,
};
use crate::run::{inspect_gpu, write_record, Args};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{Q8FfnFp32Variant, Q8FfnQ81Launch},
    evaluation::ffn_capture::FfnLayerSamples,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{json, Value};
use std::fs::File;

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    }
}

/// Resample contemporaneous pairs, not independently sorted arm timings.
pub(super) fn summarize(pairs: &[(f64, f64)]) -> Result<Value> {
    ensure!(
        pairs.len() >= 5
            && pairs
                .iter()
                .all(|(a, b)| a.is_finite() && b.is_finite() && *a > 0.0 && *b > 0.0),
        "FFN screen requires five finite positive timing pairs"
    );
    let controls: Vec<_> = pairs.iter().map(|(a, _)| *a).collect();
    let candidates: Vec<_> = pairs.iter().map(|(_, b)| *b).collect();
    let gain = median(&controls) / median(&candidates) - 1.0;
    let mut rng = StdRng::seed_from_u64(0);
    let mut gains = Vec::with_capacity(10_000);
    let mut a = vec![0.0; pairs.len()];
    let mut b = a.clone();
    for _ in 0..10_000 {
        for (a, b) in a.iter_mut().zip(&mut b) {
            (*a, *b) = pairs[rng.gen_range(0..pairs.len())];
        }
        gains.push(median(&a) / median(&b) - 1.0);
    }
    gains.sort_by(f64::total_cmp);
    let ci = [gains[249], gains[9749]];
    Ok(json!({"pairs":pairs,"native_median_ms":median(&controls),
        "candidate_median_ms":median(&candidates),"throughput_gain":gain,
        "bootstrap_gain_ci95":ci,"bootstrap_resamples":10000,"bootstrap_seed":0,
        "passes_micro_screen":gain>=0.05 && ci[0]>0.0,
        "interpretation":"complete-FFN screen only; neither runtime qualification nor E2E promotion"}))
}

fn time_stages(pipeline: &FfnPipeline<'_>, args: &Args, trial: usize) -> Result<Vec<Value>> {
    let mut stages = vec![Stage::GateUpSwiglu, Stage::IsolatedDown, Stage::Complete];
    if pipeline.kernel.has_quantization() {
        stages.insert(0, Stage::Quantize);
    }
    if trial % 2 == 1 {
        stages.reverse();
    }
    stages
        .into_iter()
        .map(|stage| pipeline.time(stage, args.iterations))
        .collect()
}

fn complete_ms(stages: &[Value]) -> Result<f64> {
    stages
        .iter()
        .find(|timing| timing["stage"] == "complete")
        .and_then(|timing| timing["gpu_ms"].as_f64())
        .context("missing complete-FFN GPU latency")
}

pub(super) fn benchmark_case(
    output: &mut File,
    pipeline: &mut FfnPipeline<'_>,
    layer: &FfnLayerSamples,
    args: &Args,
) -> Result<()> {
    let rows = pipeline.rows;
    let is_followup = args.ffn_f32_factored_scales || args.ffn_f32_coalesced_pairs;
    let control = if is_followup {
        FfnKernel::Fp32(Q8FfnFp32Variant::PackedWarp128)
    } else {
        FfnKernel::Native
    };
    let candidates: Vec<_> = if args.ffn_f32_coalesced_pairs {
        vec![FfnKernel::Fp32(
            Q8FfnFp32Variant::PackedWarp128CoalescedPairs,
        )]
    } else if args.ffn_f32_factored_scales {
        vec![FfnKernel::Fp32(
            Q8FfnFp32Variant::PackedWarp128FactoredScales,
        )]
    } else if args.ffn_q81 {
        Q8FfnQ81Launch::ALL
            .into_iter()
            .map(FfnKernel::Q81)
            .collect()
    } else {
        Q8FfnFp32Variant::ALL
            .into_iter()
            .map(FfnKernel::Fp32)
            .collect()
    };
    pipeline.select_kernel(FfnKernel::Native)?;
    let native_checks = validation::validate(pipeline, layer)?;
    let native_hidden = pipeline.hidden.read(&pipeline.model.cuda)?;
    let native_output = pipeline.output.read(&pipeline.model.cuda)?;
    write_record(
        output,
        json!({"kind":"ffn_ablation_validation","layer_id":layer.source.layer_id,
        "rows":rows,"variant":null,"checks":native_checks,"source":layer.source,
        "row_positions":layer.rows[..rows],"scratch_capacity_bytes":pipeline.scratch_bytes()}),
    )?;
    if control != FfnKernel::Native {
        pipeline.select_kernel(control)?;
        let checks = validation::validate(pipeline, layer)?;
        write_record(
            output,
            json!({"kind":"ffn_ablation_validation",
            "layer_id":layer.source.layer_id,"rows":rows,"variant":control,
            "role":"timing-control","checks":checks,"source":layer.source,
            "row_positions":layer.rows[..rows],"scratch_capacity_bytes":pipeline.scratch_bytes()}),
        )?;
    }
    let mut row_fused_reference: Option<Vec<f32>> = None;
    for &variant in &candidates {
        pipeline.select_kernel(variant)?;
        let checks = validation::validate(pipeline, layer)?;
        let hidden = pipeline.hidden.read(&pipeline.model.cuda)?;
        let result = pipeline.output.read(&pipeline.model.cuda)?;
        let compare = if variant.has_quantization() {
            validation::measure
        } else {
            validation::compare
        };
        let hidden_vs_native = compare(
            &hidden,
            &native_hidden
                .iter()
                .copied()
                .map(f64::from)
                .collect::<Vec<_>>(),
        )?;
        let output_vs_native = compare(
            &result,
            &native_output
                .iter()
                .copied()
                .map(f64::from)
                .collect::<Vec<_>>(),
        )?;
        if matches!(variant,FfnKernel::Fp32(v) if v.threads()==256) {
            let reference = row_fused_reference.get_or_insert_with(|| hidden.clone());
            ensure!(
                hidden
                    .iter()
                    .zip(reference)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "256-thread FP32 ablations changed fused row-control bits"
            );
            ensure!(
                rows != 1 || (hidden_vs_native.bitwise_equal && output_vs_native.bitwise_equal),
                "256-thread one-row FP32 ablations changed native completed FFN bits"
            );
        }
        write_record(
            output,
            json!({"kind":"ffn_ablation_validation","layer_id":layer.source.layer_id,
            "rows":rows,"variant":variant,"checks":checks,"hidden_vs_native":hidden_vs_native,
            "complete_output_vs_native":output_vs_native,"scratch_capacity_bytes":pipeline.scratch_bytes(),
            "fused_256_thread_row_control_bits_preserved":matches!(variant,FfnKernel::Fp32(v) if v.threads()==256),
            "is_numerical_track":variant.has_quantization(),
            "projection_checks_scope":if variant.has_quantization() {"candidate gate/up/SwiGLU checked against quantized-input CPU reference; original FP32 drift reported separately"} else if is_followup {"candidate fused gate/up capture, SwiGLU and completed down outputs checked against scalar CPU references; capture and normal SwiGLU bits equal"} else {"gate/up checks use the native unfused oracle; candidate fused SwiGLU and completed down outputs are checked against scalar CPU references"}}),
        )?;
    }
    // All expensive CPU references precede active warmup and primary pairs.
    let mut results = Vec::new();
    for variant in candidates {
        pipeline.select_kernel(control)?;
        let control_warmup = pipeline.warm_up(1000.0, 256)?;
        pipeline.select_kernel(variant)?;
        let candidate_warmup = pipeline.warm_up(100.0, 256)?;
        write_record(
            output,
            json!({"kind":"ffn_ablation_warmup","layer_id":layer.source.layer_id,
            "rows":rows,"variant":variant,"control_variant":control,
            "control":control_warmup,"candidate":candidate_warmup,
            "native":if control==FfnKernel::Native {Some(&control_warmup)} else {None},
            "gpu_after":inspect_gpu()?}),
        )?;
        let mut pairs = Vec::new();
        for trial in 0..args.trials {
            let gpu_before = inspect_gpu()?;
            let mut arms = [control, variant];
            if trial % 2 == 1 {
                arms.reverse();
            }
            let mut measurements = Vec::new();
            let (mut control_ms, mut candidate_ms) = (0.0, 0.0);
            let mut quantization_ms = 0.0;
            for arm in arms {
                pipeline.select_kernel(arm)?;
                let stages = time_stages(pipeline, args, trial)?;
                if arm != control {
                    candidate_ms = complete_ms(&stages)?;
                    if arm.has_quantization() {
                        quantization_ms = stages
                            .iter()
                            .find(|s| s["stage"] == "quantize")
                            .and_then(|s| s["gpu_ms"].as_f64())
                            .context("missing actual quantization cost")?;
                    }
                } else {
                    control_ms = complete_ms(&stages)?;
                }
                measurements.push(json!({"variant":arm,"stages":stages}));
            }
            pairs.push((control_ms, candidate_ms));
            write_record(
                output,
                json!({"kind":"ffn_ablation_pair","layer_id":layer.source.layer_id,
                "rows":rows,"variant":variant,"control_variant":control,"trial":trial,"measurements":measurements,
                "gpu_before":gpu_before,"gpu_after":inspect_gpu()?,"activation_quantization_ms":quantization_ms,
                "is_numerical_track":variant.has_quantization(),
                "scratch_capacity_bytes":pipeline.scratch_bytes(),"threads":variant.threads(),
                "quantization_payload_bytes":pipeline.quantized.as_ref().map_or(0,|q|q.shape.bytes()),
                "quantization_scratch_capacity_bytes":pipeline.quantized.as_ref().map_or(0,|q|q.shape.bytes()+32),
                "has_packed_loads":variant.has_packed_loads(),"has_warp_reduction":variant.has_warp_reduction(),
                "scope":if args.ffn_f32_coalesced_pairs {"adjacent-pair FP32 loads versus qualified packed/warp128; same fused geometry and immutable prepared down"} else if args.ffn_f32_factored_scales {"FP32 scale factoring versus qualified packed/warp128; same fused geometry and immutable prepared down"} else if variant.has_quantization() {"Q8_1 quantization included in candidate gate/up and complete FFN; numerical track, prepared FP32-activation down unchanged"} else if rows==1 {"native one-row split4 load/reduction ablation"} else {"genuine fused rows4 versus production two-projection/SwiGLU path; not a pure load/reduction ablation"}}),
            )?;
        }
        pipeline.verify_guards_and_inputs(layer)?;
        eprintln!(
            "[q8_execution_bench] layer={} rows={rows} variant={variant:?}: {} complete FFN pairs",
            layer.source.layer_id, args.trials
        );
        results.push((variant, pairs));
    }
    // Statistical CPU work follows all primary GPU measurements for this shape.
    for (variant, pairs) in results {
        let mut result = summarize(&pairs)?;
        if control != FfnKernel::Native {
            let median = result
                .as_object_mut()
                .unwrap()
                .remove("native_median_ms")
                .unwrap();
            result["control_median_ms"] = median;
        }
        write_record(
            output,
            json!({"kind":"ffn_ablation_screen","layer_id":layer.source.layer_id,
            "rows":rows,"variant":variant,"control_variant":control,"result":result,
            "is_decode_screen":rows==1,"runtime_integration_allowed":false}),
        )?;
    }
    Ok(())
}

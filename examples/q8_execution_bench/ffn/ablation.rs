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
    let is_followup = args.ffn_f32_role_parallel
        || args.ffn_f32_native_warp
        || args.ffn_f32_task_pairs
        || args.ffn_f32_eight_values
        || args.ffn_f32_cohorts
        || args.ffn_f32_persistent_tiles
        || args.ffn_f32_virtual_warp
        || args.ffn_f32_factored_scales
        || args.ffn_f32_coalesced_pairs
        || args.ffn_f32_split_planes
        || args.ffn_f32_split_two_columns
        || args.ffn_f32_joint
        || args.ffn_f32_fixed_k
        || args.ffn_f32_unfused
        || args.ffn_f32_weight_cg
        || args.ffn_f32_byte_permute;
    let control = if args.ffn_f32_role_parallel
        || args.ffn_f32_native_warp
        || args.ffn_f32_task_pairs
        || args.ffn_f32_eight_values
        || args.ffn_f32_cohorts
        || args.ffn_f32_persistent_tiles
        || args.ffn_f32_virtual_warp
        || args.ffn_f32_split_two_columns
        || args.ffn_f32_joint
        || args.ffn_f32_fixed_k
        || args.ffn_f32_unfused
        || args.ffn_f32_weight_cg
        || args.ffn_f32_byte_permute
    {
        FfnKernel::SplitF32x4
    } else if is_followup {
        FfnKernel::Fp32(Q8FfnFp32Variant::PackedWarp128)
    } else {
        FfnKernel::Native
    };
    let candidates: Vec<_> = if args.ffn_f32_role_parallel {
        vec![FfnKernel::RoleParallel]
    } else if args.ffn_f32_native_warp {
        vec![FfnKernel::NativeWarp]
    } else if args.ffn_f32_task_pairs {
        vec![FfnKernel::TaskPairPacket]
    } else if args.ffn_f32_eight_values {
        vec![FfnKernel::SplitF32x8]
    } else if args.ffn_f32_cohorts {
        vec![FfnKernel::SplitF32x4Cohorts]
    } else if args.ffn_f32_persistent_tiles {
        vec![FfnKernel::SplitF32x4PersistentTiles]
    } else if args.ffn_f32_virtual_warp {
        vec![FfnKernel::SplitF32x4VirtualWarp]
    } else if args.ffn_f32_byte_permute {
        vec![FfnKernel::SplitF32x4BytePermute]
    } else if args.ffn_f32_weight_cg {
        vec![FfnKernel::SplitF32x4WeightCg]
    } else if args.ffn_f32_unfused {
        vec![FfnKernel::SplitF32x4Unfused]
    } else if args.ffn_f32_fixed_k {
        vec![FfnKernel::SplitF32x4K2048]
    } else if args.ffn_f32_joint {
        vec![FfnKernel::JointF32x4]
    } else if args.ffn_f32_split_two_columns {
        vec![FfnKernel::SplitF32x4TwoColumns]
    } else if args.ffn_f32_split_planes {
        vec![FfnKernel::SplitF32x4]
    } else if args.ffn_f32_coalesced_pairs {
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
    let control_hidden = pipeline.hidden.read(&pipeline.model.cuda)?;
    let control_output = pipeline.output.read(&pipeline.model.cuda)?;
    let control_projections =
        if args.ffn_f32_role_parallel || args.ffn_f32_eight_values || args.ffn_f32_native_warp {
            Some([
                pipeline.gate.read(&pipeline.model.cuda)?,
                pipeline.up.read(&pipeline.model.cuda)?,
            ])
        } else {
            None
        };
    let mut row_fused_reference: Option<Vec<f32>> = None;
    for &variant in &candidates {
        pipeline.select_kernel(variant)?;
        let checks = validation::validate(pipeline, layer)?;
        let hidden = pipeline.hidden.read(&pipeline.model.cuda)?;
        let result = pipeline.output.read(&pipeline.model.cuda)?;
        let has_control_hidden_bits = hidden
            .iter()
            .zip(&control_hidden)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        let has_control_output_bits = result
            .iter()
            .zip(&control_output)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        let drift_vs_control = if let Some(projections) = &control_projections {
            if args.ffn_f32_role_parallel {
                for (actual, reference) in [
                    (pipeline.gate.read(&pipeline.model.cuda)?, &projections[0]),
                    (pipeline.up.read(&pipeline.model.cuda)?, &projections[1]),
                ] {
                    ensure!(
                        actual
                            .iter()
                            .zip(reference)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "role-parallel changed original gate/up bits"
                    );
                }
            }
            let compare_control = |actual: &[f32], reference: &[f32]| {
                validation::compare(
                    actual,
                    &reference.iter().copied().map(f64::from).collect::<Vec<_>>(),
                )
            };
            Some(
                json!({"gate":compare_control(&pipeline.gate.read(&pipeline.model.cuda)?,&projections[0])?,
                "up":compare_control(&pipeline.up.read(&pipeline.model.cuda)?,&projections[1])?,
                "swiglu":compare_control(&hidden,&control_hidden)?,"complete_down":compare_control(&result,&control_output)?,
                "scope":"FP32 ownership drift, not activation quantization; unchanged CPU bound also enforced"}),
            )
        } else {
            None
        };
        if args.ffn_f32_role_parallel
            || args.ffn_f32_task_pairs
            || args.ffn_f32_cohorts
            || args.ffn_f32_persistent_tiles
            || args.ffn_f32_virtual_warp
            || args.ffn_f32_split_two_columns
            || args.ffn_f32_joint
            || args.ffn_f32_fixed_k
            || args.ffn_f32_unfused
            || args.ffn_f32_weight_cg
            || args.ffn_f32_byte_permute
        {
            ensure!(
                has_control_hidden_bits && has_control_output_bits,
                "split follow-up FFN changed one-column control arithmetic"
            );
        }
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
            "timing_control_hidden_bits_preserved":has_control_hidden_bits,
            "timing_control_output_bits_preserved":has_control_output_bits,
            "drift_vs_timing_control":drift_vs_control,
            "fused_256_thread_row_control_bits_preserved":matches!(variant,FfnKernel::Fp32(v) if v.threads()==256),
            "is_numerical_track":variant.has_quantization(),
            "projection_checks_scope":if args.ffn_f32_unfused {"separate prepared gate/up projections, control-order SwiGLU and completed down checked against CPU references; captured and ordinary outputs preserve control bits"} else if variant.has_quantization() {"candidate gate/up/SwiGLU checked against quantized-input CPU reference; original FP32 drift reported separately"} else if is_followup {"candidate fused gate/up capture, SwiGLU and completed down outputs checked against scalar CPU references; capture and normal SwiGLU bits equal"} else {"gate/up checks use the native unfused oracle; candidate fused SwiGLU and completed down outputs are checked against scalar CPU references"}}),
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
                "additional_prepared_bytes":pipeline.prepared.as_ref().map_or(0,|p|p.bytes)+pipeline.joint.as_ref().map_or(0,|p|p.bytes)+pipeline.task_pair.as_ref().map_or(0,|p|p.bytes),
                "task_pair_prepared_bytes":pipeline.task_pair.as_ref().map_or(0,|p|p.bytes),
                "joint_prepared_bytes":pipeline.joint.as_ref().map_or(0,|p|p.bytes),
                "tracked_device_bytes":m40_llm::cuda::CudaContext::total_device_bytes(),
                "quantization_payload_bytes":pipeline.quantized.as_ref().map_or(0,|q|q.shape.bytes()),
                "quantization_scratch_capacity_bytes":pipeline.quantized.as_ref().map_or(0,|q|q.shape.bytes()+32),
                "has_packed_loads":variant.has_packed_loads(),"has_warp_reduction":variant.has_warp_reduction(),
                "cache_policy":if args.ffn_f32_weight_cg {json!({"control_weights":"nc","candidate_weights":"cg","activations":"nc","geometry":"one-column/four-value/128","prepared_down":"unchanged"})} else {Value::Null},
                "scope":if args.ffn_f32_role_parallel {"two physical 128-thread tensor roles per output CTA; original task/FMA and reduction order, split planes, precise SiLU and immutable prepared down; no activation quantization"} else if args.ffn_f32_native_warp {"native 32-partial FP32 association, one gate/up accumulator pair per lane; four output warps, original split planes and immutable prepared down, no activation quantization or 128-partial emulation"} else if args.ffn_f32_task_pairs {"thread-owned two-task payload packets; unchanged separate scales, original four-value task/FMA order, precise SiLU, one-column/128-thread tree and prepared down; no activation quantization"} else if args.ffn_f32_eight_values {"eight-value FP32 tasks versus qualified four-value split; one output/128 threads, unchanged planes/down, changed summation ownership and no activation quantization"} else if args.ffn_f32_cohorts {"two independent physical 128-thread groups per 256-thread CTA; original per-output FP32 tree, planes and prepared down; one publication barrier, no per-thread extra output or new scratch"} else if args.ffn_f32_persistent_tiles {"one fixed reusable 192-CTA-per-row grid; same physical per-column tree, FP32 association, split planes and prepared down; no device-wide barrier or new scratch"} else if args.ffn_f32_virtual_warp {"one fixed register-owned virtual 128-thread tree per output warp; same split planes, FP32 association, inputs and prepared down; no shared publication or block barrier"} else if args.ffn_f32_byte_permute {"exact signed-byte permutation/bias conversion versus qualified integer-to-float; identical FP32 scale/FMA association, fused geometry, split planes, read-only loads and prepared down"} else if args.ffn_f32_weight_cg {"L2-only weight/scale loads versus qualified read-only weights; identical fused math, geometry, split planes, activation loads and prepared down; no new scratch"} else if args.ffn_f32_unfused {"one fixed unfused gate/up/SwiGLU decomposition versus qualified one-column fused split; same planes, FP32 association, existing scratch and prepared down; all three launches and intermediate traffic included"} else if args.ffn_f32_fixed_k {"fixed K=2048 address/loop specialization versus qualified one-column split; same planes, FP32 arithmetic and prepared down; no new layout or scratch"} else if args.ffn_f32_joint {"one-column paired gate/up records versus qualified one-column split; both pairs resident in both arms; original FP32 arithmetic and immutable down; no activation quantization"} else if args.ffn_f32_split_two_columns {"fixed two-column split FP32/four-value/128 versus qualified one-column split; identical prepared gate/up and down; no extra allocation or quantization"} else if args.ffn_f32_split_planes {"existing split-plane Q8 weights with four-value FP32 loads versus qualified packed/warp128; same fused geometry and immutable prepared down; added preparation resident for both arms"} else if args.ffn_f32_coalesced_pairs {"adjacent-pair FP32 loads versus qualified packed/warp128; same fused geometry and immutable prepared down"} else if args.ffn_f32_factored_scales {"FP32 scale factoring versus qualified packed/warp128; same fused geometry and immutable prepared down"} else if variant.has_quantization() {"Q8_1 quantization included in candidate gate/up and complete FFN; numerical track, prepared FP32-activation down unchanged"} else if rows==1 {"native one-row split4 load/reduction ablation"} else {"genuine fused rows4 versus production two-projection/SwiGLU path; not a pure load/reduction ablation"}}),
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

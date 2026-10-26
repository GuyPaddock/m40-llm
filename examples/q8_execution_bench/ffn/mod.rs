mod ablation;
mod access;
mod fixture;
mod joint;
mod pipeline;
mod prepared;
mod prepared_projection;
mod projection;
mod reductions;
mod validation;
mod warmup;

use super::{command_output, inspect_gpu, write_record, Args};
use anyhow::{ensure, Result};
use fixture::FfnInputs;
use m40_llm::{
    cuda::CudaContext,
    evaluation::identity::file_sha256,
    gguf::load_gguf,
    infer::{execution_weights::Q8ExecutionConfig, LoadedModel},
};
use pipeline::{FfnKernel, FfnPipeline, Stage};
use serde_json::json;
use std::{
    fs::{self, OpenOptions},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub(super) fn benchmark(args: &Args) -> Result<()> {
    if args.decode_reductions {
        return reductions::benchmark(args);
    }
    if args.prepared_f32_warp || args.prepared_f32_packed_activation {
        return prepared_projection::benchmark(args);
    }
    if args.qkv_f32_warp
        || args.output_f32_warp
        || args.qkv_f32_packed
        || args.output_f32_packed
        || args.qkv_f32_split
        || args.output_f32_split
    {
        return projection::benchmark(args);
    }
    let is_fp32_ablation = args.ffn_f32_eight_values
        || args.ffn_f32_cohorts
        || args.ffn_f32_persistent_tiles
        || args.ffn_f32_virtual_warp
        || args.ffn_f32_ablation
        || args.ffn_f32_factored_scales
        || args.ffn_f32_coalesced_pairs
        || args.ffn_f32_split_planes
        || args.ffn_f32_split_two_columns
        || args.ffn_f32_joint
        || args.ffn_f32_fixed_k
        || args.ffn_f32_unfused
        || args.ffn_f32_weight_cg
        || args.ffn_f32_byte_permute;
    ensure!(
        !cfg!(debug_assertions) && args.rows == [1, 4],
        "FFN baseline requires release and --rows 1,4"
    );
    let inputs = FfnInputs::load(args)?;
    ensure!(
        !args.ffn_f32_fixed_k || inputs.artifact.plan.d_model == 2048,
        "fixed-width FFN benchmark requires K=2048 captured inputs"
    );
    inputs.validate_environment()?;
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.output)?;
    let gpu = inspect_gpu()?;
    let initial_device_bytes = CudaContext::total_device_bytes();
    let began = Instant::now();
    let model = LoadedModel::from_gguf_with_q8_execution(
        load_gguf(&args.model)?,
        fs::read(&args.model)?,
        -1,
        Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
    )?;
    inputs.validate_model(&model)?;
    let setup_ms = began.elapsed().as_secs_f64() * 1000.0;
    write_record(
        &mut output,
        json!({"kind":if args.ffn_q81 {"ffn_q81_manifest"} else if is_fp32_ablation {"ffn_ablation_manifest"} else {"ffn_baseline_manifest"},"schema_version":1,
        "model":args.model,"model_sha256":inputs.artifact.plan.model_sha256,
        "fixture":args.ffn_inputs,"fixture_sha256":inputs.artifact_sha256,"capture_report":args.ffn_capture_report,
        "capture_report_sha256":inputs.report_sha256,"capture_revision":inputs.manifest["revision"],
        "execution_configuration":inputs.manifest["execution_configuration"],"execution_sha256":inputs.artifact.plan.execution_sha256,
        "revision":command_output("git", &["rev-parse","HEAD"])?,"worktree":command_output("git", &["status","--short"])?,
        "executable_sha256":file_sha256(&std::env::current_exe()?)?,"build_profile":"release",
        "created_unix_s":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "rustc":command_output("rustc", &["--version"])?,"nvcc":command_output("nvcc", &["--version"])?,
        "host_compiler":command_output("g++", &["--version"])?,"cuda_flags":["-O3","sm_52","compute_52","no fast math"],
        "gpu_before":gpu,"prepared_weights":model.q8_execution_stats(),"setup_ms":setup_ms,
        "memory":{"original_weights":model.weights_len,"prepared_weights":model.q8_execution_stats().resident_bytes,
            "resident_kv":0,"tracked_device_bytes":CudaContext::total_device_bytes()},
        "rows":args.rows,"trials":args.trials,"iterations":args.iterations,
        "is_fp32_ablation":is_fp32_ablation,
        "request_pattern_hypothesis":if args.ffn_f32_coalesced_pairs {Some(access::describe_requests(inputs.artifact.plan.d_model))} else {None},
        "is_q81_numerical_track":args.ffn_q81,
        "q81_variants":if args.ffn_q81 {json!(m40_llm::cuda::Q8FfnQ81Launch::ALL)} else {serde_json::Value::Null},
        "fp32_variants":if args.ffn_f32_eight_values {json!([FfnKernel::SplitF32x8])} else if args.ffn_f32_cohorts {json!([FfnKernel::SplitF32x4Cohorts])} else if args.ffn_f32_persistent_tiles {json!([FfnKernel::SplitF32x4PersistentTiles])} else if args.ffn_f32_virtual_warp {json!([FfnKernel::SplitF32x4VirtualWarp])} else if args.ffn_f32_byte_permute {json!([FfnKernel::SplitF32x4BytePermute])} else if args.ffn_f32_weight_cg {json!([FfnKernel::SplitF32x4WeightCg])} else if args.ffn_f32_unfused {json!([FfnKernel::SplitF32x4Unfused])} else if args.ffn_f32_fixed_k {json!([FfnKernel::SplitF32x4K2048])} else if args.ffn_f32_joint {json!([FfnKernel::JointF32x4])} else if args.ffn_f32_split_two_columns {json!([FfnKernel::SplitF32x4TwoColumns])} else if args.ffn_f32_split_planes {json!([FfnKernel::SplitF32x4])} else if args.ffn_f32_coalesced_pairs {json!([m40_llm::cuda::Q8FfnFp32Variant::PackedWarp128CoalescedPairs])} else if args.ffn_f32_factored_scales {json!([m40_llm::cuda::Q8FfnFp32Variant::PackedWarp128FactoredScales])} else if args.ffn_f32_ablation {json!(m40_llm::cuda::Q8FfnFp32Variant::ALL)} else {serde_json::Value::Null},
        "eight_value_contract":if args.ffn_f32_eight_values {json!({"values_per_task":8,"has_changed_summation_ownership":true,"has_activation_quantization":false,"control":"qualified split-f32x4-128, original prepared down/vocabulary","acceptance":"unchanged CPU tolerance; actual-512/frozen quality and HTTP gates remain required"})} else {serde_json::Value::Null},
        "byte_conversion":if args.ffn_f32_byte_permute {json!({"control":"signed-integer-to-float","candidate":"exact-byte-permute-bias","geometry":"one-column/four-value/128","prepared_down":"unchanged"})} else {serde_json::Value::Null},
        "cache_policy":if args.ffn_f32_weight_cg {json!({"control_weights":"nc","candidate_weights":"cg","activations":"nc","scope":"fixed fused geometry, planes, FP32 association and prepared down"})} else {serde_json::Value::Null},
        "activation_quantization":{"is_enabled":args.ffn_q81,"control_gpu_ms":0.0,"candidate_gpu_ms":if args.ffn_q81 {json!("measured separately and included in gate/up and complete totals")} else {json!(0.0)}},
        "control":if args.ffn_f32_eight_values {"qualified split-f32x4-128 versus eight-value tasks on the same planes; changed FP32 summation ownership, unchanged prepared down/vocabulary, no activation quantization or KV"} else if args.ffn_f32_cohorts {"qualified one-output 128-thread CTA versus two physical 128-thread output cohorts in a 256-thread CTA; original per-output arithmetic/tree, planes and prepared down/vocabulary; no KV or attention"} else if args.ffn_f32_persistent_tiles {"qualified one-CTA-per-column split FP32/four-value/128 versus a fixed 192-CTA-per-row reusable grid; identical per-column math, split planes and prepared down/vocabulary; no KV or attention"} else if args.ffn_f32_virtual_warp {"qualified one-column split FP32/four-value/128 versus four independent output warps emulating the same virtual partials/tree; identical planes, arithmetic and prepared down/vocabulary; no KV or attention"} else if args.ffn_f32_unfused {"qualified one-column split FP32/four-value/128 versus separate fixed four-column projections and control-order SwiGLU; same planes, existing gate/up scratch and immutable down/vocabulary; all three launches included"} else if args.ffn_f32_joint {"qualified one-column split-plane FP32/four-value/128; both arms retain control and joint pairs with immutable down/vocabulary; no KV or attention"} else if args.ffn_f32_split_two_columns || args.ffn_f32_fixed_k || args.ffn_f32_weight_cg || args.ffn_f32_byte_permute {"qualified one-column split-plane FP32/four-value/128 gate/up/SwiGLU; same prepared gate/up pair and immutable down/vocabulary; no KV or attention"} else if args.ffn_f32_factored_scales || args.ffn_f32_coalesced_pairs || args.ffn_f32_split_planes {"qualified packed FP32/128 gate/up/SwiGLU; immutable prepared down/vocabulary; no KV or attention"} else {"native FP32-activation gate/up/SwiGLU; immutable prepared down/vocabulary; no KV allocated or attention executed"},
        "timing_scope":"CUDA events and host enqueue+completion per repeated stage; required cross-stream waits included; setup/validation/copies excluded",
        "interpretation":"complete FFN microbenchmark baseline, not token/HTTP throughput, a speedup, verifier cost or measured DRAM traffic"}),
    )?;
    for layer in &inputs.artifact.layers {
        for &rows in &args.rows {
            let mut pipeline = FfnPipeline::new(&model, layer, rows)?;
            if args.ffn_f32_eight_values
                || args.ffn_f32_cohorts
                || args.ffn_f32_persistent_tiles
                || args.ffn_f32_virtual_warp
                || args.ffn_f32_split_planes
                || args.ffn_f32_split_two_columns
                || args.ffn_f32_joint
                || args.ffn_f32_fixed_k
                || args.ffn_f32_unfused
                || args.ffn_f32_weight_cg
                || args.ffn_f32_byte_permute
            {
                pipeline.prepare_split_weights(layer)?;
                write_record(
                    &mut output,
                    json!({"kind":"ffn_ablation_preparation","layer_id":layer.source.layer_id,
                    "rows":rows,"preparation":pipeline.prepared.as_ref().unwrap().report,
                    "scratch_capacity_bytes":pipeline.scratch_bytes()}),
                )?;
            }
            if args.ffn_f32_joint {
                pipeline.joint = Some(joint::JointFfnWeights::new(
                    &model,
                    layer,
                    pipeline.prepared.as_ref().unwrap().bytes,
                )?);
                write_record(
                    &mut output,
                    json!({"kind":"ffn_joint_preparation","layer_id":layer.source.layer_id,
                    "rows":rows,"preparation":pipeline.joint.as_ref().unwrap().report}),
                )?;
            }
            if args.ffn_q81 {
                pipeline.reserve_quantized_scratch()?;
            }
            if is_fp32_ablation || args.ffn_q81 {
                ablation::benchmark_case(&mut output, &mut pipeline, layer, args)?;
                continue;
            }
            let checks = validation::validate(&pipeline, layer)?;
            write_record(
                &mut output,
                json!({"kind":"ffn_baseline_validation","layer_id":layer.source.layer_id,
                "rows":rows,"row_positions":layer.rows[..rows],"checks":checks,
                "scratch_capacity_bytes":pipeline.scratch_bytes(),"source":layer.source,
                "memory":{"original_weights":model.weights_len,"prepared_weights":model.q8_execution_stats().resident_bytes,
                    "resident_kv":0,"tracked_device_bytes":CudaContext::total_device_bytes()},
                "scope":"whole model weights and qualified vocabulary cache resident but only this layer FFN executes; native CUDA allocator/driver overhead excluded"}),
            )?;
            let warmup = pipeline.warm_up(1000.0, 256)?;
            write_record(
                &mut output,
                json!({"kind":"ffn_baseline_warmup","layer_id":layer.source.layer_id,"rows":rows,"warmup":warmup,"gpu_after":inspect_gpu()?}),
            )?;
            for trial in 0..args.trials {
                let gpu_before = inspect_gpu()?;
                let mut stages = [Stage::GateUpSwiglu, Stage::IsolatedDown, Stage::Complete];
                if trial % 2 == 1 {
                    stages.reverse();
                }
                let mut timings = Vec::new();
                for stage in stages {
                    timings.push(pipeline.time(stage, args.iterations)?);
                }
                write_record(
                    &mut output,
                    json!({"kind":"ffn_baseline_timing","layer_id":layer.source.layer_id,
                    "rows":rows,"trial":trial,"stages":timings,"gpu_before":gpu_before,"gpu_after":inspect_gpu()?,
                    "activation_quantization_ms":0.0,"scratch_capacity_bytes":pipeline.scratch_bytes()}),
                )?;
            }
            pipeline.verify_guards_and_inputs(layer)?;
            eprintln!(
                "[q8_execution_bench] FFN layer={} rows={rows}: validated {} baseline trials",
                layer.source.layer_id, args.trials
            );
        }
    }
    ensure!(
        model
            .materialized_weights
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned materialization cache"))?
            .is_empty(),
        "FFN baseline unexpectedly materialized FP32 weights"
    );
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial_device_bytes,
        "FFN baseline leaked tracked device allocations"
    );
    write_record(
        &mut output,
        json!({"kind":if args.ffn_q81 {"ffn_q81_complete"} else if is_fp32_ablation {"ffn_ablation_complete"} else {"ffn_baseline_complete"},"is_complete":true,"cases":6,
        "device_bytes_before":initial_device_bytes,"device_bytes_after":CudaContext::total_device_bytes(),
        "gpu_after_release":inspect_gpu()?,"scope":"Rust-tracked payload cleanup, not driver context RSS"}),
    )
}

#[cfg(test)]
mod tests;

//! Native projection screens using frozen normalized FFN input surrogates.
mod evidence;
mod pipeline;
mod prepared;
#[cfg(test)]
mod tests;
use super::super::{command_output, inspect_gpu, write_record, Args};
use super::{ablation::summarize, fixture::FfnInputs};
use anyhow::{ensure, Result};
use m40_llm::{
    cuda::CudaContext,
    evaluation::identity::file_sha256,
    gguf::load_gguf,
    infer::{execution_weights::Q8ExecutionConfig, LoadedModel},
};
use pipeline::{ProjectionComparison, ProjectionKind, ProjectionProbe};
use serde_json::json;
use std::{
    fs::{self, OpenOptions},
    time::Instant,
};

pub(super) fn benchmark(args: &Args) -> Result<()> {
    let kind = if args.output_f32_warp || args.output_f32_packed || args.output_f32_split {
        ProjectionKind::Output
    } else {
        ProjectionKind::Qkv
    };
    let comparison = if args.qkv_f32_split || args.output_f32_split {
        ProjectionComparison::SplitPlanes
    } else if args.qkv_f32_packed || args.output_f32_packed {
        ProjectionComparison::PackedLoads
    } else {
        ProjectionComparison::WarpReduction
    };
    let prefix = match (kind, comparison) {
        (ProjectionKind::Qkv, ProjectionComparison::WarpReduction) => "qkv_warp",
        (ProjectionKind::Output, ProjectionComparison::WarpReduction) => "output_warp",
        (ProjectionKind::Qkv, ProjectionComparison::PackedLoads) => "qkv_packed",
        (ProjectionKind::Output, ProjectionComparison::PackedLoads) => "output_packed",
        (ProjectionKind::Qkv, ProjectionComparison::SplitPlanes) => "qkv_split",
        (ProjectionKind::Output, ProjectionComparison::SplitPlanes) => "output_split",
    };
    ensure!(
        !cfg!(debug_assertions) && args.rows == [1, 4],
        "projection screen requires release and --rows 1,4"
    );
    let inputs = FfnInputs::load(args)?;
    inputs.validate_environment()?;
    let peer = args
        .qkv_warp_report
        .as_deref()
        .map(|path| evidence::QkvEvidence::load(path, &inputs))
        .transpose()?;
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.output)?;
    let gpu = inspect_gpu()?;
    let initial = CudaContext::total_device_bytes();
    let began = Instant::now();
    let model = LoadedModel::from_gguf_with_q8_execution(
        load_gguf(&args.model)?,
        fs::read(&args.model)?,
        -1,
        Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
    )?;
    inputs.validate_model(&model)?;
    write_record(
        &mut output,
        json!({"kind":format!("{prefix}_manifest"),"schema_version":1,
        "revision":command_output("git",&["rev-parse","HEAD"])?,"worktree":command_output("git",&["status","--short"])?,
        "executable_sha256":file_sha256(&std::env::current_exe()?)?,"build_profile":"release",
        "model":args.model,"model_sha256":inputs.artifact.plan.model_sha256,
        "fixture":args.ffn_inputs,"fixture_sha256":inputs.artifact_sha256,"capture_report":args.ffn_capture_report,
        "capture_report_sha256":inputs.report_sha256,"execution_configuration":inputs.manifest["execution_configuration"],
        "prepared_weights":model.q8_execution_stats(),"setup_ms":began.elapsed().as_secs_f64()*1000.0,
        "rows":args.rows,"trials":args.trials,"iterations":args.iterations,"gpu_before":gpu,
        "control":if comparison==ProjectionComparison::SplitPlanes {"qualified packed raw GGUF Q8_0 loads and original-tree warp finish, unchanged geometry"} else if comparison==ProjectionComparison::PackedLoads {"qualified raw GGUF Q8_0 scalar loads and original-tree warp finish; unchanged geometry"} else if kind==ProjectionKind::Qkv {"original raw GGUF Q8_0 split4 fused QKV, 256 threads, shared reduction"} else {"original raw GGUF Q8_0 coltile4 split4 output, 256 threads, shared reduction"},
        "candidate":if comparison==ProjectionComparison::SplitPlanes {"existing lossless split planes with one uint2 payload read per eight weights; qualified activation loads, geometry and FP32 tree"} else if comparison==ProjectionComparison::PackedLoads {"four U16 weights and two float4 activations per eight-value task; qualified warp tree and original FP32 association"} else {"same raw weights, task assignment and original-tree warp finish, 256 threads"},
        "comparison":if comparison==ProjectionComparison::SplitPlanes {"split-planes"} else if comparison==ProjectionComparison::PackedLoads {"packed-loads"} else {"warp-reduction"},
        "profile_source":if comparison==ProjectionComparison::SplitPlanes {json!({
            "path":"docs/experiments/2026-10-26-maxwell-packed-stage-audit.json",
            "sha256":"3ff95bf79269368145bf3044bb4d351ec826f11444b87cadafbfe7868e29177e",
            "generated_step":255,"scope":"optimistic GPU-stage fractions, not measured HTTP gains"})} else if comparison==ProjectionComparison::PackedLoads {json!({
            "path":"docs/experiments/2026-10-25-maxwell-projection-stage-audit.json",
            "sha256":"9445906c521ae61700935e0b934d8744d98eb0db247f7abeee667e2cc9e3066d",
            "generated_step":255,"scope":"optimistic GPU-stage fractions, not measured HTTP gains"})} else {serde_json::Value::Null},
        "input_scope":"frozen actual normalized FFN inputs used as projection surrogates; not captured attention-norm or attention-output inputs",
        "rows_scope":"one or four distinct independent single-token requests, one launch per row, not batched GEMM",
        "qkv_peer":peer.as_ref().map(|p|&p.binding),
        "memory":{"original_weights":model.weights_len,"prepared_weights":model.q8_execution_stats().resident_bytes,
            "tracked_device_bytes":CudaContext::total_device_bytes(),"resident_kv":0,"additional_prepared_bytes":0},
        "cuda_flags":["-O3","sm_52","compute_52","no fast math"],
        "nvcc":command_output("nvcc",&["--version"])?,"rustc":command_output("rustc",&["--version"])?}),
    )?;
    for layer in &inputs.artifact.layers {
        for &rows in &args.rows {
            let probe = ProjectionProbe::new(
                &model,
                layer.source.layer_id,
                rows,
                &layer.normalized_inputs[..rows * inputs.artifact.plan.d_model],
                kind,
                comparison,
            )?;
            let resident = CudaContext::total_device_bytes();
            write_record(
                &mut output,
                json!({"kind":format!("{prefix}_validation"),"layer_id":layer.source.layer_id,
                "rows":rows,"row_positions":layer.rows[..rows],"checks":probe.validate()?,"sources":probe.sources,
                "scratch_bytes":probe.scratch_bytes(),"tracked_device_bytes":resident,
                "preparation":probe.preparation()}),
            )?;
            let control_warm = probe.warm(false, 1000.0)?;
            let candidate_warm = probe.warm(true, 100.0)?;
            write_record(
                &mut output,
                json!({"kind":format!("{prefix}_warmup"),"layer_id":layer.source.layer_id,"rows":rows,
                "control":control_warm,"candidate":candidate_warm,"gpu_after":inspect_gpu()?}),
            )?;
            let mut pairs = Vec::new();
            for trial in 0..args.trials {
                let gpu_before = inspect_gpu()?;
                let order = if trial % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                };
                let mut times = Vec::new();
                for is_candidate in order {
                    times.push(probe.time(is_candidate, args.iterations)?);
                }
                let ms = |is_candidate| {
                    times
                        .iter()
                        .find(|v| v["is_candidate"] == is_candidate)
                        .unwrap()["gpu_ms"]
                        .as_f64()
                        .unwrap()
                };
                pairs.push((ms(false), ms(true)));
                ensure!(
                    CudaContext::total_device_bytes() == resident,
                    "projection timing changed resident allocations"
                );
                write_record(
                    &mut output,
                    json!({"kind":format!("{prefix}_pair"),"layer_id":layer.source.layer_id,"rows":rows,
                    "trial":trial,"measurements":times,"gpu_before":gpu_before,"gpu_after":inspect_gpu()?,
                    "scratch_bytes":probe.scratch_bytes(),"tracked_device_bytes":resident,
                    "additional_prepared_bytes":probe.bytes()-probe.scratch_bytes()}),
                )?;
            }
            probe.verify()?;
            let mut result = summarize(&pairs)?;
            result["control_median_ms"] = result
                .as_object_mut()
                .unwrap()
                .remove("native_median_ms")
                .unwrap();
            result["interpretation"]=json!("complete native projection micro screen; not FFN, full generation, measured bandwidth or E2E qualification");
            let gain = result["throughput_gain"].as_f64().unwrap();
            let fraction = if comparison == ProjectionComparison::SplitPlanes {
                if kind == ProjectionKind::Qkv {
                    0.070945564817451
                } else {
                    0.05158930127407999
                }
            } else if comparison == ProjectionComparison::PackedLoads {
                if kind == ProjectionKind::Qkv {
                    0.09239512146033983
                } else {
                    0.06640904254002505
                }
            } else if kind == ProjectionKind::Qkv {
                0.1172
            } else {
                0.0825
            };
            let ceiling = 1.0 / (1.0 - fraction + fraction / (1.0 + gain)) - 1.0;
            let combined = peer
                .as_ref()
                .map(|p| p.combined_gain(layer.source.layer_id, rows, gain))
                .transpose()?;
            write_record(
                &mut output,
                json!({"kind":format!("{prefix}_screen"),"layer_id":layer.source.layer_id,"rows":rows,
                "result":result,"optimistic_profile_weighted_e2e_gain":ceiling,
                "optimistic_combined_e2e_gain":combined,
                "profile_fraction":fraction,
                "runtime_integration_allowed":comparison!=ProjectionComparison::SplitPlanes && rows==1 && result["passes_micro_screen"]==true && combined.unwrap_or(ceiling)>=0.03,
                "qualification_scope":if comparison==ProjectionComparison::SplitPlanes {"independent stage screen only; both projection reports must pass all three one-request cases and clear combined profile scope before actual-512 quality and HTTP qualification"} else {"case-level feasibility only; all three one-request cases, actual-512 quality and measured HTTP gates remain required"}}),
            )?;
            eprintln!(
                "[projection_probe] layer={} rows={rows}: {} complete projection pairs",
                layer.source.layer_id, args.trials
            );
        }
    }
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "projection benchmark leaked allocations"
    );
    write_record(
        &mut output,
        json!({"kind":format!("{prefix}_complete"),"is_complete":true,"device_bytes_before":initial,
        "device_bytes_after":CudaContext::total_device_bytes(),"gpu_after_release":inspect_gpu()?}),
    )
}

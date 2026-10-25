//! Independent native-QKV screen using frozen normalized FFN input surrogates.
mod pipeline;
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
use pipeline::QkvProbe;
use serde_json::json;
use std::{
    fs::{self, OpenOptions},
    time::Instant,
};

pub(super) fn benchmark(args: &Args) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions) && args.rows == [1, 4],
        "QKV screen requires release and --rows 1,4"
    );
    let inputs = FfnInputs::load(args)?;
    inputs.validate_environment()?;
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
        json!({"kind":"qkv_warp_manifest","schema_version":1,
        "revision":command_output("git",&["rev-parse","HEAD"])?,"worktree":command_output("git",&["status","--short"])?,
        "executable_sha256":file_sha256(&std::env::current_exe()?)?,"build_profile":"release",
        "model":args.model,"model_sha256":inputs.artifact.plan.model_sha256,
        "fixture":args.ffn_inputs,"fixture_sha256":inputs.artifact_sha256,"capture_report":args.ffn_capture_report,
        "capture_report_sha256":inputs.report_sha256,"execution_configuration":inputs.manifest["execution_configuration"],
        "prepared_weights":model.q8_execution_stats(),"setup_ms":began.elapsed().as_secs_f64()*1000.0,
        "rows":args.rows,"trials":args.trials,"iterations":args.iterations,"gpu_before":gpu,
        "control":"original raw GGUF Q8_0 split4 fused QKV, 256 threads, shared reduction",
        "candidate":"same raw QKV with original-tree warp finish, 256 threads",
        "input_scope":"frozen actual normalized FFN inputs used as QKV surrogates; not captured attention-norm inputs",
        "rows_scope":"one or four distinct independent single-token requests, one fused launch per row, not batched GEMM",
        "memory":{"original_weights":model.weights_len,"prepared_weights":model.q8_execution_stats().resident_bytes,
            "tracked_device_bytes":CudaContext::total_device_bytes(),"resident_kv":0,"additional_prepared_bytes":0},
        "cuda_flags":["-O3","sm_52","compute_52","no fast math"],
        "nvcc":command_output("nvcc",&["--version"])?,"rustc":command_output("rustc",&["--version"])?}),
    )?;
    for layer in &inputs.artifact.layers {
        for &rows in &args.rows {
            let probe = QkvProbe::new(
                &model,
                layer.source.layer_id,
                rows,
                &layer.normalized_inputs[..rows * inputs.artifact.plan.d_model],
            )?;
            let resident = CudaContext::total_device_bytes();
            write_record(
                &mut output,
                json!({"kind":"qkv_warp_validation","layer_id":layer.source.layer_id,
                "rows":rows,"row_positions":layer.rows[..rows],"checks":probe.validate()?,"sources":probe.sources,
                "scratch_bytes":probe.bytes(),"tracked_device_bytes":resident}),
            )?;
            let control_warm = probe.warm(false, 1000.0)?;
            let candidate_warm = probe.warm(true, 100.0)?;
            write_record(
                &mut output,
                json!({"kind":"qkv_warp_warmup","layer_id":layer.source.layer_id,"rows":rows,
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
                for is_warp in order {
                    times.push(probe.time(is_warp, args.iterations)?);
                }
                let ms = |is_warp| {
                    times.iter().find(|v| v["is_warp"] == is_warp).unwrap()["gpu_ms"]
                        .as_f64()
                        .unwrap()
                };
                pairs.push((ms(false), ms(true)));
                ensure!(
                    CudaContext::total_device_bytes() == resident,
                    "QKV timing changed resident allocations"
                );
                write_record(
                    &mut output,
                    json!({"kind":"qkv_warp_pair","layer_id":layer.source.layer_id,"rows":rows,
                    "trial":trial,"measurements":times,"gpu_before":gpu_before,"gpu_after":inspect_gpu()?,
                    "scratch_bytes":probe.bytes(),"tracked_device_bytes":resident}),
                )?;
            }
            probe.verify()?;
            let mut result = summarize(&pairs)?;
            result["control_median_ms"] = result
                .as_object_mut()
                .unwrap()
                .remove("native_median_ms")
                .unwrap();
            result["interpretation"]=json!("complete fused QKV micro screen; not FFN, full generation, measured bandwidth or E2E qualification");
            let gain = result["throughput_gain"].as_f64().unwrap();
            let ceiling = 1.0 / (1.0 - 0.1172 + 0.1172 / (1.0 + gain)) - 1.0;
            write_record(
                &mut output,
                json!({"kind":"qkv_warp_screen","layer_id":layer.source.layer_id,"rows":rows,
                "result":result,"optimistic_profile_weighted_e2e_gain":ceiling,
                "runtime_integration_allowed":rows==1 && result["passes_micro_screen"]==true && ceiling>=0.03}),
            )?;
            eprintln!(
                "[qkv_probe] layer={} rows={rows}: {} complete QKV pairs",
                layer.source.layer_id, args.trials
            );
        }
    }
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "QKV benchmark leaked allocations"
    );
    write_record(
        &mut output,
        json!({"kind":"qkv_warp_complete","is_complete":true,"device_bytes_before":initial,
        "device_bytes_after":CudaContext::total_device_bytes(),"gpu_after_release":inspect_gpu()?}),
    )
}

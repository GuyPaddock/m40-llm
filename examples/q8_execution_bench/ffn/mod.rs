mod fixture;
mod pipeline;
mod validation;

use super::{command_output, inspect_gpu, write_record, Args};
use anyhow::{ensure, Result};
use fixture::FfnInputs;
use m40_llm::{
    cuda::CudaContext,
    evaluation::identity::file_sha256,
    gguf::load_gguf,
    infer::{execution_weights::Q8ExecutionConfig, LoadedModel},
};
use pipeline::{FfnPipeline, Stage};
use serde_json::json;
use std::{
    fs::{self, OpenOptions},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub(super) fn benchmark(args: &Args) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions) && args.rows == [1, 4],
        "FFN baseline requires release and --rows 1,4"
    );
    let inputs = FfnInputs::load(args)?;
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
        json!({"kind":"ffn_baseline_manifest","schema_version":1,
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
        "rows":args.rows,"trials":args.trials,"iterations":args.iterations,
        "activation_quantization":{"is_enabled":false,"gpu_ms":0.0},
        "control":"native FP32-activation gate/up/SwiGLU; immutable prepared down/vocabulary; no KV allocated or attention executed",
        "timing_scope":"CUDA events and host enqueue+completion per repeated stage; required cross-stream waits included; setup/validation/copies excluded",
        "interpretation":"complete FFN microbenchmark baseline, not token/HTTP throughput, a speedup, verifier cost or measured DRAM traffic"}),
    )?;
    for layer in &inputs.artifact.layers {
        for &rows in &args.rows {
            let pipeline = FfnPipeline::new(&model, layer, rows)?;
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
            for _ in 0..3 {
                pipeline.enqueue(Stage::Complete)?;
            }
            pipeline.drain()?;
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
        json!({"kind":"ffn_baseline_complete","is_complete":true,"cases":6,
        "device_bytes_before":initial_device_bytes,"device_bytes_after":CudaContext::total_device_bytes(),
        "gpu_after_release":inspect_gpu()?,"scope":"Rust-tracked payload cleanup, not driver context RSS"}),
    )
}

#[cfg(test)]
mod tests;

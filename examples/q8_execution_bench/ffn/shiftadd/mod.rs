mod paired_timing;
mod probe;
mod reference;
mod weights;

use super::{fixture::FfnInputs, warmup};
use crate::run::{command_output, inspect_gpu, write_record, Args};
use anyhow::{ensure, Result};
use m40_llm::{
    cuda::CudaContext,
    evaluation::identity::file_sha256,
    gguf::load_gguf,
    infer::{execution_weights::Q8ExecutionConfig, LoadedModel},
};
use probe::{Arm, Probe};
use serde_json::json;
use std::{
    fs::{self, OpenOptions},
    time::Instant,
};

fn validate_args(args: &Args) -> Result<()> {
    ensure!(
        args.rows == [1, 2, 4, 8] && args.trials == 5 && args.iterations == 100,
        "ShiftAdd requires --rows 1,2,4,8 --trials 5 --iterations 100"
    );
    Ok(())
}

pub(super) fn benchmark(args: &Args) -> Result<()> {
    validate_args(args)?;
    ensure!(!cfg!(debug_assertions), "ShiftAdd screen requires release");
    ensure!(
        command_output("git", &["status", "--short"])?.is_empty(),
        "ShiftAdd measurement requires clean source"
    );
    let inputs = FfnInputs::load(args)?;
    inputs.validate_environment()?;
    let gpu = inspect_gpu()?;
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.output)?;
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
        json!({"kind":"shiftadd_manifest","schema_version":1,
        "revision":command_output("git",&["rev-parse","HEAD"])?,"executable_sha256":file_sha256(&std::env::current_exe()?)?,
        "model":args.model,"model_sha256":inputs.artifact.plan.model_sha256,
        "fixture_sha256":inputs.artifact_sha256,"capture_report_sha256":inputs.report_sha256,
        "execution_configuration":inputs.manifest["execution_configuration"],"capture_revision":inputs.manifest["revision"],
        "gpu_before":gpu,"rustc":command_output("rustc",&["--version"])?,"nvcc":command_output("nvcc",&["--version"])?,
        "cuda_flags":["-O3","sm_52","compute_52","no fast math"],"model_setup_ms":began.elapsed().as_secs_f64()*1000.0,
        "model_original_bytes":model.weights_len,"model_prepared_bytes":model.q8_execution_stats().resident_bytes,
        "model_resident_bytes":CudaContext::total_device_bytes(),"rows":args.rows,"iterations":args.iterations,
        "protocol":"docs/experiments/2026-10-31-shiftadd-projection-protocol.md",
        "scope":"source-derived projection experiment; no generated tokens, KV, model quality or whole-model speed",
        "q4_k":"no native backend","fitting":"deterministic weight-only; not released checkpoint reproduction"}),
    )?;
    for layer in &inputs.artifact.layers {
        for (role, source, host_input) in [
            ("gate", &layer.source.gate, &layer.normalized_inputs),
            ("up", &layer.source.up, &layer.normalized_inputs),
            ("down", &layer.source.down, &layer.swiglu_inputs),
        ] {
            let before = CudaContext::total_device_bytes();
            eprintln!(
                "[shiftadd] fitting layer={} role={role}",
                layer.source.layer_id
            );
            let weights = weights::Weights::new(&model, source)?;
            write_record(
                &mut output,
                json!({"kind":"shiftadd_weights","layer_id":layer.source.layer_id,"role":role,"report":weights.report}),
            )?;
            for &rows in &args.rows {
                let weights_resident = CudaContext::total_device_bytes();
                let probe =
                    Probe::new(&model.cuda, &weights, &host_input[..rows * source.shape[0]])?;
                let case = json!({"layer_id":layer.source.layer_id,"role":role,"rows":rows,"shape":source.shape,"stage":"complete_projection"});
                write_record(
                    &mut output,
                    json!({"kind":"shiftadd_validation","case":case,"phase":"before","checks":probe.validate()?,
                    "preparation":probe.report,"row_positions":layer.rows[..rows],"peak_tracked_device_bytes":CudaContext::total_device_bytes()}),
                )?;
                let mut warms = Vec::new();
                for arm in Arm::ALL {
                    let duration = warmup::measure(1000.0, |count| {
                        Ok(probe.time(arm, count)?["gpu_ms"].as_f64().unwrap())
                    })?;
                    warms.push(json!({"arm":arm,"warmup":duration}));
                }
                write_record(
                    &mut output,
                    json!({"kind":"shiftadd_warmup","case":case,"arms":warms,"gpu_after":inspect_gpu()?}),
                )?;
                for control in [Arm::Q4, Arm::Q8, Arm::PreparedQ8] {
                    let mut comparison = case.clone();
                    comparison["control"] = json!(control);
                    paired_timing::compare_samples(
                        &mut output,
                        &comparison,
                        json!([control, Arm::Shiftadd]),
                        |slot| {
                            probe.time(
                                if slot == 0 { control } else { Arm::Shiftadd },
                                args.iterations,
                            )
                        },
                    )?;
                }
                write_record(
                    &mut output,
                    json!({"kind":"shiftadd_validation","case":case,"phase":"after","checks":probe.validate()?}),
                )?;
                drop(probe);
                ensure!(
                    CudaContext::total_device_bytes() == weights_resident,
                    "projection scratch cleanup failed"
                );
                eprintln!(
                    "[shiftadd] layer={} role={role} rows={rows} complete",
                    layer.source.layer_id
                );
            }
            drop(weights);
            ensure!(
                CudaContext::total_device_bytes() == before,
                "projection weights cleanup failed"
            );
            write_record(
                &mut output,
                json!({"kind":"shiftadd_cleanup","layer_id":layer.source.layer_id,"role":role,"tracked_device_bytes":before}),
            )?;
        }
    }
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "model cleanup failed"
    );
    write_record(
        &mut output,
        json!({"kind":"shiftadd_complete","is_complete":true,"device_bytes_before":initial,"device_bytes_after":CudaContext::total_device_bytes(),"gpu_after":inspect_gpu()?}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// The experiment requires frozen captures and rejects ambiguous modes or
    /// timing settings before loading a model or allocating device memory.
    #[test]
    fn test_cli_contract() -> Result<()> {
        let base = [
            "bench",
            "--model",
            "model",
            "--output",
            "out",
            "--ffn-shiftadd",
            "--ffn-inputs",
            "inputs",
            "--ffn-capture-report",
            "report",
            "--rows",
            "1,2,4,8",
            "--trials",
            "5",
            "--iterations",
            "100",
        ];
        let args = Args::try_parse_from(base)?;
        assert!(args.ffn_shiftadd);
        validate_args(&args)?;
        let mut conflict = base.to_vec();
        conflict.push("--ffn-q81");
        assert!(Args::try_parse_from(conflict).is_err());
        let mut wrong = args;
        wrong.iterations = 10;
        assert!(validate_args(&wrong).is_err());
        assert!(Args::try_parse_from([
            "bench",
            "--model",
            "model",
            "--output",
            "out",
            "--ffn-shiftadd"
        ])
        .is_err());
        Ok(())
    }
}

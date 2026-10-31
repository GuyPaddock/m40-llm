mod encoding;
mod pipeline;
mod timing;
mod validation;

use super::{
    fixture::{source_bytes, FfnInputs},
    warmup,
};
use crate::run::{command_output, inspect_gpu, write_record, Args};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, DeviceBuffer},
    evaluation::identity::{file_sha256, sha256},
    gguf::load_gguf,
    infer::{execution_weights::Q8ExecutionConfig, LoadedModel},
};
use pipeline::{Pipeline, Stage, ARMS, CHANNELS};
use serde_json::json;
use std::{
    fs::{self, OpenOptions},
    time::Instant,
};

fn upload(ctx: &CudaContext, bytes: &[u8]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new_tagged(ctx, bytes.len(), "rotated:weight")?;
    unsafe {
        ctx.memcpy_h2d(buffer.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())?;
    }
    Ok(buffer)
}

fn hash_buffer(ctx: &CudaContext, buffer: &DeviceBuffer) -> Result<String> {
    let mut bytes = vec![0; buffer.bytes()];
    unsafe {
        ctx.memcpy_d2h(bytes.as_mut_ptr().cast(), buffer.as_ptr(), bytes.len())?;
    }
    sha256(&bytes)
}

fn validate_distinct_rows(input: &[f32], columns: usize) -> Result<()> {
    ensure!(
        input.len() >= 8 * columns,
        "missing eight rotated input rows"
    );
    for a in 0..8 {
        for b in a + 1..8 {
            ensure!(
                input[a * columns..(a + 1) * columns]
                    .iter()
                    .zip(&input[b * columns..(b + 1) * columns])
                    .any(|(x, y)| x.to_bits() != y.to_bits()),
                "duplicate rotated input rows"
            );
        }
    }
    Ok(())
}

pub(super) fn benchmark(args: &Args) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "rotated preparation requires release"
    );
    ensure!(
        args.rows == [1, 2, 4, 8] && args.trials == 5 && args.iterations == 100,
        "rotated preparation requires rows 1,2,4,8, five pairs and 100 iterations"
    );
    ensure!(
        command_output("git", &["status", "--porcelain"])?.is_empty(),
        "rotated preparation requires clean source"
    );
    let inputs = FfnInputs::load(args)?;
    inputs.validate_environment()?;
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.output)?;
    let gpu = inspect_gpu()?;
    let initial_device_bytes = CudaContext::total_device_bytes();
    let setup = Instant::now();
    let model = LoadedModel::from_gguf_with_q8_execution(
        load_gguf(&args.model)?,
        fs::read(&args.model)?,
        -1,
        Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
    )?;
    inputs.validate_model(&model)?;
    write_record(
        &mut output,
        json!({"kind":"rotated_preparation_manifest","schema_version":1,
        "revision":command_output("git",&["rev-parse","HEAD"])?,"build_profile":"release",
        "executable_sha256":file_sha256(&std::env::current_exe()?)?,"model_sha256":inputs.artifact.plan.model_sha256,
        "fixture_sha256":inputs.artifact_sha256,"capture_report_sha256":inputs.report_sha256,
        "capture_execution":inputs.manifest["execution_configuration"],"model_setup_ms":setup.elapsed().as_secs_f64()*1000.0,
        "gpu_before":gpu,"rows":args.rows,"widths":[2048,11008],"channels":CHANNELS,"arms":ARMS,
        "iterations":args.iterations,"pairs":args.trials,"minimum_gain":0.03,
        "sign_seed":0,"rotation_block":128,"normalization":"after butterfly",
        "consumer":"existing Q8_1 threads128, one row per launch, identical in both arms",
        "original_weights":model.weights_len,"prepared_weights":model.q8_execution_stats(),
        "timing_boundary":"prepare or prepare plus consumer on decode stream; launches included; setup/copies/diagnostic synchronization excluded",
        "rustc":command_output("rustc",&["--version"])?,"nvcc":command_output("nvcc",&["--version"])?,
        "interpretation":"synthetic matched rotated consumer; no compatible rotated model or whole-model quality/performance claim"}),
    )?;
    let resident = CudaContext::total_device_bytes();
    for layer in &inputs.artifact.layers {
        ensure!(
            layer.rows.len() >= 8,
            "missing eight captured row identities"
        );
        for columns in [2048, 11008] {
            let activations = if columns == 2048 {
                &layer.normalized_inputs
            } else {
                &layer.swiglu_inputs
            };
            validate_distinct_rows(activations, columns)?;
            let signs = encoding::generate_signs(columns);
            let sign_hash = sha256(&signs.iter().map(|&s| s as u8).collect::<Vec<_>>())?;
            let prepare = Instant::now();
            let sources = if columns == 2048 {
                [(&layer.source.gate, 0), (&layer.source.up, 0)]
            } else {
                [(&layer.source.down, 0), (&layer.source.down, CHANNELS)]
            };
            let mut raw = Vec::new();
            let mut reports = Vec::new();
            for (source, first_row) in sources {
                let original = source_bytes(&model, source)?;
                ensure!(
                    sha256(original)? == source.source_sha256,
                    "rotated source hash mismatch"
                );
                let row_bytes = columns / 32 * 34;
                let selected = original
                    .get(first_row * row_bytes..(first_row + CHANNELS) * row_bytes)
                    .context("rotated consumer rows outside source")?;
                let (bytes, report) = encoding::encode_rotated_q8(selected, &signs)?;
                reports.push(
                    json!({"source":source,"first_row":first_row,"rows":CHANNELS,"fit":report}),
                );
                raw.push(bytes);
            }
            let raw: [Vec<u8>; 2] = raw
                .try_into()
                .map_err(|_| anyhow::anyhow!("missing rotated weights"))?;
            let prepare_ms = prepare.elapsed().as_secs_f64() * 1000.0;
            let began_upload = Instant::now();
            let weights = [upload(&model.cuda, &raw[0])?, upload(&model.cuda, &raw[1])?];
            let upload_ms = began_upload.elapsed().as_secs_f64() * 1000.0;
            let hashes = [sha256(&raw[0])?, sha256(&raw[1])?];
            for (buffer, expected) in weights.iter().zip(&hashes) {
                ensure!(
                    hash_buffer(&model.cuda, buffer)? == *expected,
                    "rotated upload changed bytes"
                );
            }
            let reference = encoding::rotate_reference(&activations[..8 * columns], &signs);
            write_record(
                &mut output,
                json!({"kind":"rotated_preparation_weights","layer":layer.source.layer_id,
                "columns":columns,"source_and_fit":reports,"sign_sha256":sign_hash,"prepare_ms":prepare_ms,"upload_ms":upload_ms,
                "weight_bytes":weights.iter().map(|b|b.bytes()).sum::<usize>(),
                "residency_policy":"original/prepared model plus both rotated weights and all diagnostic buffers retained in both arms"}),
            )?;
            for &rows in &args.rows {
                let input = &activations[..rows * columns];
                let setup_case = Instant::now();
                let pipeline = Pipeline::new(&model.cuda, &weights, input, &signs)?;
                let setup_ms = setup_case.elapsed().as_secs_f64() * 1000.0;
                ensure!(
                    hash_buffer(&model.cuda, pipeline.producer.get_sign_buffer())? == sign_hash,
                    "sign upload changed bytes"
                );
                let case = json!({"layer":layer.source.layer_id,"rows":rows,"columns":columns,"source_rows":&layer.rows[..rows]});
                let checks = validation::validate(&pipeline, &reference[..rows * columns], &raw)?;
                write_record(
                    &mut output,
                    json!({"kind":"rotated_preparation_correctness","case":case,"checks":checks,
                    "setup_upload_ms":setup_ms,"scratch_bytes":pipeline.scratch_bytes(),"sign_bytes":columns,
                    "tracked_device_bytes":CudaContext::total_device_bytes(),
                    "eliminated_logical_intermediate_bytes":rows*columns*12,
                    "traffic_interpretation":"signed scratch write/read and canonical quantizer read avoided; output writes retained; no DRAM counter or sector claim"}),
                )?;
                for stage in [Stage::Prepare, Stage::Complete] {
                    for arm in ARMS {
                        let warm = warmup::measure_with_budget(1000.0, 256, 4096, |n| {
                            pipeline.time(arm, stage, n)?["gpu_ms"]
                                .as_f64()
                                .context("missing warmup timing")
                        })?;
                        write_record(
                            &mut output,
                            json!({"kind":"rotated_preparation_warmup","case":case,"stage":stage,"arm":arm,"warmup":warm}),
                        )?;
                    }
                    let mut comparison = case.clone();
                    comparison["stage"] = json!(stage);
                    timing::compare_samples(&mut output, &comparison, json!(ARMS), |slot| {
                        pipeline.time(ARMS[slot], stage, args.iterations)
                    })?;
                }
                let checks = validation::validate(&pipeline, &reference[..rows * columns], &raw)?;
                ensure!(
                    pipeline
                        .input
                        .read(&model.cuda)?
                        .iter()
                        .zip(input)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "rotated input changed"
                );
                ensure!(
                    hash_buffer(&model.cuda, pipeline.producer.get_sign_buffer())? == sign_hash,
                    "rotated signs changed"
                );
                write_record(
                    &mut output,
                    json!({"kind":"rotated_preparation_case_complete","case":case,"checks":checks,"ownership_passed":true}),
                )?;
            }
            for (buffer, expected) in weights.iter().zip(&hashes) {
                ensure!(
                    hash_buffer(&model.cuda, buffer)? == *expected,
                    "rotated weights changed"
                );
            }
            drop(weights);
            ensure!(
                CudaContext::total_device_bytes() == resident,
                "rotated owner leak"
            );
            write_record(
                &mut output,
                json!({"kind":"rotated_preparation_width_complete","layer":layer.source.layer_id,"columns":columns,"ownership_passed":true}),
            )?;
        }
    }
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial_device_bytes,
        "rotated model cleanup leak"
    );
    write_record(
        &mut output,
        json!({"kind":"rotated_preparation_complete","is_complete":true,"cases":24,"comparisons":48,"gpu_after":inspect_gpu()?}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Concurrency fixtures require eight genuinely distinct rows, including
    /// the later rows beyond the original four-row capture validator.
    #[test]
    fn test_distinct_rows_rejects_later_duplicates() -> Result<()> {
        let mut values: Vec<f32> = (0..16).map(|v| v as f32).collect();
        validate_distinct_rows(&values, 2)?;
        values[14] = values[0];
        values[15] = values[1];
        assert!(validate_distinct_rows(&values, 2).is_err());
        assert!(validate_distinct_rows(&values[..8], 2).is_err());
        Ok(())
    }
}

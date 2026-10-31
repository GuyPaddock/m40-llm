mod pipeline;

use super::{
    fixture::{source_bytes, FfnInputs},
    paired_timing, validation, warmup,
};
use crate::run::{command_output, inspect_gpu, write_record, Args};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, DeviceBuffer},
    evaluation::{
        identity::{file_sha256, sha256},
        quantized_fixture::{decode_q4_values, requantize_q8_to_q4},
    },
    gguf::load_gguf,
    infer::{execution_weights::Q8ExecutionConfig, LoadedModel},
};
use pipeline::{Arm, Pipeline, Stage};
use serde_json::{json, Value};
use std::{
    fs::{self, OpenOptions},
    time::Instant,
};

fn upload(ctx: &CudaContext, bytes: &[u8]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new_tagged(ctx, bytes.len(), "q4_fusion:weight")?;
    unsafe {
        ctx.memcpy_h2d(buffer.as_mut_ptr(), bytes.as_ptr().cast(), bytes.len())?;
    }
    Ok(buffer)
}

fn hash_weights(ctx: &CudaContext, weights: &[DeviceBuffer; 3]) -> Result<Vec<String>> {
    weights
        .iter()
        .map(|buffer| {
            let mut bytes = vec![0; buffer.bytes()];
            unsafe {
                ctx.memcpy_d2h(bytes.as_mut_ptr().cast(), buffer.as_ptr(), bytes.len())?;
            }
            sha256(&bytes)
        })
        .collect()
}

/// Independent scalar FP64 projections on the already decoded Q4 weights.
fn compute_reference(
    input: &[f32],
    gate: &[f64],
    up: &[f64],
    down: &[f64],
    k: usize,
    h: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut hidden = Vec::new();
    let mut output = Vec::new();
    for row in input.chunks_exact(k) {
        let values: Vec<_> = gate
            .chunks_exact(k)
            .zip(up.chunks_exact(k))
            .map(|(g, u)| {
                let g: f64 = row.iter().zip(g).map(|(&x, &w)| f64::from(x) * w).sum();
                let u: f64 = row.iter().zip(u).map(|(&x, &w)| f64::from(x) * w).sum();
                g / (1.0 + (-g).exp()) * u
            })
            .collect();
        output.extend(
            down.chunks_exact(h)
                .map(|weights| values.iter().zip(weights).map(|(a, b)| a * b).sum::<f64>()),
        );
        hidden.extend(values);
    }
    (hidden, output)
}

fn validate(pipeline: &Pipeline<'_>, hidden: &[f64], expected: &[f64]) -> Result<Value> {
    let mut reports = Vec::new();
    let mut serial_hidden = None;
    let mut batched_output = None;
    for arm in Arm::ALL {
        pipeline.enqueue(arm, Stage::Complete)?;
        let (actual_hidden, actual_output) = pipeline.read()?;
        let hidden_check = validation::compare(&actual_hidden, hidden)?;
        let output_check = validation::compare(&actual_output, expected)?;
        let hidden_bits: Vec<_> = actual_hidden.iter().map(|v| v.to_bits()).collect();
        let output_bits: Vec<_> = actual_output.iter().map(|v| v.to_bits()).collect();
        if arm == Arm::SerialDown {
            serial_hidden = Some(hidden_bits.clone());
        }
        if arm == Arm::SerialGateBatchedDown {
            batched_output = Some(output_bits.clone());
        }
        if arm == Arm::FusedRows {
            ensure!(
                Some(hidden_bits) == serial_hidden,
                "multi-row fusion changed single-row hidden bits"
            );
            ensure!(
                Some(output_bits) == batched_output,
                "same-down fusion changed complete output bits"
            );
        }
        reports.push(json!({"arm":arm,"hidden":hidden_check,"output":output_check}));
    }
    Ok(
        json!({"passed":true,"all_arms":reports,"serial_fused_hidden_bits_equal":true,"same_down_complete_bits_equal":true}),
    )
}

pub(super) fn benchmark(args: &Args) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "fusion experiment requires release"
    );
    ensure!(
        args.rows == [1, 2, 4, 8] && args.trials == 5 && args.iterations == 10,
        "fusion requires rows 1,2,4,8, five pairs and ten iterations"
    );
    ensure!(
        command_output("git", &["status", "--porcelain"])?.is_empty(),
        "fusion requires clean source"
    );
    let inputs = FfnInputs::load(args)?;
    inputs.validate_environment()?;
    ensure!(
        std::env::var("M40LLM_Q8_DECODE_STREAM").as_deref() == Ok("decode"),
        "fusion controls require decode stream"
    );
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.output)?;
    let gpu = inspect_gpu()?;
    let setup = Instant::now();
    let model = LoadedModel::from_gguf_with_q8_execution(
        load_gguf(&args.model)?,
        fs::read(&args.model)?,
        -1,
        Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
    )?;
    inputs.validate_model(&model)?;
    let k = inputs.artifact.plan.d_model;
    let h = inputs.artifact.plan.hidden_dim;
    write_record(
        &mut output,
        json!({"kind":"ffn_fusion_manifest","schema_version":1,
        "revision":command_output("git",&["rev-parse","HEAD"])?,"build_profile":"release",
        "executable_sha256":file_sha256(&std::env::current_exe()?)?,"model_sha256":inputs.artifact.plan.model_sha256,
        "fixture_sha256":inputs.artifact_sha256,"capture_report_sha256":inputs.report_sha256,
        "capture_execution":inputs.manifest["execution_configuration"],"model_setup_ms":setup.elapsed().as_secs_f64()*1000.0,
        "gpu_before":gpu,"rows":args.rows,"k":k,"h":h,"arms":Arm::ALL,
        "original_weights":model.weights_len,"prepared_q8_weights":model.q8_execution_stats(),
        "timing_boundary":"gate/up/SwiGLU or complete FFN, including required launch and stream dependencies; no setup/copy/reference",
        "rustc":command_output("rustc",&["--version"])?,"nvcc":command_output("nvcc",&["--version"])?,
        "interpretation":"matched derived Q4 fixtures; no original-Q8 accuracy or whole-model claim"}),
    )?;
    let resident = CudaContext::total_device_bytes();
    for layer in &inputs.artifact.layers {
        let prepare = Instant::now();
        ensure!(
            layer.rows.len() >= 8 && layer.normalized_inputs.len() >= 8 * k,
            "missing distinct rows"
        );
        for a in 0..8 {
            for b in a + 1..8 {
                ensure!(
                    layer.normalized_inputs[a * k..(a + 1) * k]
                        .iter()
                        .zip(&layer.normalized_inputs[b * k..(b + 1) * k])
                        .any(|(x, y)| x.to_bits() != y.to_bits()),
                    "duplicate fusion inputs"
                );
            }
        }
        let mut raw = Vec::new();
        let mut reports = Vec::new();
        for source in [&layer.source.gate, &layer.source.up, &layer.source.down] {
            let (bytes, report) = requantize_q8_to_q4(source_bytes(&model, source)?)?;
            ensure!(
                report.source_sha256 == source.source_sha256,
                "fusion source changed"
            );
            raw.push(bytes);
            reports.push(report);
        }
        let expected_hashes = raw.iter().map(|b| sha256(b)).collect::<Result<Vec<_>>>()?;
        let weights = [
            upload(&model.cuda, &raw[0])?,
            upload(&model.cuda, &raw[1])?,
            upload(&model.cuda, &raw[2])?,
        ];
        ensure!(
            hash_weights(&model.cuda, &weights)? == expected_hashes,
            "fusion weight upload changed bytes"
        );
        let prepare_ms = prepare.elapsed().as_secs_f64() * 1000.0;
        let gate = decode_q4_values(&raw[0])?;
        let up = decode_q4_values(&raw[1])?;
        let down = decode_q4_values(&raw[2])?;
        let (hidden, expected) =
            compute_reference(&layer.normalized_inputs[..8 * k], &gate, &up, &down, k, h);
        write_record(
            &mut output,
            json!({"kind":"ffn_fusion_layer","layer":layer.source.layer_id,"source":layer.source,
            "encoding":reports,"prepare_upload_ms":prepare_ms,"immutable_sha256":expected_hashes,
            "immutable_bytes":weights.iter().map(|b|b.bytes()).sum::<usize>(),
            "residency_policy":"all three native Q4 weights and all diagnostic buffers retained in every arm; original model and prepared Q8 retained"}),
        )?;
        for &rows in &args.rows {
            let activation = &layer.normalized_inputs[..rows * k];
            let pipeline = Pipeline::new(&model.cuda, &weights, activation, k, h)?;
            let case =
                json!({"layer":layer.source.layer_id,"rows":rows,"source_rows":layer.rows[..rows]});
            write_record(
                &mut output,
                json!({"kind":"ffn_fusion_correctness","case":case,
                "checks":validate(&pipeline,&hidden[..rows*h],&expected[..rows*k])?,
                "scratch_bytes":pipeline.scratch_bytes(),"tracked_device_bytes":CudaContext::total_device_bytes(),
                "native_weight_payload_bytes_per_row":raw.iter().map(Vec::len).sum::<usize>(),
                "eliminated_unfused_gate_up_logical_intermediate_bytes":rows*h*16,
                "traffic_interpretation":"logical gate/up writes and SwiGLU reads avoided; no DRAM counters or cache/sector model"}),
            )?;
            for arm in Arm::ALL {
                let warm = warmup::measure(1000.0, |n| {
                    pipeline.time(arm, Stage::Complete, n)?["gpu_ms"]
                        .as_f64()
                        .context("missing warmup duration")
                })?;
                write_record(
                    &mut output,
                    json!({"kind":"ffn_fusion_warmup","case":case,"arm":arm,"warmup":warm}),
                )?;
            }
            for stage in [Stage::GateUp, Stage::Complete] {
                let controls = if stage == Stage::GateUp {
                    &Arm::ALL[..1]
                } else {
                    &Arm::ALL[..3]
                };
                // Gate-up diagnostics include serial fusion and unfused projections;
                // the two serial controls have identical gate-up work.
                let mut controls = controls.to_vec();
                if stage == Stage::GateUp {
                    controls.push(Arm::UnfusedBatched);
                }
                for control in controls {
                    let mut comparison = case.clone();
                    comparison["stage"] = json!(stage);
                    paired_timing::compare_samples(
                        &mut output,
                        &comparison,
                        json!([control, Arm::FusedRows]),
                        |slot| {
                            pipeline.time(
                                if slot == 0 { control } else { Arm::FusedRows },
                                stage,
                                args.iterations,
                            )
                        },
                    )?;
                }
            }
            validate(&pipeline, &hidden[..rows * h], &expected[..rows * k])?;
            ensure!(
                pipeline
                    .input
                    .read(&model.cuda)?
                    .iter()
                    .zip(activation)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "fusion input changed"
            );
            write_record(
                &mut output,
                json!({"kind":"ffn_fusion_case_complete","case":case,"ownership_passed":true}),
            )?;
        }
        ensure!(
            hash_weights(&model.cuda, &weights)? == expected_hashes,
            "fusion immutable owner changed"
        );
        drop(weights);
        ensure!(
            CudaContext::total_device_bytes() == resident,
            "fusion allocation leak"
        );
        write_record(
            &mut output,
            json!({"kind":"ffn_fusion_layer_complete","layer":layer.source.layer_id,"ownership_passed":true}),
        )?;
    }
    write_record(
        &mut output,
        json!({"kind":"ffn_fusion_complete","is_complete":true,"cases":12,"comparisons":60,"gpu_after":inspect_gpu()?}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distinct rows and signed weights produce the independently calculated
    /// SwiGLU and down outputs without mixing row or channel addressing.
    #[test]
    fn test_reference_preserves_rows_and_channels() {
        let (hidden, output) = compute_reference(
            &[1.0, 0.0, 0.0, 2.0],
            &[1.0, 0.0, 0.0, -1.0],
            &[2.0, 0.0, 0.0, 3.0],
            &[1.0, 2.0, -1.0, 1.0],
            2,
            2,
        );
        let a = 2.0 / (1.0 + (-1.0f64).exp());
        let b = -12.0 / (1.0 + 2.0f64.exp());
        // Closed-form multiplication groups differ from projection then SwiGLU.
        for (actual, expected) in hidden
            .iter()
            .zip([a, 0.0, 0.0, b])
            .chain(output.iter().zip([a, -a, 2.0 * b, b]))
        {
            assert!((actual - expected).abs() < 1e-14);
        }
    }
}

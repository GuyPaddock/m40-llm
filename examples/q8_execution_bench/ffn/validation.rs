use super::{
    fixture::source_bytes,
    pipeline::{FfnKernel, FfnPipeline, Stage},
};
use anyhow::{ensure, Result};
use half::f16;
use m40_llm::{
    evaluation::ffn_capture::{FfnLayerSamples, FfnTensorSource},
    infer::q8_activation::quantize_reference,
};
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Debug, Serialize)]
pub(super) struct ErrorStats {
    pub max_abs_diff: f64,
    pub mean_abs_diff: f64,
    pub bitwise_equal: bool,
    pub compared_values: usize,
}

pub(super) fn compare(actual: &[f32], expected: &[f64]) -> Result<ErrorStats> {
    let stats = measure(actual, expected)?;
    for (&a, &b) in actual.iter().zip(expected) {
        let difference = (f64::from(a) - b).abs();
        ensure!(
            difference < 1e-3 + 2e-5 * b.abs(),
            "FFN numerical drift {difference}, expected {b}"
        );
    }
    Ok(stats)
}

/// Quantization drift against original FP32 input is measured, not hidden by
/// relaxing the independent quantized-input implementation correctness gate.
pub(super) fn measure(actual: &[f32], expected: &[f64]) -> Result<ErrorStats> {
    ensure!(
        !actual.is_empty() && actual.len() == expected.len(),
        "FFN comparison shape mismatch"
    );
    let mut maximum = 0.0f64;
    let mut sum = 0.0;
    let mut bitwise_equal = true;
    for (&a, &b) in actual.iter().zip(expected) {
        ensure!(
            a.is_finite() && b.is_finite(),
            "FFN comparison contains nonfinite values"
        );
        let difference = (f64::from(a) - b).abs();
        maximum = maximum.max(difference);
        sum += difference;
        bitwise_equal &= a.to_bits() == (b as f32).to_bits();
    }
    Ok(ErrorStats {
        max_abs_diff: maximum,
        mean_abs_diff: sum / actual.len() as f64,
        bitwise_equal,
        compared_values: actual.len(),
    })
}

/// Independent scalar dequantization with f64 accumulation. Block scales are
/// applied once per block here; no GPU reduction or prepared layout is reused.
pub(super) fn project(
    input: &[f32],
    bytes: &[u8],
    source: &FfnTensorSource,
    rows: usize,
) -> Result<Vec<f64>> {
    let [k, n] = source.shape;
    ensure!(
        input.len() == rows * k && bytes.len() == source.byte_len()?,
        "FFN CPU reference shape mismatch"
    );
    let mut output = vec![0.0; rows * n];
    for row in 0..rows {
        for column in 0..n {
            let mut sum = 0.0;
            for block in 0..k / 32 {
                let offset = (column * (k / 32) + block) * 34;
                let scale =
                    f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f64();
                let mut partial = 0.0;
                for lane in 0..32 {
                    partial += f64::from(input[row * k + block * 32 + lane])
                        * f64::from(bytes[offset + 2 + lane] as i8);
                }
                sum += partial * scale;
            }
            output[row * n + column] = sum;
        }
    }
    Ok(output)
}

pub(super) fn validate(pipeline: &FfnPipeline<'_>, layer: &FfnLayerSamples) -> Result<Value> {
    let rows = pipeline.rows;
    let input = &layer.normalized_inputs[..rows * pipeline.d_model];
    let captured = &layer.swiglu_inputs[..rows * pipeline.hidden_dim];
    let quantization = if pipeline.kernel.has_quantization() {
        let shape = pipeline
            .quantized
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing quantized scratch"))?
            .shape;
        let blocks = quantize_reference(input, shape)?;
        let reconstructed: Vec<_> = blocks.iter().flat_map(|b| b.reconstruct()).collect();
        let error = measure(
            &reconstructed,
            &input.iter().copied().map(f64::from).collect::<Vec<_>>(),
        )?;
        let bytes: Vec<_> = blocks.iter().flat_map(|b| b.to_bytes()).collect();
        Some((reconstructed, error, bytes))
    } else {
        None
    };
    let reference_input = quantization
        .as_ref()
        .map_or(input, |(reconstructed, _, _)| reconstructed.as_slice());
    let gate_cpu = project(
        reference_input,
        source_bytes(pipeline.model, &layer.source.gate)?,
        &layer.source.gate,
        rows,
    )?;
    let up_cpu = project(
        reference_input,
        source_bytes(pipeline.model, &layer.source.up)?,
        &layer.source.up,
        rows,
    )?;
    let hidden_cpu: Vec<_> = gate_cpu
        .iter()
        .zip(&up_cpu)
        .map(|(&gate, &up)| gate / (1.0 + (-gate).exp()) * up)
        .collect();
    pipeline.enqueue_gate_up_projections()?;
    pipeline.drain()?;
    let gate = compare(&pipeline.gate.read(&pipeline.model.cuda)?, &gate_cpu)?;
    let up = compare(&pipeline.up.read(&pipeline.model.cuda)?, &up_cpu)?;
    let diagnostic_hidden = if pipeline.kernel.has_intermediate_capture() {
        Some(pipeline.hidden.read(&pipeline.model.cuda)?)
    } else {
        None
    };
    pipeline.enqueue(Stage::Complete)?;
    pipeline.enqueue(Stage::IsolatedDown)?;
    pipeline.drain()?;
    let hidden = pipeline.hidden.read(&pipeline.model.cuda)?;
    if let Some(diagnostic) = &diagnostic_hidden {
        ensure!(
            hidden
                .iter()
                .zip(diagnostic)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "FFN intermediate capture changed SwiGLU bits"
        );
    }
    let hidden_error = compare(&hidden, &hidden_cpu)?;
    let compare_capture = if pipeline.kernel.has_quantization() {
        measure
    } else {
        compare
    };
    let capture_error = compare_capture(
        &hidden,
        &captured.iter().copied().map(f64::from).collect::<Vec<_>>(),
    )?;
    ensure!(
        rows != 1
            || pipeline.kernel.has_quantization()
            || pipeline.kernel.has_split_weights()
            || matches!(pipeline.kernel,FfnKernel::Fp32(variant) if variant.threads()==128)
            || capture_error.bitwise_equal,
        "single-row native/256-thread FFN no longer reproduces frozen captured SwiGLU bits"
    );
    let down = source_bytes(pipeline.model, &layer.source.down)?;
    let down_cpu = project(&hidden, down, &layer.source.down, rows)?;
    let isolated_cpu = project(captured, down, &layer.source.down, rows)?;
    let output = compare(&pipeline.output.read(&pipeline.model.cuda)?, &down_cpu)?;
    let isolated = compare(
        &pipeline.isolated_output.read(&pipeline.model.cuda)?,
        &isolated_cpu,
    )?;
    let quantization_report = if let Some((_, error, bytes)) = quantization {
        let scratch = pipeline.quantized.as_ref().unwrap();
        ensure!(
            scratch.read(&pipeline.model.cuda)? == bytes,
            "GPU Q8_1 bytes differ from independent CPU encoding"
        );
        let original_gate = project(
            input,
            source_bytes(pipeline.model, &layer.source.gate)?,
            &layer.source.gate,
            rows,
        )?;
        let original_up = project(
            input,
            source_bytes(pipeline.model, &layer.source.up)?,
            &layer.source.up,
            rows,
        )?;
        let original_hidden: Vec<_> = original_gate
            .iter()
            .zip(&original_up)
            .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        json!({"activation_reconstruction":error,"encoded_bytes_match_cpu":true,"payload_bytes":scratch.shape.bytes(),
            "gate_vs_original_fp32_cpu":measure(&pipeline.gate.read(&pipeline.model.cuda)?,&original_gate)?,
            "up_vs_original_fp32_cpu":measure(&pipeline.up.read(&pipeline.model.cuda)?,&original_up)?,
            "swiglu_vs_original_fp32_cpu":measure(&hidden,&original_hidden)?,
            "scope":"quantization drift is diagnostic; strict correctness remains against quantized-input CPU references"})
    } else {
        Value::Null
    };
    pipeline.verify_guards_and_inputs(layer)?;
    Ok(
        json!({"gate_vs_cpu":gate,"up_vs_cpu":up,"swiglu_vs_cpu":hidden_error,"swiglu_vs_frozen_capture":capture_error,
        "complete_down_vs_cpu_on_actual_swiglu":output,"isolated_down_vs_cpu_on_frozen_swiglu":isolated,
        "guards_and_immutable_inputs":true,"cpu_reference":"scalar Q8_0 dequantization and FP64 accumulation for all output entries; FP64 SiLU/up multiplication",
        "variant":pipeline.kernel,"quantization":quantization_report,
        "intermediate_capture_matches_normal_kernel":diagnostic_hidden.map(|_|true),
        "tolerance":"absolute < 1e-3 + 2e-5 * abs(reference); native/256-thread M=1 captured SwiGLU additionally bit-identical"}),
    )
}

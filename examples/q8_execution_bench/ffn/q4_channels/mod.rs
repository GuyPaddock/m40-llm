mod pipeline;
mod reference;
mod timing;

use super::{
    fixture::{source_bytes, FfnInputs},
    validation, warmup,
};
use crate::run::{command_output, inspect_gpu, write_record, Args};
use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{CudaContext, DeviceQ4SelectedFfn},
    evaluation::{
        identity::{file_sha256, sha256},
        quantized_fixture::{decode_q4_values, requantize_q8_to_q4},
    },
    gguf::load_gguf,
    infer::{execution_weights::Q8ExecutionConfig, q4_channels::Q4ChannelWeights, LoadedModel},
};
use pipeline::{Arm, Pipeline};
use reference::Selection;
use serde_json::{json, Value};
use std::{
    fs::{self, OpenOptions},
    time::Instant,
};

fn immutable_hashes(pipeline: &Pipeline<'_>) -> Result<Vec<String>> {
    pipeline.drain()?;
    pipeline
        .selected
        .immutable_buffers()
        .iter()
        .map(|buffer| {
            let mut bytes = vec![0u8; buffer.bytes()];
            unsafe {
                pipeline
                    .ctx
                    .memcpy_d2h(bytes.as_mut_ptr().cast(), buffer.as_ptr(), bytes.len())?;
            }
            sha256(&bytes)
        })
        .collect()
}

/// Lower-bound unique down-weight sectors in a cold 32-byte-sector model.
/// The union includes all rows and accounts for shared scales and packed codes.
/// It is neither the number of warp requests nor a measured DRAM transaction count.
fn traffic(indices: &[u32], h: usize, k: usize) -> Value {
    let mut channels = vec![false; h];
    let mut groups = vec![false; h / 32];
    for &i in indices {
        channels[i as usize] = true;
        groups[i as usize / 32] = true;
    }
    let mut native = vec![false; (k * h / 32 * 18).div_ceil(32)];
    let mut codes = vec![false; (h * k.div_ceil(2)).div_ceil(32)];
    let mut scales = vec![false; (h / 32 * k * 2).div_ceil(32)];
    for (channel, &selected) in channels.iter().enumerate() {
        if !selected {
            continue;
        }
        for output in 0..k {
            let block = (output * (h / 32) + channel / 32) * 18;
            for address in [block, block + 1, block + 2 + channel % 16] {
                native[address / 32] = true;
            }
            codes[(channel * k.div_ceil(2) + output / 2) / 32] = true;
            scales[((channel / 32 * k + output) * 2) / 32] = true;
        }
    }
    let sectors = |v: &[bool]| v.iter().filter(|&&x| x).count();
    json!({"model":"unique cold 32-byte sectors; aligned allocations; union across rows; no warp/reload/cache model",
        "selected_unique_channels":sectors(&channels),"selected_unique_scale_groups":sectors(&groups),
        "gate_up_unique_logical_bytes":sectors(&channels)*k/32*18*2,
        "native_down_unique_sector_bytes":sectors(&native)*32,
        "packed_down_unique_sector_bytes":(sectors(&codes)+sectors(&scales))*32,
        "packed_down_unique_logical_bytes":sectors(&channels)*k.div_ceil(2)+sectors(&groups)*k*2,
        "index_payload_bytes":indices.len()*4,"measured_dram_bytes":Value::Null,"dynamic_instruction_counters":Value::Null})
}

pub(super) fn benchmark(args: &Args) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "Q4 channel experiment requires release"
    );
    ensure!(
        args.trials == 5 && (args.rows == [1] || args.rows == [1, 2, 4, 8]),
        "Q4 channel screen requires five initial pairs and rows 1 or 1,2,4,8"
    );
    ensure!(
        command_output("git", &["status", "--porcelain"])?.is_empty(),
        "Q4 channel experiment requires a clean source commit"
    );
    let inputs = FfnInputs::load(args)?;
    inputs.validate_environment()?;
    ensure!(
        std::env::var("M40LLM_Q8_DECODE_STREAM").as_deref() == Ok("decode"),
        "Q4 dense references require decode stream routing"
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
        json!({"kind":"q4_channel_manifest","schema_version":1,
        "revision":command_output("git", &["rev-parse","HEAD"])?,"worktree":"","build_profile":"release",
        "executable_sha256":file_sha256(&std::env::current_exe()?)?,"model_sha256":inputs.artifact.plan.model_sha256,
        "fixture_sha256":inputs.artifact_sha256,"capture_report_sha256":inputs.report_sha256,
        "capture_execution":inputs.manifest["execution_configuration"],"model_setup_ms":setup.elapsed().as_secs_f64()*1000.0,
        "gpu_before":gpu,"rows":args.rows,"retained_percent":[10,20,40,60,80,100],
        "k":k,"h":h,"original_weights":model.weights_len,"prepared_q8_weights":model.q8_execution_stats(),
        "correctness_only":args.ffn_q4_channels_correctness_only,
        "selector":"supplied indices; exact Q4 hidden magnitude for importance; all selection and CPU oracle costs excluded",
        "timing_boundary":"complete gate/up, SwiGLU, down, required waits and launches; no allocation/upload/reference/selector",
        "numerical_tolerance":"1e-3 + 2e-5*abs(FP64 reference), every output",
        "rustc":command_output("rustc", &["--version"])?,"nvcc":command_output("nvcc", &["--version"])?,
        "counter_availability":"DRAM and dynamic instruction counters unavailable; sector estimates only"}),
    )?;
    let baseline_bytes = CudaContext::total_device_bytes();
    let mut cases = 0;
    for layer in &inputs.artifact.layers {
        let prepare = Instant::now();
        let mut weights = Vec::new();
        let mut encoding = Vec::new();
        for source in [&layer.source.gate, &layer.source.up, &layer.source.down] {
            let (bytes, report) = requantize_q8_to_q4(source_bytes(&model, source)?)?;
            ensure!(
                report.source_sha256 == source.source_sha256,
                "Q4 source identity changed"
            );
            weights.push(bytes);
            encoding.push(json!({"source":source,"encoding":report}));
        }
        let packed = Q4ChannelWeights::pack(&weights[2], h, k)?;
        ensure!(
            packed.unpack() == weights[2],
            "Q4 repack changed native values"
        );
        let gate = decode_q4_values(&weights[0])?;
        let up = decode_q4_values(&weights[1])?;
        let down = decode_q4_values(&weights[2])?;
        let max_rows = *args.rows.last().unwrap();
        ensure!(
            layer.rows.len() >= max_rows,
            "insufficient distinct captured rows"
        );
        let activations = &layer.normalized_inputs[..max_rows * k];
        for a in 0..max_rows {
            for b in a + 1..max_rows {
                ensure!(
                    activations[a * k..(a + 1) * k]
                        .iter()
                        .zip(&activations[b * k..(b + 1) * k])
                        .any(|(x, y)| x.to_bits() != y.to_bits()),
                    "duplicate captured activation rows"
                );
            }
        }
        let hidden = reference::hidden(activations, &gate, &up, k);
        write_record(
            &mut output,
            json!({"kind":"q4_channel_fixture","layer":layer.source.layer_id,
            "encoding":encoding,"packed_down_bytes":packed.bytes(),"prepare_and_reference_ms":prepare.elapsed().as_secs_f64()*1000.0,
            "activation_rows":&layer.rows[..max_rows]}),
        )?;
        for &rows in &args.rows {
            let row_hidden = &hidden[..rows * h];
            let all = reference::select(row_hidden, h, h, Selection::Contiguous)?;
            let dense_reference = reference::down(row_hidden, &down, &all, h, h);
            for selection in [
                Selection::Contiguous,
                Selection::Scattered,
                Selection::Importance,
            ] {
                for percent in [10, 20, 40, 60, 80, 100] {
                    let count = (h * percent).div_ceil(100);
                    let indices = reference::select(row_hidden, h, count, selection)?;
                    let selected_reference = reference::down(row_hidden, &down, &indices, h, count);
                    let case = json!({"layer":layer.source.layer_id,"rows":rows,"selection":selection,"retained_percent":percent,"retained_count":count,
                        "indices_sha256":sha256(&indices.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>())?});
                    let prepare = Instant::now();
                    let owner = DeviceQ4SelectedFfn::upload(
                        &model.cuda,
                        &weights[0],
                        &weights[1],
                        &packed,
                        &indices,
                        rows,
                    )?;
                    let pipeline =
                        Pipeline::new(&model.cuda, owner, &activations[..rows * k], k, h)?;
                    let upload_and_allocation_ms = prepare.elapsed().as_secs_f64() * 1000.0;
                    let before = immutable_hashes(&pipeline)?;
                    write_record(
                        &mut output,
                        json!({"kind":"q4_channel_case","case":case,"traffic":traffic(&indices,h,k),
                        "upload_and_allocation_ms":upload_and_allocation_ms,
                        "tracked_device_bytes":CudaContext::total_device_bytes(),"shared_scratch_and_guards_bytes":pipeline.scratch_bytes(),
                        "immutable_device_bytes":pipeline.selected.immutable_buffers().iter().map(|v| v.bytes()).sum::<usize>()}),
                    )?;
                    let mut native_selected = Vec::new();
                    let mut arms = vec![Arm::DenseSerial, Arm::NativeSelected, Arm::PackedSelected];
                    if rows > 1 {
                        arms.insert(1, Arm::DenseBatched);
                    }
                    for &arm in &arms {
                        pipeline.enqueue(arm)?;
                        let actual = pipeline.read()?;
                        let reference = if matches!(arm, Arm::DenseSerial | Arm::DenseBatched) {
                            &dense_reference
                        } else {
                            &selected_reference
                        };
                        let measured = validation::measure(&actual, reference)?;
                        let correct = validation::compare(&actual, reference);
                        write_record(
                            &mut output,
                            json!({"kind":"q4_channel_correctness","case":case,"arm":arm,"error":measured,"passed":correct.is_ok(),
                            "failure":correct.as_ref().err().map(ToString::to_string)}),
                        )?;
                        correct?;
                        if matches!(arm, Arm::NativeSelected) {
                            native_selected = actual;
                        } else if matches!(arm, Arm::PackedSelected) {
                            ensure!(
                                actual
                                    .iter()
                                    .zip(&native_selected)
                                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                                "native/repacked Q4 outputs differ"
                            );
                        }
                    }
                    if !args.ffn_q4_channels_correctness_only {
                        for &arm in &arms {
                            let warm = warmup::measure_with_budget(100.0, 16, 256, |n| {
                                Ok(pipeline.time(arm, n)?["gpu_ms"].as_f64().unwrap())
                            })?;
                            write_record(
                                &mut output,
                                json!({"kind":"q4_channel_warmup","case":case,"arm":arm,"warmup":warm}),
                            )?;
                        }
                        for control in [Arm::DenseSerial, Arm::DenseBatched] {
                            if rows == 1 && matches!(control, Arm::DenseBatched) {
                                continue;
                            }
                            for candidate in [Arm::NativeSelected, Arm::PackedSelected] {
                                timing::compare(
                                    &mut output,
                                    &pipeline,
                                    args,
                                    [control, candidate],
                                    &case,
                                )?;
                            }
                        }
                    }
                    pipeline.read()?;
                    ensure!(
                        pipeline
                            .input
                            .read(&model.cuda)?
                            .iter()
                            .zip(&activations[..rows * k])
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "Q4 input changed"
                    );
                    ensure!(
                        immutable_hashes(&pipeline)? == before,
                        "Q4 immutable owner changed"
                    );
                    drop(pipeline);
                    ensure!(
                        CudaContext::total_device_bytes() == baseline_bytes,
                        "Q4 case leaked device memory"
                    );
                    cases += 1;
                    write_record(
                        &mut output,
                        json!({"kind":"q4_channel_case_complete","case":case,"ownership_passed":true}),
                    )?;
                }
            }
        }
    }
    write_record(
        &mut output,
        json!({"kind":"q4_channel_complete","cases":cases,"is_complete":true,"gpu_after":inspect_gpu()?,
        "scope":"operator correctness and optional supplied-index timing; model quality unmeasured"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Neighboring packed channels share scales; duplicate rows do not inflate unique sectors.
    #[test]
    fn test_unique_sector_accounting() {
        let first = traffic(&[0, 1], 64, 32);
        assert_eq!(first["packed_down_unique_logical_bytes"], 96);
        assert_eq!(first["packed_down_unique_sector_bytes"], 96);
        let duplicate = traffic(&[0, 1, 0, 1], 64, 32);
        assert_eq!(
            first["native_down_unique_sector_bytes"],
            duplicate["native_down_unique_sector_bytes"]
        );
        assert_eq!(duplicate["index_payload_bytes"], 16);
        assert_eq!(
            traffic(&(0..64).collect::<Vec<_>>(), 64, 32)["packed_down_unique_sector_bytes"],
            1152
        );
    }
}

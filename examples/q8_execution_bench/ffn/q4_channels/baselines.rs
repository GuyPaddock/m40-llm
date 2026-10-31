use super::super::{
    fixture::{source_bytes, FfnInputs},
    joint::JointFfnWeights,
    packet::PreparedWarpPacket,
    pipeline::{DownProjection, FfnKernel, FfnPipeline, Stage},
    validation, warmup,
};
use super::{
    immutable_hashes,
    pipeline::{Arm, Pipeline},
    reference::{self, Selection},
    timing,
};
use crate::run::{inspect_gpu, write_record, Args};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, DeviceQ4SelectedFfn},
    evaluation::{
        identity::sha256,
        quantized_fixture::{decode_q4_values, requantize_q8_to_q4},
    },
    infer::{q4_channels::Q4ChannelWeights, LoadedModel},
};
use serde_json::json;
use std::fs::File;

struct Q4Case<'a> {
    percent: usize,
    count: usize,
    indices_sha256: String,
    pipeline: Pipeline<'a>,
    expected: Vec<f64>,
    immutable_hashes: Vec<String>,
}

pub(super) fn benchmark(
    args: &Args,
    model: &LoadedModel,
    inputs: &FfnInputs,
    output: &mut File,
) -> Result<()> {
    let k = inputs.artifact.plan.d_model;
    let h = inputs.artifact.plan.hidden_dim;
    let resident_before = CudaContext::total_device_bytes();
    for layer in &inputs.artifact.layers {
        let mut weights = Vec::new();
        let mut encoding = Vec::new();
        for source in [&layer.source.gate, &layer.source.up, &layer.source.down] {
            let (bytes, report) = requantize_q8_to_q4(source_bytes(model, source)?)?;
            ensure!(
                report.source_sha256 == source.source_sha256,
                "baseline Q4 source mismatch"
            );
            weights.push(bytes);
            encoding.push(report);
        }
        let packed = Q4ChannelWeights::pack(&weights[2], h, k)?;
        let activation = &layer.normalized_inputs[..k];
        let hidden = reference::hidden(
            activation,
            &decode_q4_values(&weights[0])?,
            &decode_q4_values(&weights[1])?,
            k,
        );
        let down = decode_q4_values(&weights[2])?;
        let mut cases = Vec::new();
        for percent in [100, 10, 20] {
            let count = (h * percent).div_ceil(100);
            let indices = reference::select(&hidden, h, count, Selection::Importance)?;
            let indices_sha256 = sha256(
                &indices
                    .iter()
                    .flat_map(|i| i.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let owner = DeviceQ4SelectedFfn::upload(
                &model.cuda,
                &weights[0],
                &weights[1],
                &packed,
                &indices,
                1,
            )?;
            let pipeline = Pipeline::new(&model.cuda, owner, activation, k, h)?;
            let hashes = immutable_hashes(&pipeline)?;
            cases.push(Q4Case {
                percent,
                count,
                indices_sha256,
                pipeline,
                expected: reference::down(&hidden, &down, &indices, h, count),
                immutable_hashes: hashes,
            });
        }

        // Preserve the qualified compound/packet implementation and its exact
        // prepared source identities. All diagnostic owners remain resident.
        let mut q8 = FfnPipeline::new(model, layer, 1)?;
        q8.prepare_split_weights(layer)?;
        let split_bytes = q8.prepared.as_ref().context("missing Q8 split pair")?.bytes;
        q8.joint = Some(JointFfnWeights::new(model, layer, split_bytes)?);
        q8.select_kernel(FfnKernel::JointSharedScales)?;
        let joint_bytes = q8.joint.as_ref().context("missing Q8 joint pair")?.bytes;
        let pointer = model
            .map_standard_layer(layer.source.layer_id)?
            .w_down
            .dptr
            .cast_const();
        q8.warp_packet = Some(PreparedWarpPacket::new(
            model,
            pointer,
            128 * 1024 * 1024 - split_bytes - joint_bytes,
        )?);
        q8.select_down_projection(DownProjection::WarpPacket)?;
        let q8_checks = validation::validate(&q8, layer)?;
        let q8_expected: Vec<_> = q8
            .output
            .read(&model.cuda)?
            .into_iter()
            .map(f64::from)
            .collect();
        write_record(
            output,
            json!({"kind":"q4_baselines_layer","layer":layer.source.layer_id,
            "source":layer.source,"encoding":encoding,"row":layer.rows[0],"q8_checks":q8_checks,
            "q8_joint":q8.joint.as_ref().map(|v|&v.report),"q8_packet":q8.warp_packet.as_ref().map(|v|&v.report),
            "residency_policy":"all full/10/20 Q4 owners and Q8 split/joint/packet diagnostic owners retained in every arm",
            "tracked_device_bytes":CudaContext::total_device_bytes(),"q8_scratch_bytes":q8.scratch_bytes()}),
        )?;
        for case in &cases {
            case.pipeline.enqueue(Arm::PackedSelected)?;
            let actual = case.pipeline.read()?;
            let correct = validation::compare(&actual, &case.expected);
            write_record(
                output,
                json!({"kind":"q4_baselines_correctness","layer":layer.source.layer_id,"retained_percent":case.percent,
                "masked_fp64_error":validation::measure(&actual,&case.expected)?,"passed":correct.is_ok(),
                "failure":correct.as_ref().err().map(ToString::to_string),
                "drift_from_q8_full_output":validation::measure(&actual,&q8_expected)?,
                "interpretation":"Q4/pruning drift at one captured FFN input; not model quality or a correctness failure"}),
            )?;
            correct?;
        }
        let q8_warm = q8.warm_up(1000.0, 256)?;
        write_record(
            output,
            json!({"kind":"q4_baselines_warmup","layer":layer.source.layer_id,"arm":"qualified_q8_packet","warmup":q8_warm}),
        )?;
        for case in &cases {
            let warm = warmup::measure(1000.0, |n| {
                case.pipeline.time(Arm::PackedSelected, n)?["gpu_ms"]
                    .as_f64()
                    .context("missing Q4 warmup")
            })?;
            write_record(
                output,
                json!({"kind":"q4_baselines_warmup","layer":layer.source.layer_id,"retained_percent":case.percent,"warmup":warm}),
            )?;
        }
        for case in &cases[1..] {
            let description = json!({"layer":layer.source.layer_id,"rows":1,"selection":"importance",
                "retained_percent":case.percent,"retained_count":case.count,"indices_sha256":case.indices_sha256});
            for control in ["packed_full_q4", "qualified_q8_packet"] {
                timing::compare_samples(
                    output,
                    &description,
                    json!([control, "packed_selected"]),
                    |slot| {
                        let mut sample = if slot == 1 {
                            case.pipeline.time(Arm::PackedSelected, args.iterations)?
                        } else if control == "packed_full_q4" {
                            cases[0]
                                .pipeline
                                .time(Arm::PackedSelected, args.iterations)?
                        } else {
                            q8.time(Stage::Complete, args.iterations)?
                        };
                        sample["arm"] = json!(if slot == 0 {
                            control
                        } else {
                            "packed_selected"
                        });
                        Ok(sample)
                    },
                )?;
            }
        }
        q8.verify_guards_and_inputs(layer)?;
        q8.enqueue(Stage::Complete)?;
        q8.drain()?;
        ensure!(
            validation::measure(&q8.output.read(&model.cuda)?, &q8_expected)?.bitwise_equal,
            "Q8 baseline output changed"
        );
        for case in &cases {
            case.pipeline.enqueue(Arm::PackedSelected)?;
            validation::compare(&case.pipeline.read()?, &case.expected)?;
            ensure!(
                immutable_hashes(&case.pipeline)? == case.immutable_hashes,
                "Q4 baseline owner changed"
            );
            ensure!(
                case.pipeline
                    .input
                    .read(&model.cuda)?
                    .iter()
                    .zip(activation)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "baseline Q4 input changed"
            );
        }
        drop(q8);
        drop(cases);
        ensure!(
            CudaContext::total_device_bytes() == resident_before,
            "stronger baseline allocation leak"
        );
        write_record(
            output,
            json!({"kind":"q4_baselines_layer_complete","layer":layer.source.layer_id,"ownership_passed":true}),
        )?;
    }
    write_record(
        output,
        json!({"kind":"q4_channel_baselines_complete","is_complete":true,"comparisons":12,"gpu_after":inspect_gpu()?,
        "interpretation":"supplied-index timing versus stronger references; predictor/recovery and model quality still unmeasured"}),
    )
}

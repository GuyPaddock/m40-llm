//! Fixed two-arm numerical probe. Never changes generation dispatch.
use super::{
    ablation,
    fixture::{self, FfnInputs},
    joint::JointFfnWeights,
    pipeline::GuardedF32,
    validation,
};
use crate::run::{command_output, inspect_gpu, write_record, Args};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, CudaEvent, CudaStream},
    evaluation::{ffn_capture::FfnLayerSamples, identity::file_sha256},
    gguf::load_gguf,
    infer::{
        execution_weights::{DeviceQ8ExecutionView, Q8ExecutionConfig},
        LoadedModel,
    },
    profile,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    ffi::c_void,
    fs::{self, File, OpenOptions},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const D: usize = 2048;
const H: usize = 11008;
#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Arm {
    Qualified,
    Ilp4,
}
const ARMS: [Arm; 2] = [Arm::Qualified, Arm::Ilp4];
#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Stage {
    GateUp,
    Down,
    Complete,
}

struct Pipeline<'a> {
    model: &'a LoadedModel,
    joint: JointFfnWeights,
    down: *const c_void,
    input: GuardedF32,
    gate: GuardedF32,
    up: GuardedF32,
    hidden: GuardedF32,
    output: GuardedF32,
    frozen_hidden: GuardedF32,
    start: CudaEvent,
    stop: CudaEvent,
}
impl<'a> Pipeline<'a> {
    fn new(model: &'a LoadedModel, layer: &FfnLayerSamples) -> Result<Self> {
        let mapped = model.map_standard_layer(layer.source.layer_id)?;
        ensure!(
            mapped.d_model == D && mapped.hidden_dim == H && !layer.rows.is_empty(),
            "ILP4 requires frozen D2048/H11008"
        );
        Ok(Self {
            model,
            joint: JointFfnWeights::new(model, layer, 0)?,
            down: mapped.w_down.dptr,
            input: GuardedF32::new(&model.cuda, &layer.normalized_inputs[..D], "ilp4:input")?,
            gate: GuardedF32::new(&model.cuda, &vec![0.; H], "ilp4:gate")?,
            up: GuardedF32::new(&model.cuda, &vec![0.; H], "ilp4:up")?,
            hidden: GuardedF32::new(&model.cuda, &vec![0.; H], "ilp4:hidden")?,
            output: GuardedF32::new(&model.cuda, &vec![0.; D], "ilp4:output")?,
            frozen_hidden: GuardedF32::new(
                &model.cuda,
                &layer.swiglu_inputs[..H],
                "ilp4:frozen-hidden",
            )?,
            start: model.cuda.create_event()?,
            stop: model.cuda.create_event()?,
        })
    }
    fn gate(&self, arm: Arm, capture: bool) -> Result<()> {
        let pair = &self.joint.device;
        let intermediates = capture.then_some((self.gate.pointer(), self.up.pointer()));
        unsafe {
            match arm {
                Arm::Qualified => pair.enqueue_shared_scales_async(
                    self.input.pointer(),
                    self.hidden.pointer(),
                    1,
                    intermediates,
                ),
                Arm::Ilp4 => self.model.cuda.ffn_q8_joint_ilp4_f32_async(
                    self.input.pointer(),
                    pair.planes()[0].as_ptr(),
                    pair.planes()[1].as_ptr(),
                    self.hidden.pointer(),
                    1,
                    H as i32,
                    D as i32,
                    intermediates,
                ),
            }
        }
    }
    fn down(&self, input: *const c_void) -> Result<()> {
        unsafe {
            self.model.mlp_down_proj_f32xf16_gguf_f32_async(
                input,
                1,
                H as i32,
                self.down,
                D as i32,
                self.output.pointer(),
            )
        }
    }
    fn enqueue(&self, arm: Arm, stage: Stage, capture: bool) -> Result<()> {
        match stage {
            Stage::GateUp => self.gate(arm, capture),
            Stage::Down => self.down(self.frozen_hidden.pointer()),
            Stage::Complete => {
                self.gate(arm, capture)?;
                self.down(self.hidden.pointer())
            }
        }
    }
    fn results(&self) -> Result<[Vec<f32>; 4]> {
        self.model.cuda.synchronize_stream(CudaStream::Decode)?;
        Ok([
            self.gate.read(&self.model.cuda)?,
            self.up.read(&self.model.cuda)?,
            self.hidden.read(&self.model.cuda)?,
            self.output.read(&self.model.cuda)?,
        ])
    }
    fn verify(&self, layer: &FfnLayerSamples) -> Result<()> {
        self.model.cuda.synchronize_stream(CudaStream::Decode)?;
        self.joint.verify()?;
        require_bits(
            &self.input.read(&self.model.cuda)?,
            &layer.normalized_inputs[..D],
        )?;
        require_bits(
            &self.frozen_hidden.read(&self.model.cuda)?,
            &layer.swiglu_inputs[..H],
        )?;
        self.results()?;
        Ok(())
    }
    fn down_hash(&self) -> Result<String> {
        let weight = self
            .model
            .q8_execution
            .get_separate_weight(self.down)
            .context("missing prepared down")?;
        let DeviceQ8ExecutionView::SeparateScalesValues { scales, values, .. } = weight.view()
        else {
            anyhow::bail!("wrong down layout")
        };
        let mut digest = ring::digest::Context::new(&ring::digest::SHA256);
        for plane in [scales, values] {
            let mut b = vec![0; plane.bytes()];
            unsafe {
                self.model
                    .cuda
                    .memcpy_d2h(b.as_mut_ptr().cast(), plane.as_ptr(), b.len())?;
            }
            digest.update(&b);
        }
        Ok(digest
            .finish()
            .as_ref()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect())
    }
    fn time(&self, arm: Arm, stage: Stage, iterations: usize) -> Result<Value> {
        self.model.cuda.synchronize_stream(CudaStream::Decode)?;
        let before = profile::snapshot();
        let started_unix_us = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let began = Instant::now();
        self.start.record(CudaStream::Decode)?;
        for _ in 0..iterations {
            self.enqueue(arm, stage, false)?;
        }
        self.stop.record(CudaStream::Decode)?;
        let after = profile::snapshot();
        let counters = profile::counter_delta(&before, &after);
        let gpu_ms =
            f64::from(self.start.elapsed_sync(&self.stop, "ilp4_stage")?) / iterations as f64;
        let wall_ms = began.elapsed().as_secs_f64() * 1000.;
        let down_calls = if matches!(stage, Stage::GateUp) {
            0
        } else {
            iterations as u64
        };
        let gate_calls = if matches!(stage, Stage::Down) {
            0
        } else {
            iterations as u64
        };
        ensure!(
            counters
                == profile::OpCounters {
                    launches: down_calls + gate_calls,
                    ..Default::default()
                },
            "unexpected enqueue work: {counters:?}"
        );
        let count = |snapshot: &profile::ProfileSnapshot, name: &str| {
            snapshot.by_op.get(name).map_or(0, |c| c.launches)
        };
        ensure!(
            count(&after, "gemm_f32xq8_separate_f32") - count(&before, "gemm_f32xq8_separate_f32")
                == down_calls,
            "down dispatch changed"
        );
        let gate_name = match arm {
            Arm::Qualified => "ffn_q8_joint_f32x4_shared_scales_128",
            Arm::Ilp4 => "ffn_q8_joint_ilp4_128",
        };
        ensure!(
            count(&after, gate_name) - count(&before, gate_name) == gate_calls,
            "gate dispatch changed"
        );
        Ok(
            json!({"stage":stage,"gpu_ms":gpu_ms,"host_enqueue_and_completion_ms":wall_ms,
            "iterations":iterations,"started_unix_us":started_unix_us,
            "completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros(),
            "enqueue_counters":counters,"gate_operation":gate_name,"down_operation":"gemm_f32xq8_separate_f32"}),
        )
    }
    fn warmup(&self, arm: Arm) -> Result<Value> {
        let mut elapsed = 0.;
        for batch in 1..=256 {
            elapsed += self.time(arm, Stage::Complete, 32)?["gpu_ms"]
                .as_f64()
                .unwrap()
                * 32.;
            if elapsed >= 1000. {
                return Ok(json!({"batches":batch,"gpu_ms":elapsed}));
            }
        }
        anyhow::bail!("fixed warmup exhausted")
    }
}
impl Drop for Pipeline<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.model.cuda.synchronize_stream(CudaStream::Decode) {
            eprintln!("[ilp4] drain failed: {e:#}");
        }
    }
}
fn require_bits(a: &[f32], b: &[f32]) -> Result<()> {
    ensure!(
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| x.is_finite() && x.to_bits() == y.to_bits()),
        "unchanged control/capture bits differ"
    );
    Ok(())
}
fn validate(p: &Pipeline<'_>, layer: &FfnLayerSamples, out: &mut File) -> Result<()> {
    let x = &layer.normalized_inputs[..D];
    let gate = validation::project(
        x,
        fixture::source_bytes(p.model, &layer.source.gate)?,
        &layer.source.gate,
        1,
    )?;
    let up = validation::project(
        x,
        fixture::source_bytes(p.model, &layer.source.up)?,
        &layer.source.up,
        1,
    )?;
    let hidden: Vec<_> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| g / (1. + (-g).exp()) * u)
        .collect();
    let mut reference: Option<[Vec<f32>; 4]> = None;
    let mut isolated: Option<Vec<f32>> = None;
    for arm in ARMS {
        p.enqueue(arm, Stage::Complete, true)?;
        let actual = p.results()?;
        if matches!(arm, Arm::Qualified) {
            require_bits(&actual[2], &layer.swiglu_inputs[..H])?;
        }
        let down = validation::project(
            &actual[2],
            fixture::source_bytes(p.model, &layer.source.down)?,
            &layer.source.down,
            1,
        )?;
        let expected = [&gate, &up, &hidden, &down];
        // Persist drift before enforcing the gate, so a rejection remains diagnostic.
        let stats = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| validation::measure(a, b))
            .collect::<Result<Vec<_>>>()?;
        let drift = reference
            .as_ref()
            .map(|r| {
                actual
                    .iter()
                    .zip(r)
                    .map(|(a, b)| {
                        validation::measure(a, &b.iter().map(|v| f64::from(*v)).collect::<Vec<_>>())
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        write_record(
            out,
            json!({"kind":"ilp4_numerics","layer_id":layer.source.layer_id,"arm":arm,"stage_order":["gate","up","swiglu","down"],"cpu_errors":stats,"versus_control":drift}),
        )?;
        for (a, b) in actual.iter().zip(expected) {
            validation::compare(a, b)?;
        }
        p.enqueue(arm, Stage::Complete, false)?;
        for (a, b) in p.results()?.iter().zip(&actual) {
            require_bits(a, b)?;
        }
        p.enqueue(arm, Stage::Down, false)?;
        let down_only = p.results()?[3].clone();
        if let Some(expected) = &isolated {
            require_bits(&down_only, expected)?;
        } else {
            isolated = Some(down_only);
        }
        p.verify(layer)?;
        write_record(
            out,
            json!({"kind":"ilp4_validation","layer_id":layer.source.layer_id,"arm":arm,
            "is_passed":true,"has_ordinary_capture_parity":true,"has_unchanged_isolated_down":true}),
        )?;
        reference = Some(actual);
    }
    Ok(())
}
fn check_gpu() -> Result<Value> {
    let state = inspect_gpu()?;
    let fields: Vec<_> = state["device"]
        .as_str()
        .context("missing GPU state")?
        .split(',')
        .map(str::trim)
        .collect();
    ensure!(
        fields.len() == 9
            && fields[0] == "GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852"
            && fields[3] == "P0"
            && fields[5].trim_end_matches(" MHz") == "1113"
            && fields[6].trim_end_matches(" MHz") == "3004",
        "GPU/clock condition changed: {state}"
    );
    Ok(state)
}
pub(super) fn benchmark(args: &Args) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions)
            && args.rows == [1]
            && args.iterations == 10
            && matches!(args.trials, 5 | 10 | 20),
        "fixed ILP4 requires release, rows1, iterations10, trials5/10/20"
    );
    ensure!(
        command_output("git", &["status", "--short"])?.is_empty(),
        "probe requires clean committed source"
    );
    let inputs = FfnInputs::load(args)?;
    inputs.validate_environment()?;
    ensure!(
        inputs
            .artifact
            .layers
            .iter()
            .map(|l| l.source.layer_id)
            .collect::<Vec<_>>()
            == [0, 17, 35],
        "frozen layer set changed"
    );
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    let initial = CudaContext::total_device_bytes();
    let model = LoadedModel::from_gguf_with_q8_execution(
        load_gguf(&args.model)?,
        fs::read(&args.model)?,
        -1,
        Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
    )?;
    inputs.validate_model(&model)?;
    let owners = serde_json::to_value(model.q8_execution_stats())?;
    write_record(
        &mut out,
        json!({"kind":"ilp4_manifest","schema_version":1,
        "revision":command_output("git",&["rev-parse","HEAD"])?,"executable_sha256":file_sha256(&std::env::current_exe()?)?,
        "model_sha256":inputs.artifact.plan.model_sha256,"fixture_sha256":inputs.artifact_sha256,"capture_report_sha256":inputs.report_sha256,
        "execution_configuration":inputs.manifest["execution_configuration"],"prepared":owners,"rows":1,"threads":128,"d":D,"h":H,"shared_bytes":1536,
        "iterations":args.iterations,"trials":args.trials,"correctness_only":args.ffn_ilp4_correctness_only,"arms":ARMS,
        "build_profile":"release","nvcc":command_output("nvcc",&["--version"])?,"rustc":command_output("rustc",&["--version"])?,
        "cuda_flags":["-O3","sm_52","compute_52","no fast math"],"gpu":inspect_gpu()?}),
    )?;
    // Validate every layer before collecting any primary timing.
    for layer in &inputs.artifact.layers {
        let before = CudaContext::total_device_bytes();
        {
            let p = Pipeline::new(&model, layer)?;
            let hash = p.down_hash()?;
            write_record(
                &mut out,
                json!({"kind":"ilp4_preparation","layer_id":layer.source.layer_id,"joint":p.joint.report,
                "row_position":layer.rows[0],"down_payload_sha256":hash,"tracked_resident":CudaContext::total_device_bytes()}),
            )?;
            validate(&p, layer, &mut out)?;
            ensure!(p.down_hash()? == hash, "down payload changed");
        }
        ensure!(
            CudaContext::total_device_bytes() == before,
            "validation owners leaked"
        );
    }
    if !args.ffn_ilp4_correctness_only {
        for layer in &inputs.artifact.layers {
            let before = CudaContext::total_device_bytes();
            {
                let p = Pipeline::new(&model, layer)?;
                let hash = p.down_hash()?;
                for arm in ARMS {
                    write_record(
                        &mut out,
                        json!({"kind":"ilp4_warmup","layer_id":layer.source.layer_id,"arm":arm,"warmup":p.warmup(arm)?,"gpu_after":check_gpu()?}),
                    )?;
                }
                let mut pairs = Vec::new();
                for trial in 0..args.trials {
                    let before_gpu = check_gpu()?;
                    let order = if trial % 2 == 0 { [0, 1] } else { [1, 0] };
                    let mut complete = [0.; 2];
                    let mut measures = Vec::new();
                    for index in order {
                        let arm = ARMS[index];
                        let mut stages = [Stage::GateUp, Stage::Down, Stage::Complete];
                        if trial % 2 == 1 {
                            stages.reverse();
                        }
                        let stages = stages
                            .into_iter()
                            .map(|s| p.time(arm, s, args.iterations))
                            .collect::<Result<Vec<_>>>()?;
                        complete[index] = stages.iter().find(|s| s["stage"] == "complete").unwrap()
                            ["gpu_ms"]
                            .as_f64()
                            .unwrap();
                        measures.push(json!({"arm":arm,"stages":stages}));
                    }
                    pairs.push((complete[0], complete[1]));
                    write_record(
                        &mut out,
                        json!({"kind":"ilp4_trial","layer_id":layer.source.layer_id,"trial":trial,"order":order,"measurements":measures,"gpu_before":before_gpu,"gpu_after":check_gpu()?}),
                    )?;
                }
                let summary = ablation::summarize(&pairs)?;
                write_record(
                    &mut out,
                    json!({"kind":"ilp4_summary","layer_id":layer.source.layer_id,"summary":summary}),
                )?;
                p.verify(layer)?;
                ensure!(p.down_hash()? == hash, "down payload changed");
            }
            ensure!(
                CudaContext::total_device_bytes() == before,
                "timing owners leaked"
            );
        }
    }
    inputs.validate_model(&model)?;
    ensure!(
        serde_json::to_value(model.q8_execution_stats())? == owners,
        "prepared owners changed"
    );
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "model teardown leaked"
    );
    write_record(
        &mut out,
        json!({"kind":"ilp4_complete","is_complete":true,"device_bytes_after":CudaContext::total_device_bytes(),"is_runtime_claim":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    /// An explicit probe cannot silently run a different experimental consumer.
    #[test]
    fn test_ilp4_cli_requires_inputs_and_excludes_other_probes() {
        let args = [
            "b",
            "--model",
            "m",
            "--output",
            "o",
            "--ffn-inputs",
            "i",
            "--ffn-capture-report",
            "c",
            "--ffn-ilp4",
        ];
        assert!(Args::try_parse_from(args).is_ok());
        for other in [
            "--ffn-four-row-reuse",
            "--ffn-f32-compound",
            "--ffn-q81",
            "--prepared-warp-packet",
        ] {
            assert!(Args::try_parse_from(args.into_iter().chain([other])).is_err());
        }
        assert!(
            Args::try_parse_from(["b", "--model", "m", "--output", "o", "--ffn-ilp4"]).is_err()
        );
    }
}

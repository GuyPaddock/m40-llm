//! One fixed reader and two contemporaneous controls, all with original down.
use super::{
    ablation, fixture::FfnInputs, joint::JointFfnWeights, pipeline::GuardedF32, validation,
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

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Arm {
    Serial,
    Independent,
    SharedRead,
}
const ARMS: [Arm; 3] = [Arm::Serial, Arm::Independent, Arm::SharedRead];

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Stage {
    GateUp,
    Down,
    Complete,
}

fn arm_order(trial: usize) -> [usize; 3] {
    let mut order = [0, 1, 2];
    order.rotate_left(trial % 3);
    if trial % 2 == 1 {
        order.reverse();
    }
    order
}

fn offset(pointer: *mut c_void, row: usize, columns: usize) -> *mut c_void {
    unsafe { pointer.cast::<f32>().add(row * columns).cast() }
}

struct FourRowPipeline<'a> {
    model: &'a LoadedModel,
    joint: JointFfnWeights,
    down: *const c_void,
    input: GuardedF32,
    gate: GuardedF32,
    up: GuardedF32,
    hidden: GuardedF32,
    output: GuardedF32,
    start: CudaEvent,
    stop: CudaEvent,
}

impl<'a> FourRowPipeline<'a> {
    fn new(model: &'a LoadedModel, layer: &FfnLayerSamples) -> Result<Self> {
        let mapped = model.map_standard_layer(layer.source.layer_id)?;
        ensure!(
            mapped.d_model == D && mapped.hidden_dim == H && layer.rows.len() >= 4,
            "four-row probe requires frozen D2048/H11008 rows"
        );
        Ok(Self {
            model,
            joint: JointFfnWeights::new(model, layer, 0)?,
            down: mapped.w_down.dptr,
            input: GuardedF32::new(&model.cuda, &layer.normalized_inputs[..4 * D], "r4:input")?,
            gate: GuardedF32::new(&model.cuda, &vec![0.; 4 * H], "r4:gate")?,
            up: GuardedF32::new(&model.cuda, &vec![0.; 4 * H], "r4:up")?,
            hidden: GuardedF32::new(&model.cuda, &vec![0.; 4 * H], "r4:hidden")?,
            output: GuardedF32::new(&model.cuda, &vec![0.; 4 * D], "r4:output")?,
            start: model.cuda.create_event()?,
            stop: model.cuda.create_event()?,
        })
    }

    fn enqueue_gate(&self, arm: Arm, capture: bool) -> Result<()> {
        let pair = &self.joint.device;
        unsafe {
            match arm {
                Arm::Serial => {
                    for row in 0..4 {
                        pair.enqueue_shared_scales_async(
                            offset(self.input.pointer(), row, D),
                            offset(self.hidden.pointer(), row, H),
                            1,
                            capture.then_some((
                                offset(self.gate.pointer(), row, H),
                                offset(self.up.pointer(), row, H),
                            )),
                        )?;
                    }
                }
                Arm::Independent => pair.enqueue_shared_scales_async(
                    self.input.pointer(),
                    self.hidden.pointer(),
                    4,
                    capture.then_some((self.gate.pointer(), self.up.pointer())),
                )?,
                Arm::SharedRead => self.model.cuda.ffn_q8_joint_four_rows_f32_async(
                    self.input.pointer(),
                    pair.planes()[0].as_ptr(),
                    pair.planes()[1].as_ptr(),
                    self.hidden.pointer(),
                    4,
                    H as i32,
                    D as i32,
                    capture.then_some((self.gate.pointer(), self.up.pointer())),
                )?,
            }
        }
        Ok(())
    }

    fn enqueue_down(&self) -> Result<()> {
        for row in 0..4 {
            unsafe {
                self.model.mlp_down_proj_f32xf16_gguf_f32_async(
                    offset(self.hidden.pointer(), row, H),
                    1,
                    H as i32,
                    self.down,
                    D as i32,
                    offset(self.output.pointer(), row, D),
                )?;
            }
        }
        Ok(())
    }

    fn enqueue(&self, arm: Arm, stage: Stage, capture: bool) -> Result<()> {
        match stage {
            Stage::GateUp => self.enqueue_gate(arm, capture),
            Stage::Down => self.enqueue_down(),
            Stage::Complete => {
                self.enqueue_gate(arm, capture)?;
                self.enqueue_down()
            }
        }
    }

    fn read_results(&self) -> Result<[Vec<f32>; 4]> {
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
        ensure!(
            self.input
                .read(&self.model.cuda)?
                .iter()
                .zip(&layer.normalized_inputs[..4 * D])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "modified frozen inputs"
        );
        self.read_results()?;
        Ok(())
    }

    fn hash_down_payload(&self) -> Result<String> {
        let weight = self
            .model
            .q8_execution
            .get_separate_weight(self.down)
            .context("missing down owner")?;
        let DeviceQ8ExecutionView::SeparateScalesValues { scales, values, .. } = weight.view()
        else {
            anyhow::bail!("wrong prepared down layout");
        };
        let mut digest = ring::digest::Context::new(&ring::digest::SHA256);
        for plane in [scales, values] {
            let mut bytes = vec![0; plane.bytes()];
            unsafe {
                self.model.cuda.memcpy_d2h(
                    bytes.as_mut_ptr().cast(),
                    plane.as_ptr(),
                    bytes.len(),
                )?;
            }
            digest.update(&bytes);
        }
        Ok(digest
            .finish()
            .as_ref()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect())
    }

    fn scratch_bytes(&self) -> usize {
        [
            &self.input,
            &self.gate,
            &self.up,
            &self.hidden,
            &self.output,
        ]
        .iter()
        .map(|b| b.bytes())
        .sum()
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
        let elapsed = self.start.elapsed_sync(&self.stop, "r4_benchmark_stage")?;
        let wall_ms = began.elapsed().as_secs_f64() * 1000.;
        let gpu_ms = f64::from(elapsed) / iterations as f64;
        let gate_launches = if matches!(stage, Stage::Down) {
            0
        } else if matches!(arm, Arm::Serial) {
            4
        } else {
            1
        };
        let down_launches = if matches!(stage, Stage::GateUp) { 0 } else { 4 };
        ensure!(
            counters
                == profile::OpCounters {
                    launches: (gate_launches + down_launches) * iterations as u64,
                    ..Default::default()
                },
            "unexpected enqueue work: {counters:?}"
        );
        let down_count = |s: &profile::ProfileSnapshot| {
            s.by_op
                .get("gemm_f32xq8_separate_f32")
                .map_or(0, |c| c.launches)
        };
        // The model dispatch reports the original prepared projection name.
        let by_op: Value = serde_json::to_value(
            after
                .by_op
                .iter()
                .filter_map(|(name, c)| {
                    let old = before.by_op.get(name).copied().unwrap_or_default();
                    (c.launches > old.launches).then_some((*name, c.launches - old.launches))
                })
                .collect::<std::collections::BTreeMap<_, _>>(),
        )?;
        ensure!(
            down_count(&after) - down_count(&before) == down_launches * iterations as u64,
            "original prepared-down dispatch changed"
        );
        Ok(
            json!({"stage":stage,"gpu_ms":gpu_ms,"host_enqueue_and_completion_ms":wall_ms,
            "iterations":iterations,"started_unix_us":started_unix_us,
            "completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros(),
            "enqueue_counters":counters,"dispatch":by_op,
            "completion":"one stop-event synchronization after repeated work; outside enqueue counters"}),
        )
    }

    fn warm_up(&self, arm: Arm) -> Result<Value> {
        let mut gpu_ms = 0.;
        for batch in 1..=256 {
            let v = self.time(arm, Stage::Complete, 32)?;
            gpu_ms += v["gpu_ms"].as_f64().unwrap() * 32.;
            if gpu_ms >= 1000. {
                return Ok(json!({"batches":batch,"gpu_ms":gpu_ms,"minimum_ms":1000}));
            }
        }
        anyhow::bail!("fixed warmup exhausted")
    }
}

impl Drop for FourRowPipeline<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.model.cuda.synchronize_stream(CudaStream::Decode) {
            eprintln!("[r4] drain failed: {e:#}");
        }
    }
}

fn require_bits(actual: &[Vec<f32>; 4], expected: &[Vec<f32>; 4]) -> Result<()> {
    ensure!(
        actual.iter().zip(expected).all(|(a, b)| a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits())),
        "four-row gate/up/SwiGLU/down bits changed"
    );
    Ok(())
}

fn validate(p: &FourRowPipeline<'_>, layer: &FfnLayerSamples, out: &mut File) -> Result<()> {
    let gate = validation::project(
        &layer.normalized_inputs[..4 * D],
        super::fixture::source_bytes(p.model, &layer.source.gate)?,
        &layer.source.gate,
        4,
    )?;
    let up = validation::project(
        &layer.normalized_inputs[..4 * D],
        super::fixture::source_bytes(p.model, &layer.source.up)?,
        &layer.source.up,
        4,
    )?;
    let hidden: Vec<f64> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| g / (1. + (-g).exp()) * u)
        .collect();
    let mut reference = None;
    for arm in ARMS {
        p.enqueue(arm, Stage::Complete, true)?;
        let captured = p.read_results()?;
        if let Some(expected) = &reference {
            require_bits(&captured, expected)?;
        }
        let down = validation::project(
            &captured[2],
            super::fixture::source_bytes(p.model, &layer.source.down)?,
            &layer.source.down,
            4,
        )?;
        let cpu = json!({"gate":validation::compare(&captured[0],&gate)?,
            "up":validation::compare(&captured[1],&up)?,"hidden":validation::compare(&captured[2],&hidden)?,
            "down":validation::compare(&captured[3],&down)?});
        p.enqueue(arm, Stage::Complete, false)?;
        require_bits(&p.read_results()?, &captured)?;
        p.verify(layer)?;
        write_record(
            out,
            json!({"kind":"r4_validation","layer_id":layer.source.layer_id,"arm":arm,
            "has_exact_bits":true,"has_ordinary_capture_parity":true,"cpu":cpu}),
        )?;
        reference = Some(captured);
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
            && args.rows == [4]
            && args.iterations == 10
            && matches!(args.trials, 5 | 10 | 20),
        "fixed probe requires release, --rows 4, --iterations 10, trials 5/10/20"
    );
    ensure!(
        command_output("git", &["status", "--short"])?.is_empty(),
        "probe requires clean committed source"
    );
    let inputs = FfnInputs::load(args)?;
    inputs.validate_environment()?;
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
        json!({"kind":"r4_manifest","schema_version":1,
        "revision":command_output("git",&["rev-parse","HEAD"])?,"worktree":"",
        "executable_sha256":file_sha256(&std::env::current_exe()?)?,
        "model_sha256":inputs.artifact.plan.model_sha256,"fixture_sha256":inputs.artifact_sha256,
        "capture_report_sha256":inputs.report_sha256,"execution_configuration":inputs.manifest["execution_configuration"],
        "prepared":owners,"rows":4,"threads":128,"d":D,"h":H,"shared_bytes":4608,
        "iterations":args.iterations,"trials":args.trials,"correctness_only":args.ffn_four_row_correctness_only,
        "arms":ARMS,"source_bytes":model.weights_len,"gpu":inspect_gpu()?,
        "build_profile":"release","nvcc":command_output("nvcc",&["--version"])?,
        "rustc":command_output("rustc",&["--version"])?,"cuda_flags":["-O3","sm_52","compute_52","no fast math"]}),
    )?;
    for layer in &inputs.artifact.layers {
        let tracked_before = CudaContext::total_device_bytes();
        {
            let p = FourRowPipeline::new(&model, layer)?;
            let down_sha256 = p.hash_down_payload()?;
            let down_owner = model
                .q8_execution
                .get_separate_weight(p.down)
                .context("missing prepared down")?;
            ensure!(
                matches!(
                    down_owner.view(),
                    DeviceQ8ExecutionView::SeparateScalesValues { .. }
                ),
                "wrong down layout"
            );
            write_record(
                &mut out,
                json!({"kind":"r4_preparation","layer_id":layer.source.layer_id,
                "joint":p.joint.report,"down_identity":down_owner.key(),"row_positions":layer.rows[..4],
                "down_payload_sha256":down_sha256,"scratch_bytes":p.scratch_bytes(),
                "tracked_before":tracked_before,"tracked_resident":CudaContext::total_device_bytes()}),
            )?;
            validate(&p, layer, &mut out)?;
            if !args.ffn_four_row_correctness_only {
                for arm in ARMS {
                    write_record(
                        &mut out,
                        json!({"kind":"r4_warmup","layer_id":layer.source.layer_id,
                        "arm":arm,"warmup":p.warm_up(arm)?,"gpu_after":check_gpu()?}),
                    )?;
                }
                let mut pairs = Vec::new();
                for trial in 0..args.trials {
                    let before = check_gpu()?;
                    let mut complete = [0.; 3];
                    let mut measures = Vec::new();
                    let order = arm_order(trial);
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
                    pairs.push(complete);
                    write_record(
                        &mut out,
                        json!({"kind":"r4_trial","layer_id":layer.source.layer_id,
                        "trial":trial,"order":order,"measurements":measures,"gpu_before":before,"gpu_after":check_gpu()?,
                        "scratch_bytes":p.scratch_bytes(),"tracked_device_bytes":CudaContext::total_device_bytes()}),
                    )?;
                }
                write_record(
                    &mut out,
                    json!({"kind":"r4_summary","layer_id":layer.source.layer_id,
                    "versus_serial":ablation::summarize(&pairs.iter().map(|p|(p[0],p[2])).collect::<Vec<_>>())?,
                    "versus_independent":ablation::summarize(&pairs.iter().map(|p|(p[1],p[2])).collect::<Vec<_>>())?}),
                )?;
                p.verify(layer)?;
            }
            ensure!(
                p.hash_down_payload()? == down_sha256,
                "prepared down payload changed"
            );
        }
        ensure!(
            CudaContext::total_device_bytes() == tracked_before,
            "four-row owners leaked"
        );
        write_record(
            &mut out,
            json!({"kind":"r4_layer_cleanup","layer_id":layer.source.layer_id,"tracked_after":tracked_before}),
        )?;
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
        json!({"kind":"r4_complete","is_complete":true,
        "device_bytes_after":CudaContext::total_device_bytes(),"is_runtime_claim":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// The research entrypoint cannot select another experiment or omit frozen inputs.
    #[test]
    fn test_reuse_cli_is_exclusive() {
        let args = [
            "bench",
            "--model",
            "m",
            "--output",
            "o",
            "--ffn-inputs",
            "i",
            "--ffn-capture-report",
            "c",
            "--ffn-four-row-reuse",
        ];
        assert!(Args::try_parse_from(args).is_ok());
        for other in [
            "--ffn-f32-compound",
            "--ffn-q81",
            "--direct-q8-feasibility",
            "--ffn-layer-reuse",
        ] {
            assert!(Args::try_parse_from(args.into_iter().chain([other])).is_err());
        }
        assert!(Args::try_parse_from([
            "b",
            "--model",
            "m",
            "--output",
            "o",
            "--ffn-four-row-reuse"
        ])
        .is_err());
    }

    /// Pairing rotates arm order without omitting or duplicating a control.
    #[test]
    fn test_orders_cover_three_arms() {
        for trial in 0..20 {
            let mut order = arm_order(trial);
            order.sort();
            assert_eq!(order, [0, 1, 2]);
        }
        assert_ne!(arm_order(0), arm_order(1));
    }
}

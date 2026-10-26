//! Independent reduction screens in the existing real-model Q8 benchmark.
mod data;
mod probe;
#[cfg(test)]
mod tests;

use super::{ablation::summarize, fixture::FfnInputs};
use crate::run::{command_output, inspect_gpu, write_record, Args};
use anyhow::{ensure, Context, Result};
use data::{hash_f32, make_logits, DISTRIBUTIONS, LOGIT_SEED};
use m40_llm::{
    cuda::CudaContext,
    evaluation::identity::{file_sha256, sha256},
    gguf::{load_gguf, GgmlDType},
    infer::{execution_weights::Q8ExecutionConfig, LoadedModel},
};
use probe::{NormSource, ReductionProbe};
use serde_json::{json, Value};
use std::{
    fs::{self, File, OpenOptions},
    time::Instant,
};

const NORM_ATTN_FRACTION: f64 = 0.01985673198945407;
const NORM_FFN_FRACTION: f64 = 0.020655872424523068;
const ARGMAX_FRACTION: f64 = 0.013236154887019735;

fn run_case(
    args: &Args,
    output: &mut File,
    probe: &ReductionProbe<'_>,
    case: Value,
) -> Result<Value> {
    let resident = CudaContext::total_device_bytes();
    write_record(
        output,
        json!({"kind":"decode_reductions_validation","case":case,"checks":probe.validate()?,"geometry_and_input":probe.describe()?,"tracked_device_bytes":resident}),
    )?;
    let control = probe.warm(false, 1000.0)?;
    let candidate = probe.warm(true, 100.0)?;
    write_record(
        output,
        json!({"kind":"decode_reductions_warmup","case":case,"control":control,"candidate":candidate,"gpu_after":inspect_gpu()?}),
    )?;
    let mut pairs = Vec::new();
    for trial in 0..args.trials {
        let gpu_before = inspect_gpu()?;
        let order = if trial % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        let mut measurements = Vec::new();
        for is_candidate in order {
            measurements.push(probe.time(is_candidate, args.iterations)?);
        }
        let ms = |is_candidate| {
            measurements
                .iter()
                .find(|v| v["is_candidate"] == is_candidate)
                .unwrap()["gpu_ms"]
                .as_f64()
                .unwrap()
        };
        pairs.push((ms(false), ms(true)));
        ensure!(
            CudaContext::total_device_bytes() == resident,
            "reduction timing changed resident bytes"
        );
        write_record(
            output,
            json!({"kind":"decode_reductions_pair","case":case,"trial":trial,"measurements":measurements,"gpu_before":gpu_before,"gpu_after":inspect_gpu()?,"tracked_device_bytes":resident}),
        )?;
    }
    probe.verify()?;
    let mut result = summarize(&pairs)?;
    result["control_median_ms"] = result
        .as_object_mut()
        .unwrap()
        .remove("native_median_ms")
        .unwrap();
    result["interpretation"]=json!("complete reduction operation; both argmax kernels included, not model logits, actual token or HTTP qualification");
    write_record(
        output,
        json!({"kind":"decode_reductions_screen","case":case,"result":result,"runtime_integration_allowed":false}),
    )?;
    eprintln!(
        "[decode_reductions] {case}: {} complete-operation pairs",
        args.trials
    );
    Ok(json!({"case":case,"result":result}))
}

/// Only independently passing stages contribute, using the most conservative
/// one-request surrogate result for each stage. No measured E2E claim follows.
fn summarize_scope(screens: &[Value]) -> Result<Value> {
    let mut saved_fraction = 0.0;
    let mut stages = Vec::new();
    for (stage, fraction, expected_cases) in [
        ("attn_norm", NORM_ATTN_FRACTION, 3),
        ("ffn_norm", NORM_FFN_FRACTION, 3),
        ("argmax", ARGMAX_FRACTION, 3),
    ] {
        let cases: Vec<_> = screens
            .iter()
            .filter(|s| s["case"]["stage"] == stage && s["case"]["rows"] == 1)
            .collect();
        ensure!(
            cases.len() == expected_cases,
            "missing independent one-request reduction cases"
        );
        let passes = cases
            .iter()
            .all(|s| s["result"]["passes_micro_screen"] == true);
        let minimum = cases
            .iter()
            .map(|s| {
                s["result"]["throughput_gain"]
                    .as_f64()
                    .context("missing reduction gain")
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        let saving = if passes {
            fraction * (1.0 - 1.0 / (1.0 + minimum))
        } else {
            0.0
        };
        saved_fraction += saving;
        stages.push(json!({"stage":stage,"profile_fraction":fraction,"passes_all_stage_cases":passes,"minimum_throughput_gain":minimum,"eligible_saved_fraction":saving}));
    }
    let optimistic_gain = 1.0 / (1.0 - saved_fraction) - 1.0;
    Ok(
        json!({"stages":stages,"optimistic_profile_weighted_gain":optimistic_gain,"passes_profile_scope_gate":optimistic_gain>=0.03,
        "runtime_integration_allowed":false,"interpretation":"passing-stage surrogate feasibility only; fresh borderline confirmation, real logits, actual-512 output, frozen tasks and HTTP gates remain required"}),
    )
}

pub(super) fn benchmark(args: &Args) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions) && args.rows == [1, 4],
        "reduction screen requires release and --rows 1,4"
    );
    let inputs = FfnInputs::load(args)?;
    inputs.validate_environment()?;
    ensure!(
        std::env::var("M40LLM_CACHE_EXPERIMENT")
            .unwrap_or_default()
            .is_empty()
            || std::env::var("M40LLM_CACHE_EXPERIMENT").as_deref() == Ok("0"),
        "reduction screen requires cache experiments off"
    );
    let revision = command_output("git", &["rev-parse", "HEAD"])?;
    let worktree = command_output("git", &["status", "--short"])?;
    ensure!(
        worktree.is_empty(),
        "reduction timing requires a committed clean worktree"
    );
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
    ensure!(
        model.model_config.vocab_size == 151936,
        "reduction screen requires original Qwen vocabulary"
    );
    let prepared = serde_json::to_value(model.q8_execution_stats())?;
    let resident = CudaContext::total_device_bytes();
    let logit_inputs=DISTRIBUTIONS.iter().map(|&distribution| {
        let values=make_logits(151936,4,distribution)?;
        Ok(json!({"distribution":distribution,"rows4_sha256":hash_f32(&values)?,"rows1_sha256":hash_f32(&values[..151936])?}))
    }).collect::<Result<Vec<Value>>>()?;
    write_record(
        &mut output,
        json!({"kind":"decode_reductions_manifest","schema_version":1,
        "revision":revision,"worktree":worktree,"build_profile":"release","executable_sha256":file_sha256(&std::env::current_exe()?)?,
        "model":args.model,"model_sha256":inputs.artifact.plan.model_sha256,"fixture":args.ffn_inputs,"fixture_sha256":inputs.artifact_sha256,
        "capture_report":args.ffn_capture_report,"capture_report_sha256":inputs.report_sha256,"execution_configuration":inputs.manifest["execution_configuration"],
        "execution_sha256":inputs.artifact.plan.execution_sha256,"prepared_weights":prepared,"setup_ms":began.elapsed().as_secs_f64()*1000.0,
        "memory":{"original_weights":model.weights_len,"prepared_weights":model.q8_execution_stats().resident_bytes,"resident_kv":0,"tracked_device_bytes":resident},
        "rows":args.rows,"trials":args.trials,"iterations":args.iterations,"gpu_before":gpu,
        "cuda_flags":["-O3","sm_52","compute_52","no fast math"],"nvcc":command_output("nvcc",&["--version"])?,"rustc":command_output("rustc",&["--version"])? ,
        "norm_input_scope":"frozen actual normalized FFN vectors used as norm input surrogates, not captured pre-norm hidden states; actual original model weights and epsilon",
        "argmax_input_scope":"labeled synthetic finite logits at original vocabulary size, not captured model logits","logit_seed":LOGIT_SEED,"logit_inputs":logit_inputs,
        "timing_scope":"same-stream CUDA events over complete operation, including both argmax kernels; setup, upload, readback, guards and pre-drain excluded",
        "profile_source":{"path":"docs/experiments/2026-10-26-maxwell-packed-stage-audit.json","sha256":"3ff95bf79269368145bf3044bb4d351ec826f11444b87cadafbfe7868e29177e","generated_step":255}}),
    )?;
    let mut screens = Vec::new();
    for layer in &inputs.artifact.layers {
        let mapped = model.map_standard_layer(layer.source.layer_id)?;
        for (stage, view) in [
            ("attn_norm", mapped.attn_norm),
            ("ffn_norm", mapped.ffn_norm),
        ] {
            let view = view.context("reduction screen requires actual norm weights")?;
            ensure!(
                view.shape == [2048] && matches!(view.dtype, GgmlDType::F16 | GgmlDType::F32),
                "unsupported original norm weight layout"
            );
            let tensor = model
                .gguf
                .tensors
                .iter()
                .find(|t| {
                    t.offset == view.byte_offset && t.shape == view.shape && t.dtype == view.dtype
                })
                .context("missing original norm tensor identity")?;
            let start = usize::try_from(view.byte_offset)?;
            let end = start
                .checked_add(view.nbytes)
                .context("norm source overflow")?;
            let bytes = model
                .host_weights
                .get(start..end)
                .context("norm source outside upload")?
                .to_vec();
            ensure!(
                bytes.len() == 2048 * if view.dtype == GgmlDType::F16 { 2 } else { 4 },
                "norm source byte length mismatch"
            );
            let identity = json!({"name":tensor.name,"byte_offset":view.byte_offset,"dtype":view.dtype,"shape":view.shape,"sha256":sha256(&bytes)?});
            ensure!(
                !model.d_weights_base.is_null()
                    && !view.dptr.is_null()
                    && end <= model.weights_len
                    && (model.d_weights_base as usize).checked_add(start)
                        == Some(view.dptr as usize),
                "norm weight does not refer to the uploaded original span"
            );
            let pointer = view.dptr.cast_const();
            for &rows in &args.rows {
                let case = json!({"stage":stage,"layer_id":layer.source.layer_id,"rows":rows,"row_positions":layer.rows[..rows],"weight":identity,"eps":model.model_config.layer_norm_epsilon});
                // The model owns this original weight span through probe drop.
                let probe = unsafe {
                    ReductionProbe::new_norm(
                        &model.cuda,
                        &layer.normalized_inputs[..rows * 2048],
                        rows,
                        NormSource {
                            pointer,
                            bytes: bytes.clone(),
                            dtype: if view.dtype == GgmlDType::F16 { 0 } else { 1 },
                            eps: model.model_config.layer_norm_epsilon,
                        },
                    )?
                };
                screens.push(run_case(args, &mut output, &probe, case)?);
            }
        }
    }
    for distribution in DISTRIBUTIONS {
        let values = make_logits(151936, 4, distribution)?;
        for &rows in &args.rows {
            let probe =
                ReductionProbe::new_argmax(&model.cuda, &values[..rows * 151936], rows, 151936)?;
            screens.push(run_case(
                args,
                &mut output,
                &probe,
                json!({"stage":"argmax","distribution":distribution,"rows":rows}),
            )?);
        }
    }
    ensure!(
        screens.len() == 18
            && CudaContext::total_device_bytes() == resident
            && serde_json::to_value(model.q8_execution_stats())? == prepared,
        "reduction screen changed model or leaked case buffers"
    );
    inputs.validate_model(&model)?;
    write_record(
        &mut output,
        json!({"kind":"decode_reductions_scope","result":summarize_scope(&screens)?}),
    )?;
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "reduction benchmark leaked allocations"
    );
    write_record(
        &mut output,
        json!({"kind":"decode_reductions_complete","is_complete":true,"device_bytes_before":initial,"device_bytes_after":CudaContext::total_device_bytes(),"gpu_after_release":inspect_gpu()?}),
    )
}

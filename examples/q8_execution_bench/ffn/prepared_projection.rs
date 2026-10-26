//! Fixed prepared-projection reduction screen, with no runtime policy changes.
use super::{
    ablation::summarize,
    fixture::FfnInputs,
    pipeline::{FfnKernel, FfnPipeline, GuardedF32, Stage},
    validation,
};
use crate::run::{command_output, inspect_gpu, write_record, Args};
use anyhow::{ensure, Context, Result};
use half::f16;
use m40_llm::{
    cuda::{CudaContext, CudaStream},
    evaluation::{ffn_capture::FfnLayerSamples, identity::file_sha256},
    gguf::load_gguf,
    infer::{execution_weights::Q8ExecutionConfig, LoadedModel},
    profile,
};
use serde_json::{json, Value};
use std::{
    fs::{self, File, OpenOptions},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

fn require_identical(actual: &[f32], expected: &[f32]) -> Result<()> {
    ensure!(
        actual.len() == expected.len()
            && actual
                .iter()
                .zip(expected)
                .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits()),
        "prepared warp projection changed control bits"
    );
    Ok(())
}

fn stage_ms(timings: &[Value], name: &str) -> Result<f64> {
    timings
        .iter()
        .find(|t| t["stage"] == name)
        .and_then(|t| t["gpu_ms"].as_f64())
        .context("missing prepared projection timing")
}

fn operation_count(
    before: &profile::ProfileSnapshot,
    after: &profile::ProfileSnapshot,
    op: &str,
) -> u64 {
    after.by_op.get(op).map_or(0, |v| v.launches) - before.by_op.get(op).map_or(0, |v| v.launches)
}

fn projection_operation(is_warp: bool) -> &'static str {
    if is_warp {
        "gemm_f32xq8_separate_warp"
    } else {
        "gemm_f32xq8_separate_f32"
    }
}

fn benchmark_down(
    file: &mut File,
    model: &LoadedModel,
    layer: &FfnLayerSamples,
    rows: usize,
    args: &Args,
) -> Result<()> {
    let mut p = FfnPipeline::new(model, layer, rows)?;
    p.prepare_split_weights(layer)?;
    p.select_kernel(FfnKernel::SplitF32x4)?;
    let down_view = model.map_standard_layer(layer.source.layer_id)?.w_down;
    let down_weight = model
        .q8_execution
        .get_separate_weight(down_view.dptr.cast_const())
        .context("missing qualified prepared down owner")?;
    let original_planes = unsafe { down_weight.read_planes()? };
    let control_checks = validation::validate(&p, layer)?;
    let control_hidden = p.hidden.read(&model.cuda)?;
    let control_output = p.output.read(&model.cuda)?;
    p.enqueue(Stage::IsolatedDown)?;
    p.drain()?;
    let control_down = p.isolated_output.read(&model.cuda)?;
    p.select_warp_down(true)?;
    let candidate_checks = validation::validate(&p, layer)?;
    require_identical(&p.hidden.read(&model.cuda)?, &control_hidden)?;
    require_identical(&p.output.read(&model.cuda)?, &control_output)?;
    p.enqueue(Stage::IsolatedDown)?;
    p.drain()?;
    require_identical(&p.isolated_output.read(&model.cuda)?, &control_down)?;
    write_record(
        file,
        json!({"kind":"prepared_warp_validation","family":"down",
        "layer_id":layer.source.layer_id,"rows":rows,"source":layer.source.down,
        "row_positions":layer.rows[..rows],"control_checks":control_checks,"candidate_checks":candidate_checks,
        "control_hidden_bits_equal":true,"control_complete_output_bits_equal":true,"control_down_bits_equal":true,
        "gate_up_preparation":p.prepared.as_ref().unwrap().report,"scratch_bytes":p.scratch_bytes()}),
    )?;
    p.select_warp_down(false)?;
    let control_warm = p.warm_up(1000.0, 256)?;
    p.select_warp_down(true)?;
    let candidate_warm = p.warm_up(100.0, 256)?;
    write_record(
        file,
        json!({"kind":"prepared_warp_warmup","family":"down",
        "layer_id":layer.source.layer_id,"rows":rows,"control":control_warm,"candidate":candidate_warm}),
    )?;
    let resident = CudaContext::total_device_bytes();
    let mut complete_pairs = Vec::new();
    let mut down_pairs = Vec::new();
    for trial in 0..args.trials {
        let gpu_before = inspect_gpu()?;
        let mut measurements = Vec::new();
        for is_warp in if trial % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            p.select_warp_down(is_warp)?;
            let stages = if trial % 2 == 0 {
                [Stage::IsolatedDown, Stage::Complete]
            } else {
                [Stage::Complete, Stage::IsolatedDown]
            };
            let timings = stages
                .into_iter()
                .map(|stage| -> Result<Value> {
                    let before = profile::snapshot();
                    let mut t = p.time(stage, args.iterations)?;
                    let op = projection_operation(is_warp);
                    let count = operation_count(&before, &profile::snapshot(), op);
                    ensure!(
                        count == args.iterations as u64,
                        "wrong prepared down dispatch"
                    );
                    t["projection_operation"] = json!(op);
                    t["projection_launches"] = json!(count);
                    Ok(t)
                })
                .collect::<Result<Vec<_>>>()?;
            measurements.push(json!({"is_warp":is_warp,"timings":timings}));
        }
        let timing = |warp, stage| -> Result<f64> {
            let t = measurements
                .iter()
                .find(|m| m["is_warp"] == warp)
                .context("missing paired arm")?;
            stage_ms(
                t["timings"].as_array().context("missing stage array")?,
                stage,
            )
        };
        complete_pairs.push((timing(false, "complete")?, timing(true, "complete")?));
        down_pairs.push((
            timing(false, "isolated_down")?,
            timing(true, "isolated_down")?,
        ));
        ensure!(
            CudaContext::total_device_bytes() == resident,
            "prepared down timing allocated weights/scratch"
        );
        write_record(
            file,
            json!({"kind":"prepared_warp_pair","family":"down","layer_id":layer.source.layer_id,
            "rows":rows,"trial":trial,"measurements":measurements,"gpu_before":gpu_before,"gpu_after":inspect_gpu()?,
            "resident_bytes":resident,"scratch_bytes":p.scratch_bytes()}),
        )?;
    }
    p.verify_guards_and_inputs(layer)?;
    ensure!(
        unsafe { down_weight.read_planes()? } == original_planes,
        "prepared down planes changed"
    );
    write_record(
        file,
        json!({"kind":"prepared_warp_screen","family":"down","layer_id":layer.source.layer_id,
        "rows":rows,"complete_ffn":summarize(&complete_pairs)?,"isolated_projection":summarize(&down_pairs)?,
        "gate":"complete FFN >=5% and positive paired interval; isolated down is diagnostic only",
        "runtime_integration_allowed":false}),
    )?;
    eprintln!(
        "[prepared_warp] down layer={} rows={rows}: {} pairs",
        layer.source.layer_id, args.trials
    );
    Ok(())
}

fn benchmark_vocabulary(
    file: &mut File,
    model: &LoadedModel,
    layer: &FfnLayerSamples,
    rows: usize,
    args: &Args,
) -> Result<()> {
    let (name, view, k, n, _) = model.map_lm_head()?;
    let weight = model
        .q8_execution
        .get_separate_weight(view.dptr.cast_const())
        .context("missing qualified vocabulary planes")?;
    let input_values = &layer.normalized_inputs[..rows * k];
    let input = GuardedF32::new(&model.cuda, input_values, "prepared-head:input")?;
    let output = GuardedF32::new(&model.cuda, &vec![0.0; rows * n], "prepared-head:output")?;
    let stream = if rows == 1 {
        CudaStream::Decode
    } else {
        CudaStream::Prefill
    };
    let config = m40_llm::infer::execution_weights::Q8SeparateLaunch::new(4, 128)?;
    let enqueue = |is_warp| -> Result<()> {
        unsafe {
            if is_warp {
                weight.enqueue_separate_warp_async(
                    input.pointer(),
                    output.pointer(),
                    rows,
                    config,
                    stream,
                )
            } else {
                model.matmul_f32xf16_gguf_f32_async(
                    input.pointer(),
                    view.dptr.cast_const(),
                    output.pointer(),
                    rows as i32,
                    n as i32,
                    k as i32,
                )
            }
        }
    };
    enqueue(false)?;
    model.cuda.synchronize_stream(stream)?;
    let expected = output.read(&model.cuda)?;
    enqueue(true)?;
    model.cuda.synchronize_stream(stream)?;
    require_identical(&output.read(&model.cuda)?, &expected)?;
    let source_offset = usize::try_from(weight.key().source.byte_offset())?;
    let blocks = model
        .host_weights
        .get(
            source_offset
                ..source_offset
                    .checked_add(weight.plan().source_bytes())
                    .context("vocabulary span overflow")?,
        )
        .context("vocabulary source span")?;
    let mut cpu_max = 0.0f64;
    for row in 0..rows {
        for col in [0, n / 7, n / 2, n - 1] {
            let cpu: f64 = (0..k)
                .map(|i| {
                    let b = (col * k.div_ceil(32) + i / 32) * 34;
                    let scale =
                        f16::from_bits(u16::from_le_bytes([blocks[b], blocks[b + 1]])).to_f64();
                    f64::from(input_values[row * k + i])
                        * scale
                        * f64::from(blocks[b + 2 + i % 32] as i8)
                })
                .sum();
            let error = (f64::from(expected[row * n + col]) - cpu).abs();
            ensure!(
                error < 1e-3 + cpu.abs() * 2e-5,
                "vocabulary CPU projection drift"
            );
            cpu_max = cpu_max.max(error);
        }
    }
    write_record(
        file,
        json!({"kind":"prepared_warp_validation","family":"vocabulary","rows":rows,
        "tensor":name,"key":weight.key(),"control_bits_equal":true,"cpu_sample_max_abs_diff":cpu_max,
        "input_scope":"distinct frozen normalized FFN layer-0 input surrogates, not captured output-norm hidden vectors",
        "scratch_bytes":input.bytes()+output.bytes()}),
    )?;
    let original_planes = unsafe { weight.read_planes()? };
    let start = model.cuda.create_event()?;
    let stop = model.cuda.create_event()?;
    let time = |is_warp, iterations| -> Result<Value> {
        model.cuda.synchronize_stream(stream)?;
        let before = profile::snapshot();
        let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros();
        let began = Instant::now();
        start.record(stream)?;
        for _ in 0..iterations {
            enqueue(is_warp)?;
        }
        stop.record(stream)?;
        let after = profile::snapshot();
        let counters = profile::counter_delta(&before, &after);
        let op = projection_operation(is_warp);
        let launches = operation_count(&before, &after, op);
        ensure!(
            counters.device_allocs == 0
                && counters.device_frees == 0
                && counters.h2d_copies == 0
                && counters.d2h_copies == 0
                && counters.stream_syncs == 0
                && counters.cublas_calls == 0
                && counters.launches == iterations as u64
                && launches == iterations as u64,
            "vocabulary timed enqueue allocated, copied or synchronized"
        );
        let gpu_ms =
            f64::from(start.elapsed_sync(&stop, "prepared-head:completion")?) / iterations as f64;
        Ok(
            json!({"is_warp":is_warp,"iterations":iterations,"gpu_ms":gpu_ms,
            "host_enqueue_and_completion_ms":began.elapsed().as_secs_f64()*1000.0/iterations as f64,
            "started_unix_us":started,"completed_unix_us":SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros(),
            "enqueue_counters":counters,"projection_operation":op,"projection_launches":launches}),
        )
    };
    for is_warp in [false, true] {
        let bound = if is_warp { 100.0 } else { 1000.0 };
        let mut total = 0.0;
        for batch in 0..256 {
            let t = time(is_warp, 32)?;
            total += t["gpu_ms"].as_f64().context("warmup latency")? * 32.0;
            if total >= bound {
                break;
            }
            ensure!(batch < 255, "vocabulary warmup exhausted bound");
        }
        write_record(
            file,
            json!({"kind":"prepared_warp_warmup","family":"vocabulary","rows":rows,"is_warp":is_warp,"elapsed_gpu_ms":total}),
        )?;
    }
    let resident = CudaContext::total_device_bytes();
    let mut pairs = Vec::new();
    for trial in 0..args.trials {
        let gpu_before = inspect_gpu()?;
        let mut timings = Vec::new();
        for is_warp in if trial % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            timings.push(time(is_warp, args.iterations)?);
        }
        let ms = |warp| {
            timings.iter().find(|t| t["is_warp"] == warp).unwrap()["gpu_ms"]
                .as_f64()
                .unwrap()
        };
        pairs.push((ms(false), ms(true)));
        ensure!(
            CudaContext::total_device_bytes() == resident,
            "vocabulary timing allocated"
        );
        write_record(
            file,
            json!({"kind":"prepared_warp_pair","family":"vocabulary","rows":rows,"trial":trial,
            "measurements":timings,"gpu_before":gpu_before,"gpu_after":inspect_gpu()?,"resident_bytes":resident}),
        )?;
    }
    require_identical(&output.read(&model.cuda)?, &expected)?;
    require_identical(&input.read(&model.cuda)?, input_values)?;
    ensure!(
        unsafe { weight.read_planes()? } == original_planes,
        "vocabulary planes changed"
    );
    let mut result = summarize(&pairs)?;
    result["interpretation"] =
        json!("complete vocabulary projection micro screen; not FFN or E2E throughput");
    write_record(
        file,
        json!({"kind":"prepared_warp_screen","family":"vocabulary","rows":rows,
        "result":result,"runtime_integration_allowed":false}),
    )?;
    eprintln!(
        "[prepared_warp] vocabulary rows={rows}: {} pairs",
        args.trials
    );
    Ok(())
}

pub(super) fn benchmark(args: &Args) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions) && args.rows == [1, 4],
        "prepared warp screen requires release and --rows 1,4"
    );
    let inputs = FfnInputs::load(args)?;
    inputs.validate_environment()?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.output)?;
    let gpu = inspect_gpu()?;
    let initial = CudaContext::total_device_bytes();
    let model = LoadedModel::from_gguf_with_q8_execution(
        load_gguf(&args.model)?,
        fs::read(&args.model)?,
        -1,
        Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
    )?;
    inputs.validate_model(&model)?;
    let stats = model.q8_execution_stats();
    write_record(
        &mut file,
        json!({"kind":"prepared_warp_manifest","schema_version":1,
        "revision":command_output("git",&["rev-parse","HEAD"])?,"worktree":command_output("git",&["status","--short"])?,
        "executable_sha256":file_sha256(&std::env::current_exe()?)?,"build_profile":"release",
        "model_sha256":inputs.artifact.plan.model_sha256,"fixture_sha256":inputs.artifact_sha256,
        "capture_report_sha256":inputs.report_sha256,"execution_configuration":inputs.manifest["execution_configuration"],
        "cuda_flags":["-O3","sm_52","compute_52","no fast math"],"nvcc":command_output("nvcc",&["--version"])?,
        "rows":args.rows,"trials":args.trials,"iterations":args.iterations,"prepared_weights":stats,"gpu_before":gpu,
        "control":"qualified split-f32x4-128 gate/up; original prepared four-column down/vocabulary shared tree",
        "candidate":"same prepared planes and FP32 task order; original-tree warp finish, down M1=256 else128; vocabulary128",
        "memory_scope":"source plus full qualified down/vocabulary cache; one split gate/up pair per FFN case; both arms share all allocations; no KV",
        "runtime_defaults_changed":false,"is_e2e_claim":false}),
    )?;
    for layer in &inputs.artifact.layers {
        for rows in [1, 4] {
            benchmark_down(&mut file, &model, layer, rows, args)?;
        }
    }
    for rows in [1, 4] {
        benchmark_vocabulary(&mut file, &model, &inputs.artifact.layers[0], rows, args)?;
    }
    ensure!(
        serde_json::to_value(model.q8_execution_stats())? == serde_json::to_value(stats)?,
        "prepared cache identities changed"
    );
    drop(model);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "prepared warp screen leaked device payloads"
    );
    write_record(
        &mut file,
        json!({"kind":"prepared_warp_complete","is_complete":true,"cases":8,
        "device_bytes_before":initial,"device_bytes_after":CudaContext::total_device_bytes(),"gpu_after_release":inspect_gpu()?,
        "runtime_defaults_changed":false,"is_e2e_claim":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bit gate must reject changed values, signed zero, nonfinite output
    /// and incomplete vectors instead of accepting an approximate match.
    #[test]
    fn test_prepared_projection_requires_complete_finite_control_bits() {
        assert!(require_identical(&[0.0, 1.25], &[0.0, 1.25]).is_ok());
        for (actual, expected) in [
            (vec![0.0], vec![-0.0]),
            (vec![1.0], vec![1.000001]),
            (vec![f32::NAN], vec![f32::NAN]),
            (vec![f32::INFINITY], vec![f32::INFINITY]),
            (vec![1.0], vec![1.0, 2.0]),
        ] {
            assert!(require_identical(&actual, &expected).is_err());
        }
    }
}

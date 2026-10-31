use super::{
    append,
    capture::{self, Capture},
    fixture::{Fixture, DIMENSION},
    pipeline::{Arm, Case},
    Args,
};
use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{CudaContext, CudaStream},
    evaluation::identity::file_sha256,
    profile,
};
use serde_json::{json, Value};
use std::{fs::OpenOptions, io::Write, process::Command, time::Instant};

#[allow(dead_code)]
#[path = "../compare_ollama/protocol.rs"]
mod protocol;

const M40_UUID: &str = "GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852";

pub struct Report(std::fs::File);
impl Report {
    pub fn record(&mut self, value: Value) -> Result<()> {
        serde_json::to_writer(&mut self.0, &value)?;
        writeln!(self.0)?;
        self.0.flush()?;
        Ok(())
    }
}

pub struct Measurement {
    pub wall_ms: f64,
    pub cuda_timeline_ms: f64,
    pub detail: Value,
}

fn command(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    ensure!(
        output.status.success(),
        "{program} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

pub fn validate_environment() -> Result<()> {
    for (key, value) in std::env::vars().filter(|(key, _)| key.starts_with("M40LLM_")) {
        ensure!(matches!(key.as_str(), "M40LLM_SYSROOT"|"M40LLM_ENABLE_CUBLAS"|"M40LLM_ENABLE_NVCC")
            || (key == "M40LLM_FORCE_M40" && value == "1"), "unset {key}: this study freezes explicit attention dispatch and disables logging overrides");
    }
    Ok(())
}

pub fn check_gpu_processes() -> Result<()> {
    let processes = command(
        "nvidia-smi",
        &[
            "-i",
            M40_UUID,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
    )?;
    for line in processes.lines().filter(|v| !v.trim().is_empty()) {
        ensure!(
            line.trim().parse::<u32>()? == std::process::id(),
            "another compute process owns the M40: {line}"
        );
    }
    Ok(())
}

fn gpu_state() -> Result<String> {
    command("nvidia-smi", &["-i", M40_UUID, "--query-gpu=uuid,name,driver_version,pstate,temperature.gpu,clocks.current.graphics,clocks.current.memory,memory.used,power.draw", "--format=csv,noheader,nounits"])
}

pub fn run(args: &Args) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "real-input study requires --release"
    );
    ensure!(
        (1..=100).contains(&args.iterations),
        "iterations must be 1..100"
    );
    ensure!(
        command("git", &["status", "--porcelain"])?.is_empty(),
        "commit the source before measuring real fixtures"
    );
    validate_environment()?;
    let mut capture = Capture::open(&args.capture_report, &args.kv_inputs)?;
    check_gpu_processes()?;
    let ctx = CudaContext::new_m40()?;
    let props = ctx.current_device_props()?;
    ensure!(props.major == 5 && props.minor == 2, "requires sm_52");
    let baseline = CudaContext::total_device_bytes();
    let mut report = Report(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args.output)?,
    );
    report.record(json!({"kind":"lossless_attention_manifest","schema_version":1,
        "revision":command("git", &["rev-parse","HEAD"])?,"executable_sha256":file_sha256(&std::env::current_exe()?)?,
        "capture_report_sha256":file_sha256(&args.capture_report)?,"kv_artifact_sha256":capture.artifact_sha256(),
        "capture":capture.manifest(),"gpu_state":gpu_state()?,"device_name":props.name,"device_id":props.device_id,
        "is_correctness_only":args.correctness_only,"iterations":args.iterations,"generated_token_count":0,
        "query_source":"deterministic synthetic FP32 vectors; not captured model queries",
        "prototype":"head128, eight contiguous partitions, warp per head/partition, full-context GQA",
        "native":"existing dense head128, eight partitions, 256 threads, serial row dispatch",
        "append":"CPU raw-history mirror plus new GPU token readback, full re-encode and replacement upload",
        "model_weights_resident":false,"is_model_quality_measured":false,"measured_dram_bytes":Value::Null}))?;
    let mut cases = 0;
    for layer in capture::LAYERS {
        for length in capture::CONTEXTS {
            check_gpu_processes()?;
            let sample = capture.read_layer(layer, length, DIMENSION)?;
            let key = json!({"layer":layer,"length":length,"head_dim":DIMENSION,"input_sha256":sample.sha256()?});
            let fixture = Fixture::from_sample(&sample, length)?;
            let case = Case::new(&ctx, fixture)?;
            report.record(
                json!({"kind":"lossless_attention_setup","case":key,"setup":case.setup,
                "retained_fp32_capture_view_bytes":sample.bytes()}),
            )?;
            let began = Instant::now();
            let validation = case.validate();
            if let Err(error) = &validation {
                report.record(json!({"kind":"lossless_attention_validation_failure","case":key,"error":format!("{error:#}"),"timing_admitted":false}))?;
            }
            report.record(json!({"kind":"lossless_attention_validation","case":key,
                "validation_wall_ms":began.elapsed().as_secs_f64()*1000.0,"result":validation?}))?;
            for batch in [1, 2, 4, 8] {
                let batch_key = json!({"fixture":key,"batch":batch});
                // Append is validated even in correctness-only mode. This is
                // an untimed admission, not a paired performance observation.
                append::validate(&case, batch)?;
                report.record(
                    json!({"kind":"lossless_append_validation","case":batch_key,"is_valid":true}),
                )?;
                if args.correctness_only {
                    continue;
                }
                report.record(json!({"kind":"lossless_attention_traffic","case":batch_key,"traffic":logical_traffic(&case,batch)?}))?;
                for (control, candidate, name) in [
                    (Arm::Dense, Arm::Staged, "dense_vs_staged"),
                    (Arm::Staged, Arm::Fused, "staged_vs_fused"),
                    (Arm::Dense, Arm::Fused, "dense_vs_fused"),
                    (Arm::Native, Arm::Fused, "native_serial_vs_fused_batched"),
                    (Arm::SerialFused, Arm::Fused, "fused_serial_vs_batched"),
                ] {
                    let expected = [
                        case.read_arm(control, batch)?,
                        case.read_arm(candidate, batch)?,
                    ];
                    compare(&batch_key, name, &mut report, |is_candidate| {
                        let index = usize::from(is_candidate);
                        let arm = [control, candidate][index];
                        let result = measure_attention(&case, arm, batch, args.iterations)?;
                        ensure!(
                            case.read_output(batch)? == expected[index],
                            "timed output changed for {arm:?}"
                        );
                        Ok(result)
                    })?;
                    case.verify_storage()?;
                }
                append::compare(&case, batch, &batch_key, &mut report)?;
            }
            drop(case);
            ensure!(
                CudaContext::total_device_bytes() == baseline,
                "case leaked tracked device storage"
            );
            report.record(json!({"kind":"lossless_attention_case_complete","case":key,
                "tracked_device_bytes_before":baseline,"tracked_device_bytes_after":CudaContext::total_device_bytes(),"gpu_state":gpu_state()?}))?;
            cases += 1;
        }
    }
    report.record(json!({"kind":"lossless_attention_complete","is_complete":true,"cases":cases,
        "generated_token_count":0,"stop_reason":"fixed_context_grid_complete","is_model_speedup_measured":false}))
}

fn measure_attention(
    case: &Case,
    arm: Arm,
    batch: usize,
    iterations: usize,
) -> Result<Measurement> {
    case.ctx.synchronize_stream(CudaStream::Decode)?;
    let before = profile::snapshot();
    let live = CudaContext::total_device_bytes();
    let began = Instant::now();
    case.start.record(CudaStream::Decode)?;
    for _ in 0..iterations {
        case.enqueue(arm, batch)?;
    }
    case.stop.record(CudaStream::Decode)?;
    let cuda_timeline_ms = f64::from(
        case.start
            .elapsed_sync(&case.stop, "lossless_attention_timing")?,
    ) / iterations as f64;
    let wall_ms = began.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    let counters = profile::counter_delta(&before, &profile::snapshot());
    ensure!(
        CudaContext::total_device_bytes() == live
            && counters.device_allocs == 0
            && counters.device_frees == 0
            && counters.h2d_copies == 0
            && counters.d2h_copies == 0,
        "attention interval allocated/freed or copied host memory"
    );
    Ok(Measurement {
        wall_ms,
        cuda_timeline_ms,
        detail: json!({"arm":arm,"iterations":iterations,"wrapper_counters":counters,
        "boundary":"complete B-row attention, staged decode if selected, partition and merge; wall includes amortized event submission/completion"}),
    })
}

pub fn compare(
    key: &Value,
    name: &str,
    report: &mut Report,
    mut measure: impl FnMut(bool) -> Result<Measurement>,
) -> Result<()> {
    check_gpu_processes()?;
    for _ in 0..3 {
        measure(false)?;
        measure(true)?;
    }
    let mut pairs = Vec::new();
    let mut gpu_pairs = Vec::new();
    let mut target = 5;
    let mut is_confirmation = false;
    loop {
        for index in pairs.len()..target {
            let order = if index % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            let mut observations = [None, None];
            for is_candidate in order {
                let observation = measure(is_candidate)?;
                ensure!(
                    observation.wall_ms.is_finite()
                        && observation.wall_ms > 0.0
                        && observation.cuda_timeline_ms.is_finite()
                        && observation.cuda_timeline_ms > 0.0,
                    "empty timing observation"
                );
                observations[usize::from(is_candidate)] = Some(observation);
            }
            let [Some(control), Some(candidate)] = observations else {
                unreachable!()
            };
            pairs.push(protocol::Pair {
                control_tps: 1000.0 / control.wall_ms,
                candidate_tps: 1000.0 / candidate.wall_ms,
            });
            gpu_pairs.push(protocol::Pair {
                control_tps: 1000.0 / control.cuda_timeline_ms,
                candidate_tps: 1000.0 / candidate.cuda_timeline_ms,
            });
            report.record(json!({"kind":"lossless_pair","case":key,"comparison":name,"pair":index,"is_confirmation":is_confirmation,"candidate_order":order,
                "wall_ms_per_operation":[control.wall_ms,candidate.wall_ms],"cuda_timeline_ms_per_operation":[control.cuda_timeline_ms,candidate.cuda_timeline_ms],
                "detail":[control.detail,candidate.detail]}))?;
        }
        let comparison = protocol::compare_pairs(&pairs)?;
        let next = protocol::next_trial_count(&comparison, is_confirmation);
        let mut summary = serde_json::to_value(&comparison)?;
        let object = summary.as_object_mut().unwrap();
        object.remove("reaches_original_target");
        let qualifies = object.remove("passes_promotion_gate").unwrap();
        object.insert("passes_operator_speed_gate".into(), qualifies);
        let gpu = protocol::compare_pairs(&gpu_pairs)?;
        report.record(json!({"kind":"lossless_comparison","case":key,"comparison":name,"is_confirmation":is_confirmation,
            "wall_comparison":summary,"cuda_timeline_gain":gpu.gain,"cuda_timeline_gain_ci95":gpu.bootstrap_gain_ci95,
            "cuda_timeline_control":gpu.control,"cuda_timeline_candidate":gpu.candidate,
            "timing_error":next.as_ref().err().map(|e|format!("{e:#}")),"units":"complete B-row operations/second, not model tokens/second",
            "is_model_speedup":false}))?;
        match next {
            Ok(Some(10)) => target = 10,
            Ok(Some(20)) => {
                target = 20;
                pairs.clear();
                gpu_pairs.clear();
                is_confirmation = true;
            }
            Ok(None) | Err(_) => break,
            Ok(Some(_)) => anyhow::bail!("unexpected paired trial policy"),
        }
    }
    Ok(())
}

fn logical_traffic(case: &Case, batch: usize) -> Result<Value> {
    let shape = case.get_shape(batch)?;
    let encoded = case.setup["representation"]["streams"].as_array().unwrap();
    let stream_count = case.fixture.sequences * capture::KV_HEADS;
    let active: Vec<_> = encoded[..batch * capture::KV_HEADS]
        .iter()
        .chain(&encoded[stream_count..stream_count + batch * capture::KV_HEADS])
        .collect();
    let footprint: u64 = active
        .iter()
        .map(|v| v["encoded_bytes"].as_u64().unwrap())
        .sum();
    let sectors: u64 = active
        .iter()
        .map(|v| v["encoded_bytes"].as_u64().unwrap().div_ceil(32) * 32)
        .sum();
    Ok(
        json!({"dense_unique_kv_bytes":2*shape.get_kv_bytes(),"dense_per_query_head_kv_bytes":2*shape.get_kv_bytes()*capture::QUERY_HEADS/capture::KV_HEADS,
        "encoded_unique_stream_bytes":footprint,"pointer_directory_bytes":batch*capture::KV_HEADS*2*8,
        "encoded_per_query_head_footprint_sum_bytes":footprint*(capture::QUERY_HEADS/capture::KV_HEADS) as u64,
        "encoded_aligned_stream_sector_footprint_sum_bytes":sectors,
        "staged_dense_write_bytes":2*shape.get_kv_bytes(),"staged_attention_reads_bytes":2*shape.get_kv_bytes()*capture::QUERY_HEADS/capture::KV_HEADS,
        "scope":"logical references and unique aligned stream footprints; not physical transactions; excludes duplicate field/directory loads, cache reuse and allocator overhead",
        "measured_dram_bytes":Value::Null}),
    )
}

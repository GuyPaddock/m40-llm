use super::{
    capture::{self, Capture},
    pipeline::{Arm, Case},
    reference::{Backing, Shape},
    validation, Args,
};
use anyhow::{ensure, Result};
use m40_llm::{
    cuda::{CudaContext, CudaStream},
    evaluation::identity::file_sha256,
    kv_compression::{KvCompressionConfig, ScopedRuntimeConfig},
    profile,
};
use serde_json::{json, Value};
use std::{fs::OpenOptions, io::Write, process::Command, time::Instant};

#[cfg_attr(all(feature = "cuda", feature = "server", nvcc), allow(dead_code))]
#[path = "../compare_ollama/protocol.rs"]
mod protocol;

const M40_UUID: &str = "GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852";

struct Report(std::fs::File);
impl Report {
    fn record(&mut self, value: Value) -> Result<()> {
        serde_json::to_writer(&mut self.0, &value)?;
        writeln!(self.0)?;
        self.0.flush()?;
        Ok(())
    }
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
        ensure!(matches!(key.as_str(),"M40LLM_SYSROOT"|"M40LLM_ENABLE_CUBLAS"|"M40LLM_ENABLE_NVCC")
            || (key=="M40LLM_FORCE_M40" && value=="1"),"unset {key}: the attention probe freezes native dispatch, plain top-k, score order and no anchors/masks");
    }
    Ok(())
}

fn check_gpu_processes() -> Result<String> {
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
    Ok(processes)
}

fn gpu_state() -> Result<String> {
    command("nvidia-smi",&["-i",M40_UUID,"--query-gpu=uuid,name,driver_version,pstate,temperature.gpu,clocks.current.graphics,clocks.current.memory,memory.used,power.draw","--format=csv,noheader,nounits"])
}

pub fn run(args: &Args) -> Result<()> {
    ensure!(
        !cfg!(debug_assertions),
        "attention research requires --release"
    );
    ensure!(
        (1..=100).contains(&args.iterations),
        "iterations must be 1..100"
    );
    ensure!(
        command("git", &["status", "--porcelain"])?.is_empty(),
        "commit source before real-fixture execution"
    );
    validate_environment()?;
    let mut capture = Capture::open(&args.capture_report, &args.kv_inputs)?;
    check_gpu_processes()?;
    let ctx = CudaContext::new_m40()?;
    let props = ctx.current_device_props()?;
    ensure!(props.major == 5 && props.minor == 2, "requires sm_52");
    let _dense_reference = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    let before = CudaContext::total_device_bytes();
    let mut report = Report(
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&args.output)?,
    );
    report.record(json!({"kind":"kv_attention_benchmark_manifest","schema_version":1,
        "revision":command("git",&["rev-parse","HEAD"])?,"executable_sha256":file_sha256(&std::env::current_exe()?)?,
        "capture_report_sha256":file_sha256(&args.capture_report)?,"kv_artifact_sha256":capture.artifact_sha256(),
        "capture_manifest":capture.manifest(),"gpu_state":gpu_state()?,"device_name":props.name,"device_id":props.device_id,
        "is_correctness_only":args.correctness_only,"iterations":args.iterations,"generated_token_count":0,
        "query_source":"deterministic synthetic FP32 vectors, not captured model queries",
        "head64_scope":"first 64 channels of captured heads; diagnostic, not Qwen execution",
        "dense_reference":"full-context native FP16 GQA; explicit dense_reference configuration",
        "selection":"plain top8, score order, recent1024, block32, no anchors or masks",
        "model_weights_resident":false,"is_model_quality_measured":false,
        "counter_scope":"host wrapper events; a staged wrapper submits multiple physical kernels",
        "memory_counter_scope":"DRAM transactions not collected; prior CUPTI admission failed, see docs/maxwell_counters.md"}))?;
    let mut cases = 0;
    for layer in capture::LAYERS {
        for length in capture::CONTEXTS {
            for head_dim in [64, 128] {
                let sample = capture.read_layer(layer, length, head_dim)?;
                let input_sha = sample.sha256()?;
                for backing in [Backing::Q8, Backing::Mixed] {
                    if backing == Backing::Q8 && head_dim == 128 {
                        continue;
                    }
                    check_gpu_processes()?;
                    let shape = Shape {
                        length,
                        head_dim,
                        query_heads: capture::QUERY_HEADS,
                        kv_heads: capture::KV_HEADS,
                        recent: 1024,
                        block: 32,
                        top: 8,
                    };
                    let key = json!({"layer":layer,"shape":shape,"backing":backing,"input_sha256":input_sha});
                    let case = Case::new(&ctx, shape, backing, &sample)?;
                    report.record(json!({"kind":"kv_attention_setup","case":key,"setup":case.setup,
                        "fp32_host_fixture_payload_bytes":sample.bytes(),"allocated_sequences":case.sequences}))?;
                    let began = Instant::now();
                    let checked = validation::validate(&case, &sample);
                    if let Err(error) = &checked {
                        report.record(json!({"kind":"kv_attention_validation_failure","case":key,"error":format!("{error:#}"),"timing_admitted":false}))?;
                    }
                    let checked = checked?;
                    report.record(json!({"kind":"kv_attention_validation","case":key,
                        "validation_wall_ms":began.elapsed().as_secs_f64()*1000.0,"result":checked}))?;
                    if !args.correctness_only {
                        for batch in [1, 2, 4, 8] {
                            let batch_key = json!({"fixture":key,"batch":batch});
                            report.record(
                                json!({"kind":"kv_attention_logical_traffic","case":batch_key,
                                "traffic":logical_traffic(shape,backing,batch,&checked)?}),
                            )?;
                            compare(
                                &case,
                                Arm::Staged,
                                Arm::Direct,
                                batch,
                                args.iterations,
                                &batch_key,
                                "staged_vs_direct",
                                &mut report,
                            )?;
                            if backing == Backing::Mixed {
                                compare(
                                    &case,
                                    Arm::Direct,
                                    Arm::BatchedDirect,
                                    batch,
                                    args.iterations,
                                    &batch_key,
                                    "serial_vs_batched_direct",
                                    &mut report,
                                )?;
                            }
                            // Dense coverage differs. Keep its paired observations
                            // for cost context while explicitly suppressing adoption.
                            compare(
                                &case,
                                Arm::Dense,
                                Arm::Direct,
                                batch,
                                args.iterations,
                                &batch_key,
                                "dense_context_only",
                                &mut report,
                            )?;
                        }
                    }
                    drop(case);
                    ensure!(
                        CudaContext::total_device_bytes() == before,
                        "case leaked tracked auxiliary buffers"
                    );
                    report.record(json!({"kind":"kv_attention_case_complete","case":key,
                        "tracked_device_bytes_before":before,"tracked_device_bytes_after":CudaContext::total_device_bytes(),
                        "gpu_state":gpu_state()?}))?;
                    cases += 1;
                }
            }
        }
    }
    report.record(json!({"kind":"kv_attention_benchmark_complete","is_complete":true,"cases":cases,
        "generated_token_count":0,"stop_reason":"fixed_context_format_grid_complete",
        "is_model_performance_measured":false,"tracked_device_bytes_after":CudaContext::total_device_bytes()}))
}

fn measure(
    case: &Case,
    arm: Arm,
    batch: usize,
    iterations: usize,
) -> Result<(f64, f64, profile::OpCounters)> {
    case.ctx.synchronize_stream(CudaStream::Decode)?;
    let live = CudaContext::total_device_bytes();
    let counters = profile::snapshot();
    let began = Instant::now();
    case.start.record(CudaStream::Decode)?;
    for _ in 0..iterations {
        case.enqueue(arm, batch)?;
    }
    case.stop.record(CudaStream::Decode)?;
    let gpu =
        f64::from(case.start.elapsed_sync(&case.stop, "kv_attention_timing")?) / iterations as f64;
    let wall = began.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    let delta = profile::counter_delta(&counters, &profile::snapshot());
    ensure!(gpu > 0.0 && wall > 0.0, "empty timing interval");
    ensure!(
        CudaContext::total_device_bytes() == live
            && delta.device_allocs == 0
            && delta.device_frees == 0,
        "attention timing allocated or freed memory"
    );
    ensure!(
        delta.h2d_copies == 0 && delta.d2h_copies == 0,
        "attention timing transferred host data"
    );
    Ok((wall, gpu, delta))
}

#[allow(clippy::too_many_arguments)]
fn compare(
    case: &Case,
    control: Arm,
    candidate: Arm,
    batch: usize,
    iterations: usize,
    key: &Value,
    name: &str,
    report: &mut Report,
) -> Result<()> {
    let expected = [
        case.read_arm(control, batch)?,
        case.read_arm(candidate, batch)?,
    ];
    for _ in 0..3 {
        case.enqueue(control, batch)?;
        case.enqueue(candidate, batch)?;
    }
    case.ctx.synchronize_stream(CudaStream::Decode)?;
    let mut pairs = Vec::new();
    let mut gpu_pairs = Vec::new();
    let mut target = 5;
    let mut is_confirmation = false;
    loop {
        for index in pairs.len()..target {
            let order = if index % 2 == 0 { [0, 1] } else { [1, 0] };
            let mut wall = [0.0; 2];
            let mut gpu = [0.0; 2];
            let mut counters = [profile::OpCounters::default(); 2];
            for arm in order {
                (wall[arm], gpu[arm], counters[arm]) =
                    measure(case, [control, candidate][arm], batch, iterations)?;
                // Readback is outside both timing boundaries and their counters.
                // Every timed output must match its own untimed reference.
                let actual = case.read_output(batch)?;
                ensure!(
                    actual
                        .iter()
                        .zip(&expected[arm])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "timed attention output changed for {:?}",
                    [control, candidate][arm]
                );
            }
            pairs.push(protocol::Pair {
                control_tps: 1000.0 / wall[0],
                candidate_tps: 1000.0 / wall[1],
            });
            gpu_pairs.push(protocol::Pair {
                control_tps: 1000.0 / gpu[0],
                candidate_tps: 1000.0 / gpu[1],
            });
            report.record(json!({"kind":"kv_attention_pair","case":key,"comparison":name,"pair":index,
                "is_confirmation":is_confirmation,"control":control,"candidate":candidate,"order":order,
                "iterations":iterations,"wall_ms_per_operation":wall,"gpu_ms_per_operation":gpu,"wrapper_counters":counters,
                "boundary":"complete selection/gather-or-direct/attention; cached events; wall includes event submission and completion, amortized over iterations"}))?;
        }
        let comparison = protocol::compare_pairs(&pairs)?;
        let next = protocol::next_trial_count(&comparison, is_confirmation).map(|next| {
            // Dense coverage differs, so the adoption threshold has no meaning.
            // Retain its variation extension, but omit a borderline confirmation.
            if name == "dense_context_only" && next == Some(20) {
                None
            } else {
                next
            }
        });
        let mut summary = serde_json::to_value(&comparison)?;
        let object = summary.as_object_mut().unwrap();
        object.remove("reaches_original_target");
        let qualifies = object.remove("passes_promotion_gate").unwrap();
        object.insert(
            "passes_operator_speed_gate".into(),
            if name == "dense_context_only" {
                Value::Null
            } else {
                qualifies
            },
        );
        let gpu_comparison = protocol::compare_pairs(&gpu_pairs)?;
        report.record(json!({"kind":"kv_attention_comparison","case":key,"comparison":name,"is_confirmation":is_confirmation,
            "wall_comparison":summary,"gpu_gain":gpu_comparison.gain,"gpu_gain_ci95":gpu_comparison.bootstrap_gain_ci95,
            "gpu_control":gpu_comparison.control,"gpu_candidate":gpu_comparison.candidate,
            "units":"complete B-row attention operations/second, not tokens/second",
            "timing_error":next.as_ref().err().map(|e|format!("{e:#}")),"coverage_matches":name!="dense_context_only",
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
            Ok(Some(_)) => anyhow::bail!("unexpected trial policy"),
        }
    }
    case.verify_queries()?;
    Ok(())
}

fn logical_traffic(s: Shape, backing: Backing, batch: usize, validation: &Value) -> Result<Value> {
    let counts: Vec<Vec<usize>> =
        serde_json::from_value(validation["selected_old_counts"].clone())?;
    let selected: usize = counts[..batch].iter().flatten().sum();
    let recent = batch * s.query_heads * s.length.min(s.recent);
    let old_scans = batch * s.query_heads * s.old_len();
    let (k_bytes, v_bytes, scale_bytes) = match backing {
        Backing::Q8 => (s.head_dim, s.head_dim, 8),
        Backing::Mixed => (s.head_dim * 2, s.head_dim / 2, 4),
    };
    let k_scale = if backing == Backing::Q8 { 4 } else { 0 };
    Ok(
        json!({"old_key_selection_logical_bytes":old_scans*(k_bytes+k_scale),
        "selected_old_kv_logical_bytes":selected*(k_bytes+v_bytes+scale_bytes),
        "recent_kv_logical_bytes":recent*s.head_dim*4,
        "staged_kv_write_plus_read_bytes":(selected+recent)*s.head_dim*8,
        "staged_position_and_count_write_bytes":(selected+recent+batch*s.query_heads)*4,
        "dense_full_context_kv_logical_bytes":batch*s.query_heads*s.length*s.head_dim*4,
        "old_selection_aligned_sector_footprint_sum_bytes":old_scans*(k_bytes.div_ceil(32)+usize::from(k_scale>0))*32,
        "selected_old_aligned_sector_footprint_sum_bytes":selected*(k_bytes.div_ceil(32)+v_bytes.div_ceil(32)+if backing==Backing::Q8 {2}else{1})*32,
        "sector_scope":"sum of aligned 32-byte vector/scale footprints; ignores cache reuse and duplicate warp requests",
        "logical_scope":"per-query-head references; repeated GQA reads can hit cache; not observed DRAM traffic",
        "measured_dram_bytes":Value::Null}),
    )
}

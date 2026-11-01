use super::probe::{Arm, Observation, Probe};
#[path = "../common/cpu_placement.rs"]
mod cpu_placement;
#[allow(dead_code)]
#[path = "../compare_ollama/protocol.rs"]
mod protocol;

use anyhow::{ensure, Result};
use clap::Parser;
use m40_llm::{cuda::CudaContext, evaluation::identity::file_sha256};
use serde_json::{json, Value};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    process::Command,
    time::Instant,
};

const M40_UUID: &str = "GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852";

#[derive(Parser)]
struct Args {
    /// New JSONL report.
    #[arg(long)]
    output: PathBuf,
    /// Exactly one externally pinned logical CPU.
    #[arg(long)]
    cpu: usize,
    /// Local memory node, bound before first-touch allocation.
    #[arg(long)]
    numa_node: usize,
    /// Completed operations per transfer timing interval.
    #[arg(long, default_value_t = 20)]
    iterations: usize,
    /// Check all sizes and paths without recording paired cost estimates.
    #[arg(long)]
    validate_only: bool,
}

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
        ensure!(
            matches!(
                key.as_str(),
                "M40LLM_SYSROOT" | "M40LLM_ENABLE_CUBLAS" | "M40LLM_ENABLE_NVCC"
            ) || (key == "M40LLM_FORCE_M40" && value == "1"),
            "unset {key}: transfer probe excludes runtime/logging overrides"
        );
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
            "another process owns M40: {line}"
        );
    }
    Ok(())
}

fn gpu_state() -> Result<String> {
    command("nvidia-smi",&["-i",M40_UUID,"--query-gpu=uuid,name,pci.bus_id,driver_version,pstate,temperature.gpu,clocks.current.graphics,clocks.current.memory,memory.used,power.draw","--format=csv,noheader,nounits"])
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.validate_only || !cfg!(debug_assertions),
        "timing requires --release"
    );
    ensure!(
        (1..=100).contains(&args.iterations),
        "iterations must be 1..100"
    );
    ensure!(
        command("git", &["status", "--porcelain"])?.is_empty(),
        "commit source before hardware runs"
    );
    ensure!(!args.output.exists(), "output must be new");
    validate_environment()?;
    let placement = cpu_placement::inspect(args.cpu, args.numa_node)?;
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
    report.record(json!({"kind":"cooperative_transfer_manifest","schema_version":1,
        "revision":command("git",&["rev-parse","HEAD"])?,"executable_sha256":file_sha256(&std::env::current_exe()?)?,
        "placement":placement,"gpu_state":gpu_state()?,"topology":command("nvidia-smi",&["topo","-m"])?,
        "iterations":args.iterations,"is_validation_only":args.validate_only,"generated_token_count":0,
        "allocation":"pageable first-touched host vectors and persistent device buffers; no pinned memory",
        "input_source":"deterministic byte mixing of index and recorded generation; representative sizes, not captured activation values",
        "host_process_status_after_cuda_context":fs::read_to_string("/proc/self/status")?,
        "scope":"isolated complete transfer/API costs, not projection or model speed",
        "recommended_simulator_policy":"schema 2 with copy/compute overlap false; completed cost already includes synchronization",
        "cuda_event_scope":"ordered device timeline envelope includes host gaps and event submission; not active DMA/kernel time",
        "wall_boundary":"only repeated completed operations; event setup/completion, allocation, first touch, preparation and verification excluded",
        "measured_pcie_transactions":Value::Null,"has_overlap_measurement":false}))?;
    let mut cases = Vec::new();
    for width in [2048usize, 11008] {
        for batch in [1usize, 2, 4, 8] {
            cases.push((
                format!("activation_{width}_batch_{batch}"),
                width * batch * 4,
            ));
        }
    }
    cases.extend([
        ("fp16_kv_token_one_layer".into(), 2 * 2 * 128 * 2),
        ("fp16_kv_token_36_layers".into(), 36 * 2 * 2 * 128 * 2),
        ("fp16_kv_1k_one_layer".into(), 1024 * 2 * 2 * 128 * 2),
        ("fp16_kv_8k_one_layer".into(), 8192 * 2 * 2 * 128 * 2),
        ("one_original_q8_projection".into(), 2048 * 11008 / 32 * 34),
    ]);
    for (name, bytes) in &cases {
        check_gpu_processes()?;
        let allocation = Instant::now();
        let mut probe = Probe::new(&ctx, *bytes)?;
        let allocation_ms = allocation.elapsed().as_secs_f64() * 1000.0;
        let key = json!({"name":name,"bytes":bytes});
        report.record(json!({"kind":"cooperative_transfer_setup","case":key,"allocation_and_host_first_touch_ms":allocation_ms,
            "host_bytes":probe.get_host_bytes(),"tracked_device_bytes":CudaContext::total_device_bytes()-baseline,
            "numa_maps":fs::read_to_string("/proc/self/numa_maps")?}))?;
        for arm in [Arm::H2d, Arm::D2h, Arm::SameStream, Arm::Handoff] {
            probe.measure(arm, 1)?;
        }
        report.record(json!({"kind":"cooperative_transfer_validation","case":key,"is_bit_exact":true,"guards_unchanged":true}))?;
        if !args.validate_only {
            compare(
                &mut probe,
                &key,
                "pageable_h2d_vs_d2h",
                [Arm::H2d, Arm::D2h],
                args.iterations,
                &mut report,
            )?;
            compare(
                &mut probe,
                &key,
                "same_stream_vs_handoff",
                [Arm::SameStream, Arm::Handoff],
                args.iterations,
                &mut report,
            )?;
        }
        drop(probe);
        ensure!(
            CudaContext::total_device_bytes() == baseline,
            "transfer case leaked allocations"
        );
        report.record(json!({"kind":"cooperative_transfer_case_complete","case":key,"gpu_state":gpu_state()?,
            "tracked_device_bytes_before":baseline,"tracked_device_bytes_after":CudaContext::total_device_bytes()}))?;
    }
    let mut idle = Probe::new(&ctx, 1)?;
    for arm in [Arm::IdleDecode, Arm::IdlePrefill] {
        idle.measure(arm, 1)?;
    }
    if !args.validate_only {
        compare(
            &mut idle,
            &json!({"name":"idle_synchronization","bytes":0}),
            "decode_vs_prefill_idle_sync",
            [Arm::IdleDecode, Arm::IdlePrefill],
            1000,
            &mut report,
        )?;
    }
    drop(idle);
    ensure!(
        CudaContext::total_device_bytes() == baseline,
        "idle probe leaked allocations"
    );
    report.record(json!({"kind":"cooperative_transfer_complete","is_complete":true,"transfer_cases":cases.len(),
        "generated_token_count":0,"stop_reason":"fixed_transfer_grid_complete","tracked_device_bytes_after":CudaContext::total_device_bytes(),
        "gpu_state":gpu_state()?,"is_model_performance_measured":false}))
}

fn compare(
    probe: &mut Probe,
    key: &Value,
    name: &str,
    arms: [Arm; 2],
    iterations: usize,
    report: &mut Report,
) -> Result<()> {
    for _ in 0..3 {
        probe.measure(arms[0], iterations)?;
        probe.measure(arms[1], iterations)?;
    }
    let mut pairs = Vec::new();
    let mut wall = [Vec::new(), Vec::new()];
    let mut timeline = [Vec::new(), Vec::new()];
    let mut target = 5;
    loop {
        for index in pairs.len()..target {
            let order = if index % 2 == 0 { [0, 1] } else { [1, 0] };
            let mut observations: [Option<Observation>; 2] = [None, None];
            for arm in order {
                observations[arm] = Some(probe.measure(arms[arm], iterations)?);
            }
            let [Some(control), Some(candidate)] = observations else {
                unreachable!()
            };
            pairs.push(protocol::Pair {
                control_tps: 1e9 / control.wall_ns_per_operation,
                candidate_tps: 1e9 / candidate.wall_ns_per_operation,
            });
            wall[0].push(control.wall_ns_per_operation);
            wall[1].push(candidate.wall_ns_per_operation);
            timeline[0].push(control.cuda_timeline_ns_per_operation);
            timeline[1].push(candidate.cuda_timeline_ns_per_operation);
            report.record(
                json!({"kind":"cooperative_transfer_pair","case":key,"comparison":name,"pair":index,
                "order":order,"observations":[control,candidate]}),
            )?;
        }
        let comparison = protocol::compare_pairs(&pairs)?;
        // These are cost profiles, not competing implementations of the same
        // operation. Use the variation extension, not a speed-adoption gate.
        let next = protocol::next_trial_count(&comparison, false).map(|next| {
            if next == Some(20) {
                None
            } else {
                next
            }
        });
        let mut summary = serde_json::to_value(&comparison)?;
        let object = summary.as_object_mut().unwrap();
        object.remove("reaches_original_target");
        object.remove("passes_promotion_gate");
        object.remove("promotion_threshold");
        object.remove("needs_fresh_confirmation");
        report.record(json!({"kind":"cooperative_transfer_costs","case":key,"comparison":name,"arms":arms,
            "wall_ns":[protocol::distribution(&wall[0])?,protocol::distribution(&wall[1])?],
            "cuda_timeline_ns":[protocol::distribution(&timeline[0])?,protocol::distribution(&timeline[1])?],
            "paired_operation_rate_comparison":summary,"timing_error":next.as_ref().err().map(|e|format!("{e:#}")),
            "is_cost_profile":true,"is_speed_gate_evaluated":false,"is_model_speedup":false}))?;
        match next {
            Ok(Some(10)) => target = 10,
            Ok(None) | Err(_) => break,
            Ok(Some(_)) => anyhow::bail!("unexpected trial policy"),
        }
    }
    Ok(())
}

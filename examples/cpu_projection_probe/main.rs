//! Bounded single-core Q8 projection costs for cooperative decode research.

mod fixture;
mod kernel;
#[cfg_attr(all(feature = "cuda", feature = "server", nvcc), allow(dead_code))]
#[path = "../compare_ollama/protocol.rs"]
mod protocol;
#[cfg(test)]
mod tests;
mod timing;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use fixture::Fixture;
use m40_llm::evaluation::identity::{file_sha256, sha256};
use serde_json::{json, Value};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    process::Command,
    time::Instant,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    ffn_inputs: PathBuf,
    #[arg(long)]
    capture_report: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// One pinned logical CPU; the run uses exactly one thread.
    #[arg(long)]
    cpu: usize,
    /// Bind allocations to this NUMA node before starting the process.
    #[arg(long)]
    numa_node: usize,
    /// Fixed operations per trial; cache eviction precedes every operation.
    #[arg(long, default_value_t = 3)]
    iterations: usize,
    /// Validate all real matrix/input cases without recording performance.
    #[arg(long)]
    validate_only: bool,
}

pub struct Report(std::fs::File);
impl Report {
    pub fn record(&mut self, value: Value) -> Result<()> {
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

fn placement(args: &Args) -> Result<Value> {
    let status = fs::read_to_string("/proc/self/status")?;
    let affinity = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .context("CPU affinity unavailable")?
        .trim();
    ensure!(
        affinity == args.cpu.to_string(),
        "pin exactly --cpu with numactl before running"
    );
    let nodes = command("numactl", &["--show"])?;
    let policy = nodes
        .lines()
        .find_map(|line| line.strip_prefix("policy:"))
        .unwrap_or("")
        .trim();
    let memory = nodes
        .lines()
        .find_map(|line| line.strip_prefix("membind:"))
        .unwrap_or("")
        .trim();
    ensure!(
        policy == "bind" && memory == args.numa_node.to_string(),
        "bind memory to --numa-node before running"
    );
    let cpus = fs::read_to_string(format!(
        "/sys/devices/system/node/node{}/cpulist",
        args.numa_node
    ))?;
    ensure!(cpus.trim().split(',').any(|part| {
        let (first, last) = part.split_once('-').unwrap_or((part, part));
        matches!((first.parse::<usize>(), last.parse::<usize>()), (Ok(a), Ok(b)) if a <= args.cpu && args.cpu <= b)
    }), "CPU and memory must be on the same NUMA node");
    Ok(
        json!({"cpu":args.cpu,"threads":1,"numa_node":args.numa_node,"numactl":nodes,
        "status":status,"lscpu":command("lscpu", &["-J"])?,
        "thread_siblings":fs::read_to_string(format!("/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list",args.cpu))?}),
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(kernel::has_avx2(), "AVX2 is required");
    ensure!(
        (1..=20).contains(&args.iterations),
        "iterations must be 1..20"
    );
    ensure!(
        args.validate_only || !cfg!(debug_assertions),
        "timings require --release"
    );
    ensure!(
        command("git", &["status", "--porcelain"])?.is_empty(),
        "commit source before real-fixture runs"
    );
    let placement = placement(&args)?;
    ensure!(!args.output.exists(), "output must be new");
    let setup = Instant::now();
    let mut fixture = Fixture::load(&args.model, &args.ffn_inputs, &args.capture_report)?;
    let fixture_load_ms = setup.elapsed().as_secs_f64() * 1000.0;
    let mut report = Report(
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&args.output)?,
    );
    report.record(json!({"kind":"cpu_projection_manifest","schema_version":1,
        "revision":command("git", &["rev-parse","HEAD"])?,
        "executable_sha256":file_sha256(&std::env::current_exe()?)?,
        "model_sha256":fixture::MODEL_SHA,"input_sha256":fixture::INPUT_SHA,"capture_report_sha256":fixture::REPORT_SHA,
        "placement":placement,"fixture_load_ms":fixture_load_ms,"iterations":args.iterations,
        "is_validation_only":args.validate_only,"generated_token_count":0,
        "dispatch":["scalar_q8_fp32_sequential","avx2_q8_fp32_eight_lanes_no_fma"],
        "scope":"isolated original-Q8 projections, not FFN or complete CPU decoding",
        "cache_policy":"128MiB read eviction before each measured operation, untimed; no DRAM counters",
        "weight_residency":"one original Q8 matrix, no materialized FP32 weights",
        "cpu_event_time":null,"gpu_event_time":null}))?;
    let eviction_setup = Instant::now();
    let mut eviction = timing::Eviction::new();
    report.record(
        json!({"kind":"cpu_projection_eviction_setup","bytes":eviction.bytes(),
        "allocation_first_touch_ms":eviction_setup.elapsed().as_secs_f64()*1000.0}),
    )?;
    let mut cases = 0;
    for layer_index in 0..fixture.artifact.layers.len() {
        for family in ["gate", "up", "down"] {
            let layer = &fixture.artifact.layers[layer_index];
            let source = match family {
                "gate" => &layer.source.gate,
                "up" => &layer.source.up,
                _ => &layer.source.down,
            }
            .clone();
            let input = if family == "down" {
                &layer.swiglu_inputs
            } else {
                &layer.normalized_inputs
            };
            let input = input[..8 * source.shape[0]].to_vec();
            let row_positions = layer.rows[..8].to_vec();
            let layer_id = layer.source.layer_id;
            let began = Instant::now();
            let matrix = fixture.read_matrix(&source)?;
            report.record(json!({"kind":"cpu_projection_setup","layer":layer_id,"family":family,
                "source":source,"tensor_load_validate_ms":began.elapsed().as_secs_f64()*1000.0,
                "q8_payload_bytes":matrix.bytes.len(),"eviction_bytes":eviction.bytes(),
                "retained_input_bytes":input.len()*4,"capture_payload_bytes":fixture.artifact.observation_payload_bytes,
                "numa_maps":fs::read_to_string("/proc/self/numa_maps")?}))?;
            for batch in [1, 2, 4, 8] {
                let values = &input[..batch * matrix.columns];
                let key = json!({"layer":layer_id,"family":family,"batch":batch,
                    "columns":matrix.columns,"rows":matrix.rows,"row_positions":row_positions[..batch],
                    "input_sha256":sha256(&values.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?});
                let validation_start = Instant::now();
                let mut scalar = vec![0.0; batch * matrix.rows];
                let mut avx2 = vec![0.0; batch * matrix.rows];
                let reference = matrix.reference(values, batch)?;
                matrix.project_scalar(values, batch, &mut scalar)?;
                matrix.project_avx2(values, batch, &mut avx2)?;
                report.record(json!({"kind":"cpu_projection_validation","case":key,
                    "scalar":timing::validate_output(&scalar,&reference)?,
                    "avx2":timing::validate_output(&avx2,&reference)?,
                    "allocation_reference_validation_ms":validation_start.elapsed().as_secs_f64()*1000.0,
                    "two_fp32_outputs_and_fp64_reference_bytes":batch*matrix.rows*16,
                    "tolerance":"abs(error) <= 1e-3 + 2e-5 * abs(FP64 reference)"}))?;
                if !args.validate_only {
                    timing::run_case(
                        &matrix,
                        values,
                        batch,
                        args.iterations,
                        &key,
                        &mut eviction,
                        &mut report,
                    )?;
                }
                cases += 1;
            }
        }
    }
    report.record(
        json!({"kind":"cpu_projection_complete","is_complete":true,"cases":cases,
        "generated_token_count":0,"stop_reason":"fixed_36_case_grid_complete"}),
    )
}

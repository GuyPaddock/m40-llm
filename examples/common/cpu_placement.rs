//! Shared affinity and NUMA admission for isolated hardware-cost probes.
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{fs, process::Command};

fn read_command(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    ensure!(
        output.status.success(),
        "{program} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

pub fn inspect(cpu: usize, numa_node: usize) -> Result<Value> {
    let status = fs::read_to_string("/proc/self/status")?;
    let affinity = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .context("CPU affinity unavailable")?
        .trim();
    ensure!(
        affinity == cpu.to_string(),
        "pin exactly --cpu with numactl before running"
    );
    let nodes = read_command("numactl", &["--show"])?;
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
        policy == "bind" && memory == numa_node.to_string(),
        "bind memory to --numa-node before running"
    );
    let cpus = fs::read_to_string(format!("/sys/devices/system/node/node{numa_node}/cpulist"))?;
    ensure!(cpus.trim().split(',').any(|part| {
        let (first,last) = part.split_once('-').unwrap_or((part,part));
        matches!((first.parse::<usize>(),last.parse::<usize>()),(Ok(a),Ok(b)) if a <= cpu && cpu <= b)
    }), "CPU and memory must be on the same NUMA node");
    Ok(
        json!({"cpu":cpu,"threads":1,"numa_node":numa_node,"numactl":nodes,"status":status,
        "lscpu":read_command("lscpu", &["-J"])?,
        "thread_siblings":fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"))?}),
    )
}

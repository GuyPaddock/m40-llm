use crate::input::Trace;
use anyhow::{ensure, Context, Result};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    GpuOnly,
    Independent,
    Assisted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Resource {
    Cpu,
    Gpu,
    Copy,
}

impl Resource {
    fn index(self) -> usize {
        match self {
            Self::Cpu => 0,
            Self::Gpu => 1,
            Self::Copy => 2,
        }
    }
}

#[derive(Debug, Clone)]
struct Step {
    resource: Resource,
    stage: String,
    operation: &'static str,
    duration_ns: u64,
    transfer_bytes: u64,
}

struct RequestState {
    is_cpu: bool,
    tokens: Vec<Vec<Step>>,
    token: usize,
    step: usize,
    ready_ns: u64,
    is_running: bool,
    token_start_ns: Option<u64>,
    completions: Vec<u64>,
}

#[derive(Debug, Serialize)]
pub struct Interval {
    pub request_id: String,
    pub token: usize,
    pub stage: String,
    pub operation: &'static str,
    pub resource: Resource,
    pub ready_ns: u64,
    pub start_ns: u64,
    pub end_ns: u64,
    pub queue_wait_ns: u64,
    pub transfer_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct RequestResult {
    pub id: String,
    pub is_cpu_resident: bool,
    pub completion_ns: Vec<u64>,
    pub first_decode_latency_ns: u64,
}

#[derive(Debug, Serialize)]
pub struct Quantiles {
    pub samples: usize,
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
}

#[derive(Debug, Serialize)]
pub struct ResultTrace {
    pub mode: Mode,
    pub simulated_tokens: usize,
    pub elapsed_ns: u64,
    pub simulated_tokens_per_second: f64,
    pub host_reserved_bytes: u64,
    pub gpu_reserved_bytes: u64,
    pub cpu_compute_busy_ns: u64,
    pub cpu_worker_occupied_ns: u64,
    pub gpu_busy_ns: u64,
    pub copy_busy_ns: u64,
    pub transfer_bytes: u64,
    pub inter_token_latency: Option<Quantiles>,
    pub first_decode_latency: Quantiles,
    pub requests: Vec<RequestResult>,
    pub intervals: Vec<Interval>,
}

pub fn simulate(trace: &Trace, mode: Mode) -> Result<ResultTrace> {
    trace.validate()?;
    let mut host_reserved = trace.memory.host_fixed_bytes;
    let mut gpu_reserved = trace.memory.gpu_fixed_bytes;
    let mut states = Vec::new();
    for request in &trace.requests {
        let is_cpu = mode != Mode::GpuOnly && request.is_cpu_resident;
        let (reserved, peak) = if is_cpu {
            (&mut host_reserved, request.host_peak_kv_bytes)
        } else {
            (&mut gpu_reserved, request.gpu_peak_kv_bytes)
        };
        *reserved = reserved
            .checked_add(peak)
            .context("memory accounting overflow")?;
        let tokens = request
            .token_profiles
            .iter()
            .map(|name| {
                let mut steps = Vec::new();
                for stage in &trace.profiles[name] {
                    let mut append = |resource, operation, duration_ns, transfer_bytes| {
                        steps.push(Step {
                            resource,
                            stage: stage.name.clone(),
                            operation,
                            duration_ns,
                            transfer_bytes,
                        });
                    };
                    if is_cpu {
                        if let Some(assist) =
                            stage.assist.as_ref().filter(|_| mode == Mode::Assisted)
                        {
                            append(
                                Resource::Copy,
                                "h2d",
                                assist.h2d.duration_ns,
                                assist.h2d_bytes,
                            );
                            append(Resource::Gpu, "assist", stage.gpu.duration_ns, 0);
                            append(
                                Resource::Copy,
                                "d2h",
                                assist.d2h.duration_ns,
                                assist.d2h_bytes,
                            );
                            if assist.synchronization.duration_ns > 0 {
                                append(
                                    Resource::Cpu,
                                    "synchronize",
                                    assist.synchronization.duration_ns,
                                    0,
                                );
                            }
                        } else if stage.cpu.duration_ns > 0 {
                            append(Resource::Cpu, "local", stage.cpu.duration_ns, 0);
                        }
                    } else if stage.gpu.duration_ns > 0 {
                        append(Resource::Gpu, "main", stage.gpu.duration_ns, 0);
                    }
                }
                steps
            })
            .collect();
        states.push(RequestState {
            is_cpu,
            tokens,
            token: 0,
            step: 0,
            ready_ns: request.ready_ns,
            is_running: false,
            token_start_ns: None,
            completions: Vec::new(),
        });
    }
    ensure!(
        host_reserved <= trace.memory.host_capacity_bytes,
        "host finish-peak reservation exceeds capacity"
    );
    ensure!(
        gpu_reserved <= trace.memory.gpu_capacity_bytes,
        "GPU finish-peak reservation exceeds capacity"
    );

    let start_ns = trace.requests.iter().map(|r| r.ready_ns).min().unwrap();
    let mut now = start_ns;
    let mut active_cpu = None;
    let mut active_gpu = None;
    let mut running: [Option<(usize, u64)>; 3] = [None; 3];
    let mut intervals = Vec::new();
    let mut cpu_occupied = 0;
    loop {
        // Finish every simultaneous event before dispatching any new stage.
        for slot in &mut running {
            if let Some((id, end)) = *slot {
                if end == now {
                    *slot = None;
                    let state = &mut states[id];
                    state.is_running = false;
                    state.ready_ns = now;
                    state.step += 1;
                    if state.step == state.tokens[state.token].len() {
                        state.completions.push(now);
                        state.token += 1;
                        state.step = 0;
                        let began = state.token_start_ns.take().unwrap();
                        if state.is_cpu {
                            cpu_occupied += now - began;
                            active_cpu = None;
                        } else {
                            active_gpu = None;
                        }
                    }
                }
            }
        }
        if states.iter().all(|s| s.token == s.tokens.len()) {
            break;
        }
        acquire_request(&mut active_cpu, true, &mut states, now);
        acquire_request(&mut active_gpu, false, &mut states, now);

        let has_overlap = trace.has_effective_copy_compute_overlap();
        let resources = if has_overlap {
            [Resource::Cpu, Resource::Gpu, Resource::Copy]
        } else {
            // A ready assist copy gets the next shared GPU/copy boundary;
            // do not starve it behind the GPU main request's remaining stages.
            [Resource::Cpu, Resource::Copy, Resource::Gpu]
        };
        for resource in resources {
            if !has_overlap
                && ((resource == Resource::Gpu && running[Resource::Copy.index()].is_some())
                    || (resource == Resource::Copy && running[Resource::Gpu.index()].is_some()))
            {
                continue;
            }
            let slot = &mut running[resource.index()];
            if slot.is_some() {
                continue;
            }
            // CPU-owned assistance has priority over the next GPU main stage.
            let id = [active_cpu, active_gpu].into_iter().flatten().find(|&id| {
                let state = &states[id];
                !state.is_running
                    && state.ready_ns <= now
                    && state.tokens[state.token][state.step].resource == resource
            });
            if let Some(id) = id {
                let state = &mut states[id];
                let step = &state.tokens[state.token][state.step];
                let end = now
                    .checked_add(step.duration_ns)
                    .context("clock overflow")?;
                intervals.push(Interval {
                    request_id: trace.requests[id].id.clone(),
                    token: state.token,
                    stage: step.stage.clone(),
                    operation: step.operation,
                    resource,
                    ready_ns: state.ready_ns,
                    start_ns: now,
                    end_ns: end,
                    queue_wait_ns: now - state.ready_ns,
                    transfer_bytes: step.transfer_bytes,
                });
                state.is_running = true;
                *slot = Some((id, end));
            }
        }
        let next_finish = running.iter().flatten().map(|(_, end)| *end).min();
        let next_arrival = states
            .iter()
            .filter(|s| s.token < s.tokens.len() && s.ready_ns > now)
            .map(|s| s.ready_ns)
            .min();
        now = next_finish
            .into_iter()
            .chain(next_arrival)
            .min()
            .context("simulation deadlock")?;
    }
    let requests: Vec<_> = states
        .into_iter()
        .zip(&trace.requests)
        .map(|(state, request)| RequestResult {
            id: request.id.clone(),
            is_cpu_resident: state.is_cpu,
            first_decode_latency_ns: state.completions[0] - request.ready_ns,
            completion_ns: state.completions,
        })
        .collect();
    let itls = requests
        .iter()
        .flat_map(|r| r.completion_ns.windows(2).map(|w| w[1] - w[0]))
        .collect();
    let first = requests.iter().map(|r| r.first_decode_latency_ns).collect();
    let busy = |resource| {
        intervals
            .iter()
            .filter(|i| i.resource == resource)
            .map(|i| i.end_ns - i.start_ns)
            .sum()
    };
    let simulated_tokens = requests.iter().map(|r| r.completion_ns.len()).sum();
    Ok(ResultTrace {
        mode,
        simulated_tokens,
        elapsed_ns: now - start_ns,
        simulated_tokens_per_second: simulated_tokens as f64 * 1e9 / (now - start_ns) as f64,
        host_reserved_bytes: host_reserved,
        gpu_reserved_bytes: gpu_reserved,
        cpu_compute_busy_ns: busy(Resource::Cpu),
        cpu_worker_occupied_ns: cpu_occupied,
        gpu_busy_ns: busy(Resource::Gpu),
        copy_busy_ns: busy(Resource::Copy),
        transfer_bytes: intervals.iter().map(|i| i.transfer_bytes).sum(),
        inter_token_latency: calculate_quantiles(itls),
        first_decode_latency: calculate_quantiles(first).unwrap(),
        requests,
        intervals,
    })
}

fn acquire_request(
    active: &mut Option<usize>,
    is_cpu: bool,
    states: &mut [RequestState],
    now: u64,
) {
    if active.is_some() {
        return;
    }
    // Earliest ready time, then stable input order. Finished tokens rejoin FIFO.
    *active = states
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_cpu == is_cpu && s.token < s.tokens.len() && s.ready_ns <= now)
        .min_by_key(|(id, s)| (s.ready_ns, *id))
        .map(|(id, _)| id);
    if let Some(id) = *active {
        states[id].token_start_ns = Some(now);
    }
}

fn calculate_quantiles(mut values: Vec<u64>) -> Option<Quantiles> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let at = |percent: usize| values[(values.len() * percent).div_ceil(100) - 1];
    Some(Quantiles {
        samples: values.len(),
        p50_ns: at(50),
        p95_ns: at(95),
        p99_ns: at(99),
    })
}

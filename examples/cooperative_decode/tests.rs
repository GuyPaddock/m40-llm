use crate::{
    input::{Assist, Cost, Evidence, Memory, Request, Stage, Trace},
    simulation::{simulate, Mode, Resource},
};
use std::collections::BTreeMap;

fn cost(duration_ns: u64) -> Cost {
    Cost {
        duration_ns,
        evidence: "hand_calculation".into(),
    }
}

fn fixture() -> Trace {
    Trace {
        schema_version: 1,
        has_copy_compute_overlap: None,
        description: "Synthetic event-order fixture; not hardware measurements".into(),
        cpu_threads: 1,
        cpu_numa_node: 0,
        memory: Memory {
            host_capacity_bytes: 1000,
            gpu_capacity_bytes: 1000,
            host_fixed_bytes: 100,
            gpu_fixed_bytes: 100,
        },
        evidence: BTreeMap::from([(
            "hand_calculation".into(),
            Evidence::Assumption {
                description: "Explicit integer costs for independent hand calculations".into(),
            },
        )]),
        profiles: BTreeMap::from([(
            "token".into(),
            vec![
                Stage {
                    name: "prefix".into(),
                    cpu: cost(4),
                    gpu: cost(10),
                    assist: None,
                },
                Stage {
                    name: "dense".into(),
                    cpu: cost(20),
                    gpu: cost(10),
                    assist: Some(Assist {
                        h2d: cost(2),
                        d2h: cost(2),
                        synchronization: cost(1),
                        h2d_bytes: 16,
                        d2h_bytes: 16,
                    }),
                },
            ],
        )]),
        requests: vec![request("gpu", false, 0, 1), request("cpu", true, 0, 1)],
    }
}

fn request(id: &str, is_cpu_resident: bool, ready_ns: u64, tokens: usize) -> Request {
    Request {
        id: id.into(),
        ready_ns,
        is_cpu_resident,
        host_peak_kv_bytes: 200,
        gpu_peak_kv_bytes: 200,
        token_profiles: vec!["token".into(); tokens],
    }
}

/// Independent engines overlap work; GPU-only serializes both complete tokens.
#[test]
fn test_independent_engines_and_single_token_latency() {
    let trace = fixture();
    let gpu = simulate(&trace, Mode::GpuOnly).unwrap();
    assert_eq!(gpu.requests[0].completion_ns, [20]);
    assert_eq!(gpu.requests[1].completion_ns, [40]);
    assert_eq!(gpu.elapsed_ns, 40);
    assert_eq!(gpu.gpu_busy_ns, 40);
    assert_eq!(gpu.cpu_worker_occupied_ns, 0);
    assert_eq!(gpu.gpu_reserved_bytes, 500);
    assert_eq!(gpu.host_reserved_bytes, 100);
    assert!(gpu.inter_token_latency.is_none());
    let independent = simulate(&trace, Mode::Independent).unwrap();
    assert_eq!(independent.requests[0].completion_ns, [20]);
    assert_eq!(independent.requests[1].completion_ns, [24]);
    assert_eq!(independent.elapsed_ns, 24);
    assert_eq!(independent.cpu_worker_occupied_ns, 24);
    assert_eq!(independent.first_decode_latency.p50_ns, 20);
    assert_eq!(independent.first_decode_latency.p99_ns, 24);
    assert_eq!(independent.gpu_reserved_bytes, 300);
    assert_eq!(independent.host_reserved_bytes, 300);
}

/// An assist waits for a running GPU stage, takes priority at its boundary,
/// and charges both copies and CPU synchronization before releasing its worker.
#[test]
fn test_nonpreemptive_assist_delays_gpu_main() {
    let result = simulate(&fixture(), Mode::Assisted).unwrap();
    assert_eq!(result.requests[0].completion_ns, [30]);
    assert_eq!(result.requests[1].completion_ns, [23]);
    let gpu_intervals: Vec<_> = result
        .intervals
        .iter()
        .filter(|i| i.resource == Resource::Gpu)
        .map(|i| (i.operation, i.start_ns, i.end_ns, i.queue_wait_ns))
        .collect();
    assert_eq!(
        gpu_intervals,
        [
            ("main", 0, 10, 0),
            ("assist", 10, 20, 4),
            ("main", 20, 30, 10)
        ]
    );
    assert_eq!(result.cpu_compute_busy_ns, 5);
    assert_eq!(result.cpu_worker_occupied_ns, 23);
    assert_eq!(result.copy_busy_ns, 4);
    assert_eq!(result.transfer_bytes, 32);
    assert_eq!(result.gpu_busy_ns, 30);
}

/// Blocking transfer profiles forbid all GPU/copy overlap and still give
/// assistance priority at stage boundaries. Already-included synchronization
/// costs zero extra time rather than adding a fabricated nanosecond.
#[test]
fn test_serialized_copy_profile_preserves_assist_priority() {
    let mut trace = fixture();
    trace.schema_version = 2;
    trace.has_copy_compute_overlap = Some(false);
    trace.profiles.get_mut("token").unwrap()[1]
        .assist
        .as_mut()
        .unwrap()
        .synchronization
        .duration_ns = 0;
    let result = simulate(&trace, Mode::Assisted).unwrap();
    assert_eq!(result.requests[0].completion_ns, [34]);
    assert_eq!(result.requests[1].completion_ns, [24]);
    assert_eq!(result.cpu_compute_busy_ns, 4);
    assert_eq!(result.cpu_worker_occupied_ns, 24);
    assert_eq!(result.copy_busy_ns, 4);
    for copy in result
        .intervals
        .iter()
        .filter(|v| v.resource == Resource::Copy)
    {
        for gpu in result
            .intervals
            .iter()
            .filter(|v| v.resource == Resource::Gpu)
        {
            assert!(copy.end_ns <= gpu.start_ns || gpu.end_ns <= copy.start_ns);
        }
    }
    let first_copy = result
        .intervals
        .iter()
        .find(|v| v.operation == "h2d")
        .unwrap();
    assert_eq!(
        (
            first_copy.start_ns,
            first_copy.end_ns,
            first_copy.queue_wait_ns
        ),
        (10, 12, 6)
    );
    assert!(!result
        .intervals
        .iter()
        .any(|v| v.operation == "synchronize"));
}

/// New profiles must declare overlap, and only synchronization may be free
/// when its service is already included in measured completed-operation costs.
#[test]
fn test_transfer_policy_requires_explicit_schema_two() {
    let mut trace = fixture();
    trace.schema_version = 2;
    assert!(trace.validate().is_err());
    trace.has_copy_compute_overlap = Some(true);
    assert!(trace.validate().is_ok());
    assert_eq!(simulate(&trace, Mode::Assisted).unwrap().elapsed_ns, 30);
    trace.schema_version = 1;
    assert!(trace.validate().is_err());
    trace.schema_version = 2;
    trace.profiles.get_mut("token").unwrap()[1]
        .assist
        .as_mut()
        .unwrap()
        .h2d
        .duration_ns = 0;
    assert!(trace.validate().is_err());
}

/// Simultaneous copy and native-stage completion makes the assist eligible
/// before the next main stage, independent of resource iteration order.
#[test]
fn test_assist_priority_at_simultaneous_completion() {
    let mut trace = fixture();
    trace.profiles.get_mut("token").unwrap()[0].cpu.duration_ns = 8;
    let result = simulate(&trace, Mode::Assisted).unwrap();
    let assist = result
        .intervals
        .iter()
        .find(|i| i.operation == "assist")
        .unwrap();
    assert_eq!(assist.start_ns, 10);
    assert_eq!(assist.queue_wait_ns, 0);
    assert_eq!(result.requests[0].completion_ns, [30]);
}

/// The serialized input rejects misspelled fields, fractional/negative time,
/// and non-finite numbers rather than silently applying a default cost.
#[test]
fn test_serialized_cost_contract() {
    let value = serde_json::to_value(fixture()).unwrap();
    assert!(serde_json::from_value::<Trace>(value.clone()).is_ok());
    for invalid in [
        serde_json::json!(-1),
        serde_json::json!(1.5),
        serde_json::json!("NaN"),
    ] {
        let mut changed = value.clone();
        changed["profiles"]["token"][0]["cpu"]["duration_ns"] = invalid;
        assert!(serde_json::from_value::<Trace>(changed).is_err());
    }
    let mut changed = value;
    changed["cpu_thread"] = serde_json::json!(4);
    assert!(serde_json::from_value::<Trace>(changed).is_err());
}

/// CPU requests cannot run through another request's assist wait. Completed
/// tokens rejoin FIFO so a waiting request advances before that owner's next token.
#[test]
fn test_worker_ownership_and_token_fairness() {
    let mut trace = fixture();
    trace.requests = vec![request("first", true, 0, 2), request("second", true, 1, 1)];
    let result = simulate(&trace, Mode::Assisted).unwrap();
    assert_eq!(result.requests[0].completion_ns, [19, 57]);
    assert_eq!(result.requests[1].completion_ns, [38]);
    assert_eq!(result.requests[1].first_decode_latency_ns, 37);
    assert_eq!(result.cpu_worker_occupied_ns, 57);
    assert_eq!(result.inter_token_latency.as_ref().unwrap().samples, 1);
    assert_eq!(result.inter_token_latency.as_ref().unwrap().p50_ns, 38);
    let second_first = result
        .intervals
        .iter()
        .find(|i| i.request_id == "second")
        .unwrap();
    assert_eq!(second_first.start_ns, 19);
    assert_eq!(second_first.queue_wait_ns, 18);
}

/// Profile changes, late arrivals and idle gaps remain visible in completion
/// times, while pre-arrival time is excluded from the reported decode window.
#[test]
fn test_context_profiles_and_arrivals() {
    let mut trace = fixture();
    trace.profiles.insert(
        "long".into(),
        vec![Stage {
            name: "all".into(),
            cpu: cost(80),
            gpu: cost(50),
            assist: None,
        }],
    );
    trace.requests = vec![request("gpu", false, 100, 1), request("late", true, 200, 1)];
    trace.requests[0].token_profiles.push("long".into());
    let result = simulate(&trace, Mode::Independent).unwrap();
    assert_eq!(result.requests[0].completion_ns, [120, 170]);
    assert_eq!(result.requests[1].completion_ns, [224]);
    assert_eq!(result.elapsed_ns, 124);
    assert_eq!(result.simulated_tokens, 3);
    assert_eq!(result.inter_token_latency.unwrap().p99_ns, 50);
}

/// Capacity uses finish-peak KV and fixed allocations, with checked sums.
/// A GPU-only scenario may fail while the same requests fit two engines.
#[test]
fn test_rejects_memory_overcommit_and_overflow() {
    let mut trace = fixture();
    trace.memory.gpu_capacity_bytes = 300;
    assert!(simulate(&trace, Mode::GpuOnly).is_err());
    assert!(simulate(&trace, Mode::Independent).is_ok());
    trace.memory.host_capacity_bytes = 299;
    assert!(simulate(&trace, Mode::Independent).is_err());
    trace.memory.host_capacity_bytes = u64::MAX;
    trace.memory.host_fixed_bytes = u64::MAX;
    assert!(simulate(&trace, Mode::Independent).is_err());
}

/// Invalid traces fail before simulation instead of producing plausible output.
#[test]
fn test_rejects_unpriced_or_invalid_work() {
    let mut trace = fixture();
    trace.evidence.clear();
    assert!(trace.validate().is_err());
    let mut trace = fixture();
    trace.requests[1].id = trace.requests[0].id.clone();
    assert!(trace.validate().is_err());
    let mut trace = fixture();
    trace.requests[0].token_profiles[0] = "missing".into();
    assert!(trace.validate().is_err());
    let mut trace = fixture();
    trace.profiles.get_mut("token").unwrap()[0].gpu.duration_ns = 0;
    assert!(trace.validate().is_err());
    let mut trace = fixture();
    trace.requests[0].ready_ns = u64::MAX;
    assert!(trace.validate().is_err());
    let mut trace = fixture();
    trace.requests[0].token_profiles = vec!["token".into(); 20_001];
    assert!(trace.validate().is_err());
}

/// Across different trace sizes, engine service is conserved, resource
/// intervals never overlap, and every request's token completions are causal.
#[test]
fn test_event_invariants_across_workloads() {
    for count in 1..=8 {
        let mut trace = fixture();
        trace.memory.host_capacity_bytes = 10_000;
        trace.memory.gpu_capacity_bytes = 10_000;
        trace.requests = (0..count)
            .map(|i| request(&format!("r{i}"), i % 2 == 0, i as u64 * 7, i + 1))
            .collect();
        for mode in [Mode::GpuOnly, Mode::Independent, Mode::Assisted] {
            let result = simulate(&trace, mode).unwrap();
            let expected_tokens: usize = (1..=count).sum();
            assert_eq!(result.simulated_tokens, expected_tokens);
            for resource in [Resource::Cpu, Resource::Gpu, Resource::Copy] {
                let intervals: Vec<_> = result
                    .intervals
                    .iter()
                    .filter(|i| i.resource == resource)
                    .collect();
                for pair in intervals.windows(2) {
                    assert!(pair[0].end_ns <= pair[1].start_ns);
                }
                for interval in intervals {
                    assert!(interval.start_ns >= interval.ready_ns);
                }
            }
            for (request, output) in trace.requests.iter().zip(&result.requests) {
                assert_eq!(request.token_profiles.len(), output.completion_ns.len());
                assert!(output.completion_ns[0] > request.ready_ns);
                assert!(output
                    .completion_ns
                    .windows(2)
                    .all(|pair| pair[1] > pair[0]));
            }
            let cpu_tokens: usize = trace
                .requests
                .iter()
                .filter(|r| r.is_cpu_resident)
                .map(|r| r.token_profiles.len())
                .sum();
            let expected_gpu = match mode {
                Mode::GpuOnly => expected_tokens * 20,
                Mode::Independent => (expected_tokens - cpu_tokens) * 20,
                Mode::Assisted => (expected_tokens - cpu_tokens) * 20 + cpu_tokens * 10,
            };
            assert_eq!(result.gpu_busy_ns, expected_gpu as u64);
            assert!(result.cpu_worker_occupied_ns <= result.elapsed_ns);
        }
    }
}

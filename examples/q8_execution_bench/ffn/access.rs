//! Static per-warp instruction footprints, not measured cache/DRAM traffic.
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeSet;

#[derive(Default, Serialize)]
struct Footprint {
    input_sector_requests: usize,
    weight_sector_requests: usize,
    scale_sector_requests: usize,
}

fn count_sectors(addresses: impl Iterator<Item = usize>, width: usize) -> usize {
    addresses
        .flat_map(|address| (address..address + width).map(|byte| byte / 32))
        .collect::<BTreeSet<_>>()
        .len()
}

fn estimate_requests(k: usize, phase: usize, has_coalesced_pairs: bool) -> Footprint {
    let values_per_task = if has_coalesced_pairs { 2 } else { 8 };
    let tasks_per_block = 32 / values_per_task;
    let tasks = k / values_per_task;
    let mut result = Footprint::default();
    for warp in 0..4 {
        for iteration in 0..tasks.div_ceil(128) {
            let active: Vec<_> = (0..32)
                .map(|lane| iteration * 128 + warp * 32 + lane)
                .filter(|task| *task < tasks)
                .collect();
            let block = |task: usize| task / tasks_per_block;
            let offset = |task: usize| task % tasks_per_block * values_per_task;
            result.scale_sector_requests +=
                2 * count_sectors(active.iter().map(|&task| phase + block(task) * 34), 2);
            if has_coalesced_pairs {
                result.input_sector_requests += count_sectors(
                    active
                        .iter()
                        .map(|&task| (block(task) * 32 + offset(task)) * 4),
                    8,
                );
                result.weight_sector_requests += 2 * count_sectors(
                    active
                        .iter()
                        .map(|&task| phase + block(task) * 34 + 2 + offset(task)),
                    2,
                );
            } else {
                for pair in 0..4 {
                    result.weight_sector_requests += 2 * count_sectors(
                        active
                            .iter()
                            .map(|&task| phase + block(task) * 34 + 2 + offset(task) + pair * 2),
                        2,
                    );
                    for lane in 0..2 {
                        result.input_sector_requests += count_sectors(
                            active.iter().map(|&task| {
                                (block(task) * 32 + offset(task) + pair * 2 + lane) * 4
                            }),
                            4,
                        );
                    }
                }
            }
        }
    }
    result
}

pub(super) fn describe_requests(k: usize) -> Value {
    let phases = [0, 2].map(|phase| {
        json!({"phase_bytes":phase,
        "packed_warp128":estimate_requests(k,phase,false),
        "coalesced_pairs128":estimate_requests(k,phase,true)})
    });
    json!({"scope":"predicted 32-byte sectors summed per warp memory instruction, one output column and one input row; NOT cache misses, measured transactions or DRAM bytes",
        "threads":128,"k":k,"input_base_phase_bytes":0,"input_alignment":32,
        "kernel_minimum_input_alignment":8,"activation_storage":"FP32",
        "weight_base_phases":phases,
        "assumptions":"scalar FP32 input loads in split4 control; float2 input load in adjacent-pair candidate; U16 Q8 payload and scale loads; gate and up counted separately; no cache reuse or replay modeled"})
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Adjacent float2 lanes cover four input sectors for one 32-value block;
    /// split4 scalar requests revisit those sectors across eight instructions.
    #[test]
    fn test_sector_model_counts_issued_requests_not_unique_bytes() {
        let old = estimate_requests(32, 0, false);
        let candidate = estimate_requests(32, 0, true);
        assert_eq!(
            (
                old.input_sector_requests,
                old.weight_sector_requests,
                old.scale_sector_requests
            ),
            (32, 10, 2)
        );
        assert_eq!(
            (
                candidate.input_sector_requests,
                candidate.weight_sector_requests,
                candidate.scale_sector_requests
            ),
            (4, 4, 2)
        );
    }

    /// Coalesced pairs reduce the modeled input/payload request footprint at
    /// every legal halfword phase, without claiming a physical DRAM reduction.
    #[test]
    fn test_sector_model_covers_qwen_and_two_byte_weight_alignment() {
        for phase in (0..32).step_by(2) {
            let old = estimate_requests(2048, phase, false);
            let candidate = estimate_requests(2048, phase, true);
            assert!(candidate.input_sector_requests < old.input_sector_requests);
            assert!(candidate.weight_sector_requests < old.weight_sector_requests);
        }
    }
}

//! Fixed exact portfolio qualification. No process-global consumer switch.
pub(super) mod http;
pub(super) mod quality;

use super::run::{self, Report};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaContext,
    infer::{execution_weights::Q8PortfolioArm, LoadedModel, Q8FfnOwnership},
    profile,
};
use serde_json::{json, Value};
use std::{fs, path::Path, time::Instant};

const ARMS: [Q8PortfolioArm; 5] = [
    Q8PortfolioArm::Control,
    Q8PortfolioArm::Down,
    Q8PortfolioArm::Vocabulary,
    Q8PortfolioArm::Argmax,
    Q8PortfolioArm::All,
];
const FROZEN: &str = "docs/experiments/2026-10-27-maxwell-compound-runtime-quality.jsonl";
const ORIGINAL_QUALITY: &str = "docs/experiments/2026-10-27-joint-only-final-quality.jsonl";

fn read_rows(path: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()
        .map_err(Into::into)
}

fn load(path: &Path, arm: Q8PortfolioArm, report: &mut Report) -> Result<LoadedModel> {
    let initial = CudaContext::total_device_bytes();
    let start = Instant::now();
    let mut model = run::load(path, Q8FfnOwnership::JointOnly)?;
    let original_load_ms = start.elapsed().as_secs_f64() * 1000.;
    let before_bytes = CudaContext::total_device_bytes();
    let original_split = serde_json::to_value(model.q8_execution_stats())?;
    let original_joint = serde_json::to_value(model.q8_ffn_pair_stats())?;
    let before = profile::snapshot();
    model.prepare_q8_portfolio(arm)?;
    let counters = profile::counter_delta(&before, &profile::snapshot());
    let stats = model.q8_execution.portfolio_stats().context("packets")?;
    ensure!(
        stats.tensors.len() == 37
            && stats.resident_bytes == 1_192_935_424
            && stats.combined_prepared_bytes == 4_110_516_224
            && CudaContext::total_device_bytes() - before_bytes == stats.resident_bytes
            && counters.device_allocs == 74
            && counters.device_frees == 0
            && original_split == serde_json::to_value(model.q8_execution_stats())?
            && original_joint == serde_json::to_value(model.q8_ffn_pair_stats())?,
        "portfolio changed original ownership or packet allocation accounting"
    );
    model.q8_execution.verify_portfolio_payloads()?;
    let status = fs::read_to_string("/proc/self/status")?;
    report.record(json!({"kind":"portfolio_load","arm":arm,"model":run::identity(&model),
        "packets":stats,"original_load_ms":original_load_ms,"packet_setup_counters":counters,
        "tracked_device_delta":CudaContext::total_device_bytes()-initial,
        "packet_setup_device_high_water":CudaContext::total_device_bytes(),
        "packet_setup_is_monotonic":true,
        "process_lifetime_host_high_water":status.lines().find(|s|s.starts_with("VmHWM:")),
        "temporary_host_scope":"packet and inverse payload only; excludes original host model and allocator overhead"}))?;
    Ok(model)
}

/// Build the only admitted dispatch delta. This also rejects extra copies,
/// synchronizations, allocations or unrelated launch changes.
fn expected_dispatch(original: &Value, arm: Q8PortfolioArm) -> Result<Value> {
    let mut expected = original.clone();
    for counter in ["device_allocs", "device_frees"] {
        let count = expected["totals"][counter].as_u64().context("counter")?;
        expected["totals"][counter] = json!(count + 1);
    }
    let packets =
        512 * (36 * u64::from(arm.has_packet_down()) + u64::from(arm.has_packet_vocabulary()));
    if packets > 0 {
        let ops = expected["operations"]
            .as_object_mut()
            .context("operations")?;
        let separate = ops["gemm_f32xq8_separate_f32"]["launches"]
            .as_u64()
            .context("separate")?;
        let remaining = separate.checked_sub(packets).context("packet coverage")?;
        if remaining == 0 {
            ops.remove("gemm_f32xq8_separate_f32");
        } else {
            ops.get_mut("gemm_f32xq8_separate_f32").unwrap()["launches"] = json!(remaining);
        }
        ops.insert(
            "gemm_f32xq8_warp_packet".into(),
            json!({"launches":packets,"cublas_calls":0}),
        );
    }
    if arm.has_partitioned_argmax() {
        let launches = expected["totals"]["launches"]
            .as_u64()
            .context("launches")?;
        expected["totals"]["launches"] = json!(launches + 512);
        let ops = expected["operations"]
            .as_object_mut()
            .context("operations")?;
        ensure!(
            ops.remove("argmax_f32") == Some(json!({"launches":512,"cublas_calls":0})),
            "argmax coverage"
        );
        for name in [
            "argmax_f32_partition_partials",
            "argmax_f32_partition_reduce",
        ] {
            ops.insert(name.into(), json!({"launches":512,"cublas_calls":0}));
        }
    }
    Ok(expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only named consumers and the equal session scratch may change the fixed
    /// workload's operation counts; other counters must survive unchanged.
    #[test]
    fn test_portfolio_dispatch_deltas_are_bounded() -> Result<()> {
        let original = json!({"totals":{"launches":50000,"device_allocs":2,"device_frees":2,"d2h_copies":512},
            "operations":{"argmax_f32":{"launches":512,"cublas_calls":0},
            "gemm_f32xq8_separate_f32":{"launches":18944,"cublas_calls":0},
            "unchanged":{"launches":42,"cublas_calls":7}}});
        for arm in ARMS {
            let actual = expected_dispatch(&original, arm)?;
            assert_eq!(actual["totals"]["device_allocs"], 3);
            assert_eq!(actual["totals"]["device_frees"], 3);
            assert_eq!(actual["totals"]["d2h_copies"], 512);
            assert_eq!(
                actual["operations"]["unchanged"],
                original["operations"]["unchanged"]
            );
            assert_eq!(
                actual["totals"]["launches"],
                50000 + 512 * u64::from(arm.has_partitioned_argmax())
            );
            let packets = actual["operations"]["gemm_f32xq8_warp_packet"]["launches"]
                .as_u64()
                .unwrap_or(0);
            assert_eq!(
                packets,
                512 * (36 * u64::from(arm.has_packet_down())
                    + u64::from(arm.has_packet_vocabulary()))
            );
            assert_eq!(
                packets
                    + actual["operations"]["gemm_f32xq8_separate_f32"]["launches"]
                        .as_u64()
                        .unwrap_or(0),
                18944
            );
        }
        let mut broken = original;
        broken["operations"]["argmax_f32"]["launches"] = json!(511);
        assert!(expected_dispatch(&broken, Q8PortfolioArm::All).is_err());
        Ok(())
    }
}

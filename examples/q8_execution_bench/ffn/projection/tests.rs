use super::pipeline::{ProjectionComparison, ProjectionKind, ProjectionProbe};
use anyhow::Result;
use m40_llm::{
    cuda::CudaContext,
    infer::{execution_weights::Q8ExecutionConfig, LoadedModel},
    profile,
};
#[path = "../../../../tests/common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "../../../../tests/common/env_restore.rs"]
mod env_restore;
#[path = "../../../../tests/common/q8_model.rs"]
#[allow(dead_code)]
mod fixture;

/// The shared-runner QKV/output adapters preserve CPU/control results with distinct
/// rows and reuses its guarded buffers without timed allocation or copies.
/// Dispatch counters identify the actual arm and all allocations are released.
/// Packed-load comparisons use the qualified warp control, not the original
/// shared reduction, and preserve both variants without growing scratch/cache.
/// Split-plane comparisons use the qualified packed control and account for
/// their extra immutable preparation separately from reusable scratch.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_projection_probe_preserves_outputs_dispatch_and_ownership() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _stream = env_restore::EnvRestore::set("M40LLM_Q8_DECODE_STREAM", "decode");
    let _split = env_restore::EnvRestore::set("M40LLM_Q8_DECODE_SPLIT_QBLOCK", "4");
    let _columns = env_restore::EnvRestore::set("M40LLM_Q8_DECODE_COL_TILE", "4");
    let _diag = env_restore::EnvRestore::set("M40LLM_Q8_DECODE_EVENT_LOG", "0");
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let (gguf, bytes) = fixture::build(0);
        let model = LoadedModel::from_gguf_with_q8_execution(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig::parse("split", Some("16"), Some("down,vocabulary"))?,
        )?;
        let resident = CudaContext::total_device_bytes();
        let cache = serde_json::to_value(model.q8_execution_stats())?;
        for (kind, rows) in [ProjectionKind::Qkv, ProjectionKind::Output]
            .into_iter()
            .flat_map(|kind| [1, 4].map(|rows| (kind, rows)))
        {
            for comparison in [
                ProjectionComparison::WarpReduction,
                ProjectionComparison::PackedLoads,
                ProjectionComparison::SplitPlanes,
            ] {
                {
                    let input = fixture::projection::make_input(rows, 64);
                    let probe = ProjectionProbe::new(&model, 0, rows, &input, kind, comparison)?;
                    assert_eq!(CudaContext::total_device_bytes(), resident + probe.bytes());
                    let checked = probe.validate()?;
                    if comparison == ProjectionComparison::SplitPlanes {
                        let expected_bytes = if kind == ProjectionKind::Qkv {
                            13056
                        } else {
                            4352
                        };
                        assert_eq!(
                            probe.preparation().unwrap()["additional_prepared_bytes"],
                            expected_bytes
                        );
                        assert_eq!(probe.bytes() - probe.scratch_bytes(), expected_bytes);
                    } else {
                        assert!(probe.preparation().is_none());
                        assert_eq!(probe.bytes(), probe.scratch_bytes());
                    }
                    for is_candidate in [false, true] {
                        let warmup = probe.warm(is_candidate, 1.0)?;
                        assert!(warmup["gpu_ms"].as_f64().unwrap() >= 1.0);
                        assert_eq!(warmup["minimum_gpu_ms"], 1.0);
                        assert!(
                            warmup["batches"].as_u64().unwrap()
                                <= warmup["maximum_batches"].as_u64().unwrap()
                        );
                    }
                    assert_eq!(checked["control_bits_equal"], true);
                    assert_eq!(checked["guards_and_immutable_read_spans"], true);
                    let before = profile::snapshot();
                    assert!(probe.time(true, 0).is_err());
                    assert_eq!(
                        profile::counter_delta(&before, &profile::snapshot()),
                        profile::counter_delta(&before, &before)
                    );
                    for is_candidate in [false, true, false, true] {
                        let timing = probe.time(is_candidate, 8)?;
                        assert_eq!(timing["operation_launches"], rows * 8);
                        assert_eq!(timing["operation"], probe.operation(is_candidate));
                        assert_eq!(timing["is_candidate"], is_candidate);
                        assert_eq!(
                            timing["has_packed_loads"],
                            comparison == ProjectionComparison::SplitPlanes
                                || (is_candidate
                                    && comparison == ProjectionComparison::PackedLoads)
                        );
                        if comparison == ProjectionComparison::SplitPlanes && is_candidate {
                            assert_eq!(timing["split_projection_dispatch"]["launches"], rows * 8);
                            assert_eq!(timing["split_projection_dispatch"]["threads"], 256);
                            assert_eq!(
                                timing["split_projection_dispatch"]["columns"],
                                if kind == ProjectionKind::Qkv { 1 } else { 4 }
                            );
                            assert_eq!(
                                timing["split_projection_dispatch"]["additional_prepared_bytes"],
                                probe.bytes() - probe.scratch_bytes()
                            );
                        } else {
                            assert!(timing["split_projection_dispatch"].is_null());
                        }
                        for key in [
                            "device_allocs",
                            "device_frees",
                            "h2d_copies",
                            "d2h_copies",
                            "stream_syncs",
                            "stream_waits",
                            "cublas_calls",
                        ] {
                            assert_eq!(timing["enqueue_counters"][key], 0);
                        }
                    }
                    probe.validate()?;
                    probe.verify()?;
                    assert_eq!(serde_json::to_value(model.q8_execution_stats())?, cache);
                }
                assert_eq!(CudaContext::total_device_bytes(), resident);
            }
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

use super::pipeline::{ProjectionKind, ProjectionProbe};
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
            {
                let input = fixture::projection::make_input(rows, 64);
                let probe = ProjectionProbe::new(&model, 0, rows, &input, kind)?;
                assert_eq!(CudaContext::total_device_bytes(), resident + probe.bytes());
                let checked = probe.validate()?;
                assert_eq!(checked["control_bits_equal"], true);
                assert_eq!(checked["guards_and_immutable_read_spans"], true);
                let before = profile::snapshot();
                assert!(probe.time(true, 0).is_err());
                assert_eq!(
                    profile::counter_delta(&before, &profile::snapshot()),
                    profile::counter_delta(&before, &before)
                );
                for is_warp in [false, true, false, true] {
                    let timing = probe.time(is_warp, 8)?;
                    assert_eq!(timing["operation_launches"], rows * 8);
                    assert_eq!(timing["operation"], kind.operation(is_warp));
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
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

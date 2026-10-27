#![cfg(all(feature = "cuda", nvcc))]
#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/env_restore.rs"]
mod env_restore;
#[path = "common/q8_model.rs"]
#[allow(dead_code)]
mod fixture;
use anyhow::Result;
use env_restore::EnvRestore;
use m40_llm::{
    cuda::{CudaContext, DeviceBuffer},
    decode_session::DecodeSession,
    infer::{
        execution_weights::Q8ExecutionConfig, q8_prefill::Q8PrefillBackend, LoadedModel,
        Q8AttentionProjectionBackend, Q8FfnConfig,
    },
    kv_compression::{KvCompressionConfig, ScopedRuntimeConfig},
    profile,
};

fn configure() -> Vec<EnvRestore> {
    [
        ("M40LLM_ENABLE_NVCC", "1"),
        ("M40LLM_PROJECTION_BACKEND", "large-model"),
        ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_ATTN_HEAD128_KERNEL", "native"),
        ("M40LLM_FUSED_Q8_QKV", "1"),
        ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
        ("M40LLM_DECODE_GRAPH", "0"),
    ]
    .into_iter()
    .map(|(k, v)| EnvRestore::set(k, v))
    .collect()
}
fn model(is_qwen: bool, ffn: Q8FfnConfig, execution: Q8ExecutionConfig) -> Result<LoadedModel> {
    let (gguf, bytes) = fixture::build_prefill_fixture(is_qwen);
    LoadedModel::from_gguf_with_q8_runtime(
        gguf,
        bytes,
        -1,
        execution,
        ffn,
        Q8PrefillBackend::Native,
        Q8AttentionProjectionBackend::PackedWarp256,
    )
}
fn config() -> Result<Q8ExecutionConfig> {
    Q8ExecutionConfig::parse("split", Some("6144"), Some("gate-up,down,vocabulary"))
}

/// The model-local compound preserves full-forward split-control logits and
/// original split keys for head64 and biased head128. Warm decode reuses all
/// owners without allocation/copy/sync in FFN dispatch or environment reselection.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_compound_runtime_matches_split_and_retains_owners() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let _scope = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    let initial = CudaContext::total_device_bytes();
    for is_qwen in [false, true] {
        let mut control = model(is_qwen, Q8FfnConfig::split_f32x4(), config()?)?;
        let mut candidate = model(is_qwen, Q8FfnConfig::joint_shared_scales(), config()?)?;
        let stats = candidate.q8_ffn_pair_stats();
        assert_eq!(stats.pair_count, 2);
        assert!(stats.is_enabled);
        assert_eq!(stats.retained_host_pair_bytes, 0);
        let bytes: usize = stats
            .pairs
            .iter()
            .map(|p| p.gate.shape().source_bytes().unwrap() * 2)
            .sum();
        assert_eq!(stats.resident_bytes, bytes);
        assert_eq!(
            stats.combined_prepared_bytes,
            candidate.q8_execution_stats().resident_bytes + bytes
        );
        let normalize = |stats| {
            let mut v = serde_json::to_value(stats).unwrap();
            v.as_object_mut().unwrap().remove("preparation_elapsed_ms");
            v.as_object_mut().unwrap().remove("model_hash_elapsed_ms");
            v
        };
        assert_eq!(
            normalize(control.q8_execution_stats()),
            normalize(candidate.q8_execution_stats())
        );
        assert_eq!(control.q8_ffn_pair_stats().resident_bytes, 0);
        control.allocate_kv_cache_for_layers(32)?;
        candidate.allocate_kv_cache_for_layers(32)?;
        let d = control.model_config.embedding_length as usize;
        let mut a = DecodeSession::new(
            &control,
            d,
            true,
            "compound-control",
            "compound:a-x",
            "compound:a-out",
        )?;
        let mut b = DecodeSession::new(
            &candidate,
            d,
            true,
            "compound-candidate",
            "compound:b-x",
            "compound:b-out",
        )?;
        let _changed_env = EnvRestore::set("M40LLM_Q8_FFN_KERNEL", "native");
        let frozen = serde_json::to_value(candidate.q8_ffn_pair_stats())?;
        for ids in [&[1, 7, 3][..], &[1, 7, 3, 4][..], &[1, 7, 3, 4, 5][..]] {
            let expected = a.logits_for_ids(ids, |_| {})?;
            let before = profile::snapshot();
            let resident = CudaContext::total_device_bytes();
            let actual = b.logits_for_ids(ids, |_| {})?;
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            let after = profile::snapshot();
            let launches = after
                .by_op
                .get("ffn_q8_joint_f32x4_shared_scales_128")
                .map_or(0, |v| v.launches)
                - before
                    .by_op
                    .get("ffn_q8_joint_f32x4_shared_scales_128")
                    .map_or(0, |v| v.launches);
            assert_eq!(launches, if ids.len() == 3 { 6 } else { 2 });
            if ids.len() > 3 {
                let delta = profile::counter_delta(&before, &after);
                assert_eq!(delta.device_allocs, 0);
                assert_eq!(delta.device_frees, 0);
                assert_eq!(CudaContext::total_device_bytes(), resident);
            }
        }
        assert_eq!(candidate.q8_ffn, Q8FfnConfig::joint_shared_scales());
        assert_eq!(serde_json::to_value(candidate.q8_ffn_pair_stats())?, frozen);
        let input = DeviceBuffer::new_tagged(&candidate.cuda, d * 4, "compound:invalid-input")?;
        let output = DeviceBuffer::new_tagged(
            &candidate.cuda,
            candidate.model_config.feed_forward_length as usize * 4,
            "compound:invalid-output",
        )?;
        let weights = candidate.map_standard_layer(0)?;
        let before = profile::snapshot();
        unsafe {
            assert!(candidate
                .try_mlp_gate_up_swiglu_f32xf16_gguf_decode_async(
                    input.as_ptr(),
                    2,
                    d as i32,
                    weights.w_gate.dptr,
                    weights.w_up.dptr,
                    candidate.model_config.feed_forward_length as i32,
                    output.as_mut_ptr(),
                    None
                )
                .is_err());
            assert!(candidate
                .try_mlp_gate_up_swiglu_f32xf16_gguf_decode_async(
                    input.as_ptr(),
                    1,
                    d as i32,
                    weights.w_up.dptr,
                    weights.w_gate.dptr,
                    candidate.model_config.feed_forward_length as i32,
                    output.as_mut_ptr(),
                    None
                )
                .is_err());
        }
        assert_eq!(
            profile::counter_delta(&before, &profile::snapshot()).launches,
            0
        );
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Candidate joint storage must be charged in addition to the retained split
/// cache. A one-byte-short budget fails cleanly and releases earlier owners.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_compound_combined_budget_failure_releases_model() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let initial = CudaContext::total_device_bytes();
    let required = {
        let loaded = model(false, Q8FfnConfig::joint_shared_scales(), config()?)?;
        loaded.q8_ffn_pair_stats().combined_prepared_bytes
    };
    assert_eq!(CudaContext::total_device_bytes(), initial);
    let mut limited = config()?;
    limited.budget_bytes = required - 1;
    let error = model(false, Q8FfnConfig::joint_shared_scales(), limited)
        .err()
        .expect("combined budget must reject");
    assert!(error.to_string().contains("budget"));
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

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
    cuda::CudaContext,
    infer::{
        execution_weights::Q8ExecutionConfig, q8_prefill::Q8PrefillBackend, LoadedModel,
        Q8FfnConfig,
    },
    verification::VerificationSession,
};

fn configure() -> Vec<EnvRestore> {
    [
        ("M40LLM_PROJECTION_BACKEND", "large-model"),
        ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_ATTN_HEAD128_KERNEL", "native"),
        ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
        ("M40LLM_FUSED_Q8_QKV", "1"),
        ("M40LLM_Q8_LM_HEAD_ARGMAX", "0"),
        ("M40LLM_F16_LM_HEAD_ARGMAX", "0"),
        ("M40LLM_DECODE_GRAPH", "0"),
    ]
    .into_iter()
    .map(|(k, v)| EnvRestore::set(k, v))
    .collect()
}
fn model(is_qwen: bool, backend: Q8PrefillBackend, capacity: u32) -> Result<LoadedModel> {
    let (gguf, bytes) = fixture::build_prefill_fixture(is_qwen);
    let mut model = LoadedModel::from_gguf_with_q8_backends(
        gguf,
        bytes,
        -1,
        Q8ExecutionConfig::parse("split", Some("6144"), Some("gate-up,down,vocabulary"))?,
        Q8FfnConfig::split_f32x4(),
        backend,
    )?;
    model.allocate_kv_cache_for_layers(capacity)?;
    Ok(model)
}
fn compare_logits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-3 + 2e-5 * b.abs(),
            "logit {i}: {a} vs {b}"
        );
    }
}
fn compare_kv(actual: &(Vec<u16>, Vec<u16>), expected: &(Vec<u16>, Vec<u16>)) {
    for (&a, &b) in actual
        .0
        .iter()
        .chain(&actual.1)
        .zip(expected.0.iter().chain(&expected.1))
    {
        let a = half::f16::from_bits(a).to_f32();
        let b = half::f16::from_bits(b).to_f32();
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-3,
            "KV {a} vs {b}"
        );
    }
}

/// Real causal continuation rows retain serial logits, greedy IDs and dense
/// KV at absolute positions for head64 and biased split-half-RoPE head128.
/// Rollback clears every rejected suffix without changing the old prefix.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_verification_causal_rows_and_rollback_match_serial() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let initial = CudaContext::total_device_bytes();
    let mut backends = vec![Q8PrefillBackend::Native];
    if cfg!(have_cublas) {
        backends.push(Q8PrefillBackend::DequantSgemm);
    }
    for is_qwen in [false, true] {
        for backend in backends.iter().copied() {
            let mut serial_model = model(is_qwen, backend, 32)?;
            let mut packed_model = model(is_qwen, backend, 32)?;
            let mut serial = VerificationSession::new(&mut serial_model, &[])?;
            let mut packed = VerificationSession::new(&mut packed_model, &[])?;
            let prompt = [1, 7, 3, 9, 4];
            for count in [2, 4, 8] {
                let mut next = serial.prefill(&prompt)?;
                let mut candidates = Vec::new();
                let mut logits = Vec::new();
                let mut predictions = Vec::new();
                for _ in 0..count {
                    candidates.push(next);
                    let (n, l) = serial.serial_step_with_logits(next)?;
                    next = n;
                    predictions.push(n);
                    logits.push(l);
                }
                packed.prefill(&prompt)?;
                let prefix = packed.debug_kv_token(0, 1)?;
                let attempt = packed.begin(&candidates, true)?;
                assert_eq!(attempt.accepted_len(), count);
                assert_eq!(attempt.predictions(), predictions);
                for (a, b) in attempt.captured_logits().unwrap().iter().zip(&logits) {
                    compare_logits(a, b);
                }
                let outcome = attempt.commit()?;
                assert_eq!(outcome.accepted_tokens, candidates);
                assert_eq!(outcome.next_token, next);
                assert_eq!(
                    packed.debug_lengths()?,
                    vec![(prompt.len() + count) as u32; 2]
                );
                for layer in 0..2 {
                    for position in 0..prompt.len() + count {
                        compare_kv(
                            &packed.debug_kv_token(layer, position as u32)?,
                            &serial.debug_kv_token(layer, position as u32)?,
                        );
                    }
                }
                for reject in 0..count {
                    packed.prefill(&prompt)?;
                    let mut wrong = candidates.clone();
                    wrong[reject] = (wrong[reject] + 1) % 96;
                    if wrong[reject] == candidates[reject] {
                        wrong[reject] = (wrong[reject] + 1) % 96;
                    }
                    let attempt = packed.begin(&wrong, false)?;
                    assert_eq!(attempt.accepted_len(), reject);
                    let out = attempt.commit()?;
                    assert_eq!(out.accepted_tokens, &candidates[..reject]);
                    assert_eq!(out.next_token, candidates[reject]);
                    assert_eq!(
                        packed.debug_lengths()?,
                        vec![(prompt.len() + reject) as u32; 2]
                    );
                    assert_eq!(packed.debug_kv_token(0, 1)?, prefix);
                    for layer in 0..2 {
                        for position in prompt.len() + reject..prompt.len() + count {
                            let (k, v) = packed.debug_kv_token(layer, position as u32)?;
                            assert!(k.iter().chain(&v).all(|&b| b == 0));
                        }
                    }
                    assert_eq!(packed.serial_step(candidates[reject])?, predictions[reject]);
                }
                packed.prefill(&prompt)?;
                drop(packed.begin(&candidates, false)?);
                assert_eq!(packed.committed_len(), prompt.len() as u32);
                assert_eq!(packed.debug_lengths()?, vec![prompt.len() as u32; 2]);
                let attempt = packed.begin(&candidates, false)?;
                let out = attempt.commit_prefix(1)?;
                assert_eq!(out.accepted_tokens, &candidates[..1]);
            }
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// A stop token caps acceptance; invalid commits and capacity overruns cannot
/// publish speculative state, and aliased caches are rejected before work.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_verification_rejects_aliases_capacity_and_unverified_commits() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env = configure();
    let initial = CudaContext::total_device_bytes();
    {
        let mut loaded = model(false, Q8PrefillBackend::Native, 8)?;
        let alias = loaded.kv_cache.as_ref().unwrap().clone();
        assert!(VerificationSession::new(&mut loaded, &[]).is_err());
        drop(alias);
        let prompt = [1, 2, 3, 4, 5];
        let first = {
            let mut session = VerificationSession::new(&mut loaded, &[])?;
            session.prefill(&prompt)?
        };
        let mut session = VerificationSession::new(&mut loaded, &[first])?;
        session.prefill(&prompt)?;
        assert!(session.begin(&[first; 4], false).is_err());
        assert_eq!(session.debug_lengths()?, vec![5; 2]);
        let wrong = (first + 1) % 96;
        assert!(session
            .begin(&[wrong, first], false)?
            .commit_prefix(1)
            .is_err());
        assert_eq!(session.debug_lengths()?, vec![5; 2]);
        let out = session.begin(&[first, first], false)?.commit()?;
        assert_eq!(out.accepted_tokens, vec![first]);
        assert!(out.has_stopped);
        assert_eq!(session.debug_lengths()?, vec![6; 2]);
        let (k, v) = session.debug_kv_token(0, 6)?;
        assert!(k.iter().chain(&v).all(|&b| b == 0));
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

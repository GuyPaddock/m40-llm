#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "common/tiny_gguf.rs"]
mod fixture;
use anyhow::Result;
use m40_llm::{
    cuda::CudaContext,
    evaluation::teacher_forced::{compare_loaded_models, score_loaded_model},
    infer::LoadedModel,
    kv_compression::{KvCompressionConfig, ScopedRuntimeConfig},
};

fn model() -> Result<LoadedModel> {
    let (gguf, weights) = fixture::make_tied_identity_tiny_gguf(fixture::TinyGgufConfig {
        vocab: 96,
        d_model: 64,
        hidden: 128,
        head_count: 1,
        block_count: 2,
        context_length: 32,
    });
    let mut model = LoadedModel::from_gguf(gguf, weights, -1)?;
    model.allocate_kv_cache_for_layers(32)?;
    Ok(model)
}

/// The single-model scorer must reset dense KV, agree exactly with the paired
/// evaluator's reference, reject invalid input and preserve subsequent usability.
#[test]
#[ignore = "requires CUDA hardware; run with --include-ignored --test-threads=1"]
fn test_single_model_cuda_loss_matches_paired_reference() -> Result<()> {
    let _scope = cuda_test_scope::lock_cuda();
    let _dense = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let mut reference = model()?;
        let mut candidate = model()?;
        let ids = [1, 4, 3, 8, 7];
        let expected = compare_loaded_models(&mut reference, &mut candidate, &ids)?.reference;
        assert_eq!(score_loaded_model(&mut reference, &ids)?, expected);
        score_loaded_model(&mut reference, &[2, 9, 6])?;
        assert_eq!(score_loaded_model(&mut reference, &ids)?, expected);
        let before = CudaContext::total_device_bytes();
        assert!(score_loaded_model(&mut reference, &[1]).is_err());
        assert!(score_loaded_model(&mut reference, &[96, 1]).is_err());
        assert!(score_loaded_model(&mut reference, &[1; 33]).is_err());
        assert_eq!(CudaContext::total_device_bytes(), before);
        assert_eq!(score_loaded_model(&mut reference, &ids)?, expected);
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

/// Independent CUDA sessions must produce identical full logits for the same
/// model, reset KV between passages, preserve failure cleanup, and release scratch.
#[test]
#[ignore = "requires CUDA hardware; run with --include-ignored --test-threads=1"]
fn test_cuda_teacher_forcing_isolated_sessions_and_reset() -> Result<()> {
    let _scope = cuda_test_scope::lock_cuda();
    let _dense = ScopedRuntimeConfig::new(KvCompressionConfig::dense_reference());
    let initial = CudaContext::total_device_bytes();
    {
        let mut reference = model()?;
        let mut candidate = model()?;
        let ids = [1, 4, 3, 8, 7];
        let first = compare_loaded_models(&mut reference, &mut candidate, &ids)?;
        assert_eq!(first.max_abs_logit_diff, 0.0);
        assert_eq!(first.reference, first.candidate);
        assert_eq!(first.reference.scored_tokens, 4);
        compare_loaded_models(&mut reference, &mut candidate, &[2, 9, 6])?;
        let repeated = compare_loaded_models(&mut reference, &mut candidate, &ids)?;
        assert_eq!(first, repeated);
        let before = CudaContext::total_device_bytes();
        assert!(compare_loaded_models(&mut reference, &mut candidate, &[1; 33]).is_err());
        assert_eq!(CudaContext::total_device_bytes(), before);
        assert!(compare_loaded_models(&mut reference, &mut candidate, &[96, 1]).is_err());
        assert_eq!(
            compare_loaded_models(&mut reference, &mut candidate, &ids)?,
            first
        );
        let own_cache = candidate.kv_cache.take();
        candidate.kv_cache = reference.kv_cache.clone();
        let before = CudaContext::total_device_bytes();
        assert!(compare_loaded_models(&mut reference, &mut candidate, &ids).is_err());
        assert_eq!(CudaContext::total_device_bytes(), before);
        candidate.kv_cache = own_cache;
        assert_eq!(
            compare_loaded_models(&mut reference, &mut candidate, &ids)?,
            first
        );
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

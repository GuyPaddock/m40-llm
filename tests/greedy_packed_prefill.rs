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
    generate::{generate_text, GenerateOptions, PromptFormat},
    infer::{execution_weights::Q8ExecutionConfig, LoadedModel},
    kv_compression::KvCompressionConfig,
    profile,
};

/// An explicit prefix bound must reach native packed attention even with GPU
/// greedy enabled. Zero keeps sequential dispatch, and no implicit packed
/// admission changes the reference. Exercise actual generation and launches.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_gpu_greedy_honors_explicit_packed_prefix_bound() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _env: Vec<_> = [
        ("M40LLM_ENABLE_NVCC", "1"),
        ("M40LLM_PROJECTION_BACKEND", "large-model"),
        ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_Q8_DECODE_COL_TILE", "4"),
        ("M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK", "4"),
        ("M40LLM_FUSED_Q8_QKV", "1"),
        ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
        ("M40LLM_CUDA_GREEDY_ARGMAX", "1"),
        ("M40LLM_DECODE_GRAPH", "0"),
    ]
    .into_iter()
    .map(|(key, value)| EnvRestore::set(key, value))
    .collect();
    let _hardware = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let (mut gguf, bytes) = fixture::build_with_layers(0, 2);
        gguf.metadata.remove("tokenizer.ggml.model");
        let mut model = LoadedModel::from_gguf_with_q8_execution(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
        )?;
        model.allocate_kv_cache_for_layers(32)?;
        for (bound, mode, launches) in [("0", "sequential", 0), ("8", "packed-prefix", 2)] {
            let _bound = EnvRestore::set("M40LLM_PREFILL_CHUNK_SIZE", bound);
            let before = profile::snapshot();
            let generated = generate_text(
                &model,
                GenerateOptions {
                    prompt: "ABC".to_string(),
                    max_tokens: Some(1),
                    top_k: Some(1),
                    seed: Some(0),
                    kv_compression: KvCompressionConfig::dense_reference(),
                    prompt_format: PromptFormat::Raw,
                    ..Default::default()
                },
            )?;
            let after = profile::snapshot();
            assert_eq!(generated.metrics.prompt_tokens, 3);
            assert_eq!(generated.metrics.generated_tokens, 1);
            assert_eq!(generated.prefill_mode, mode);
            let operation = "attention_prefill_f32_gqa_varlen";
            assert_eq!(
                after.by_op.get(operation).map_or(0, |v| v.launches)
                    - before.by_op.get(operation).map_or(0, |v| v.launches),
                launches
            );
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

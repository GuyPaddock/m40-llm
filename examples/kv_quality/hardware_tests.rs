#[path = "../../tests/common/cuda_test_scope.rs"]
mod cuda_test_scope;
#[path = "../../tests/common/env_restore.rs"]
mod env_restore;
#[path = "../../tests/common/q8_model.rs"]
#[allow(dead_code)]
mod q8_model;

use super::{
    config::{CacheArm, ENVIRONMENT},
    prefix::collect_prefixes,
};
use anyhow::Result;
use m40_llm::{
    cuda::{CudaContext, CudaStream},
    infer::{
        execution_weights::Q8ExecutionConfig, q8_prefill::Q8PrefillBackend, LoadedModel,
        Q8AttentionProjectionBackend, Q8FfnConfig,
    },
};

/// The real packed-prefix/session path populates both recent and old mixed KV,
/// identical coverage gives identical results, and a future target cannot alter
/// its own prediction. All tracked model/workspace/cache ownership is released.
#[test]
#[ignore = "serial M40 real-model KV evaluation pipeline smoke"]
fn test_prefix_pipeline_preserves_coverage_and_causality() -> Result<()> {
    let _lock = cuda_test_scope::lock_cuda();
    let _environment: Vec<_> = ENVIRONMENT
        .iter()
        .map(|&(key, value)| env_restore::EnvRestore::set(key, value))
        .collect();
    let ctx = CudaContext::new_m40()?;
    let initial = CudaContext::total_device_bytes();
    {
        let (gguf, bytes) = q8_model::build_prefill_fixture(true);
        let mut model = LoadedModel::from_gguf_with_q8_runtime(
            gguf,
            bytes,
            -1,
            Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
            Q8FfnConfig::native(),
            Q8PrefillBackend::DequantSgemm,
            Q8AttentionProjectionBackend::Native,
        )?;
        let prompt: Vec<u32> = (0..1057)
            .map(|index| (index * 13 % 300 + 1) as u32)
            .collect();
        let dense = collect_prefixes(&mut model, CacheArm::Dense, &prompt, &[4, 7, 9])?;
        let all = collect_prefixes(&mut model, CacheArm::QuantizedAll, &prompt, &[4, 7, 9])?;
        let top8 = collect_prefixes(&mut model, CacheArm::QuantizedTop8, &prompt, &[4, 7, 9])?;
        assert_eq!(all.state["old_tokens"], 35);
        assert_eq!(all.state["old_blocks"], 2);
        assert_eq!(
            all.logits, top8.logits,
            "same two old blocks must give identical observations"
        );
        assert_eq!(dense.steps, vec![0, 1, 2]);
        let changed_target =
            collect_prefixes(&mut model, CacheArm::QuantizedAll, &prompt, &[4, 7, 10])?;
        assert_eq!(
            all.logits, changed_target.logits,
            "future target leaked into its prediction"
        );
        let short_dense =
            collect_prefixes(&mut model, CacheArm::Dense, &[1, 2, 3, 4, 5, 6], &[7, 8, 9])?;
        let short_mixed = collect_prefixes(
            &mut model,
            CacheArm::QuantizedAll,
            &[1, 2, 3, 4, 5, 6],
            &[7, 8, 9],
        )?;
        assert_eq!(short_mixed.state["old_tokens"], 0);
        for (a, b) in short_dense
            .logits
            .iter()
            .flatten()
            .zip(short_mixed.logits.iter().flatten())
        {
            assert!(
                (a - b).abs() <= 2e-3,
                "no-old fixture changed logits beyond existing forward tolerance"
            );
        }
    }
    ctx.synchronize_stream(CudaStream::Decode)?;
    ctx.synchronize_stream(CudaStream::Prefill)?;
    assert_eq!(
        CudaContext::total_device_bytes(),
        initial,
        "tracked allocation leak"
    );
    Ok(())
}

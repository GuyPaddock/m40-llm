#![cfg(all(feature = "cuda", nvcc))]

#[path = "common/env_restore.rs"]
mod env_restore;
#[path = "common/q8_model.rs"]
mod fixture;
#[path = "common/q8_greedy.rs"]
mod greedy;

use anyhow::{ensure, Result};
use env_restore::EnvRestore;
use m40_llm::{
    cuda::CudaContext,
    infer::{
        execution_weights::{Q8ExecutionConfig, Q8ExecutionFamilies, Q8ExecutionMode},
        LoadedModel,
    },
};

/// Prepared vocabulary weights must override native fused-head dispatch without
/// disabling device greedy sampling, changing tie handling, or copying logits.
/// Both streams and cache modes agree with native and full-logits references.
#[test]
fn test_prepared_greedy_uses_cached_vocabulary_and_one_token_readback() -> Result<()> {
    let _fused = EnvRestore::set("M40LLM_Q8_LM_HEAD_ARGMAX", "1");
    let initial = CudaContext::total_device_bytes();
    for is_shared in [false, true] {
        let _shared = EnvRestore::set(
            "M40LLM_Q8_LM_HEAD_ARGMAX_SHARED_A",
            if is_shared { "1" } else { "0" },
        );
        for stream in ["prefill", "decode"] {
            let _stream = EnvRestore::set("M40LLM_Q8_DECODE_STREAM", stream);
            let (gguf, bytes) = fixture::build(0);
            let native = LoadedModel::from_gguf_with_q8_execution(
                gguf,
                bytes,
                -1,
                Q8ExecutionConfig::disabled(),
            )?;
            let props = native.cuda.current_device_props()?;
            ensure!(
                props.major == 5 && props.minor == 2,
                "greedy test requires sm_52"
            );
            for mode in [Q8ExecutionMode::Split, Q8ExecutionMode::Mixed] {
                let (gguf, bytes) = fixture::build(0);
                let model = LoadedModel::from_gguf_with_q8_execution(
                    gguf,
                    bytes,
                    -1,
                    Q8ExecutionConfig {
                        mode,
                        budget_bytes: 1024 * 1024,
                        families: Q8ExecutionFamilies::all(),
                    },
                )?;
                assert!(model.q8_execution_stats().has_prepared_vocabulary);
                for hidden in [vec![0.0; 64], fixture::projection::make_input(1, 64)] {
                    let (expected, _) = greedy::compare_greedy_with_logits(&native, &hidden)?;
                    let (actual, logits) = greedy::compare_greedy_with_logits(&model, &hidden)?;
                    assert_eq!(actual, expected);
                    if hidden.iter().all(|&value| value == 0.0) {
                        assert_eq!(actual, 0, "ties choose the first vocabulary token");
                    }
                    let (name, _, _, vocab, _) = model.map_lm_head()?;
                    let tensor = &model.device_tensors[&name];
                    let source = &model.host_weights
                        [tensor.byte_offset as usize..tensor.byte_offset as usize + tensor.nbytes];
                    let mut best = (0_u32, f64::NEG_INFINITY);
                    let mut cpu_logits = Vec::new();
                    for col in 0..vocab {
                        let value = fixture::projection::project_cpu(&hidden, source, 0, col, 64);
                        cpu_logits.push(value as f32);
                        if value > best.1 {
                            best = (col as u32, value);
                        }
                    }
                    let (maximum, mean) = fixture::projection::measure_error(&logits, &cpu_logits);
                    ensure!(
                        maximum < 1e-3 && mean < 1e-4,
                        "vocabulary projection CPU mismatch"
                    );
                    assert_eq!(actual, best.0, "greedy differs from CPU Q8 projection");
                }
            }
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), initial);
    Ok(())
}

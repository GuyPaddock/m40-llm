use super::{
    config::CacheArm,
    execution::{allocate_cache, read_cache_state, summarize_dispatch, verify_dispatch},
    metrics::select_observation_steps,
};
use anyhow::{ensure, Result};
use m40_llm::{
    decode_session::DecodeSession, infer::LoadedModel, kv_compression::ScopedRuntimeConfig, profile,
};
use serde_json::Value;

pub struct PrefixObservations {
    pub steps: Vec<usize>,
    pub logits: Vec<Vec<f32>>,
    pub dispatch: Value,
    pub state: Value,
}

/// Keep one exclusively borrowed model/session per arm. Each prediction sees
/// the prompt and only the earlier dense-generated tokens, including at gaps.
pub fn collect_prefixes(
    model: &mut LoadedModel,
    arm: CacheArm,
    prompt: &[u32],
    continuation: &[u32],
) -> Result<PrefixObservations> {
    allocate_cache(model, arm)?;
    let steps = select_observation_steps(continuation.len())?;
    let config = arm.create_config();
    let _scope = ScopedRuntimeConfig::new(config.clone());
    let dimension = model.validate_full_layer_decode()?.0;
    let before = profile::snapshot();
    let mut logits = Vec::new();
    {
        let mut session = DecodeSession::new_for_sequence_with_kv_config(
            model,
            0,
            config,
            dimension,
            true,
            "kv_quality",
            "kv_quality:x",
            "kv_quality:output",
        )?;
        session.prefill_packed_prefix_ids(prompt)?;
        for &step in &steps {
            let mut ids = prompt.to_vec();
            ids.extend_from_slice(&continuation[..step]);
            let values = session.logits_for_ids(&ids, |_| {})?;
            ensure!(
                values.len() == model.model_config.vocab_size as usize
                    && values.iter().all(|v| v.is_finite()),
                "invalid full logits"
            );
            ensure!(
                session.processed_len() == ids.len(),
                "prefix did not consume exactly its history"
            );
            logits.push(values);
        }
    }
    let dispatch = summarize_dispatch(&before);
    verify_dispatch(
        &dispatch,
        arm,
        continuation.len(),
        model.model_config.block_count,
    )?;
    let state = read_cache_state(model, arm, prompt.len() + continuation.len() - 1)?;
    Ok(PrefixObservations {
        steps,
        logits,
        dispatch,
        state,
    })
}

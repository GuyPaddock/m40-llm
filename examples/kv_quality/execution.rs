use super::{
    config::CacheArm,
    fixture::{Case, BLOCK, CAPACITY, RECENT},
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::{CudaContext, CudaStream},
    generate::{generate_text, GenerateOptions, GeneratedText, PromptFormat, PromptPrefill},
    gguf,
    infer::{
        execution_weights::{Q8ExecutionConfig, Q8SeparateLaunch},
        q8_prefill::Q8PrefillBackend,
        LoadedModel, Q8AttentionProjectionBackend, Q8FfnConfig,
    },
    profile::{self, ProfileSnapshot},
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path};

pub fn load_model(path: &Path) -> Result<LoadedModel> {
    let mut model = LoadedModel::from_gguf_with_q8_runtime(
        gguf::load_gguf(path)?,
        fs::read(path)?,
        -1,
        Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
        Q8FfnConfig::native(),
        Q8PrefillBackend::DequantSgemm,
        Q8AttentionProjectionBackend::Native,
    )?;
    let props = model.cuda.current_device_props()?;
    ensure!(
        props.major == 5 && props.minor == 2,
        "requires Maxwell sm_52"
    );
    let config = &model.model_config;
    ensure!(
        config.architecture == "qwen2"
            && config.block_count == 36
            && config.embedding_length == 2048
            && config.feed_forward_length == 11008
            && config.attention_key_length == 128
            && config.context_length as usize >= CAPACITY,
        "unexpected Qwen model shape"
    );
    let stats = model.q8_execution_stats();
    ensure!(
        stats.has_prepared_vocabulary
            && stats.single_row_down_split_launch == Some(Q8SeparateLaunch::new(4, 256)?),
        "missing prepared control"
    );
    ensure!(
        model.q8_ffn.is_native()
            && model.q8_prefill == Q8PrefillBackend::DequantSgemm
            && model.q8_attention_projection.backend() == Q8AttentionProjectionBackend::Native,
        "unexpected projection/prefill configuration"
    );
    allocate_cache(&mut model, CacheArm::Dense)?;
    model.validate_full_layer_decode()?;
    Ok(model)
}

pub fn allocate_cache(model: &mut LoadedModel, arm: CacheArm) -> Result<()> {
    model.cuda.synchronize_stream(CudaStream::Decode)?;
    model.cuda.synchronize_stream(CudaStream::Prefill)?;
    let config = arm.create_config();
    ensure!(
        config.recent_window as usize == RECENT && config.block_size as usize == BLOCK,
        "cache bounds changed"
    );
    if arm == CacheArm::Dense {
        model.allocate_kv_cache_for_layers(CAPACITY as u32)?;
    } else {
        model.allocate_compressed_kv_cache_for_layers(CAPACITY as u32, &config)?;
    }
    let cache = model.kv_cache.as_ref().context("allocated cache")?;
    ensure!(
        cache.is_compressed() == (arm != CacheArm::Dense),
        "wrong cache allocation"
    );
    ensure!(
        cache.max_seq_len() as usize == CAPACITY
            && cache.max_batch_size() == model.model_config.block_count,
        "wrong cache capacity/slots"
    );
    if arm != CacheArm::Dense {
        ensure!(
            cache.exact_old_backing() == "fp16-k-q4-v",
            "wrong mixed backing"
        );
    }
    Ok(())
}

pub fn summarize_dispatch(before: &ProfileSnapshot) -> Value {
    let after = profile::snapshot();
    let operations: BTreeMap<_, _> = after
        .by_op
        .iter()
        .filter_map(|(name, counts)| {
            let previous = before.by_op.get(name).copied().unwrap_or_default();
            let launches = counts.launches - previous.launches;
            let cublas = counts.cublas_calls - previous.cublas_calls;
            (launches != 0 || cublas != 0)
                .then_some((*name, json!({"launches":launches,"cublas_calls":cublas})))
        })
        .collect();
    json!({"totals":profile::counter_delta(before,&after),"operations":operations})
}

pub fn verify_dispatch(
    dispatch: &Value,
    arm: CacheArm,
    one_row_forwards: usize,
    layers: u32,
) -> Result<()> {
    ensure!(
        dispatch["operations"][arm.get_attention_operation()]["launches"].as_u64()
            == Some(one_row_forwards as u64 * u64::from(layers)),
        "requested attention did not execute every one-row layer forward"
    );
    ensure!(
        dispatch["totals"]["cublas_calls"].as_u64() == Some(7 * u64::from(layers)),
        "prefill did not execute every dequant-SGEMM projection"
    );
    Ok(())
}

pub fn read_cache_state(model: &LoadedModel, arm: CacheArm, consumed: usize) -> Result<Value> {
    let cache = model.kv_cache.as_ref().context("allocated cache")?;
    let old_tokens = consumed.saturating_sub(RECENT);
    let old_blocks = old_tokens.div_ceil(BLOCK);
    ensure!(
        consumed <= CAPACITY && old_blocks <= 64,
        "lost full old-block coverage"
    );
    let mut lengths = Vec::new();
    for layer in 0..model.model_config.block_count {
        let physical = model.kv_cache_physical_slot_for_layer_sequence(layer, 0)?;
        let length = cache.debug_sequence_length(&model.cuda, physical)?;
        ensure!(
            length as usize == consumed,
            "layer {layer} has incorrect consumed length"
        );
        lengths.push(length);
        if arm != CacheArm::Dense {
            let snapshot = cache.debug_compressed_snapshot(&model.cuda, physical)?;
            ensure!(snapshot.seq_len == length, "compressed length disagrees");
            for (index, &count) in snapshot.block_counts.iter().enumerate() {
                let expected = old_tokens.saturating_sub(index * BLOCK).min(BLOCK);
                ensure!(
                    count as usize == expected,
                    "layer {layer} old block {index} has {count}, expected {expected}"
                );
            }
            ensure!(
                snapshot
                    .block_counts
                    .iter()
                    .map(|&n| n as usize)
                    .sum::<usize>()
                    == old_tokens,
                "old tokens missing from cache"
            );
        }
    }
    Ok(
        json!({"consumed_tokens":consumed,"physical_lengths":lengths,"is_compressed":cache.is_compressed(),
        "old_tokens":old_tokens,"old_blocks":old_blocks,"all_old_blocks_fit_top64":old_blocks<=64,
        "configured_old_block_budget":if arm==CacheArm::Dense {old_blocks} else {old_blocks.min(arm.create_config().top_blocks as usize)},
        "allocated_bytes":cache.actual_bytes(),"dense_equivalent_bytes":cache.dense_equivalent_bytes(),
        "tracked_device_bytes":CudaContext::total_device_bytes(),"selection_scope":"budget under the fixed tested plain-top-k kernel contract, not a telemetry dump of selected indices"}),
    )
}

pub fn generate_case(
    model: &mut LoadedModel,
    case: &Case,
    arm: CacheArm,
) -> Result<(GeneratedText, Value, Value)> {
    allocate_cache(model, arm)?;
    let before = profile::snapshot();
    let generated = generate_text(
        model,
        GenerateOptions {
            prompt: case.prompt.clone(),
            prompt_format: PromptFormat::Raw,
            top_k: Some(1),
            seed: Some(0),
            max_tokens: Some(case.task.max_tokens),
            kv_compression: arm.create_config(),
            reset_kv_cache: true,
            prompt_prefill: PromptPrefill::PackedPrefix {
                max_prompt_tokens: 2112,
            },
            ..Default::default()
        },
    )?;
    ensure!(
        generated.prompt_token_len == case.prompt_ids.len()
            && generated.token_ids[..generated.prompt_token_len] == case.prompt_ids,
        "changed prompt tokenization"
    );
    let count = generated.token_ids.len() - generated.prompt_token_len;
    ensure!(
        count > 0 && count <= case.task.max_tokens && count == generated.metrics.generated_tokens,
        "invalid generated token accounting"
    );
    let expected_prefill = if arm == CacheArm::Dense {
        "packed-prefix-dequant-sgemm"
    } else {
        "packed-prefix-block-select-exact-dequant-sgemm"
    };
    ensure!(
        generated.prefill_mode == expected_prefill
            && generated.q8_prefill_temporary_weight_bytes.is_some(),
        "unexpected prefill path: {}",
        generated.prefill_mode
    );
    let dispatch = summarize_dispatch(&before);
    verify_dispatch(&dispatch, arm, count, model.model_config.block_count)?;
    let state = read_cache_state(model, arm, generated.token_ids.len() - 1)?;
    Ok((generated, dispatch, state))
}

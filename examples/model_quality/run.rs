use super::{
    compact::{self, CompactComparison},
    records::{self, Report},
    step_timing, tasks, Args,
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaContext,
    evaluation::{
        corpus::{Corpus, Split},
        loss::LossMetrics,
        teacher_forced::compare_loaded_models,
    },
    gguf,
    infer::{execution_weights::Q8ExecutionConfig, LoadedModel},
    tokenizer::Tokenizer,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub(super) fn environment() -> Result<BTreeMap<String, String>> {
    let environment: BTreeMap<_, _> = std::env::vars()
        .filter(|(key, _)| {
            key.starts_with("M40LLM_") || key.starts_with("CUDA_") || key == "OMP_NUM_THREADS"
        })
        .collect();
    for (key, expected) in [
        ("M40LLM_ENABLE_NVCC", "1"),
        ("M40LLM_PROJECTION_BACKEND", "large-model"),
        ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
        ("M40LLM_Q8_EXECUTION_WEIGHTS", "off"),
        ("M40LLM_DECODE_GRAPH", "0"),
    ] {
        ensure!(
            environment.get(key).map(String::as_str) == Some(expected),
            "quality checkpoint requires {key}={expected}"
        );
    }
    ensure!(
        !environment.keys().any(|key| key.starts_with("M40LLM_KV_")),
        "unset KV diagnostics/policies for dense model-quality evaluation"
    );
    Ok(environment)
}

pub(super) fn load_model(
    path: &Path,
    context_tokens: u32,
    skipped_layers: &[usize],
    activation: super::activations::ActivationSetup<'_>,
) -> Result<(LoadedModel, Tokenizer)> {
    load_model_with_execution(
        path,
        context_tokens,
        skipped_layers,
        activation,
        Q8ExecutionConfig::disabled(),
    )
}

pub(super) fn load_model_with_execution(
    path: &Path,
    context_tokens: u32,
    skipped_layers: &[usize],
    activation: super::activations::ActivationSetup<'_>,
    execution: Q8ExecutionConfig,
) -> Result<(LoadedModel, Tokenizer)> {
    let gguf = gguf::load_gguf(path)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata)?;
    ensure!(
        tokenizer.is_qwen2() || tokenizer.is_llama3(),
        "quality runner requires supported Qwen2 or Llama3 tokenizer/chat formatting"
    );
    let mut model = LoadedModel::from_gguf_with_q8_execution(gguf, fs::read(path)?, -1, execution)?;
    let props = model.cuda.current_device_props()?;
    ensure!(
        props.major == 5 && props.minor == 2,
        "quality checkpoint requires sm_52"
    );
    ensure!(
        context_tokens <= model.model_config.context_length,
        "requested context exceeds model context"
    );
    model.set_diagnostic_layer_mask(skipped_layers)?;
    activation.configure(&mut model)?;
    model.allocate_kv_cache_for_layers(context_tokens)?;
    model.validate_full_layer_decode()?;
    Ok((model, tokenizer))
}

pub fn memory(model: &LoadedModel) -> Result<Value> {
    let cache = model.kv_cache.as_ref().context("allocated KV")?;
    let uploaded_weights = model.cuda.uploaded_weights_bytes()?;
    ensure!(
        uploaded_weights == model.weights_len && !model.d_weights_base.is_null(),
        "quality memory accounting requires successfully uploaded device weights"
    );
    let materialized = model
        .materialized_weights
        .lock()
        .map_err(|_| anyhow::anyhow!("poisoned weight cache"))?;
    let materialized_bytes: usize = materialized.values().map(|weight| weight.bytes).sum();
    Ok(json!({
        "uploaded_weight_payload_bytes":model.weights_len,
        "uploaded_weight_allocation_bytes":uploaded_weights,
        "source_layer_count":model.model_config.block_count,
        "active_layer_count":model.active_layer_count()?,
        "has_in_process_compaction":false,
        "resident_kv_bytes":cache.actual_bytes(),
        "kv_accounting_scope":"native cache K/V payload capacity after successful allocation; excludes native length metadata and CUDA allocator overhead",
        "physical_kv_slots":cache.max_batch_size(),
        "allocated_context_tokens":cache.max_seq_len(),
        "head128_attention_workspace_bytes":cache.head128_attention_workspace_bytes(),
        "prepared_weight_bytes":model.q8_execution_stats().resident_bytes,
        "activation_sparsity":model.activation_research_stats()?,
        "materialized_f32_bytes":materialized_bytes,
        "combined_tracked_device_bytes":CudaContext::total_device_bytes(),
        "scope":"per-model requested storage; tracked total covers ALL models' Rust-tracked weights/workspaces, with native KV separate; excludes driver/library overhead; not standalone peak VRAM"
    }))
}

pub(super) fn model_identity(
    path: &Path,
    model: &LoadedModel,
    tokenizer: &Tokenizer,
) -> Result<Value> {
    let config = &model.model_config;
    let dtype_counts =
        model
            .gguf
            .tensors
            .iter()
            .fold(BTreeMap::<String, usize>::new(), |mut counts, tensor| {
                *counts.entry(format!("{:?}", tensor.dtype)).or_default() += 1;
                counts
            });
    Ok(
        json!({"path":fs::canonicalize(path)?,"sha256":records::file_sha256(path)?,
        "file_bytes":fs::metadata(path)?.len(),"architecture":config.architecture,"layers":config.block_count,
        "skipped_layer_ids":model.layer_mask.skipped_layers(),
        "active_layer_ids":model.layer_mask.active_layers(config.block_count)?.collect::<Vec<_>>(),
        "active_layer_count":model.active_layer_count()?,
        "model_context_tokens":config.context_length,"d_model":config.embedding_length,
        "ffn_dim":config.feed_forward_length,"query_heads":config.attention_head_count,
        "kv_heads":config.attention_head_count_kv,"head_dim":config.attention_key_length,
        "vocabulary":config.vocab_size,"rope_layout":config.rope_layout_code(),
        "tokenizer_kind":format!("{:?}",tokenizer.kind()),"bos_id":tokenizer.bos_id(),
        "stop_ids":tokenizer.stop_ids(),"tokenizer_metadata_sha256":tokenizer_sha256(model)?,
        "tensor_dtype_counts":dtype_counts,"memory":memory(model)?}),
    )
}

fn tokenizer_sha256(model: &LoadedModel) -> Result<String> {
    let metadata: BTreeMap<_, _> = model
        .gguf
        .metadata
        .iter()
        .filter(|(key, _)| key.starts_with("tokenizer."))
        .collect();
    records::sha256(&serde_json::to_vec(&metadata)?)
}

/// Preflight every chosen input before forward execution. Tokenization and full output
/// bounds must agree; evaluation never silently truncates passages or tasks.
pub(super) fn validate_inputs(
    corpus: &Corpus,
    splits: &[Split],
    a: &Tokenizer,
    b: &Tokenizer,
    capacity: u32,
    vocabulary: u32,
) -> Result<()> {
    ensure!(
        a.bos_id() == b.bos_id() && a.stop_ids() == b.stop_ids(),
        "tokenizer special IDs differ"
    );
    for split in splits {
        for task in &corpus.split(*split).tasks {
            let (_, ids) = tasks::encode_task(a, &task.prompt)?;
            let (_, other_ids) = tasks::encode_task(b, &task.prompt)?;
            ensure!(ids == other_ids, "task tokenizer mismatch: {}", task.id);
            ensure!(
                !ids.is_empty() && ids.len() + task.max_tokens <= capacity as usize,
                "task {} exceeds allocated context; no truncation",
                task.id
            );
            ensure!(
                ids.iter().all(|&id| id < vocabulary),
                "task token outside model vocabulary: {}",
                task.id
            );
        }
        for passage in &corpus.split(*split).passages {
            let ids = tasks::encode_passage(a, &passage.text)?;
            ensure!(
                ids == tasks::encode_passage(b, &passage.text)?,
                "passage tokenizer mismatch: {}",
                passage.id
            );
            ensure!(
                (2..=capacity as usize).contains(&ids.len()),
                "passage {} has invalid length; no truncation",
                passage.id
            );
            ensure!(
                ids.iter().all(|&id| id < vocabulary),
                "passage token outside model vocabulary: {}",
                passage.id
            );
        }
    }
    Ok(())
}

pub fn evaluate(args: Args) -> Result<()> {
    if args.q8_prefix_drift.is_some() {
        return super::prepared_drift::diagnose(args);
    }
    if args.rank_layers.is_some() {
        return super::layers::rank(&args);
    }
    if args.capture_activations.is_some() {
        return super::activations::capture(&args);
    }
    let environment = environment()?;
    let corpus_bytes = fs::read(&args.corpus)?;
    let corpus = Corpus::parse(std::str::from_utf8(&corpus_bytes)?)?;
    let compact_preflight = Instant::now();
    let compact = CompactComparison::load(&args, &corpus_bytes)?;
    let compact_preflight_ms = compact
        .as_ref()
        .map(|_| compact_preflight.elapsed().as_secs_f64() * 1000.0);
    if compact.is_some() {
        super::research::validate_execution(&environment)?;
    }
    let activation = super::activations::load_variant(&args, &environment, &corpus, &corpus_bytes)?;
    let layer_ranking = super::layers::load_variant(&args, &environment, &corpus, &corpus_bytes)?;
    let skipped_layers = layer_ranking
        .as_ref()
        .map(|value| value.skipped_layers())
        .transpose()?
        .unwrap_or_else(|| args.skip_layers.clone());
    let splits = args.split.selected();
    let mut report = Report::create(&args.output)?;
    let initial_device_bytes = CudaContext::total_device_bytes();
    let load_start = Instant::now();
    let (mut reference, reference_tokenizer) = load_model(
        &args.reference_model,
        args.context_tokens,
        compact
            .as_ref()
            .map_or(&[], |value| value.reference_skipped_layers()),
        super::activations::ActivationSetup::Disabled,
    )?;
    let reference_load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let load_start = Instant::now();
    let (mut candidate, candidate_tokenizer) = load_model(
        &args.model,
        args.context_tokens,
        &skipped_layers,
        activation.as_ref().map_or(
            super::activations::ActivationSetup::Disabled,
            super::activations::ActivationSetup::Apply,
        ),
    )?;
    let candidate_load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    if let Some(ranking) = &layer_ranking {
        ranking.validate_model(&candidate, &candidate_tokenizer, &corpus)?;
    }
    if let Some(compact) = &compact {
        compact.validate_models(&reference, &candidate)?;
    }
    ensure!(
        reference.model_config.vocab_size == candidate.model_config.vocab_size,
        "reference/candidate vocabulary sizes differ"
    );
    ensure!(
        tokenizer_sha256(&reference)? == tokenizer_sha256(&candidate)?,
        "full tokenizer metadata differs; vocabulary-logit comparison is invalid"
    );
    validate_inputs(
        &corpus,
        &splits,
        &reference_tokenizer,
        &candidate_tokenizer,
        args.context_tokens,
        reference.model_config.vocab_size,
    )?;
    if compact.is_some() {
        for split in &splits {
            for task in &corpus.split(*split).tasks {
                let (_, ids) = tasks::encode_task(&reference_tokenizer, &task.prompt)?;
                compact::validate_diagnostic_capacity(ids.len(), args.context_tokens)?;
            }
        }
    }
    let reference_identity =
        model_identity(&args.reference_model, &reference, &reference_tokenizer)?;
    let candidate_identity = model_identity(&args.model, &candidate, &candidate_tokenizer)?;
    if let Some(compact) = &compact {
        compact.validate_identities(&reference_identity, &candidate_identity)?;
    }
    let configuration = json!({"environment":environment,"kv_mode":"off","allocated_context_tokens":args.context_tokens,
        "sampling":"greedy top_k=1 seed=0, no penalties","cache_reset":"before each task/passage",
        "task_template":"canonical Qwen system+user ChatML; Llama3 auto chat",
        "passage_tokenization":"plain text, metadata BOS if available, no EOS or chat wrapper; no truncation",
        "teacher_forcing":"identical observed prefixes, full vocabulary logits, no sampling",
        "timing_scope":"resident in-process generation admission-to-return including detokenization; excludes loading/HTTP/queue; loss includes full-logit readback and CPU scoring",
        "quality_timing_trials":1,"candidate_skipped_layer_ids":candidate.layer_mask.skipped_layers(),
        "reference_skipped_layer_ids":reference.layer_mask.skipped_layers(),
        "comparison_purpose":if compact.is_some() {"masked_source_vs_compact_export"} else {"unchanged_reference_vs_model_variant"},
        "compact_export":compact.as_ref().map(CompactComparison::describe),
        "step_timing_trials":args.step_timing_trials,"step_timing_warm_prefix_tokens":args.step_timing_warm_prefix_tokens,
        "candidate_activation_calibration":activation.as_ref().map(|value|value.describe()),
        "candidate_layer_ranking":layer_ranking.as_ref().map(|value|value.describe()).transpose()?});
    let mut transforms = if candidate.layer_mask.has_skipped_layers() {
        vec![
            json!({"kind":"diagnostic_layer_mask","skipped_layer_ids":candidate.layer_mask.skipped_layers(),
            "active_layer_ids":candidate_identity["active_layer_ids"],"original_layer_count":candidate.model_config.block_count,
            "has_in_process_compaction":false,
            "calibration_ranking":layer_ranking.as_ref().map(|value|value.describe()).transpose()?}),
        ]
    } else {
        Vec::new()
    };
    if let Some(activation) = &activation {
        transforms.push(activation.describe());
    }
    report.append(json!({"kind":"model_quality_manifest","schema_version":1,
        "created_unix_s":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "variant_id":args.variant_id,"reference":reference_identity,"candidate":candidate_identity,
        "has_identical_model_source":reference_identity["sha256"] == candidate_identity["sha256"],
        "has_identical_execution_model":reference_identity["sha256"] == candidate_identity["sha256"] && transforms.is_empty() && !reference.layer_mask.has_skipped_layers(),
        "reference_in_process_transforms":if reference.layer_mask.has_skipped_layers() {vec![json!({"kind":"diagnostic_layer_mask","skipped_layer_ids":reference.layer_mask.skipped_layers()})]} else {vec![]},
        "in_process_transforms":transforms,"external_model_difference":if compact.is_some() {"verified compact export; live comparison is against masked source, not unchanged source"} else {"only file hashes/metadata verified; no pruning provenance inferred from label"},
        "corpus_path":args.corpus,"corpus_id":corpus.corpus_id,"corpus_sha256":records::sha256(&corpus_bytes)?,
        "selected_splits":splits,"configuration":configuration,"configuration_sha256":records::sha256(&serde_json::to_vec(&configuration)?)?,
        "revision":records::command_text("git",&["rev-parse","HEAD"])?,"worktree":records::command_text("git",&["status","--short"])?,
        "executable_sha256":records::file_sha256(&std::env::current_exe()?)?,
        "build_profile":if cfg!(debug_assertions) {"dev-test"} else {"release"},
        "rustc":records::command_text("rustc",&["--version"])?,"nvcc":records::command_text("nvcc",&["--version"])?,
        "cuda_flags":super::research::CUDA_FLAGS,
        "host_compiler":records::command_text("g++",&["--version"])?,
        "device":format!("{:?}",reference.cuda.current_device_props()?),
        "cold_load_reference_ms":reference_load_ms,"cold_load_candidate_ms":candidate_load_ms,
        "compact_preflight_ms":compact_preflight_ms,
        "load_scope":"GGUF parsing/file read/upload/dense KV/validation; excludes preflight and later identity hashing; not fresh-process cold I/O"}))?;
    for split in splits {
        let inputs = corpus.split(split);
        report.append(json!({"kind":"model_quality_split","split":split,
            "split_sha256":records::sha256(&serde_json::to_vec(inputs)?)?}))?;
        // Explicitly separate first-use warmup from task timing. It never becomes
        // calibration data and every subsequent case still resets KV.
        tasks::warmup(&reference, &reference_tokenizer)?;
        tasks::warmup(&candidate, &candidate_tokenizer)?;
        report.append(json!({"kind":"model_quality_warmup","split":split,
            "prompt":"Say ready.","max_tokens":2,"reference":memory(&reference)?,"candidate":memory(&candidate)?}))?;
        let mut task_results = Vec::new();
        for task in &inputs.tasks {
            let result = tasks::evaluate_task(
                &mut report,
                split,
                task,
                &reference,
                &candidate,
                &reference_tokenizer,
            )?;
            if let Some(compact) = &compact {
                let (_, ids) = tasks::encode_task(&reference_tokenizer, &task.prompt)?;
                compact.record_task_boundaries(
                    &mut report,
                    split,
                    &task.id,
                    (&mut reference, &mut candidate),
                    &ids,
                    &result,
                )?;
            }
            task_results.push(result);
        }
        let mut losses = Vec::new();
        for passage in &inputs.passages {
            let ids = tasks::encode_passage(&reference_tokenizer, &passage.text)?;
            candidate.reset_activation_statistics()?;
            let start = Instant::now();
            let comparison = compare_loaded_models(&mut reference, &mut candidate, &ids)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            report.append(json!({"kind":"model_quality_passage","split":split,"id":passage.id,
                "text":passage.text,"token_ids":ids,"comparison":comparison,"elapsed_ms":elapsed_ms,
                "timing_scope":"both models interleaved serially, full-logit D2H and CPU scoring; NOT decode throughput",
                "reference_memory":memory(&reference)?,"candidate_memory":memory(&candidate)?}))?;
            if compact.is_some() {
                compact::require_identical_logits(&comparison)?;
            }
            eprintln!(
                "{split:?} {}: reference PPL={:?}, candidate PPL={:?}, max drift={:.6}",
                passage.id,
                comparison.reference.perplexity,
                comparison.candidate.perplexity,
                comparison.max_abs_logit_diff
            );
            losses.push(comparison);
        }
        if args.step_timing_trials > 0 {
            let timing_passage = &inputs.passages[0];
            let ids = tasks::encode_passage(&reference_tokenizer, &timing_passage.text)?;
            let expected_reference_skipped_layers = compact
                .as_ref()
                .map(|value| value.reference_skipped_layers().to_vec())
                .unwrap_or_default();
            step_timing::measure_pairs(
                &mut report,
                split,
                &timing_passage.id,
                &mut reference,
                &mut candidate,
                &ids,
                step_timing::StepTimingConfig {
                    warm_prefix_tokens: args.step_timing_warm_prefix_tokens,
                    trials: args.step_timing_trials,
                    expected_reference_skipped_layers,
                    should_require_equivalent_predictions: compact.is_some(),
                },
            )?;
        }
        let mut summary = json!({"kind":"model_quality_summary","split":split,
            "tasks":task_results.len(),"dense_reference_passes":task_results.iter().filter(|result|result.reference_passed).count(),
            "candidate_passes":task_results.iter().filter(|result|result.candidate_passed).count(),
            "dense_valid_regressions":task_results.iter().filter(|result|result.reference_passed && !result.candidate_passed).count(),
            "inconclusive_tasks":task_results.iter().filter(|result|!result.reference_passed).count(),
            "reference_loss":LossMetrics::combine(&losses.iter().map(|r|r.reference.clone()).collect::<Vec<_>>())?,
            "candidate_loss":LossMetrics::combine(&losses.iter().map(|r|r.candidate.clone()).collect::<Vec<_>>())?,
            "max_abs_logit_diff":losses.iter().map(|r|r.max_abs_logit_diff).fold(0.0_f64,f64::max),
            "reference_memory":memory(&reference)?,"candidate_memory":memory(&candidate)?});
        if compact.is_some() {
            let fields = summary.as_object_mut().expect("summary object");
            for (old, new) in [
                ("dense_reference_passes", "masked_reference_passes"),
                ("dense_valid_regressions", "masked_reference_regressions"),
                ("inconclusive_tasks", "masked_reference_failed_tasks"),
            ] {
                let value = fields.remove(old).expect("summary field");
                fields.insert(new.to_owned(), value);
            }
            fields.insert(
                "comparison_purpose".into(),
                json!("compact equivalence, not quality versus unchanged source"),
            );
        }
        report.append(summary)?;
    }
    ensure!(
        memory(&reference)?["materialized_f32_bytes"] == 0
            && memory(&candidate)?["materialized_f32_bytes"] == 0,
        "unexpected materialized FP32 weights in compact quality run"
    );
    if let Some(compact) = &compact {
        compact.validate_models(&reference, &candidate)?;
    }
    drop(reference);
    drop(candidate);
    ensure!(
        CudaContext::total_device_bytes() == initial_device_bytes,
        "evaluation device allocation leak"
    );
    report.append(
        json!({"kind":"model_quality_complete","device_bytes_before":initial_device_bytes,
        "device_bytes_after":CudaContext::total_device_bytes(),"is_complete":true,
        "compact_equivalence_verified":compact.as_ref().map(|_|true)}),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frozen inputs are usable unchanged, but a short allocation, invalid
    /// vocabulary, or incompatible BOS policy must fail before forward execution.
    #[test]
    fn test_preflight_rejects_truncation_and_tokenizer_mismatch() -> Result<()> {
        let corpus = Corpus::parse(include_str!("../../data/research_quality/v1.json"))?;
        let tokenizer = Tokenizer::byte_level();
        let splits = [Split::Calibration, Split::HeldOut];
        validate_inputs(&corpus, &splits, &tokenizer, &tokenizer, 1024, 256)?;
        assert!(validate_inputs(&corpus, &splits, &tokenizer, &tokenizer, 16, 256).is_err());
        assert!(validate_inputs(&corpus, &splits, &tokenizer, &tokenizer, 1024, 64).is_err());
        let mut other = Tokenizer::byte_level();
        other.set_bos_id(Some(1));
        assert!(validate_inputs(&corpus, &splits, &tokenizer, &other, 1024, 256).is_err());
        assert_eq!(tasks::encode_passage(&other, "abc")?, vec![1, 97, 98, 99]);
        Ok(())
    }
}

use super::{
    q8_runtime::{self, PrefillChoice, RuntimeConfig},
    qwen_workload,
    records::{self, Report},
    run, tasks, Args,
};
use anyhow::{ensure, Context, Result};
use m40_llm::{
    cuda::CudaContext,
    decode_session::DecodeSession,
    evaluation::{
        corpus::Corpus,
        loss::{score_logits, LossMetrics},
        teacher_forced::compare_loaded_models,
    },
    generate::{GeneratedText, PromptPrefill},
    infer::LoadedModel,
    kv_compression::KvCompressionConfig,
    profile,
};
use serde_json::{json, Value};
use std::{fs, time::Instant};

fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(a, b)| a != b)
        .or_else(|| (a.len() != b.len()).then_some(a.len().min(b.len())))
}

fn prepared_identity(stats: &m40_llm::infer::execution_weights::Q8ExecutionStats) -> Result<Value> {
    let mut value = serde_json::to_value(stats)?;
    let object = value.as_object_mut().context("prepared stats object")?;
    object.remove("preparation_elapsed_ms");
    object.remove("model_hash_elapsed_ms");
    Ok(value)
}

fn dispatch(before: &profile::ProfileSnapshot) -> Value {
    let after = profile::snapshot();
    let operations: std::collections::BTreeMap<_, _> = after
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
    json!({"totals":profile::counter_delta(before, &after),"operations":operations})
}

fn record_generation(
    report: &mut Report,
    label: &str,
    model: &LoadedModel,
    config: RuntimeConfig,
    prompt: &str,
) -> Result<GeneratedText> {
    let before = profile::snapshot();
    let start = Instant::now();
    let generated = tasks::generate_with_prefill(
        model,
        prompt,
        qwen_workload::MAX_TOKENS,
        config.prompt_prefill(),
    )?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let counters = dispatch(&before);
    let expected_ffn = match config.ffn {
        q8_runtime::FfnChoice::Native => None,
        q8_runtime::FfnChoice::PackedF32_256 => Some("ffn_q8_f32_packed_warp_256"),
        q8_runtime::FfnChoice::PackedF32_128 => Some("ffn_q8_f32_packed_warp_128"),
        q8_runtime::FfnChoice::Q81_128 => Some("ffn_q8_1_threads128"),
        q8_runtime::FfnChoice::Q81_256 => Some("ffn_q8_1_threads256"),
    };
    if let Some(operation) = expected_ffn {
        ensure!(
            counters["operations"][operation]["launches"]
                .as_u64()
                .is_some_and(|n| n > 0),
            "requested FFN candidate was not dispatched"
        );
    }
    if config.prefill == PrefillChoice::DequantSgemm {
        ensure!(
            counters["operations"]["dequantize_q8_0_to_f32_colmajor"]["launches"]
                == 7 * model.model_config.block_count,
            "temporary SGEMM did not convert all transformer projections"
        );
        ensure!(
            generated.q8_prefill_temporary_weight_bytes.is_some(),
            "missing actual temporary capacity"
        );
    }
    report.append(json!({"kind":"q8_runtime_browser_generation","label":label,"config":config,
        "prompt_token_ids":generated.token_ids[..generated.prompt_token_len],
        "generation":tasks::generation_record(&generated, elapsed_ms, model)?,"dispatch":counters,
        "scope":"one untimed qualification run, natural stopping and actual GPU-greedy tokens; not HTTP throughput"}))?;
    Ok(generated)
}

fn create_session(model: &mut LoadedModel) -> Result<DecodeSession> {
    model.reset_kv_cache()?;
    let dimension = model.validate_full_layer_decode()?.0;
    DecodeSession::new_for_sequence_with_kv_config(
        model,
        0,
        KvCompressionConfig::dense_reference(),
        dimension,
        true,
        "runtime_quality",
        "runtime_quality:x",
        "runtime_quality:out",
    )
}

/// Recompute full logits from production greedy hidden states on identical
/// reference prefixes. The generic head is diagnostic, never a replacement for
/// the actual greedy head used to establish continuation equality.
fn record_prefix_logits(
    report: &mut Report,
    reference: &mut LoadedModel,
    candidate: &mut LoadedModel,
    config: RuntimeConfig,
    generated: &GeneratedText,
    candidate_generated: &GeneratedText,
) -> Result<()> {
    let prompt_len = generated.prompt_token_len;
    let reference_ids = &generated.token_ids[prompt_len..];
    ensure!(
        !reference_ids.is_empty(),
        "reference produced no continuation"
    );
    let candidate_ids = &candidate_generated.token_ids[candidate_generated.prompt_token_len..];
    let divergence = first_divergence(reference_ids, candidate_ids);
    let mut steps = vec![0, 1, 8, 32, 128, 256, 511];
    if let Some(step) = divergence {
        steps.push(step);
    }
    steps.retain(|&step| step < reference_ids.len());
    steps.sort_unstable();
    steps.dedup();
    let last = *steps.last().context("no sampled prefixes")?;
    let mut a = create_session(reference)?;
    let mut b = create_session(candidate)?;
    if config.prefill != PrefillChoice::Sequential {
        b.prefill_packed_prefix_ids(&generated.token_ids[..prompt_len])?;
    }
    for (step, &expected) in reference_ids.iter().enumerate().take(last + 1) {
        let prefix = &generated.token_ids[..prompt_len + step];
        let actual_a = a.greedy_token_for_ids(prefix)?;
        let actual_b = b.greedy_token_for_ids(prefix)?;
        ensure!(
            actual_a == expected,
            "reference replay changed actual greedy token at {step}"
        );
        if !steps.contains(&step) {
            continue;
        }
        let logits_a = unsafe { reference.logits_from_hidden(a.d_out_ptr()?.cast_const())? };
        let logits_b = unsafe { candidate.logits_from_hidden(b.d_out_ptr()?.cast_const())? };
        ensure!(
            logits_a.len() == logits_b.len(),
            "logit vocabulary mismatch"
        );
        let scored_a = score_logits(&logits_a, expected)?;
        let scored_b = score_logits(&logits_b, expected)?;
        let (maximum, sum) =
            logits_a
                .iter()
                .zip(&logits_b)
                .fold((0.0_f64, 0.0), |(max, sum), (&a, &b)| {
                    let diff = (f64::from(a) - f64::from(b)).abs();
                    (max.max(diff), sum + diff)
                });
        let overlap = scored_a
            .top_indices
            .iter()
            .filter(|id| scored_b.top_indices.contains(id))
            .count();
        report.append(json!({"kind":"q8_runtime_prefix_logits","generated_step_zero_based":step,
            "prefix_length":prefix.len(),"expected_actual_reference_token_id":expected,
            "actual_reference_greedy_token_id":actual_a,"actual_candidate_greedy_token_id":actual_b,
            "reference":scored_a,"candidate":scored_b,"max_abs_logit_diff":maximum,
            "mean_abs_logit_diff":sum/logits_a.len() as f64,"top10_overlap":overlap,
            "scope":"identical absolute reference prefix; generic full-vocabulary head diagnostic; not timing or sampled continuation"}))?;
    }
    Ok(())
}

pub(super) fn evaluate(args: Args) -> Result<()> {
    ensure!(
        args.context_tokens == qwen_workload::CONTEXT_TOKENS,
        "M08 requires context 1024"
    );
    ensure!(
        records::file_sha256(&args.reference_model)? == records::file_sha256(&args.model)?,
        "runtime candidates must use the identical source model"
    );
    let environment = run::environment()?;
    q8_runtime::validate_environment(&environment)?;
    let config = RuntimeConfig {
        ffn: args.q8_runtime_candidate.context("runtime candidate")?,
        prefill: args.q8_runtime_prefill.unwrap_or_default(),
    };
    let control = RuntimeConfig::default();
    let corpus_bytes = fs::read(&args.corpus)?;
    let corpus = Corpus::parse(std::str::from_utf8(&corpus_bytes)?)?;
    let splits = args.split.selected();
    let mut report = Report::create(&args.output)?;
    let initial = CudaContext::total_device_bytes();
    let (mut reference, tokenizer) =
        control.load_model(&args.reference_model, args.context_tokens)?;
    let (mut candidate, candidate_tokenizer) =
        config.load_model(&args.model, args.context_tokens)?;
    run::validate_inputs(
        &corpus,
        &splits,
        &tokenizer,
        &candidate_tokenizer,
        args.context_tokens,
        reference.model_config.vocab_size,
    )?;
    ensure!(
        run::tokenizer_sha256(&reference)? == run::tokenizer_sha256(&candidate)?,
        "tokenizer mismatch"
    );
    let a_stats = reference.q8_execution_stats();
    let b_stats = candidate.q8_execution_stats();
    ensure!(
        prepared_identity(&a_stats)? == prepared_identity(&b_stats)?,
        "prepared control differs"
    );
    let prompt = qwen_workload::render_user_prompt(qwen_workload::USER_PROMPT);
    let prompt_ids = tokenizer.encode_with_specials(&prompt, false, false)?;
    ensure!(
        prompt_ids.len() == 65,
        "canonical browser prompt must contain 65 IDs"
    );
    for split in &splits {
        for task in &corpus.split(*split).tasks {
            let (_, ids) = tasks::encode_task(&tokenizer, &task.prompt)?;
            if let PromptPrefill::PackedPrefix { max_prompt_tokens } = config.prompt_prefill() {
                ensure!(
                    ids.len() <= max_prompt_tokens,
                    "task {} exceeds explicit packed bound",
                    task.id
                );
            }
        }
    }
    report.append(json!({"kind":"q8_runtime_manifest","schema_version":1,"variant_id":args.variant_id,
        "reference":run::model_identity(&args.reference_model,&reference,&tokenizer)?,
        "candidate":run::model_identity(&args.model,&candidate,&candidate_tokenizer)?,
        "control_config":control,"candidate_config":config,"is_numerical_track":config.is_numerical(),
        "original_exact_goal_eligible":!config.is_numerical(),"environment":environment,
        "prepared_control":a_stats,"corpus_sha256":records::sha256(&corpus_bytes)?,"selected_splits":splits,
        "reference_memory":run::memory(&reference)?,"candidate_memory":run::memory(&candidate)?,
        "revision":records::command_text("git", &["rev-parse","HEAD"])?,
        "worktree":records::command_text("git", &["status","--short"])?,
        "executable_sha256":records::file_sha256(&std::env::current_exe()?)?,
        "build_profile":"release","cuda_flags":super::research::CUDA_FLAGS,
        "browser_prompt":prompt,"browser_prompt_ids":prompt_ids,
        "cache_reset":"fresh explicit dense KV per generation/task/passage",
        "timing_scope":"diagnostic resident generation only; no HTTP/E2E promotion from these timings"}))?;
    for (model, choice) in [(&reference, control), (&candidate, config)] {
        let (prompt, _) = tasks::encode_task(&tokenizer, "Say ready.")?;
        tasks::generate_with_prefill(model, &prompt, 2, choice.prompt_prefill())?;
    }
    let a = record_generation(&mut report, "control", &reference, control, &prompt)?;
    let b = record_generation(&mut report, "candidate", &candidate, config, &prompt)?;
    ensure!(
        a.token_ids[..a.prompt_token_len] == prompt_ids
            && b.token_ids[..b.prompt_token_len] == prompt_ids,
        "browser generation changed prompt IDs"
    );
    let a_ids = &a.token_ids[a.prompt_token_len..];
    let b_ids = &b.token_ids[b.prompt_token_len..];
    let divergence = first_divergence(a_ids, b_ids);
    let has_identical_generation = divergence.is_none()
        && a.output == b.output
        && a.metrics.stop_reason == b.metrics.stop_reason;
    let has_fixed_workload = [&a, &b].iter().all(|g| {
        g.metrics.generated_tokens == 512
            && g.metrics.stop_reason == m40_llm::decode::StopReason::Length
    });
    report.append(json!({"kind":"q8_runtime_browser_comparison","has_identical_generation":has_identical_generation,
        "has_fixed_workload":has_fixed_workload,"first_generated_divergence_step_zero_based":divergence,
        "control_generated_tokens":a.metrics.generated_tokens,"candidate_generated_tokens":b.metrics.generated_tokens}))?;
    record_prefix_logits(&mut report, &mut reference, &mut candidate, config, &a, &b)?;
    let mut regressions = 0;
    let mut dense_valid = 0;
    for split in splits {
        for task in &corpus.split(split).tasks {
            let result = tasks::evaluate_task_with_prefill(
                &mut report,
                split,
                task,
                &reference,
                &candidate,
                &tokenizer,
                (control.prompt_prefill(), config.prompt_prefill()),
            )?;
            dense_valid += usize::from(result.reference_passed);
            regressions += usize::from(result.reference_passed && !result.candidate_passed);
        }
        if config.prefill == PrefillChoice::Sequential {
            let mut reference_losses = Vec::new();
            let mut candidate_losses = Vec::new();
            for passage in &corpus.split(split).passages {
                let ids = tasks::encode_passage(&tokenizer, &passage.text)?;
                let comparison = compare_loaded_models(&mut reference, &mut candidate, &ids)?;
                reference_losses.push(comparison.reference.clone());
                candidate_losses.push(comparison.candidate.clone());
                report.append(json!({"kind":"q8_runtime_passage","split":split,"id":passage.id,
                    "token_ids":ids,"comparison":comparison,"scope":"sequential full-logit teacher forcing; not GPU-greedy timing"}))?;
            }
            report.append(json!({"kind":"q8_runtime_loss_summary","split":split,
                "reference":LossMetrics::combine(&reference_losses)?,"candidate":LossMetrics::combine(&candidate_losses)?}))?;
        } else {
            report.append(json!({"kind":"q8_runtime_loss_scope","split":split,
                "status":"not_applicable","reason":"sequential passage loss would not exercise packed prompt prefill; browser prefix logit diagnostics and actual task generation do"}))?;
        }
    }
    let reference_stats_after = reference.q8_execution_stats();
    let candidate_stats_after = candidate.q8_execution_stats();
    ensure!(
        prepared_identity(&reference_stats_after)? == prepared_identity(&a_stats)?
            && prepared_identity(&candidate_stats_after)? == prepared_identity(&a_stats)?,
        "prepared control changed during evaluation"
    );
    drop(reference);
    drop(candidate);
    ensure!(
        CudaContext::total_device_bytes() == initial,
        "runtime comparison leaked tracked allocations"
    );
    report.append(json!({"kind":"q8_runtime_summary","dense_valid_tasks":dense_valid,"dense_valid_regressions":regressions,
        "has_identical_browser_generation":has_identical_generation,"has_fixed_browser_workload":has_fixed_workload,
        "passes_exact_quality_gate":!config.is_numerical() && has_identical_generation && has_fixed_workload && dense_valid > 0 && regressions == 0,
        "is_numerical_track":config.is_numerical(),"recommendation":"none; controlled HTTP timing and remaining acceptance gates are separate",
        "tracked_device_bytes_initial":initial,"tracked_device_bytes_after":CudaContext::total_device_bytes()}))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Detect both a changed token and a changed stop length, including prefix
    /// continuations whose zipped token pairs would otherwise all compare equal.
    #[test]
    fn test_first_divergence_includes_unequal_lengths() {
        assert_eq!(first_divergence(&[1, 2], &[1, 2]), None);
        assert_eq!(first_divergence(&[1, 2], &[1, 3]), Some(1));
        assert_eq!(first_divergence(&[1, 2], &[1, 2, 3]), Some(2));
        assert_eq!(first_divergence(&[1], &[]), Some(0));
    }

    /// Full-logit diagnostics replay actual greedy prefixes, report every
    /// vocabulary element, dispatch the requested FFN/prefill path and release
    /// all fixture storage. Preparation timings are not part of weight identity.
    #[test]
    #[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
    fn test_runtime_prefix_diagnostics_exercise_candidates() -> Result<()> {
        use super::q8_runtime::FfnChoice;
        use m40_llm::{
            generate::{generate_text, GenerateOptions, PromptFormat},
            infer::execution_weights::Q8ExecutionConfig,
        };
        let _lock = crate::cuda_test_scope::lock_cuda();
        let _env: Vec<_> = [
            ("M40LLM_ENABLE_NVCC", "1"),
            ("M40LLM_Q8_DECODE_STREAM", "decode"),
            ("M40LLM_Q8_MLP_GATE_UP_SPLIT_QBLOCK", "4"),
            ("M40LLM_FUSED_Q8_QKV", "1"),
            ("M40LLM_FUSED_Q8_MLP_SWIGLU", "1"),
            ("M40LLM_CUDA_GREEDY_ARGMAX", "1"),
            ("M40LLM_PROJECTION_BACKEND", "large-model"),
            ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
            ("M40LLM_DECODE_GRAPH", "0"),
            ("M40LLM_PREFILL_CHUNK_SIZE", "invalid"),
        ]
        .into_iter()
        .map(|(k, v)| crate::env_restore::EnvRestore::set(k, v))
        .collect();
        let _hardware = CudaContext::new_m40()?;
        let initial = CudaContext::total_device_bytes();
        for config in [
            RuntimeConfig {
                ffn: FfnChoice::PackedF32_256,
                prefill: PrefillChoice::Sequential,
            },
            RuntimeConfig {
                ffn: FfnChoice::Q81_128,
                prefill: PrefillChoice::Sequential,
            },
            RuntimeConfig {
                ffn: FfnChoice::Native,
                prefill: PrefillChoice::DequantSgemm,
            },
        ] {
            let load = |config: RuntimeConfig| -> Result<LoadedModel> {
                let (mut gguf, bytes) = crate::q8_fixture::build_prefill_fixture(false);
                gguf.metadata.remove("tokenizer.ggml.model");
                let mut model = LoadedModel::from_gguf_with_q8_backends(
                    gguf,
                    bytes,
                    -1,
                    Q8ExecutionConfig::parse("split", Some("6144"), Some("down,vocabulary"))?,
                    config.ffn_config()?,
                    config.prefill_backend(),
                )?;
                model.allocate_kv_cache_for_layers(16)?;
                Ok(model)
            };
            let mut reference = load(RuntimeConfig::default())?;
            let mut candidate = load(config)?;
            let a_stats = reference.q8_execution_stats();
            let mut changed_times = a_stats.clone();
            changed_times.preparation_elapsed_ms += 1000.0;
            assert_eq!(
                prepared_identity(&a_stats)?,
                prepared_identity(&changed_times)?
            );
            changed_times.resident_bytes += 1;
            assert_ne!(
                prepared_identity(&a_stats)?,
                prepared_identity(&changed_times)?
            );
            let generate = |model: &LoadedModel, config: RuntimeConfig| {
                generate_text(
                    model,
                    GenerateOptions {
                        prompt: "ABC".into(),
                        prompt_format: PromptFormat::Raw,
                        max_tokens: Some(3),
                        top_k: Some(1),
                        kv_compression: KvCompressionConfig::dense_reference(),
                        prompt_prefill: config.prompt_prefill(),
                        ..Default::default()
                    },
                )
            };
            let a = generate(&reference, RuntimeConfig::default())?;
            let before = profile::snapshot();
            let b = generate(&candidate, config)?;
            let counters = dispatch(&before);
            let operation = match config.ffn {
                FfnChoice::PackedF32_256 => "ffn_q8_f32_packed_warp_256",
                FfnChoice::Q81_128 => "ffn_q8_1_threads128",
                _ => "dequantize_q8_0_to_f32_colmajor",
            };
            assert!(
                counters["operations"][operation]["launches"]
                    .as_u64()
                    .unwrap_or(0)
                    > 0
            );
            let destination = crate::test_files::TestReportPath::new();
            let mut report = Report::create(destination.path())?;
            record_prefix_logits(&mut report, &mut reference, &mut candidate, config, &a, &b)?;
            let records: Vec<Value> = fs::read_to_string(destination.path())?
                .lines()
                .map(serde_json::from_str)
                .collect::<Result<_, _>>()?;
            assert!(!records.is_empty());
            assert_eq!(records[0]["generated_step_zero_based"], 0);
            for record in records {
                assert!(record["max_abs_logit_diff"].as_f64().unwrap().is_finite());
                assert!(record["top10_overlap"].as_u64().unwrap() <= 10);
                assert_eq!(
                    record["actual_reference_greedy_token_id"],
                    record["expected_actual_reference_token_id"]
                );
            }
        }
        assert_eq!(CudaContext::total_device_bytes(), initial);
        Ok(())
    }
}

use super::*;
use clap::Parser;
use m40_llm::{evaluation::loss::compare_teacher_forced, gguf::compact::export_compact_gguf};
use std::path::Path;

fn arguments(source: &Path, candidate: &Path, record: &Path, corpus: &Path) -> Result<Args> {
    let output = record.with_extension("comparison.jsonl");
    Ok(Args::try_parse_from([
        "model_quality",
        "--reference-model",
        source.to_str().context("source path")?,
        "--model",
        candidate.to_str().context("candidate path")?,
        "--variant-id",
        "compact-test",
        "--compact-export-report",
        record.to_str().context("record path")?,
        "--corpus",
        corpus.to_str().context("corpus path")?,
        "--output",
        output.to_str().context("output path")?,
    ])?)
}

/// A real exported fixture binds the reference mask, untouched tokenizer and
/// requested storage without a GPU. Adverse upstream quality is kept visible.
#[test]
fn test_compact_preflight_binds_actual_export_and_source_mask() -> Result<()> {
    let fixture = crate::compact_fixture::CompactFixture::new("qwen2", 32, false)?;
    let result = export_compact_gguf(&fixture.source, &fixture.destination, &fixture.evidence()?)?;
    let record = crate::test_files::TestReportPath::new();
    fs::write(
        record.path(),
        serde_json::to_vec(
            &json!({"kind":"compact_export_complete","is_complete":true,"export":result}),
        )?,
    )?;
    let args = arguments(
        &fixture.source,
        &fixture.destination,
        record.path(),
        &fixture.corpus,
    )?;
    let comparison = CompactComparison::load(&args, &fs::read(&fixture.corpus)?)?.unwrap();
    assert_eq!(comparison.reference_skipped_layers(), &[1]);
    assert_eq!(comparison.retained_original_layers, [0, 2, 3, 4]);
    assert!(comparison.source_weight_bytes > comparison.compact_weight_bytes);
    let description = comparison.describe();
    assert_eq!(description["is_unchanged_model"], false);
    assert_eq!(
        description["prior_quality_costs"][1]["dense_valid_regressions"],
        2
    );
    assert!(!args.output.exists());
    assert_eq!(fs::read(&fixture.source)?, fixture.source_bytes);
    Ok(())
}

/// Record labels cannot override file identities, protected masks, corpus,
/// copy ranges or embedded quality lineage. Ordinary evaluation stays opt-in.
#[test]
fn test_compact_preflight_rejects_stale_or_altered_export_records() -> Result<()> {
    let fixture = crate::compact_fixture::CompactFixture::new("qwen2", 32, false)?;
    let result = export_compact_gguf(&fixture.source, &fixture.destination, &fixture.evidence()?)?;
    let original = json!({"kind":"compact_export_complete","is_complete":true,"export":result});
    let record = crate::test_files::TestReportPath::new();
    let mut args = arguments(
        &fixture.source,
        &fixture.destination,
        record.path(),
        &fixture.corpus,
    )?;
    let corpus = fs::read(&fixture.corpus)?;
    let mutations: [fn(&mut Value); 7] = [
        |record| record["is_complete"] = json!(false),
        |record| record["export"]["output_sha256"] = json!("a".repeat(64)),
        |record| record["export"]["evidence"]["corpus_sha256"] = json!("a".repeat(64)),
        |record| record["export"]["evidence"]["removed_original_layers"] = json!([0]),
        |record| record["export"]["plan"]["tensors"][0]["destination_offset"] = json!(32),
        |record| record["export"]["evidence"]["quality_report_sha256"] = json!("b".repeat(64)),
        |record| record["export"]["output_file_bytes"] = json!(1),
    ];
    for mutate in mutations {
        let mut changed = original.clone();
        mutate(&mut changed);
        fs::write(record.path(), serde_json::to_vec(&changed)?)?;
        assert!(CompactComparison::load(&args, &corpus).is_err());
    }
    fs::write(record.path(), serde_json::to_vec(&original)?)?;
    args.model = fixture.source.clone();
    assert!(CompactComparison::load(&args, &corpus).is_err());
    args.compact_export_report = None;
    assert!(CompactComparison::load(&args, &corpus)?.is_none());
    assert_eq!(fs::read(&fixture.source)?, fixture.source_bytes);
    Ok(())
}

/// Identical top tokens do not prove equivalence. A changed non-top logit must
/// fail even when greedy output would conceal the numeric drift.
#[test]
fn test_compact_logit_gate_rejects_hidden_non_top_drift() -> Result<()> {
    validate_diagnostic_capacity(5, 7)?;
    assert!(validate_diagnostic_capacity(5, 6).is_err());
    assert!(validate_diagnostic_capacity(0, 7).is_err());
    let equal = compare_teacher_forced(
        &[0, 1, 2],
        |_| Ok(vec![4.0, 2.0, 1.0]),
        |_| Ok(vec![4.0, 2.0, 1.0]),
    )?;
    require_identical_logits(&equal)?;
    let mut incomplete = equal.clone();
    incomplete.tokens.pop();
    assert!(require_identical_logits(&incomplete).is_err());
    let drift = compare_teacher_forced(
        &[0, 1, 2],
        |_| Ok(vec![4.0, 2.0, 1.0]),
        |_| Ok(vec![4.0, 2.0, 1.125]),
    )?;
    assert_eq!(
        drift.tokens[0].reference.top_indices[0],
        drift.tokens[0].candidate.top_indices[0]
    );
    assert!(require_identical_logits(&drift).is_err());
    Ok(())
}

/// Explicit real-model paths exercise the file-backed runtime, source-only mask,
/// full prompt/decode logits, generation, reset, physical KV reclamation and RAII.
/// Missing models/device fail when requested; GPU-less CI ignores this test.
#[test]
#[ignore = "requires M40 and explicit M40LLM_COMPACT_SOURCE_MODEL/MODEL/EXPORT_REPORT paths"]
fn test_real_compact_model_matches_masked_source_logits_and_allocations() -> Result<()> {
    let _scope = crate::cuda_test_scope::lock_cuda();
    let _hardware = m40_llm::cuda::CudaContext::new_m40()?;
    let initial = m40_llm::cuda::CudaContext::total_device_bytes();
    let path = |key: &str| -> Result<PathBuf> {
        Ok(std::env::var_os(key)
            .with_context(|| format!("set explicit {key}"))?
            .into())
    };
    let corpus_path = PathBuf::from("data/research_quality/v1.json");
    let mut args = arguments(
        &path("M40LLM_COMPACT_SOURCE_MODEL")?,
        &path("M40LLM_COMPACT_MODEL")?,
        &path("M40LLM_COMPACT_EXPORT_REPORT")?,
        &corpus_path,
    )?;
    args.context_tokens = 128;
    let _environment: Vec<_> = [
        ("M40LLM_ENABLE_NVCC", "1"),
        ("M40LLM_PROJECTION_BACKEND", "large-model"),
        ("M40LLM_MATERIALIZE_F32_WEIGHTS", "0"),
        ("M40LLM_Q8_EXECUTION_WEIGHTS", "off"),
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_DECODE_GRAPH", "0"),
        ("M40LLM_PREFILL_CHUNK_SIZE", "0"),
        ("M40LLM_KV_COMPRESSED_PREFILL_CHUNK_SIZE", "0"),
    ]
    .into_iter()
    .map(|(key, value)| crate::env_restore::EnvRestore::set(key, value))
    .collect();
    {
        let comparison = CompactComparison::load(&args, &fs::read(&corpus_path)?)?.unwrap();
        let (mut reference, tokenizer) = run::load_model(
            &args.reference_model,
            args.context_tokens,
            comparison.reference_skipped_layers(),
            super::super::activations::ActivationSetup::Disabled,
        )?;
        let (mut candidate, _) = run::load_model(
            &args.model,
            args.context_tokens,
            &[],
            super::super::activations::ActivationSetup::Disabled,
        )?;
        let _attention = crate::env_restore::EnvRestore::set(
            "M40LLM_ATTN_HEAD128_KERNEL",
            if reference.model_config.attention_key_length == 128 {
                "parallel"
            } else {
                "auto"
            },
        );
        let _partitions =
            crate::env_restore::EnvRestore::set("M40LLM_ATTN_HEAD128_PARTITIONS", "8");
        let _threads = crate::env_restore::EnvRestore::set("M40LLM_ATTN_HEAD128_THREADS", "256");
        comparison.validate_models(&reference, &candidate)?;
        assert_eq!(
            reference.cuda.uploaded_weights_bytes()? - candidate.cuda.uploaded_weights_bytes()?,
            comparison.source_weight_bytes - comparison.compact_weight_bytes
        );
        assert!(
            reference.kv_cache.as_ref().unwrap().actual_bytes()
                > candidate.kv_cache.as_ref().unwrap().actual_bytes()
        );
        let original_mask = std::mem::take(&mut reference.layer_mask);
        let before = m40_llm::profile::snapshot();
        assert!(comparison.validate_models(&reference, &candidate).is_err());
        assert_eq!(
            m40_llm::profile::counter_delta(&before, &m40_llm::profile::snapshot()),
            m40_llm::profile::OpCounters::default()
        );
        reference.layer_mask = original_mask;
        let destination = crate::test_files::TestReportPath::new();
        let mut report = Report::create(destination.path())?;
        let task = m40_llm::evaluation::corpus::Task {
            id: "compact-generation".into(),
            family: m40_llm::evaluation::corpus::TaskFamily::Chat,
            prompt: "Reply with exactly OK.".into(),
            max_tokens: 4,
            scoring: m40_llm::evaluation::scoring::ScoringRule::Exact {
                answers: vec!["OK".into()],
            },
        };
        let result = super::super::tasks::evaluate_task(
            &mut report,
            Split::HeldOut,
            &task,
            &reference,
            &candidate,
            &tokenizer,
        )?;
        assert!(result.has_identical_generation);
        let (_, ids) = super::super::tasks::encode_task(&tokenizer, &task.prompt)?;
        comparison.record_task_boundaries(
            &mut report,
            Split::HeldOut,
            &task.id,
            (&mut reference, &mut candidate),
            &ids,
            &result,
        )?;
        let repeated = compare_loaded_models(&mut reference, &mut candidate, &[1, 2, 3, 4, 5])?;
        require_identical_logits(&repeated)?;
        comparison.validate_models(&reference, &candidate)?;
        let records: Vec<Value> = fs::read_to_string(destination.path())?
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        assert_eq!(records[1]["kind"], "compact_task_logit_parity");
        assert_eq!(records[1]["max_abs_logit_diff"], 0.0);
        assert!(records[1]["first_decode"].is_object());
    }
    assert_eq!(m40_llm::cuda::CudaContext::total_device_bytes(), initial);
    Ok(())
}

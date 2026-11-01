use super::*;
use crate::records;
use m40_llm::infer::execution_weights::Q8ExecutionConfig;
use serde_json::json;
use std::{collections::BTreeMap, path::Path};

fn trace_rows() -> Result<Vec<Value>> {
    let prompt_ids = vec![2_u32, 3];
    Ok(vec![
        json!({"kind":"manifest","model_sha256":"a".repeat(64),"context_tokens":16,
            "kv_mode":"off","build_profile":"release","worktree":"","candidate_q8_config":candidate_config()?,
            "control_q8_execution":{"mode":"off"},"environment":{},"prompt":"hello",
            "prompt_token_ids":prompt_ids,"prompt_sha256":records::sha256(b"hello")?,
            "prompt_token_ids_sha256":records::sha256(&serde_json::to_vec(&prompt_ids)?)?,"max_tokens":8}),
        json!({"kind":"q8_execution_setup","candidate_execution":{"mode":"split","families":["down","vocabulary"]}}),
        json!({"kind":"token_trace","runtime":"m40-llm","model_variant":"control",
            "matches_measured_output":true,"is_timed":false,"stage_profile_enabled":false,
            "prompt_token_ids":prompt_ids,"generated_token_ids":[4,5,6,7,8,9,10,11]}),
        json!({"kind":"token_trace","runtime":"m40-llm","model_variant":"prepared",
            "matches_measured_output":true,"is_timed":false,"stage_profile_enabled":false,
            "prompt_token_ids":prompt_ids,"generated_token_ids":[4,9,6,7,8,9,10,11]}),
        json!({"kind":"q8_token_comparison","is_equal":false,"first_different_generated_step":1,"different_id_count":1,
            "control_generated_tokens":8,"candidate_generated_tokens":8}),
        json!({"kind":"q8_promotion_decision","actual_generated_ids_equal":false}),
    ])
}

fn encoded(rows: &[Value]) -> Result<Vec<u8>> {
    Ok(rows
        .iter()
        .map(serde_json::to_string)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .join("\n")
        .into_bytes())
}

fn parse(rows: &[Value]) -> Result<PinnedTrace> {
    PinnedTrace::parse(
        Path::new("trace.jsonl"),
        &encoded(rows)?,
        &"a".repeat(64),
        16,
        &BTreeMap::new(),
    )
}

/// Only original verified IDs and the pinned model/context/backend may define
/// the replay. Missing, duplicate, altered or falsely completed evidence fails
/// before any model/device allocation; explicit off normalizes the same base.
#[test]
fn test_prefix_trace_preflight_binds_identity_and_actual_branch() -> Result<()> {
    let rows = trace_rows()?;
    let parsed = parse(&rows)?;
    assert_eq!(parsed.first_difference, 1);
    assert_eq!(
        parsed.sample_steps.iter().copied().collect::<Vec<_>>(),
        (0..8).collect::<Vec<_>>()
    );
    assert_eq!(parsed.reference_ids, [4, 5, 6, 7, 8, 9, 10, 11]);
    let explicit = BTreeMap::from([("M40LLM_Q8_EXECUTION_WEIGHTS".to_owned(), "off".to_owned())]);
    PinnedTrace::parse(
        Path::new("trace.jsonl"),
        &encoded(&rows)?,
        &"a".repeat(64),
        16,
        &explicit,
    )?;
    for (index, key, value) in [
        (0, "model_sha256", json!("b".repeat(64))),
        (0, "context_tokens", json!(17)),
        (0, "prompt_sha256", json!("f".repeat(64))),
        (0, "prompt_token_ids_sha256", json!("f".repeat(64))),
        (0, "worktree", json!("dirty")),
        (0, "max_tokens", json!(9)),
        (2, "matches_measured_output", json!(false)),
        (2, "generated_token_ids", json!([4, 5])),
        (3, "model_variant", json!("control")),
        (3, "prompt_token_ids", json!([2, 9])),
        (4, "first_different_generated_step", json!(2)),
        (4, "different_id_count", json!(2)),
    ] {
        let mut invalid = rows.clone();
        invalid[index][key] = value;
        assert!(parse(&invalid).is_err(), "accepted altered {index}/{key}");
    }
    let mut duplicate = rows.clone();
    duplicate.push(rows[0].clone());
    assert!(parse(&duplicate).is_err());
    assert!(parse(&rows[..5]).is_err());
    let changed_execution =
        BTreeMap::from([("M40LLM_Q8_DECODE_STREAM".to_owned(), "decode".to_owned())]);
    assert!(PinnedTrace::parse(
        Path::new("trace.jsonl"),
        &encoded(&rows)?,
        &"a".repeat(64),
        16,
        &changed_execution
    )
    .is_err());
    Ok(())
}

/// An overlarge or malformed trace must fail without allocating unbounded
/// decoded records, and errors must not be mistaken for an empty successful run.
#[test]
fn test_prefix_trace_rejects_malformed_and_oversized_input() -> Result<()> {
    let destination = crate::test_files::TestReportPath::new();
    std::fs::write(destination.path(), vec![b' '; 8 * 1024 * 1024 + 1])?;
    assert!(PinnedTrace::load(destination.path(), &"a".repeat(64), 16, &BTreeMap::new()).is_err());
    assert!(PinnedTrace::parse(
        Path::new("bad.jsonl"),
        b"{\n",
        &"a".repeat(64),
        16,
        &BTreeMap::new()
    )
    .is_err());
    let too_many = "{}\n".repeat(4097);
    assert!(PinnedTrace::parse(
        Path::new("many.jsonl"),
        too_many.as_bytes(),
        &"a".repeat(64),
        16,
        &BTreeMap::new()
    )
    .is_err());
    Ok(())
}

/// Same-model controls have zero hidden/full-logit drift. Prepared probes must
/// preserve each actual greedy result and original prefix position. Expected
/// target IDs vary with the forced history; two complete resets reproduce all
/// records without leaking device storage or silently advancing KV in a probe.
#[test]
#[ignore = "requires M40 hardware; run with --include-ignored --test-threads=1"]
fn test_prefix_head_probes_preserve_forced_history_and_reset() -> Result<()> {
    let _scope = crate::cuda_test_scope::lock_cuda();
    let _env: Vec<_> = [
        ("M40LLM_Q8_DECODE_STREAM", "decode"),
        ("M40LLM_Q8_LM_HEAD_ARGMAX", "1"),
        ("M40LLM_Q8_LM_HEAD_ARGMAX_SHARED_A", "1"),
    ]
    .into_iter()
    .map(|(key, value)| crate::env_restore::EnvRestore::set(key, value))
    .collect();
    let _hardware = CudaContext::new_m40()?;
    let before = CudaContext::total_device_bytes();
    let trace = parse(&trace_rows()?)?;
    for execution in [Q8ExecutionConfig::disabled(), candidate_config()?] {
        let build = |config| -> Result<LoadedModel> {
            let (gguf, bytes) = crate::q8_fixture::build(0);
            let mut model = LoadedModel::from_gguf_with_q8_execution(gguf, bytes, -1, config)?;
            model.allocate_kv_cache_for_layers(16)?;
            Ok(model)
        };
        let mut reference = build(Q8ExecutionConfig::disabled())?;
        let mut candidate = build(execution)?;
        let owned_cache = candidate.kv_cache.take();
        candidate.kv_cache = reference.kv_cache.clone();
        assert!(reference
            .kv_cache
            .as_ref()
            .unwrap()
            .has_same_storage(candidate.kv_cache.as_ref().unwrap()));
        let rejected = crate::test_files::TestReportPath::new();
        let mut report = Report::create(rejected.path())?;
        let counts = m40_llm::profile::snapshot();
        assert!(probe_models(&mut report, &mut reference, &mut candidate, &trace).is_err());
        assert_eq!(
            m40_llm::profile::counter_delta(&counts, &m40_llm::profile::snapshot()),
            m40_llm::profile::OpCounters::default()
        );
        candidate.kv_cache = owned_cache;
        assert!(!reference
            .kv_cache
            .as_ref()
            .unwrap()
            .has_same_storage(candidate.kv_cache.as_ref().unwrap()));
        let mut previous = None;
        for _ in 0..2 {
            let destination = crate::test_files::TestReportPath::new();
            let mut report = Report::create(destination.path())?;
            let summary = probe_models(&mut report, &mut reference, &mut candidate, &trace)?;
            assert_eq!(summary["predictions"], 8);
            assert_eq!(summary["head_probes"], 8);
            assert_eq!(summary["qualifies_for_promotion"], false);
            let rows: Vec<Value> = std::fs::read_to_string(destination.path())?
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
            assert_eq!(rows.len(), 16);
            for (step, pair) in rows.chunks_exact(2).enumerate() {
                assert_eq!(pair[0]["prefix_tokens"], 2 + step);
                assert_eq!(
                    pair[0]["expected_reference_token_id"],
                    trace.reference_ids[step]
                );
                assert_eq!(
                    pair[1]["expected_reference_token_id"],
                    trace.reference_ids[step]
                );
                assert_eq!(
                    pair[0]["reference_greedy_token_id"],
                    pair[1]["reference_hidden_native_head"]["actual_greedy_token_id"]
                );
                assert_eq!(
                    pair[0]["prepared_greedy_token_id"],
                    pair[1]["prepared_hidden_prepared_head"]["actual_greedy_token_id"]
                );
                if step > trace.first_difference {
                    assert!(pair[0]["historical_prepared_token_on_same_prefix"].is_null());
                }
                if execution.mode == m40_llm::infer::execution_weights::Q8ExecutionMode::Off {
                    assert_eq!(pair[1]["hidden_difference"]["max_abs_diff"], 0.0);
                    assert_eq!(
                        pair[1]["generic_whole_path_difference"]["max_abs_diff"],
                        0.0
                    );
                    assert_eq!(pair[0]["has_same_actual_greedy"], true);
                }
            }
            if let Some(previous) = &previous {
                assert_eq!(previous, &rows);
            }
            previous = Some(rows);
        }
    }
    assert_eq!(CudaContext::total_device_bytes(), before);
    Ok(())
}

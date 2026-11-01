use super::*;
use m40_llm::{
    evaluation::{
        corpus::Corpus,
        ffn_capture::{
            FfnCaptureArtifact, FfnCaptureCollector, FfnCapturePlan, FfnLayerSource,
            FfnTensorSource,
        },
        identity::sha256,
    },
    gguf::GgmlDType,
};

/// Equal scores use channel identity, independent of rank implementation;
/// malformed counts/nonfinite data cannot produce a plausible selection.
#[test]
fn test_top_k_ties_and_invalid_scores() -> Result<()> {
    assert_eq!(predictor::select_top_k(&[2.0, 4.0, 4.0, 1.0], 2)?, [1, 2]);
    assert_eq!(predictor::select_top_k(&[1.0; 5], 3)?, [0, 1, 2]);
    assert!(predictor::select_top_k(&[1.0], 0).is_err());
    assert!(predictor::select_top_k(&[1.0], 2).is_err());
    assert!(predictor::select_top_k(&[f64::NAN], 1).is_err());
    let selected = predictor::random_selection(31, 7, 19)?;
    assert_eq!(selected.len(), 7);
    assert!(selected.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(selected.iter().all(|&channel| channel < 31));
    assert_eq!(selected, predictor::random_selection(31, 7, 19)?);
    Ok(())
}

/// Cosine sketches preserve sign/scale relations, normalize finite vectors,
/// and distinguish a zero feature from an invalid nonfinite feature.
#[test]
fn test_sketch_scale_sign_zero_and_finiteness() -> Result<()> {
    let values: Vec<_> = (0..32).map(|index| index as f32 - 15.0).collect();
    let original = sketch(&values)?;
    let doubled = sketch(&values.iter().map(|value| value * 2.0).collect::<Vec<_>>())?;
    let opposite = sketch(&values.iter().map(|value| -value).collect::<Vec<_>>())?;
    for index in 0..16 {
        assert!((original[index] - doubled[index]).abs() < 1e-12);
        assert!((original[index] + opposite[index]).abs() < 1e-12);
    }
    assert!((original.iter().map(|value| value * value).sum::<f64>() - 1.0).abs() < 1e-12);
    assert_eq!(sketch(&[0.0; 32])?, [0.0; 16]);
    assert!(sketch(&[f32::INFINITY]).is_err());
    Ok(())
}

fn feature(axis: usize) -> [f64; 16] {
    let mut value = [0.0; 16];
    value[axis] = 1.0;
    value
}

/// A skipped token cannot teach the history; predictions require a prior
/// scheduled refresh and cannot cross an unobserved refresh boundary.
#[test]
fn test_history_accepts_only_causal_scheduled_labels() -> Result<()> {
    let mut history = History::new(4, 2, 8)?;
    assert!(history.predict(1, Signal::Latest, None).is_err());
    assert!(history
        .observe_refresh(1, &[0, 1], feature(0), None)
        .is_err());
    history.observe_refresh(0, &[0, 1], feature(0), None)?;
    assert!(history.predict(0, Signal::Latest, None).is_err());
    assert!(history.predict(8, Signal::Latest, None).is_err());
    assert!(history
        .observe_refresh(16, &[2, 3], feature(1), None)
        .is_err());
    assert!(history
        .observe_refresh(8, &[2, 2], feature(1), None)
        .is_err());
    assert_eq!(history.predict(7, Signal::Latest, None)?.channels, [0, 1]);
    history.observe_refresh(8, &[2, 3], feature(1), None)?;
    let latest = history.predict(9, Signal::Latest, None)?;
    assert_eq!(latest.channels, [2, 3]);
    assert_eq!(latest.source_rows, [8]);
    let cumulative = history.predict(9, Signal::Frequency, None)?;
    assert_eq!(cumulative.channels, [0, 1]);
    assert_eq!(cumulative.source_rows, [0, 8]);
    Ok(())
}

/// Current and preceding-layer features retrieve different labels when their
/// nearest states differ. Ties and zero queries choose the latest refresh;
/// bounded eviction prevents access to older labels through the sketch bank.
#[test]
fn test_nearest_feature_recency_fallback_and_eviction() -> Result<()> {
    let mut history = History::new(4, 1, 8)?;
    history.observe_refresh(0, &[0], feature(0), Some(feature(1)))?;
    history.observe_refresh(8, &[1], feature(1), Some(feature(0)))?;
    assert_eq!(
        history
            .predict(9, Signal::CurrentSketch, Some(feature(0)))?
            .channels,
        [0]
    );
    assert_eq!(
        history
            .predict(9, Signal::PreviousSketch, Some(feature(0)))?
            .channels,
        [1]
    );
    assert_eq!(
        history
            .predict(9, Signal::CurrentSketch, Some(feature(2)))?
            .source_rows,
        [8]
    );
    assert_eq!(
        history
            .predict(9, Signal::CurrentSketch, Some([0.0; 16]))?
            .source_rows,
        [8]
    );
    for row in (16..=64).step_by(8) {
        history.observe_refresh(row, &[2], feature(1), Some(feature(0)))?;
    }
    assert_eq!(
        history
            .predict(65, Signal::CurrentSketch, Some(feature(0)))?
            .source_rows,
        [64]
    );
    let fresh_task = History::new(4, 1, 8)?;
    assert!(fresh_task
        .predict(65, Signal::CurrentSketch, Some(feature(0)))
        .is_err());
    Ok(())
}

/// Weighted contribution can disagree with set overlap; zero oracle mass is
/// explicit and never turns into a passing row or a NaN.
#[test]
fn test_metrics_distinguish_overlap_mass_and_zero() -> Result<()> {
    let metrics = measure(
        &[4.0, 3.0, 2.0, 1.0],
        &[1.0, 2.0, 3.0, 4.0],
        &[0, 1],
        &[1, 2],
    )?;
    assert_eq!(metrics.recall, 0.5);
    assert_eq!(metrics.hidden_l1_fraction, Some(0.5));
    assert_eq!(metrics.weighted_fraction, Some(0.6));
    assert_eq!(metrics.weighted_oracle_fraction, Some(1.2));
    assert!(!metrics.passes_row_screen);
    let zero = measure(&[0.0; 4], &[1.0; 4], &[0, 1], &[0, 1])?;
    assert!(zero.has_zero_oracle_mass);
    assert_eq!(zero.weighted_oracle_fraction, None);
    assert!(!zero.passes_row_screen);
    assert!(measure(&[1.0; 4], &[1.0; 4], &[0, 1], &[1, 1]).is_err());
    Ok(())
}

/// Dense refresh successes do not inflate the sparse screen, and bootstrap
/// plus refresh latency is charged in the remaining-budget arithmetic.
#[test]
fn test_summary_excludes_refresh_success_and_prices_dense_steps() -> Result<()> {
    let success = measure(&[1.0; 4], &[1.0; 4], &[0, 1], &[0, 1])?;
    let failure = measure(&[1.0; 4], &[1.0; 4], &[0, 1], &[2, 3])?;
    let mut summary = Summary::default();
    summary.observe(&success, true);
    assert!(!summary.passes_screen());
    assert_eq!(remaining_budget(1.0, 0.5, &summary, 1), None);
    summary.observe(&failure, false);
    assert_eq!(summary.passing_sparse_rows, 0);
    assert!(!summary.passes_screen());
    let expected = (3.0 / 1.05 - 2.0 - 0.5) * 1000.0;
    assert!((remaining_budget(1.0, 0.5, &summary, 1).unwrap() - expected).abs() < 1e-10);
    Ok(())
}

/// Signed Q8 payloads and unequal per-output scales contribute to each
/// logical input column, including both groups at the block boundary.
#[test]
fn test_down_norms_preserve_q8_coordinates_and_scales() -> Result<()> {
    let mut bytes = Vec::new();
    for (scale, code) in [(3.0, 1i8), (5.0, -2), (4.0, -1), (12.0, 2)] {
        bytes.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        bytes.extend(std::iter::repeat_n(code as u8, 32));
    }
    let norms = evidence::calculate_down_norms(&bytes, 64, 2)?;
    assert_eq!(&norms[..32], &[5.0; 32]);
    assert_eq!(&norms[32..], &[26.0; 32]);
    assert!(evidence::calculate_down_norms(&bytes[..bytes.len() - 1], 64, 2).is_err());
    Ok(())
}

fn paired_evidence(generated_count: usize) -> Result<(FfnCaptureArtifact, Vec<Value>, String)> {
    let corpus = Corpus::parse(include_str!("../../data/research_quality/v1.json"))?;
    let configuration = json!({"fixture":true});
    let plan = FfnCapturePlan::new(
        "1".repeat(64),
        "2".repeat(64),
        sha256(&serde_json::to_vec(&configuration)?)?,
        &corpus,
        36,
        vec![0, 16, 17, 34, 35],
        32,
        64,
        160,
    )?;
    let sources = [0, 16, 17, 34, 35]
        .into_iter()
        .map(|layer| FfnLayerSource {
            layer_id: layer,
            gate: FfnTensorSource {
                name: format!("{layer}.gate"),
                byte_offset: (layer * 3 * 2176) as u64,
                dtype: GgmlDType::Q8_0,
                shape: [32, 64],
                source_sha256: "3".repeat(64),
            },
            up: FfnTensorSource {
                name: format!("{layer}.up"),
                byte_offset: ((layer * 3 + 1) * 2176) as u64,
                dtype: GgmlDType::Q8_0,
                shape: [32, 64],
                source_sha256: "3".repeat(64),
            },
            down: FfnTensorSource {
                name: format!("{layer}.down"),
                byte_offset: ((layer * 3 + 2) * 2176) as u64,
                dtype: GgmlDType::Q8_0,
                shape: [64, 32],
                source_sha256: "3".repeat(64),
            },
        })
        .collect();
    let mut collector =
        FfnCaptureCollector::new(plan.clone(), "4".repeat(64), sources, 4 * 1024 * 1024)?;
    let mut records = vec![
        json!({"kind":"ffn_capture_manifest", "schema_version":1,"build_profile":"release",
        "revision":"d6f6ae9898db16b31956a3a07a6511de4b7ace95","worktree":"","should_capture_complete_generations":true,
        "model":{"sha256":plan.model_sha256},"corpus_sha256":plan.corpus_sha256,"execution_sha256":plan.execution_sha256,
        "execution_configuration":configuration}),
    ];
    for (index, task) in corpus.calibration.tasks.iter().enumerate() {
        let count = if index == 1 { 1 } else { generated_count };
        collector.begin_task(&task.id, &[11, 12])?;
        for row in 0..count - 1 {
            for layer in [0, 16, 17, 34, 35] {
                collector.observe(layer, (2 + row) as u32, &[1.0; 32], &[2.0; 64])?;
            }
        }
        let generated = vec![17; count];
        collector.finish_task(&generated)?;
        let generation = json!({"generated_token_ids":generated,"output":"paired","metrics":{"generated_tokens":count,"stop_reason":"stop"}});
        records.push(json!({"kind":"ffn_capture_task","id":task.id,"has_identical_generation":true,
            "prompt_token_ids":[11,12],"baseline":generation,"captured":generation,"captured_rows_per_layer":count-1}));
    }
    let artifact = collector.finish()?;
    let digest = sha256(&serde_json::to_vec(&artifact)?)?;
    records.push(json!({"kind":"ffn_capture_complete","is_complete":true,"artifact_sha256":digest,"device_bytes_before":0,"device_bytes_after":0}));
    Ok((artifact, records, digest))
}

/// Incorrect provenance, unequal actual generations, counts, stopping reasons,
/// incomplete export and owner leaks are rejected before any metrics are used.
#[test]
fn test_report_rejects_forged_pairing_and_incomplete_capture() -> Result<()> {
    let (artifact, records, digest) = paired_evidence(2)?;
    evidence::validate_report(&artifact, &records, &digest)?;
    for (index, path, value) in [
        (0, "/worktree", json!("modified")),
        (0, "/build_profile", json!("debug")),
        (0, "/model/sha256", json!("wrong")),
        (0, "/should_capture_complete_generations", json!(false)),
        (1, "/captured/generated_token_ids", json!([17, 19])),
        (1, "/captured/metrics/generated_tokens", json!(3)),
        (1, "/captured/metrics/stop_reason", json!("length")),
        (1, "/captured_rows_per_layer", json!(0)),
        (7, "/is_complete", json!(false)),
        (7, "/device_bytes_after", json!(64)),
    ] {
        let mut changed = records.clone();
        *changed[index].pointer_mut(path).unwrap() = value;
        assert!(
            evidence::validate_report(&artifact, &changed, &digest).is_err(),
            "{path}"
        );
    }
    assert!(evidence::validate_report(&artifact, &records[..7], &digest).is_err());
    assert!(evidence::validate_report(&artifact, &records, "wrong").is_err());
    Ok(())
}

fn analyze_fixture(artifact: FfnCaptureArtifact) -> Result<Vec<Value>> {
    artifact.validate_complete_generations()?;
    let evidence = Evidence {
        artifact,
        down_norms: [0, 17, 35]
            .into_iter()
            .map(|layer| (layer, vec![1.0; 64]))
            .collect(),
        artifact_sha256: String::new(),
        report_sha256: String::new(),
        manifest: json!({}),
    };
    let path = std::env::temp_dir().join(format!(
        "selector-causality-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    let mut report = Report {
        output: BufWriter::new(output),
        summaries: BTreeMap::new(),
    };
    analyze(&evidence, &mut report)?;
    drop(report);
    let text = std::fs::read_to_string(&path)?;
    std::fs::remove_file(path)?;
    text.lines()
        .map(|line| Ok(serde_json::from_str(line)?))
        .collect()
}

/// Changing a skipped token's true hidden values changes the diagnostic
/// previous-oracle result, but cannot change any causal policy's labels or
/// source rows. Empty histories remain explicit rather than disappearing.
#[test]
fn test_complete_analysis_cannot_learn_from_skipped_hidden_values() -> Result<()> {
    let (artifact, _, _) = paired_evidence(18)?;
    let mut altered = artifact.clone();
    for layer in &mut altered.layers {
        layer.swiglu_inputs[64..96].fill(0.0);
        layer.swiglu_inputs[96..128].fill(100.0);
    }
    let original = analyze_fixture(artifact)?;
    let changed = analyze_fixture(altered)?;
    assert_eq!(original.len(), changed.len());
    let mut causal_rows = 0;
    let mut changed_previous_oracles = 0;
    let mut empty_histories = 0;
    for (a, b) in original.iter().zip(&changed) {
        if a["kind"] == "selector_row" {
            let method = a["method"].as_str().unwrap();
            if method == "first_token"
                || method.starts_with("refresh_")
                || method.contains("sketch_")
            {
                assert_eq!(a["selected_channels"], b["selected_channels"]);
                assert_eq!(a["source_rows"], b["source_rows"]);
                if a["is_dense_refresh"] == false {
                    assert!(a["source_rows"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|source| source.as_u64().unwrap() < a["row"].as_u64().unwrap()));
                }
                causal_rows += 1;
            }
            if method == "previous_oracle" && a["selected_channels"] != b["selected_channels"] {
                changed_previous_oracles += 1;
            }
        }
        if a["kind"] == "selector_summary" && a["scope"] == "cal-fact-depot" {
            assert_eq!(a["summary"]["observed_rows"], 0);
            assert_eq!(a["summary"]["eligible_sparse_rows"], 0);
            assert_eq!(a["bootstrap_dense_steps"], 1);
            assert_eq!(a["passes_fixed_screen"], false);
            empty_histories += 1;
        }
    }
    assert!(causal_rows > 0 && changed_previous_oracles == 6 && empty_histories == 68);
    Ok(())
}

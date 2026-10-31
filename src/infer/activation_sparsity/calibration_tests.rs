use super::*;

fn collector(budget: usize) -> Result<ActivationCalibrationCollector> {
    let corpus = Corpus::parse(include_str!("../../../data/research_quality/v1.json"))?;
    let plan =
        ActivationCalibrationPlan::new("1".repeat(64), "2".repeat(64), "3".repeat(64), &corpus, 2)?;
    let layers = (0..2)
        .map(|id| CalibrationLayer {
            layer_id: id,
            tensor_name: format!("blk.{id}.ffn_down.weight"),
            byte_offset: id as u64 * 136,
            input_dim: 4,
            output_dim: 2,
        })
        .collect();
    ActivationCalibrationCollector::new(plan, "4".repeat(64), layers, budget)
}

fn complete() -> Result<ActivationCalibration> {
    let mut collector = collector(384)?;
    for id in collector.plan.calibration_task_ids.clone() {
        collector.begin_task(&id, 3)?;
        assert!(!collector.should_observe(2));
        for position in 3..5 {
            collector.observe(0, position, &[0.0, 1.0, 2.0, 3.0])?;
            collector.observe(1, position, &[0.0, 10.0, 20.0, 30.0])?;
        }
        assert!(!collector.should_observe(5));
        collector.finish_task(3)?;
    }
    collector.finish()
}

/// Only generated-token inputs are captured, with the same range at every
/// layer; fitted thresholds retain per-layer scale and round-trip identities.
#[test]
fn test_calibration_artifact_tracks_generated_ranges_and_layer_thresholds() -> Result<()> {
    let artifact = complete()?;
    assert_eq!(artifact.tasks.len(), 6);
    assert!(artifact
        .tasks
        .iter()
        .all(|task| task.captured_absolute_positions == [3, 4]));
    assert_eq!(artifact.observation_payload_bytes, 384);
    assert!(artifact.observation_capacity_bytes >= 384);
    assert_eq!(artifact.layers[0].observed_rows, 12);
    let thresholds = artifact.thresholds(0.5)?;
    assert_eq!(thresholds[0].magnitude(), 1.0);
    assert_eq!(thresholds[1].magnitude(), 10.0);
    assert_eq!(artifact.layers[1].thresholds[2].achieved_sparsity, 0.5);
    assert!(artifact
        .thresholds(0.0)?
        .iter()
        .all(|threshold| threshold.magnitude() == 0.0));
    let parsed = ActivationCalibration::parse(&serde_json::to_string(&artifact)?)?;
    assert_eq!(
        serde_json::to_value(parsed)?,
        serde_json::to_value(artifact)?
    );
    Ok(())
}

/// Held-out/duplicate tasks, wrong dimensions, nonfinite values and missing
/// layer rows must fail before a partial population can be called calibrated.
#[test]
fn test_calibration_rejects_leakage_incomplete_and_out_of_order_rows() -> Result<()> {
    assert!(collector(383).is_err());
    let mut collector = collector(384)?;
    assert!(collector.begin_task("held-out-task", 3).is_err());
    let id = collector.plan.calibration_task_ids[0].clone();
    collector.begin_task(&id, 3)?;
    assert!(collector.observe(0, 2, &[1.0; 4]).is_err());
    assert!(collector.observe(0, 4, &[1.0; 4]).is_err());
    assert!(collector.observe(0, 3, &[1.0; 3]).is_err());
    assert!(collector.observe(0, 3, &[f32::NAN; 4]).is_err());
    collector.observe(0, 3, &[1.0; 4])?;
    assert!(collector.observe(0, 3, &[1.0; 4]).is_err());
    assert!(collector.finish_task(2).is_err());
    collector.observe(1, 3, &[1.0; 4])?;
    collector.finish_task(2)?;
    assert!(collector.begin_task(&id, 3).is_err());
    assert!(collector.finish().is_err());
    Ok(())
}

/// External artifacts cannot change split, token positions, hashes, layer IDs,
/// zero control or memory counts and still be accepted for application.
#[test]
fn test_calibration_parser_rejects_inconsistent_external_artifacts() -> Result<()> {
    let valid = serde_json::to_value(complete()?)?;
    for (path, replacement) in [
        ("/plan/split", serde_json::json!("held_out")),
        ("/plan/model_sha256", serde_json::json!("not-a-hash")),
        (
            "/tasks/0/captured_absolute_positions",
            serde_json::json!([0, 1]),
        ),
        ("/tasks/0/prompt_tokens", serde_json::json!(usize::MAX)),
        ("/layers/0/source/layer_id", serde_json::json!(5)),
        ("/layers/0/thresholds/0/magnitude", serde_json::json!(1.0)),
        ("/layers/0/thresholds/1/magnitude", serde_json::json!(-1.0)),
        ("/layers/0/observed_rows", serde_json::json!(0)),
        ("/observation_payload_bytes", serde_json::json!(0)),
    ] {
        let mut invalid = valid.clone();
        *invalid.pointer_mut(path).unwrap() = replacement;
        assert!(
            ActivationCalibration::parse(&serde_json::to_string(&invalid)?).is_err(),
            "{path}"
        );
    }
    assert!(complete()?.thresholds(0.7).is_err());
    Ok(())
}

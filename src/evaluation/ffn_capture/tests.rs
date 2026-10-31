use super::*;
use crate::{evaluation::corpus::Corpus, gguf::GgmlDType};
use anyhow::Result;
use serde_json::json;

const BUDGET: usize = 6 * 2 * (32 + 64) * 2 * 4;

fn corpus() -> Result<Corpus> {
    Corpus::parse(include_str!("../../../data/research_quality/v1.json"))
}

fn plan() -> Result<FfnCapturePlan> {
    FfnCapturePlan::new(
        "1".repeat(64),
        "2".repeat(64),
        "3".repeat(64),
        &corpus()?,
        4,
        vec![0, 3],
        32,
        64,
        2,
    )
}

fn sources() -> Vec<FfnLayerSource> {
    [0, 3]
        .into_iter()
        .map(|layer_id| {
            let tensor = |kind: &str, index: u64, shape| FfnTensorSource {
                name: format!("blk.{layer_id}.ffn_{kind}.weight"),
                byte_offset: (layer_id as u64 * 3 + index) * 2176,
                dtype: GgmlDType::Q8_0,
                shape,
                source_sha256: "4".repeat(64),
            };
            FfnLayerSource {
                layer_id,
                gate: tensor("gate", 0, [32, 64]),
                up: tensor("up", 1, [32, 64]),
                down: tensor("down", 2, [64, 32]),
            }
        })
        .collect()
}

fn collector(budget: usize) -> Result<FfnCaptureCollector> {
    FfnCaptureCollector::new(plan()?, "5".repeat(64), sources(), budget)
}

fn observe_task(collector: &mut FfnCaptureCollector, task_index: usize) -> Result<()> {
    for position in 3..5 {
        for layer in [0, 3] {
            let value = (task_index * 100 + layer * 10 + position as usize) as f32;
            let mut normalized = vec![value; 32];
            normalized[0] = -0.0;
            normalized[1] = f32::from_bits(1);
            collector.observe(layer, position, &normalized, &vec![-value; 64])?;
        }
    }
    collector.finish_task(&[17, 18, 19])
}

fn complete() -> Result<FfnCaptureArtifact> {
    let mut collector = collector(BUDGET)?;
    for (index, task) in plan()?.calibration_task_ids.iter().enumerate() {
        collector.begin_task(task, &[11, 12, 13])?;
        observe_task(&mut collector, index)?;
    }
    collector.finish()
}

/// Paired inputs retain exact FP32 values, original layer IDs, generated input
/// positions and real row differences, without thresholding either vector.
#[test]
fn test_paired_rows_preserve_values_positions_and_source_identity() -> Result<()> {
    let artifact = complete()?;
    assert_eq!(artifact.observation_payload_bytes, BUDGET);
    assert_eq!(artifact.observation_capacity_bytes, BUDGET);
    assert_eq!(artifact.layers[0].source.layer_id, 0);
    assert_eq!(artifact.layers[1].source.layer_id, 3);
    assert_eq!(artifact.layers[0].source.down.shape, [64, 32]);
    assert_eq!(artifact.layers[0].source.down.byte_len()?, 2176);
    assert_eq!(artifact.layers[1].source.gate.byte_offset, 9 * 2176);
    for (index, layer) in artifact.layers.iter().enumerate() {
        assert_eq!(layer.rows.len(), 12);
        assert_eq!(layer.normalized_inputs.len(), 12 * 32);
        assert_eq!(layer.swiglu_inputs.len(), 12 * 64);
        let layer_offset = [0, 30][index];
        for row in 0..12 {
            let value = ((row / 2) * 100 + layer_offset + 3 + row % 2) as f32;
            assert_eq!(
                layer.normalized_inputs[row * 32].to_bits(),
                (-0.0f32).to_bits()
            );
            assert_eq!(layer.normalized_inputs[row * 32 + 1].to_bits(), 1);
            assert_eq!(
                &layer.normalized_inputs[row * 32 + 2..(row + 1) * 32],
                &[value; 30]
            );
            assert_eq!(
                &layer.swiglu_inputs[row * 64..(row + 1) * 64],
                &[-value; 64]
            );
            assert_eq!(layer.rows[row].absolute_position, (3 + row % 2) as u32);
            assert_eq!(layer.rows[row].task_id, artifact.tasks[row / 2].id);
        }
        assert!(layer
            .normalized_inputs
            .chunks_exact(32)
            .take(4)
            .collect::<Vec<_>>()
            .windows(2)
            .all(|pair| pair[0] != pair[1]));
    }
    assert!(artifact
        .tasks
        .iter()
        .all(|task| task.prompt_token_ids == [11, 12, 13]
            && task.generated_token_ids == [17, 18, 19]));
    Ok(())
}

/// JSON round-trips the finite FP32 observations bit-for-bit, including signed
/// zero and subnormals, while retaining the pairing of the two matrix inputs.
#[test]
fn test_artifact_round_trips_paired_fp32_bits() -> Result<()> {
    let original = complete()?;
    let parsed = FfnCaptureArtifact::parse(&serde_json::to_string(&original)?)?;
    for (a, b) in original.layers.iter().zip(&parsed.layers) {
        assert_eq!(a.source, b.source);
        assert_eq!(a.rows, b.rows);
        for (x, y) in a
            .normalized_inputs
            .iter()
            .chain(&a.swiglu_inputs)
            .zip(b.normalized_inputs.iter().chain(&b.swiglu_inputs))
        {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }
    assert_eq!(
        serde_json::to_value(original)?,
        serde_json::to_value(parsed)?
    );
    Ok(())
}

/// Held-out, duplicate and out-of-order tasks cannot contribute inputs. Prompt
/// positions and rows beyond the bound are excluded before any read is needed.
#[test]
fn test_only_bounded_generated_inputs_from_calibration_are_eligible() -> Result<()> {
    let mut collector = collector(BUDGET)?;
    assert!(!collector.should_observe(3));
    assert!(collector
        .begin_task(&corpus()?.held_out.tasks[0].id, &[1, 2, 3])
        .is_err());
    let ids = plan()?.calibration_task_ids;
    assert!(collector.begin_task(&ids[1], &[1, 2, 3]).is_err());
    assert!(collector.begin_task(&ids[0], &[]).is_err());
    collector.begin_task(&ids[0], &[1, 2, 3])?;
    assert!(collector.begin_task(&ids[0], &[1, 2, 3]).is_err());
    assert!(!collector.should_observe(2));
    assert!(collector.should_observe(3));
    assert!(collector.should_observe(4));
    assert!(!collector.should_observe(5));
    assert!(!collector.should_observe(u32::MAX));
    observe_task(&mut collector, 0)?;
    assert!(!collector.should_observe(3));
    assert!(collector.begin_task(&ids[0], &[1, 2, 3]).is_err());
    assert!(collector.validate_complete().is_err());
    Ok(())
}

/// A malformed pair or position publishes neither vector. Failed completion is
/// recoverable, and missing layers/tasks cannot be exported as valid evidence.
#[test]
fn test_invalid_rows_and_incomplete_capture_do_not_publish_partial_pairs() -> Result<()> {
    let mut collector = collector(BUDGET)?;
    let ids = plan()?.calibration_task_ids;
    collector.begin_task(&ids[0], &[11, 12, 13])?;
    for (layer, position, normalized, swiglu) in [
        (0, 2, vec![1.0; 32], vec![2.0; 64]),
        (0, 4, vec![1.0; 32], vec![2.0; 64]),
        (1, 3, vec![1.0; 32], vec![2.0; 64]),
        (0, 3, vec![1.0; 31], vec![2.0; 64]),
        (0, 3, vec![1.0; 32], vec![2.0; 63]),
        (0, 3, vec![f32::NAN; 32], vec![2.0; 64]),
        (0, 3, vec![1.0; 32], vec![f32::INFINITY; 64]),
    ] {
        assert!(collector
            .observe(layer, position, &normalized, &swiglu)
            .is_err());
    }
    assert!(collector.finish_task(&[]).is_err());
    assert!(collector.finish_task(&[1, 2]).is_err());
    observe_task(&mut collector, 0)?;
    assert!(collector.validate_complete().is_err());
    for (index, id) in ids.iter().enumerate().skip(1) {
        collector.begin_task(id, &[11, 12, 13])?;
        observe_task(&mut collector, index)?;
    }
    assert_eq!(
        serde_json::to_value(collector.finish()?)?,
        serde_json::to_value(complete()?)?
    );
    Ok(())
}

/// Emitted tokens are not all processed inputs: one actual generated token has
/// zero generated-input rows; a two-token answer has exactly one per layer.
#[test]
fn test_short_generations_use_actual_processed_input_count() -> Result<()> {
    let mut capture = collector(BUDGET)?;
    for (index, id) in plan()?.calibration_task_ids.iter().enumerate() {
        capture.begin_task(id, &[11, 12, 13])?;
        if index == 0 {
            for layer in [0, 3] {
                capture.observe(layer, 3, &[1.0; 32], &[2.0; 64])?;
                assert!(capture.observe(layer, 3, &[1.0; 32], &[2.0; 64]).is_err());
            }
            capture.finish_task(&[17, 18])?;
        } else {
            capture.finish_task(&[17])?;
        }
    }
    let artifact = capture.finish()?;
    assert_eq!(artifact.layers[0].rows.len(), 1);
    assert_eq!(artifact.observation_payload_bytes, 2 * (32 + 64) * 4);
    assert_eq!(artifact.observation_capacity_bytes, BUDGET);
    let mut empty = collector(BUDGET)?;
    for id in plan()?.calibration_task_ids {
        empty.begin_task(&id, &[1])?;
        empty.finish_task(&[17])?;
    }
    assert!(empty.finish().is_err());
    Ok(())
}

/// Pre-reservation enforces the full bounded FP32 budget, and invalid source
/// geometry/aliasing is rejected before a collector can observe anything.
#[test]
fn test_capture_rejects_budget_identity_and_geometry_errors() -> Result<()> {
    assert!(collector(BUDGET - 1).is_err());
    for bad_sources in [
        {
            let mut value = sources();
            value[0].gate.dtype = GgmlDType::F16;
            value
        },
        {
            let mut value = sources();
            value[0].gate.shape = [64, 32];
            value
        },
        {
            let mut value = sources();
            value[0].gate.name.clear();
            value
        },
        {
            let mut value = sources();
            value[0].gate.source_sha256 = "bad".to_owned();
            value
        },
        {
            let mut value = sources();
            value[0].gate.byte_offset = u64::MAX;
            value
        },
        {
            let mut value = sources();
            value[0].up.byte_offset = 1;
            value
        },
        {
            let mut value = sources();
            value[1].layer_id = 1;
            value
        },
    ] {
        assert!(FfnCaptureCollector::new(plan()?, "5".repeat(64), bad_sources, BUDGET).is_err());
    }
    let mut overflow = plan()?;
    overflow.d_model = usize::MAX - 31;
    let mut big_sources = sources();
    for source in &mut big_sources {
        source.gate.shape[0] = overflow.d_model;
        source.up.shape[0] = overflow.d_model;
        source.down.shape[1] = overflow.d_model;
    }
    assert!(FfnCaptureCollector::new(overflow, "5".repeat(64), big_sources, usize::MAX).is_err());
    assert!(FfnCaptureCollector::new(plan()?, "bad".to_owned(), sources(), BUDGET).is_err());
    Ok(())
}

/// Imported observations cannot relabel split, task/layer order, source types,
/// row positions, dimensions or allocated/payload counts and remain acceptable.
#[test]
fn test_parser_rejects_inconsistent_external_observations() -> Result<()> {
    let valid = serde_json::to_value(complete()?)?;
    for (path, replacement) in [
        ("/schema_version", json!(2)),
        ("/plan/split", json!("held_out")),
        ("/plan/model_sha256", json!("bad")),
        ("/plan/selected_layer_ids", json!([3, 0])),
        ("/plan/selected_layer_ids", json!([0, 0])),
        ("/plan/source_layer_count", json!(3)),
        ("/plan/max_generated_rows_per_task", json!(0)),
        ("/plan/max_generated_rows_per_task", json!(33)),
        ("/plan/d_model", json!(31)),
        ("/tasks/0/id", json!("held-out")),
        ("/tasks/0/prompt_token_ids", json!([])),
        ("/tasks/0/generated_token_ids", json!([17])),
        ("/layers/0/source/layer_id", json!(1)),
        ("/layers/0/source/gate/source_sha256", json!("bad")),
        ("/layers/0/rows/0/absolute_position", json!(2)),
        ("/layers/1/rows/1/task_id", json!("other-task")),
        ("/layers/0/normalized_inputs", json!([0.0])),
        ("/layers/0/swiglu_inputs", json!([0.0])),
        ("/observation_payload_bytes", json!(0)),
        ("/observation_capacity_bytes", json!(0)),
    ] {
        let mut invalid = valid.clone();
        *invalid.pointer_mut(path).unwrap() = replacement;
        assert!(
            FfnCaptureArtifact::parse(&serde_json::to_string(&invalid)?).is_err(),
            "{path}"
        );
    }
    let mut nonfinite = complete()?;
    nonfinite.layers[0].swiglu_inputs[0] = f32::NEG_INFINITY;
    assert!(nonfinite.validate().is_err());
    Ok(())
}

/// A syntactically valid plan still needs the actual corpus hash and ordered
/// calibration task IDs; a self-consistent held-out relabel is not provenance.
#[test]
fn test_consumers_must_bind_observations_to_frozen_corpus() -> Result<()> {
    let corpus = corpus()?;
    let mut plan = plan()?;
    plan.validate_corpus(&corpus, &"2".repeat(64))?;
    assert!(plan.validate_corpus(&corpus, &"9".repeat(64)).is_err());
    plan.calibration_task_ids = corpus
        .held_out
        .tasks
        .iter()
        .map(|task| task.id.clone())
        .collect();
    plan.validate()?;
    assert!(plan.validate_corpus(&corpus, &"2".repeat(64)).is_err());
    Ok(())
}

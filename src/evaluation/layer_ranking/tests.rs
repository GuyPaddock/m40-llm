use super::*;
use crate::evaluation::{
    corpus::{Corpus, Split},
    loss::LossMetrics,
};
use anyhow::Result;

fn plan() -> Result<LayerRankingPlan> {
    let corpus = Corpus::parse(include_str!("../../../data/research_quality/v1.json"))?;
    let passages = corpus
        .calibration
        .passages
        .iter()
        .enumerate()
        .map(|(index, passage)| LayerRankingPassage {
            id: passage.id.clone(),
            token_ids: if index == 0 {
                vec![1, 2, 3]
            } else {
                vec![4, 5, 6, 7]
            },
        })
        .collect();
    LayerRankingPlan::new(
        "1".repeat(64),
        "2".repeat(64),
        "3".repeat(64),
        6,
        2,
        &corpus,
        passages,
    )
}

fn influence() -> Result<LayerInfluence> {
    let plan = plan()?;
    let mut collector = LayerInfluenceCollector::new(plan.clone(), 1024)?;
    for passage in &plan.passages {
        collector.begin_passage(&passage.id)?;
        for position in 0..passage.token_ids.len() - 1 {
            for layer in (0..6).rev() {
                let output = match layer {
                    1 | 4 => [2.0, 0.0],
                    2 => [0.0, 1.0],
                    _ => [-1.0, 0.0],
                };
                collector.observe(layer, position as u32, &[1.0, 0.0], &output)?;
            }
        }
        collector.finish_passage()?;
    }
    collector.finish()
}

fn trial(layer_id: usize, mean: f64) -> Result<LayerRemovalTrial> {
    Ok(LayerRemovalTrial {
        layer_id,
        calibration_loss: LossMetrics::from_sum(mean * 5.0, 5)?,
    })
}

/// Influence measures angular residual change, not magnitude. FP64 handles the
/// full finite FP32 range; undefined and malformed vectors cannot look redundant.
#[test]
fn test_influence_matches_geometry_and_rejects_undefined_states() -> Result<()> {
    assert_eq!(measure_block_influence(&[1.0, 0.0], &[0.0, 2.0])?, 1.0);
    assert_eq!(measure_block_influence(&[1.0, 0.0], &[-2.0, 0.0])?, 2.0);
    for values in [
        [1.0, 2.0],
        [f32::MAX, f32::MAX],
        [f32::from_bits(1), f32::from_bits(2)],
    ] {
        assert!(measure_block_influence(&values, &values)? < 1e-12);
    }
    for (input, output) in [
        (vec![], vec![]),
        (vec![0.0], vec![1.0]),
        (vec![1.0], vec![0.0]),
        (vec![1.0], vec![1.0, 2.0]),
        (vec![f32::NAN], vec![1.0]),
        (vec![1.0], vec![f32::INFINITY]),
    ] {
        assert!(measure_block_influence(&input, &output).is_err());
    }
    Ok(())
}

/// Only calibration passages, original order and valid identities are eligible.
/// Observation memory is reserved against a checked budget before capture.
#[test]
fn test_influence_plan_excludes_held_out_and_rejects_invalid_capacity() -> Result<()> {
    let original = plan()?;
    assert_eq!(original.observed_rows()?, 5);
    let mut bad = original.clone();
    bad.split = Split::HeldOut;
    assert!(LayerInfluenceCollector::new(bad, 1024).is_err());
    let mut bad = original.clone();
    bad.passages[0].id = "held-passage-market".into();
    let corpus = Corpus::parse(include_str!("../../../data/research_quality/v1.json"))?;
    assert!(bad.validate_corpus(&corpus).is_err());
    let mut bad = original.clone();
    bad.execution_sha256 = "not-a-hash".into();
    assert!(bad.validate().is_err());
    let mut bad = original.clone();
    bad.passages[0].token_ids.clear();
    assert!(bad.validate().is_err());
    let mut bad = original.clone();
    bad.layer_count = 2;
    assert!(bad.validate().is_err());
    assert!(LayerInfluenceCollector::new(original, 6 * 5 * 8 - 1).is_err());
    Ok(())
}

/// Capture requires every original layer/position exactly once and excludes the
/// final teacher-forced target. Invalid submissions must not advance collection.
#[test]
fn test_capture_requires_complete_original_layer_positions() -> Result<()> {
    let plan = plan()?;
    let mut collector = LayerInfluenceCollector::new(plan.clone(), 1024)?;
    assert!(!collector.has_active_passage());
    assert!(collector.begin_passage(&plan.passages[1].id).is_err());
    collector.begin_passage(&plan.passages[0].id)?;
    assert!(collector.has_active_passage());
    assert!(collector.validate_complete().is_err());
    assert!(collector.observe(0, 1, &[1.0, 0.0], &[1.0, 0.0]).is_err());
    assert!(collector.observe(0, 0, &[1.0], &[1.0]).is_err());
    assert!(collector.observe(6, 0, &[1.0, 0.0], &[1.0, 0.0]).is_err());
    collector.observe(0, 0, &[1.0, 0.0], &[1.0, 0.0])?;
    assert!(collector.observe(0, 0, &[1.0, 0.0], &[1.0, 0.0]).is_err());
    assert!(collector.finish_passage().is_err());
    for layer in 0..6 {
        if layer != 0 {
            collector.observe(layer, 0, &[1.0, 0.0], &[1.0, 0.0])?;
        }
        collector.observe(layer, 1, &[1.0, 0.0], &[1.0, 0.0])?;
    }
    assert!(collector.observe(0, 2, &[1.0, 0.0], &[1.0, 0.0]).is_err());
    collector.finish_passage()?;
    assert!(collector.finish().is_err());
    let artifact = influence()?;
    assert!(artifact
        .layers
        .iter()
        .all(|layer| layer.block_influence.len() == 5));
    assert_eq!(artifact.ranked_layers()?, [1, 4, 2, 3]);
    let mut bad = artifact;
    bad.layers[2].block_influence.pop();
    assert!(bad.ranked_layers().is_err());
    Ok(())
}

/// Each removal is chosen after re-evaluating every remaining interior layer.
/// Exact loss ties use BI then original ID, independent of trial submission order.
#[test]
fn test_iterative_loss_ranking_recomputes_support_and_freezes_masks() -> Result<()> {
    let mut ranker = LayerRemovalRanker::new(influence()?, 3)?;
    assert_eq!(ranker.candidate_layers(), [1, 2, 3, 4]);
    assert_eq!(
        ranker.record_stage(vec![
            trial(4, 3.0)?,
            trial(3, 2.0)?,
            trial(2, 1.0)?,
            trial(1, 4.0)?
        ])?,
        2
    );
    assert_eq!(ranker.candidate_layers(), [1, 3, 4]);
    assert_eq!(
        ranker.record_stage(vec![trial(3, 2.0)?, trial(1, 0.5)?, trial(4, 1.0)?])?,
        1
    );
    assert_eq!(
        ranker.record_stage(vec![trial(3, 1.0)?, trial(4, 1.0)?])?,
        4
    );
    assert!(ranker.record_stage(Vec::new()).is_err());
    let ranking = ranker.finish()?;
    assert_eq!(ranking.selected_layers(1)?, [2]);
    assert_eq!(ranking.selected_layers(2)?, [1, 2]);
    assert_eq!(ranking.selected_layers(3)?, [1, 2, 4]);
    assert!(ranking.selected_layers(0).is_err());
    let parsed = LayerRemovalRanking::parse(&serde_json::to_string(&ranking)?)?;
    assert_eq!(parsed.selected_layers(3)?, [1, 2, 4]);
    let mut corrupt = ranking;
    corrupt.stages[0].winner_layer_id = 4;
    assert!(corrupt.validate().is_err());
    Ok(())
}

/// Missing/duplicate/protected trials and inconsistent loss populations must
/// fail without freezing a partial stage or changing the next candidate set.
#[test]
fn test_loss_ranking_rejects_incomplete_or_mismatched_evidence() -> Result<()> {
    let mut ranker = LayerRemovalRanker::new(influence()?, 3)?;
    for mut trials in [
        vec![trial(1, 1.0)?, trial(2, 1.0)?, trial(3, 1.0)?],
        vec![
            trial(1, 1.0)?,
            trial(1, 1.0)?,
            trial(3, 1.0)?,
            trial(4, 1.0)?,
        ],
        vec![
            trial(0, 1.0)?,
            trial(2, 1.0)?,
            trial(3, 1.0)?,
            trial(4, 1.0)?,
        ],
        vec![
            trial(1, 1.0)?,
            trial(2, 1.0)?,
            trial(3, 1.0)?,
            trial(4, 1.0)?,
        ],
    ] {
        if trials.len() == 4 && trials[0].layer_id == 1 && trials[1].layer_id == 2 {
            trials[0].calibration_loss = LossMetrics::from_sum(4.0, 4)?;
        }
        assert!(ranker.record_stage(trials).is_err());
        assert!(ranker.selected_layers().is_empty());
        assert_eq!(ranker.candidate_layers(), [1, 2, 3, 4]);
    }
    assert!(ranker.finish().is_err());
    Ok(())
}

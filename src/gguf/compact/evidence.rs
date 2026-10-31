use crate::evaluation::{
    corpus::{Corpus, EvaluationSplit, Split},
    identity::sha256,
    layer_ranking::LayerRemovalRanking,
    loss::LossMetrics,
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::{collections::HashSet, fs::File, io::Read, path::Path};

const MAX_REPORT_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Serialize)]
pub struct CompactQualitySummary {
    pub split: Split,
    pub tasks: usize,
    pub dense_reference_passes: usize,
    pub candidate_passes: usize,
    pub dense_valid_regressions: usize,
    pub inconclusive_tasks: usize,
    pub reference_loss: LossMetrics,
    pub candidate_loss: LossMetrics,
}

/// Completed measurements, not a quality approval. Private fields prevent
/// callers from substituting a mask without revalidating its frozen evidence.
#[derive(Debug, Clone, Serialize)]
pub struct CompactExportEvidence {
    pub(super) source_sha256: String,
    pub(super) ranking_sha256: String,
    pub(super) quality_report_sha256: String,
    pub(super) corpus_sha256: String,
    pub(super) original_layer_count: u32,
    pub(super) hidden_dim: usize,
    pub(super) removed_original_layers: Vec<usize>,
    pub(super) quality: Vec<CompactQualitySummary>,
}

fn read_bounded(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(MAX_REPORT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_REPORT_BYTES,
        "compact export evidence exceeds 64 MiB"
    );
    Ok(bytes)
}

fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= b.abs().max(1.0) * 1e-12
}

fn check_loss(value: &Value, tokens: usize) -> Result<LossMetrics> {
    let loss: LossMetrics = serde_json::from_value(value.clone())?;
    let checked = LossMetrics::from_sum(loss.nll_sum, loss.scored_tokens)?;
    ensure!(
        loss.scored_tokens == tokens
            && close(loss.mean_nll, checked.mean_nll)
            && loss.perplexity_overflow == checked.perplexity_overflow
            && match (loss.perplexity, checked.perplexity) {
                (Some(a), Some(b)) => close(a, b),
                (None, None) => true,
                _ => false,
            },
        "invalid compact export quality loss"
    );
    Ok(loss)
}

fn one_row<'a>(
    rows: &'a [Value],
    kind: &str,
    split: Option<Split>,
    id: Option<&str>,
) -> Result<&'a Value> {
    let split = split.map(|value| serde_json::to_value(value).expect("serializable split"));
    let mut matching = rows.iter().filter(|row| {
        row["kind"] == kind
            && split.as_ref().is_none_or(|split| &row["split"] == split)
            && id.is_none_or(|id| row["id"] == id)
    });
    let row = matching
        .next()
        .with_context(|| format!("missing quality row {kind} {id:?}"))?;
    ensure!(
        matching.next().is_none(),
        "duplicate quality row {kind} {id:?}"
    );
    Ok(row)
}

fn check_split(
    rows: &[Value],
    split: Split,
    corpus: &EvaluationSplit,
    ranking: &LayerRemovalRanking,
) -> Result<CompactQualitySummary> {
    let mut reference_passes = 0;
    let mut candidate_passes = 0;
    let mut regressions = 0;
    let mut inconclusive = 0;
    for task in &corpus.tasks {
        let row = one_row(rows, "model_quality_task", Some(split), Some(&task.id))?;
        ensure!(
            row["user_prompt"] == task.prompt
                && row["scoring_rule"] == serde_json::to_value(&task.scoring)?
                && row["family"] == serde_json::to_value(task.family)?
                && row["max_tokens"].as_u64() == Some(task.max_tokens as u64),
            "quality task/rubric differs from frozen corpus"
        );
        let reference = task.scoring.score(
            row["reference"]["output"]
                .as_str()
                .context("reference output")?,
        )?;
        let candidate = task.scoring.score(
            row["candidate"]["output"]
                .as_str()
                .context("candidate output")?,
        )?;
        ensure!(
            row["reference_score"] == serde_json::to_value(&reference)?
                && row["candidate_score"] == serde_json::to_value(&candidate)?,
            "quality score differs from frozen rubric"
        );
        let conclusion = if !reference.passed {
            "inconclusive"
        } else if candidate.passed {
            "preserved"
        } else {
            "regression"
        };
        ensure!(
            row["quality_conclusion"] == conclusion,
            "invalid dense-reference quality conclusion"
        );
        reference_passes += usize::from(reference.passed);
        candidate_passes += usize::from(candidate.passed);
        regressions += usize::from(reference.passed && !candidate.passed);
        inconclusive += usize::from(!reference.passed);
    }
    let (mut reference_nll, mut candidate_nll, mut scored_tokens) = (0.0, 0.0, 0usize);
    for passage in &corpus.passages {
        let row = one_row(
            rows,
            "model_quality_passage",
            Some(split),
            Some(&passage.id),
        )?;
        let tokens: Vec<u32> = serde_json::from_value(row["token_ids"].clone())?;
        ensure!(
            row["text"] == passage.text && tokens.len() >= 2,
            "invalid quality loss passage"
        );
        if split == Split::Calibration {
            let frozen = ranking
                .influence
                .plan
                .passages
                .iter()
                .find(|item| item.id == passage.id)
                .context("missing frozen calibration passage")?;
            ensure!(
                tokens == frozen.token_ids,
                "quality calibration tokens differ from ranking"
            );
        }
        let count = tokens.len() - 1;
        let comparison = &row["comparison"];
        ensure!(
            comparison["input_tokens"].as_u64() == Some(tokens.len() as u64),
            "quality passage input count differs"
        );
        reference_nll += check_loss(&comparison["reference"], count)?.nll_sum;
        candidate_nll += check_loss(&comparison["candidate"], count)?.nll_sum;
        scored_tokens = scored_tokens
            .checked_add(count)
            .context("quality token count overflow")?;
    }
    let summary = CompactQualitySummary {
        split,
        tasks: corpus.tasks.len(),
        dense_reference_passes: reference_passes,
        candidate_passes,
        dense_valid_regressions: regressions,
        inconclusive_tasks: inconclusive,
        reference_loss: LossMetrics::from_sum(reference_nll, scored_tokens)?,
        candidate_loss: LossMetrics::from_sum(candidate_nll, scored_tokens)?,
    };
    let recorded = one_row(rows, "model_quality_summary", Some(split), None)?;
    let expected = serde_json::to_value(&summary)?;
    for key in [
        "tasks",
        "dense_reference_passes",
        "candidate_passes",
        "dense_valid_regressions",
        "inconclusive_tasks",
    ] {
        ensure!(
            recorded[key] == expected[key],
            "quality split summary differs for {key}"
        );
    }
    for (key, loss) in [
        ("reference_loss", &summary.reference_loss),
        ("candidate_loss", &summary.candidate_loss),
    ] {
        ensure!(
            close(
                check_loss(&recorded[key], scored_tokens)?.nll_sum,
                loss.nll_sum
            ),
            "quality split loss differs from passages"
        );
    }
    Ok(summary)
}

impl CompactExportEvidence {
    /// Bind a requested frozen mask to a completed C19/C24 quality report over
    /// both corpus splits. Dense failures stay inconclusive, not model-change wins.
    pub fn load(
        ranking_path: &Path,
        quality_path: &Path,
        corpus_path: &Path,
        removal_count: usize,
    ) -> Result<Self> {
        let ranking_bytes = read_bounded(ranking_path)?;
        let quality_bytes = read_bounded(quality_path)?;
        let corpus_bytes = read_bounded(corpus_path)?;
        let ranking = LayerRemovalRanking::parse(std::str::from_utf8(&ranking_bytes)?)?;
        let corpus = Corpus::parse(std::str::from_utf8(&corpus_bytes)?)?;
        ranking.influence.plan.validate_corpus(&corpus)?;
        let removed_original_layers = ranking.selected_layers(removal_count)?;
        let ranking_sha256 = sha256(&ranking_bytes)?;
        let corpus_sha256 = sha256(&corpus_bytes)?;
        ensure!(
            ranking.influence.plan.corpus_sha256 == corpus_sha256,
            "ranking corpus identity mismatch"
        );
        let rows = std::str::from_utf8(&quality_bytes)?
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<Result<Vec<_>, _>>()?;
        let manifest = one_row(&rows, "model_quality_manifest", None, None)?;
        let complete = one_row(&rows, "model_quality_complete", None, None)?;
        ensure!(
            rows.first() == Some(manifest)
                && rows.last() == Some(complete)
                && complete["is_complete"] == true
                && complete["device_bytes_after"].as_u64().is_some()
                && complete["device_bytes_before"] == complete["device_bytes_after"],
            "quality report is incomplete or leaked tracked device storage"
        );
        let plan = &ranking.influence.plan;
        let active: Vec<_> = (0..plan.layer_count as usize)
            .filter(|id| !removed_original_layers.contains(id))
            .collect();
        let candidate = &manifest["candidate"];
        let reference = &manifest["reference"];
        ensure!(
            manifest["schema_version"] == 1
                && manifest["build_profile"] == "release"
                && manifest["selected_splits"] == json!(["calibration", "held_out"])
                && manifest["corpus_sha256"] == corpus_sha256
                && manifest["has_identical_model_source"] == true
                && manifest["configuration"]["kv_mode"] == "off"
                && reference["sha256"] == plan.model_sha256
                && candidate["sha256"] == plan.model_sha256
                && reference["layers"] == plan.layer_count
                && candidate["layers"] == plan.layer_count
                && candidate["d_model"].as_u64() == Some(plan.hidden_dim as u64)
                && reference["skipped_layer_ids"] == json!([])
                && candidate["skipped_layer_ids"] == json!(removed_original_layers)
                && candidate["active_layer_ids"] == json!(active),
            "quality report source/mask/reference identity mismatch"
        );
        let configured = &manifest["configuration"]["candidate_layer_ranking"];
        ensure!(
            configured["artifact_sha256"] == ranking_sha256
                && configured["removed_layer_count"].as_u64() == Some(removal_count as u64)
                && configured["skipped_original_layer_ids"] == json!(removed_original_layers)
                && configured["fit_split"] == "calibration"
                && manifest["configuration"]["candidate_activation_calibration"].is_null(),
            "quality report does not evaluate this frozen layer ranking alone"
        );
        let transforms = manifest["in_process_transforms"]
            .as_array()
            .context("quality transforms")?;
        ensure!(
            transforms.len() == 1
                && transforms[0]["kind"] == "diagnostic_layer_mask"
                && transforms[0]["skipped_layer_ids"] == json!(removed_original_layers)
                && transforms[0]["calibration_ranking"]["artifact_sha256"] == ranking_sha256,
            "quality report contains another model transform"
        );
        let mut actual_items = HashSet::new();
        for row in &rows {
            if matches!(
                row["kind"].as_str(),
                Some("model_quality_task" | "model_quality_passage")
            ) {
                let identity = (
                    row["split"].as_str().context("quality split")?,
                    row["id"].as_str().context("quality item ID")?,
                );
                ensure!(actual_items.insert(identity), "duplicate quality item");
            }
        }
        let expected_count = [&corpus.calibration, &corpus.held_out]
            .iter()
            .map(|split| split.tasks.len() + split.passages.len())
            .sum::<usize>();
        ensure!(
            actual_items.len() == expected_count,
            "quality report has missing/extra corpus items"
        );
        let calibration = check_split(&rows, Split::Calibration, &corpus.calibration, &ranking)?;
        let held_out = check_split(&rows, Split::HeldOut, &corpus.held_out, &ranking)?;
        let stage = &ranking.stages[removal_count - 1];
        let frozen = &stage
            .trials
            .iter()
            .find(|trial| trial.layer_id == stage.winner_layer_id)
            .context("frozen winner loss")?
            .calibration_loss;
        ensure!(
            calibration.candidate_loss.scored_tokens == frozen.scored_tokens
                && close(calibration.candidate_loss.nll_sum, frozen.nll_sum),
            "applied quality mask does not reproduce its frozen calibration loss"
        );
        Ok(Self {
            source_sha256: plan.model_sha256.clone(),
            ranking_sha256,
            quality_report_sha256: sha256(&quality_bytes)?,
            corpus_sha256,
            original_layer_count: plan.layer_count,
            hidden_dim: plan.hidden_dim,
            removed_original_layers,
            quality: vec![calibration, held_out],
        })
    }

    pub fn removed_original_layers(&self) -> &[usize] {
        &self.removed_original_layers
    }

    pub fn quality(&self) -> &[CompactQualitySummary] {
        &self.quality
    }
}

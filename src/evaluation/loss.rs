use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LossMetrics {
    pub scored_tokens: usize,
    pub nll_sum: f64,
    pub mean_nll: f64,
    pub perplexity: Option<f64>,
    pub perplexity_overflow: bool,
}

impl LossMetrics {
    pub fn from_sum(nll_sum: f64, scored_tokens: usize) -> Result<Self> {
        ensure!(
            scored_tokens > 0 && nll_sum.is_finite() && nll_sum >= 0.0,
            "invalid/empty loss aggregate"
        );
        let mean_nll = nll_sum / scored_tokens as f64;
        let perplexity = mean_nll.exp();
        Ok(Self {
            scored_tokens,
            nll_sum,
            mean_nll,
            perplexity: perplexity.is_finite().then_some(perplexity),
            perplexity_overflow: !perplexity.is_finite(),
        })
    }

    /// Weight by scored tokens, not by passages and never by averaging PPL.
    pub fn combine(metrics: &[Self]) -> Result<Self> {
        let mut count = 0_usize;
        let mut sum = 0.0;
        for metric in metrics {
            ensure!(
                metric.scored_tokens > 0 && metric.nll_sum.is_finite() && metric.nll_sum >= 0.0,
                "invalid passage loss"
            );
            count = count
                .checked_add(metric.scored_tokens)
                .ok_or_else(|| anyhow::anyhow!("loss count overflow"))?;
            sum += metric.nll_sum;
        }
        Self::from_sum(sum, count)
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TokenPrediction {
    pub target_logit: f32,
    /// Rank is one-based, ties resolved by the first vocabulary index.
    pub target_rank: usize,
    pub nll: f64,
    pub top_indices: Vec<usize>,
}

pub fn score_logits(logits: &[f32], target: u32) -> Result<TokenPrediction> {
    ensure!(
        !logits.is_empty() && logits.iter().all(|v| v.is_finite()),
        "empty/nonfinite logits"
    );
    let index = target as usize;
    ensure!(index < logits.len(), "target token outside vocabulary");
    let maximum = logits
        .iter()
        .map(|&v| f64::from(v))
        .fold(f64::NEG_INFINITY, f64::max);
    let shifted_log_partition = logits
        .iter()
        .map(|&v| (f64::from(v) - maximum).exp())
        .sum::<f64>()
        .ln();
    let target_logit = logits[index];
    // Subtract before adding log(sum) to preserve ln(vocab) at huge equal logits.
    let nll = shifted_log_partition + (maximum - f64::from(target_logit));
    let target_rank = 1 + logits
        .iter()
        .enumerate()
        .filter(|(i, v)| **v > target_logit || (**v == target_logit && *i < index))
        .count();
    let mut top_indices: Vec<_> = (0..logits.len()).collect();
    let compare = |a: &usize, b: &usize| {
        if logits[*a] == logits[*b] {
            a.cmp(b)
        } else {
            logits[*b].total_cmp(&logits[*a])
        }
    };
    let top_count = 10.min(logits.len());
    if top_count < top_indices.len() {
        top_indices.select_nth_unstable_by(top_count, compare);
        top_indices.truncate(top_count);
    }
    top_indices.sort_unstable_by(compare);
    Ok(TokenPrediction {
        target_logit,
        target_rank,
        nll,
        top_indices,
    })
}

/// Score one model on the observed sequence without sampling or introducing
/// targets into their own prefixes. Uses the same full-logit NLL as comparisons.
pub fn score_teacher_forced(
    ids: &[u32],
    mut predict: impl FnMut(&[u32]) -> Result<Vec<f32>>,
) -> Result<LossMetrics> {
    ensure!(
        ids.len() >= 2,
        "teacher forcing requires a prefix and target"
    );
    let mut vocabulary = None;
    let mut sum = 0.0;
    for position in 1..ids.len() {
        let logits = predict(&ids[..position])?;
        ensure!(
            vocabulary.is_none_or(|count| count == logits.len()),
            "vocabulary changed during evaluation"
        );
        vocabulary = Some(logits.len());
        sum += score_logits(&logits, ids[position])?.nll;
    }
    LossMetrics::from_sum(sum, ids.len() - 1)
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TokenComparison {
    pub position: usize,
    pub target_id: u32,
    pub reference: TokenPrediction,
    pub candidate: TokenPrediction,
    pub max_abs_logit_diff: f64,
    pub mean_abs_logit_diff: f64,
    pub top10_overlap: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TeacherForcedReport {
    pub input_tokens: usize,
    pub reference: LossMetrics,
    pub candidate: LossMetrics,
    pub max_abs_logit_diff: f64,
    pub mean_abs_logit_diff: f64,
    pub tokens: Vec<TokenComparison>,
}

/// Predict token i from exactly ids[..i], never from the target or a sampled
/// continuation. Closures may cache prior prefixes but must produce full logits.
/// Inputs include any caller-selected BOS; no implicit EOS/BOS/windowing occurs.
pub fn compare_teacher_forced(
    ids: &[u32],
    mut predict_reference: impl FnMut(&[u32]) -> Result<Vec<f32>>,
    mut predict_candidate: impl FnMut(&[u32]) -> Result<Vec<f32>>,
) -> Result<TeacherForcedReport> {
    ensure!(
        ids.len() >= 2,
        "teacher forcing requires a prefix and target"
    );
    let mut reference_sum = 0.0;
    let mut candidate_sum = 0.0;
    let mut maximum = 0.0_f64;
    let mut total_mean = 0.0;
    let mut vocabulary = None;
    let mut tokens = Vec::with_capacity(ids.len() - 1);
    for position in 1..ids.len() {
        let reference_logits = predict_reference(&ids[..position])?;
        let candidate_logits = predict_candidate(&ids[..position])?;
        ensure!(
            reference_logits.len() == candidate_logits.len(),
            "reference/candidate vocabulary sizes differ"
        );
        if let Some(size) = vocabulary {
            ensure!(
                reference_logits.len() == size,
                "vocabulary changed during evaluation"
            );
        }
        vocabulary = Some(reference_logits.len());
        let reference = score_logits(&reference_logits, ids[position])?;
        let candidate = score_logits(&candidate_logits, ids[position])?;
        reference_sum += reference.nll;
        candidate_sum += candidate.nll;
        let mut token_max = 0.0_f64;
        let mut token_sum = 0.0;
        for (&a, &b) in reference_logits.iter().zip(&candidate_logits) {
            let difference = (f64::from(a) - f64::from(b)).abs();
            token_max = token_max.max(difference);
            token_sum += difference;
        }
        let mean = token_sum / reference_logits.len() as f64;
        maximum = maximum.max(token_max);
        total_mean += mean;
        let top10_overlap = reference
            .top_indices
            .iter()
            .filter(|id| candidate.top_indices.contains(id))
            .count();
        tokens.push(TokenComparison {
            position,
            target_id: ids[position],
            reference,
            candidate,
            max_abs_logit_diff: token_max,
            mean_abs_logit_diff: mean,
            top10_overlap,
        });
    }
    Ok(TeacherForcedReport {
        input_tokens: ids.len(),
        reference: LossMetrics::from_sum(reference_sum, tokens.len())?,
        candidate: LossMetrics::from_sum(candidate_sum, tokens.len())?,
        max_abs_logit_diff: maximum,
        mean_abs_logit_diff: total_mean / tokens.len() as f64,
        tokens,
    })
}

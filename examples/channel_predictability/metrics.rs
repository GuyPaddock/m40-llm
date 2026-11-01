use anyhow::{ensure, Result};
use serde::Serialize;

#[derive(Clone, Serialize)]
pub struct Metrics {
    pub recall: f64,
    pub hidden_l1_fraction: Option<f64>,
    pub weighted_fraction: Option<f64>,
    pub weighted_oracle_fraction: Option<f64>,
    pub has_zero_oracle_mass: bool,
    pub passes_row_screen: bool,
}

pub fn measure(
    hidden: &[f32],
    norms: &[f64],
    oracle: &[usize],
    selected: &[usize],
) -> Result<Metrics> {
    ensure!(
        !hidden.is_empty()
            && norms.len() == hidden.len()
            && !oracle.is_empty()
            && oracle.len() == selected.len()
            && hidden.iter().all(|value| value.is_finite())
            && norms.iter().all(|value| value.is_finite() && *value >= 0.0),
        "invalid metric inputs"
    );
    for channels in [oracle, selected] {
        ensure!(
            channels.iter().all(|&channel| channel < hidden.len())
                && channels.windows(2).all(|pair| pair[0] < pair[1]),
            "invalid metric selection"
        );
    }
    let total: f64 = hidden.iter().map(|value| f64::from(value.abs())).sum();
    let weighted_total: f64 = hidden
        .iter()
        .zip(norms)
        .map(|(value, norm)| f64::from(value.abs()) * norm)
        .sum();
    let selected_mass: f64 = selected
        .iter()
        .map(|&channel| f64::from(hidden[channel].abs()))
        .sum();
    let weighted = |channels: &[usize]| {
        channels
            .iter()
            .map(|&channel| f64::from(hidden[channel].abs()) * norms[channel])
            .sum::<f64>()
    };
    let oracle_mass = weighted(oracle);
    let selected_weighted = weighted(selected);
    let recall = selected
        .iter()
        .filter(|channel| oracle.binary_search(channel).is_ok())
        .count() as f64
        / oracle.len() as f64;
    let ratio = |numerator: f64, denominator: f64| {
        if denominator > 0.0 {
            Some(numerator / denominator)
        } else {
            None
        }
    };
    let weighted_oracle_fraction = ratio(selected_weighted, oracle_mass);
    Ok(Metrics {
        recall,
        hidden_l1_fraction: ratio(selected_mass, total),
        weighted_fraction: ratio(selected_weighted, weighted_total),
        weighted_oracle_fraction,
        has_zero_oracle_mass: oracle_mass == 0.0,
        passes_row_screen: recall >= 0.80
            && weighted_oracle_fraction.is_some_and(|value| value >= 0.95),
    })
}

#[derive(Default, Serialize)]
pub struct Summary {
    pub observed_rows: usize,
    pub dense_refresh_rows: usize,
    pub eligible_sparse_rows: usize,
    pub passing_sparse_rows: usize,
    pub zero_oracle_sparse_rows: usize,
    pub recall_sum: f64,
    pub minimum_recall: Option<f64>,
    pub minimum_weighted_oracle_fraction: Option<f64>,
}

impl Summary {
    pub fn observe(&mut self, metrics: &Metrics, is_dense: bool) {
        self.observed_rows += 1;
        if is_dense {
            self.dense_refresh_rows += 1;
            return;
        }
        self.eligible_sparse_rows += 1;
        self.passing_sparse_rows += usize::from(metrics.passes_row_screen);
        self.zero_oracle_sparse_rows += usize::from(metrics.has_zero_oracle_mass);
        self.recall_sum += metrics.recall;
        self.minimum_recall = Some(
            self.minimum_recall
                .map_or(metrics.recall, |minimum| minimum.min(metrics.recall)),
        );
        if let Some(value) = metrics.weighted_oracle_fraction {
            self.minimum_weighted_oracle_fraction = Some(
                self.minimum_weighted_oracle_fraction
                    .map_or(value, |minimum| minimum.min(value)),
            );
        }
    }

    pub fn passes_screen(&self) -> bool {
        self.eligible_sparse_rows > 0
            && self.passing_sparse_rows * 10 >= self.eligible_sparse_rows * 9
    }
}

/// Optimistic historical timing bound. Bootstrap and scheduled refreshes use
/// dense time; only sparse actions receive the selected-Q4 time estimate.
pub fn remaining_budget(
    dense_ms: f64,
    selected_ms: f64,
    summary: &Summary,
    tasks: usize,
) -> Option<f64> {
    if summary.eligible_sparse_rows == 0 {
        return None;
    }
    let steps = summary.observed_rows + tasks;
    let dense_steps = summary.dense_refresh_rows + tasks;
    Some(
        (steps as f64 * dense_ms / 1.05
            - dense_steps as f64 * dense_ms
            - summary.eligible_sparse_rows as f64 * selected_ms)
            * 1000.0
            / summary.eligible_sparse_rows as f64,
    )
}

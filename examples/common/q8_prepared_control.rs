use super::{FfnChoice, RuntimeConfig};
use anyhow::{ensure, Context, Result};
use m40_llm::infer::execution_weights::{Q8ExecutionLayout, Q8ExecutionMode, Q8ExecutionStats};

/// Allow only an intentional new split-FFN gate/up delta. Every existing control
/// key, including gate/up for the current split control, stays pinned with its
/// source identity, payload bytes, launch geometry and accounting.
pub fn validate_prepared_control(
    control: &Q8ExecutionStats,
    candidate: &Q8ExecutionStats,
    control_config: RuntimeConfig,
    config: RuntimeConfig,
) -> Result<()> {
    config.validate_comparison(control_config)?;
    let expected_control = control_config.execution_config()?;
    let expected_candidate = config.execution_config()?;
    ensure!(
        control.mode == Q8ExecutionMode::Split
            && candidate.mode == control.mode
            && control.families == expected_control.families
            && candidate.families == expected_candidate.families
            && control.budget_bytes == expected_control.budget_bytes
            && candidate.budget_bytes == expected_candidate.budget_bytes
            && control.source_weights_bytes == candidate.source_weights_bytes
            && control.has_prepared_vocabulary
            && candidate.has_prepared_vocabulary
            && control.single_row_down_split_launch == candidate.single_row_down_split_launch,
        "runtime comparison changed prepared down/vocabulary configuration"
    );
    let source_hash = control
        .tensors
        .first()
        .context("empty prepared control")?
        .key
        .source
        .model_sha256();
    for stats in [control, candidate] {
        let identities: std::collections::HashSet<_> =
            stats.tensors.iter().map(|tensor| &tensor.key).collect();
        ensure!(
            identities.len() == stats.tensors.len(),
            "duplicate prepared tensor identity"
        );
        let sum: usize = stats
            .tensors
            .iter()
            .map(|tensor| tensor.resident_bytes)
            .sum();
        ensure!(
            sum == stats.resident_bytes
                && sum == stats.estimated_bytes
                && stats.total_weight_payload_bytes == stats.source_weights_bytes + sum,
            "prepared cache accounting mismatch"
        );
        ensure!(
            stats
                .tensors
                .iter()
                .all(|tensor| tensor.key.source.model_sha256() == source_hash
                    && tensor.key.layout == Q8ExecutionLayout::SeparateScalesValues),
            "prepared model identity/layout changed"
        );
    }
    for old in &control.tensors {
        let matching: Vec<_> = candidate
            .tensors
            .iter()
            .filter(|new| new.key == old.key)
            .collect();
        ensure!(
            matching.len() == 1
                && matching[0].resident_bytes == old.resident_bytes
                && matching[0].source_bytes == old.source_bytes,
            "prepared down/vocabulary tensor changed"
        );
    }
    let extras: Vec<_> = candidate
        .tensors
        .iter()
        .filter(|new| !control.tensors.iter().any(|old| new.key == old.key))
        .collect();
    if config.ffn == FfnChoice::SplitF32x4 && control_config.ffn != FfnChoice::SplitF32x4 {
        ensure!(
            control.tensors.len() > 1 && extras.len() == 2 * (control.tensors.len() - 1),
            "split FFN requires exactly two extra gate/up tensors per layer"
        );
        ensure!(
            extras.iter().all(|tensor| {
                let name = tensor.key.source.tensor_name();
                [
                    ".ffn_gate.weight",
                    ".ffn_up.weight",
                    ".feed_forward.w1.weight",
                    ".feed_forward.w3.weight",
                ]
                .iter()
                .any(|suffix| name.ends_with(suffix))
                    && tensor.source_bytes == tensor.resident_bytes
            }),
            "split FFN added non-gate/up weights or changed payload storage"
        );
    } else {
        ensure!(extras.is_empty(), "unexpected prepared runtime weights");
    }
    Ok(())
}

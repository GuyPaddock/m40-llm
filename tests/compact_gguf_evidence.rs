#[path = "common/compact_gguf.rs"]
mod fixture;

use anyhow::Result;
use fixture::CompactFixture;
use m40_llm::{evaluation::corpus::Split, gguf::compact::CompactExportEvidence};
use serde_json::{json, Value};
use std::fs;

/// Complete evaluation is not quality approval. Retain dense-valid regressions
/// and held-out loss rather than hiding them to authorize an artifact.
#[test]
fn test_compact_evidence_retains_quality_costs_and_frozen_mask() -> Result<()> {
    let fixture = CompactFixture::new("qwen2", 32, false)?;
    let evidence = fixture.evidence()?;
    assert_eq!(evidence.removed_original_layers(), &[1]);
    assert_eq!(evidence.quality().len(), 2);
    assert_eq!(evidence.quality()[0].split, Split::Calibration);
    assert_eq!(evidence.quality()[0].candidate_loss.nll_sum, 12.0);
    assert_eq!(evidence.quality()[1].split, Split::HeldOut);
    assert_eq!(evidence.quality()[1].candidate_loss.nll_sum, 14.0);
    for split in evidence.quality() {
        assert_eq!(split.tasks, 6);
        assert_eq!(split.dense_reference_passes, 2);
        assert_eq!(split.dense_valid_regressions, 2);
        assert_eq!(split.candidate_passes, 0);
        assert_eq!(split.inconclusive_tasks, 4);
        assert!(split.candidate_loss.mean_nll > split.reference_loss.mean_nll);
    }
    assert!(!fixture.destination.exists());
    assert_eq!(fs::read(&fixture.source)?, fixture.source_bytes);
    Ok(())
}

/// Incomplete, mismatched, altered-rubric or leaked reports do not qualify a
/// source/mask for export, even if their variant label looks legitimate.
#[test]
fn test_compact_evidence_rejects_invalid_quality_claims() -> Result<()> {
    let fixture = CompactFixture::new("qwen2", 32, false)?;
    let mutations: [fn(&mut Vec<Value>); 12] = [
        |rows| {
            rows.pop();
        },
        |rows| {
            rows.insert(1, rows[1].clone());
        },
        |rows| {
            let index = rows
                .iter()
                .position(|row| row["kind"] == "model_quality_passage")
                .unwrap();
            rows.remove(index);
        },
        |rows| rows[0]["candidate"]["sha256"] = json!("b".repeat(64)),
        |rows| rows[0]["candidate"]["skipped_layer_ids"] = json!([2]),
        |rows| {
            rows[0]["configuration"]["candidate_layer_ranking"]["artifact_sha256"] =
                json!("c".repeat(64))
        },
        |rows| rows[1]["user_prompt"] = json!("different prompt"),
        |rows| rows[1]["candidate_score"]["passed"] = json!(true),
        |rows| rows[0]["selected_splits"] = json!(["calibration"]),
        |rows| {
            rows[0]["in_process_transforms"]
                .as_array_mut()
                .unwrap()
                .push(json!({"kind":"activation_threshold"}))
        },
        |rows| rows.last_mut().unwrap()["device_bytes_after"] = json!(1),
        |rows| {
            let row = rows
                .iter_mut()
                .find(|row| row["kind"] == "model_quality_passage")
                .unwrap();
            row["token_ids"] = json!([3, 2, 1]);
        },
    ];
    for mutate in mutations {
        let mut rows = fixture.report_rows.clone();
        mutate(&mut rows);
        fixture.write_report(&rows)?;
        assert!(fixture.evidence().is_err());
        assert!(!fixture.destination.exists());
    }
    fixture.write_report(&fixture.report_rows)?;
    assert!(
        CompactExportEvidence::load(&fixture.ranking, &fixture.report, &fixture.corpus, 2).is_err()
    );
    fixture.evidence()?;
    assert_eq!(fs::read(&fixture.source)?, fixture.source_bytes);
    Ok(())
}

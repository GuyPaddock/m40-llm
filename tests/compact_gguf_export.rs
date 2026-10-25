#[path = "common/compact_gguf.rs"]
mod fixture;

use anyhow::Result;
use fixture::{get_u64, CompactFixture};
use m40_llm::{
    evaluation::identity::file_sha256,
    gguf::{compact::export_compact_gguf, load_gguf, GgufValue},
};
use std::{fs, sync::Arc};

/// Export preserves every retained payload and untouched metadata wire value,
/// including empty-array element types and NaN bits lost by naive reserialization.
#[test]
fn test_compact_export_round_trips_payloads_metadata_and_evaluated_lineage() -> Result<()> {
    for (architecture, alignment, has_u64_count) in [("llama", 32, false), ("qwen2", 64, true)] {
        let fixture = CompactFixture::new(architecture, alignment, has_u64_count)?;
        let evidence = fixture.evidence()?;
        assert_eq!(evidence.removed_original_layers(), &[1]);
        assert!(evidence
            .quality()
            .iter()
            .all(|split| split.dense_valid_regressions > 0));
        let result = export_compact_gguf(&fixture.source, &fixture.destination, &evidence)?;
        let exported = load_gguf(&fixture.destination)?;
        let output = fs::read(&fixture.destination)?;
        assert_eq!(fs::read(&fixture.source)?, fixture.source_bytes);
        assert_eq!(
            get_u64(&exported, &format!("{architecture}.block_count")),
            4
        );
        assert_eq!(get_u64(&exported, "m40llm.pruning.original_layer_count"), 5);
        assert_eq!(
            exported.metadata["general.name"].as_str(),
            Some("Fixture instruct m40-compact-4of5L")
        );
        assert!(!exported.metadata.contains_key("general.size_label"));
        assert_eq!(
            get_u64(&exported, "general.parameter_count"),
            32 * 64 + 4 * 32 * 32 + 32 + 32 * 64
        );
        assert_eq!(
            exported.metadata["m40llm.pruning.source_sha256"].as_str(),
            Some(file_sha256(&fixture.source)?.as_str())
        );
        assert_eq!(
            exported.metadata["m40llm.pruning.ranking_sha256"].as_str(),
            Some(file_sha256(&fixture.ranking)?.as_str())
        );
        assert_eq!(
            exported.metadata["m40llm.pruning.quality_report_sha256"].as_str(),
            Some(file_sha256(&fixture.report)?.as_str())
        );
        let quality: serde_json::Value = serde_json::from_str(
            exported.metadata["m40llm.pruning.quality_summary"]
                .as_str()
                .unwrap(),
        )?;
        assert!(
            quality[1]["candidate_loss"]["perplexity"].as_f64().unwrap()
                > quality[1]["reference_loss"]["perplexity"].as_f64().unwrap()
        );
        assert_eq!(result.plan.retained_original_layers(), &[0, 2, 3, 4]);
        for (actual, copy) in exported.tensors.iter().zip(result.plan.tensors()) {
            let from = copy.source_offset() as usize;
            let to = exported.data_offset as usize + actual.offset as usize;
            let length = copy.payload_bytes() as usize;
            assert_eq!(
                &output[to..to + length],
                &fixture.source_bytes[from..from + length]
            );
            assert_eq!(actual.name, copy.destination_name());
            assert_eq!(actual.offset % alignment as u64, 0);
        }
        for (key, bytes) in &fixture.source_metadata {
            if ![
                "general.name",
                "general.parameter_count",
                "general.size_label",
                &format!("{architecture}.block_count"),
            ]
            .contains(&key.as_str())
            {
                assert!(
                    output.windows(bytes.len()).any(|window| window == bytes),
                    "metadata wire value changed: {key}"
                );
            }
        }
        assert!(
            matches!(exported.metadata["fixture.empty_float64"], GgufValue::Array(ref values) if values.is_empty())
        );
        assert_eq!(result.output_file_bytes, output.len() as u64);
        assert_eq!(
            result.output_tensor_data_bytes,
            output.len() as u64 - exported.data_offset
        );
        assert_eq!(result.output_sha256, file_sha256(&fixture.destination)?);
        assert!(
            result.plan.retained_weight_payload_bytes() < result.plan.source_weight_payload_bytes()
        );
        assert!(!fixture.has_partial_files()?);
    }
    Ok(())
}

/// Existing outputs and aliases of the source are never overwritten or modified.
#[test]
fn test_compact_export_refuses_existing_files_and_source_aliases() -> Result<()> {
    let fixture = CompactFixture::new("qwen2", 32, false)?;
    let evidence = fixture.evidence()?;
    fs::write(&fixture.destination, b"keep existing result")?;
    assert!(export_compact_gguf(&fixture.source, &fixture.destination, &evidence).is_err());
    assert_eq!(fs::read(&fixture.destination)?, b"keep existing result");
    assert!(export_compact_gguf(&fixture.source, &fixture.source, &evidence).is_err());
    let alias = fixture.directory.join("source-alias.gguf");
    fs::hard_link(&fixture.source, &alias)?;
    assert!(export_compact_gguf(&fixture.source, &alias, &evidence).is_err());
    #[cfg(unix)]
    {
        let link = fixture.directory.join("dangling.gguf");
        std::os::unix::fs::symlink(fixture.directory.join("missing"), &link)?;
        assert!(export_compact_gguf(&fixture.source, &link, &evidence).is_err());
        assert!(fs::symlink_metadata(&link)?.file_type().is_symlink());
    }
    assert_eq!(fs::read(&fixture.source)?, fixture.source_bytes);
    assert!(!fixture.has_partial_files()?);
    Ok(())
}

/// An evaluated identity cannot be applied to a changed source. Reject before
/// publishing a destination rather than attaching another model's quality claims.
#[test]
fn test_compact_export_rejects_changed_source_without_publication() -> Result<()> {
    let fixture = CompactFixture::new("qwen2", 32, false)?;
    let evidence = fixture.evidence()?;
    let mut changed = fixture.source_bytes.clone();
    *changed.last_mut().unwrap() ^= 1;
    fs::write(&fixture.source, &changed)?;
    assert!(export_compact_gguf(&fixture.source, &fixture.destination, &evidence).is_err());
    assert!(!fixture.destination.exists());
    assert_eq!(fs::read(&fixture.source)?, changed);
    assert!(!fixture.has_partial_files()?);
    Ok(())
}

/// Concurrent publishers can produce exactly one complete artifact, never
/// overwrite the winner or leave their unique temporary files behind.
#[test]
fn test_compact_export_publication_is_create_new_under_race() -> Result<()> {
    let fixture = Arc::new(CompactFixture::new("qwen2", 32, false)?);
    let evidence = Arc::new(fixture.evidence()?);
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let jobs: Vec<_> = (0..2)
        .map(|_| {
            let fixture = fixture.clone();
            let evidence = evidence.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                export_compact_gguf(&fixture.source, &fixture.destination, &evidence)
            })
        })
        .collect();
    let results: Vec<_> = jobs.into_iter().map(|job| job.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(load_gguf(&fixture.destination)?.tensors.len(), 7);
    assert_eq!(fs::read(&fixture.source)?, fixture.source_bytes);
    assert!(!fixture.has_partial_files()?);
    Ok(())
}

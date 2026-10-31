use super::*;
use std::{
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    directory: PathBuf,
    report: PathBuf,
    artifact: PathBuf,
    rows: Vec<Value>,
}

impl Fixture {
    fn new() -> Result<Self> {
        let directory = std::env::temp_dir().join(format!(
            "m40-kv-fixture-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory)?;
        let mut fixture = Self {
            report: directory.join("capture.jsonl"),
            artifact: directory.join("capture.bin"),
            directory,
            rows: complete_rows()?,
        };
        let mut artifact = File::create(&fixture.artifact)?;
        artifact.set_len(ARTIFACT_BYTES as u64)?;
        // A sparse artifact limits physical storage while exercising the real
        // full-length hash and offset checks, including the last stream.
        let mut changed = vec![0; STREAM_BYTES];
        for (word, value) in [
            (0, 1.25_f32),
            (SOURCE_HEAD_DIM + 64, -0.0),
            (2 * SOURCE_HEAD_DIM + 127, 2.0),
        ] {
            changed[word * 2..word * 2 + 2]
                .copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
        }
        artifact.seek(SeekFrom::Start((2 * STREAM_BYTES) as u64))?;
        artifact.write_all(&changed)?;
        artifact.flush()?;
        drop(artifact);
        fixture.rows[3]["raw_sha256"] = json!(sha256(&changed)?);
        fixture.rows.last_mut().unwrap()["artifact_sha256"] =
            json!(file_sha256(&fixture.artifact)?);
        fixture.write_report()?;
        Ok(fixture)
    }

    fn write_report(&self) -> Result<()> {
        let mut file = File::create(&self.report)?;
        for row in &self.rows {
            serde_json::to_writer(&mut file, row)?;
            writeln!(file)?;
        }
        Ok(())
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

fn complete_rows() -> Result<Vec<Value>> {
    let mut inputs = Vec::new();
    for sequence in 0..SEQUENCES {
        let tokens: Vec<u32> = (0..TOKENS)
            .map(|token| (token + sequence * TOKENS) as u32)
            .collect();
        let bytes: Vec<u8> = tokens.iter().flat_map(|id| id.to_le_bytes()).collect();
        inputs.push(
            json!({"sequence":sequence,"source_token_start":sequence*TOKENS,
            "tokens":tokens,"token_sha256":sha256(&bytes)?}),
        );
    }
    let mut rows = vec![
        json!({"kind":"lossless_kv_capture_manifest","schema_version":1,
        "sequences":SEQUENCES,"layers":LAYERS,"contexts":CONTEXTS,
        "model":{"sha256":MODEL_SHA,"head_dim":SOURCE_HEAD_DIM,"kv_heads":KV_HEADS,"query_heads":QUERY_HEADS},
        "source_sha256":SOURCE_SHA,"source":"0eee578:docs/perf_baselines.md",
        "kv_mode":"explicit dense FP16 off","graphs":false,"inputs":inputs}),
    ];
    let zero_hash = sha256(&vec![0; STREAM_BYTES])?;
    let mut offset = 0;
    for sequence in 0..SEQUENCES {
        for layer in LAYERS {
            for head in 0..KV_HEADS {
                for role in ["key", "value"] {
                    rows.push(json!({"kind":"lossless_kv_stream","case":{
                        "sequence":sequence,"layer":layer,"head":head,"role":role},
                        "raw_offset":offset,"raw_bytes":STREAM_BYTES,"raw_sha256":zero_hash,
                        "head_dim":SOURCE_HEAD_DIM,"tokens":TOKENS}));
                    offset += STREAM_BYTES;
                }
            }
        }
        rows.push(json!({"kind":"lossless_kv_sequence","sequence":sequence,
            "processed_input_tokens":TOKENS,"generated_token_count":0}));
    }
    rows.push(
        json!({"kind":"lossless_kv_capture_complete","is_complete":true,
        "artifact_bytes":ARTIFACT_BYTES,"artifact_sha256":"0".repeat(64),
        "tracked_device_bytes_before":0,"tracked_device_bytes_after":0}),
    );
    Ok(rows)
}

/// Real file admission must preserve token/head ordering and signed zero,
/// distinguish the diagnostic head64 slice, and reject corrupted raw bytes
/// even when corruption lies beyond the first inspected context prefix.
#[test]
fn test_capture_reads_exact_head_layout_and_checks_full_artifact() -> Result<()> {
    let fixture = Fixture::new()?;
    let mut capture = Capture::open(&fixture.report, &fixture.artifact)?;
    let full = capture.read_layer(0, 1024, 128)?;
    assert_eq!(full.keys[0][128], 1.25);
    assert_eq!(full.keys[0][256 + 128 + 64].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(full.keys[0][2 * 256 + 128 + 127], 2.0);
    assert!(full.values.iter().flatten().all(|&value| value == 0.0));
    assert!(full.keys[1..].iter().flatten().all(|&value| value == 0.0));
    let sliced = capture.read_layer(0, 1024, 64)?;
    assert_eq!(sliced.keys[0][64], 1.25);
    assert_eq!(sliced.keys[0][2 * 128 + 127], 0.0);
    assert!(capture.read_layer(1, 1024, 128).is_err());
    assert!(capture.read_layer(0, 1000, 128).is_err());
    assert!(capture.read_layer(0, 1024, 32).is_err());
    let mut artifact = fs::OpenOptions::new().write(true).open(&fixture.artifact)?;
    artifact.seek(SeekFrom::End(-1))?;
    artifact.write_all(&[1])?;
    drop(artifact);
    assert!(Capture::open(&fixture.report, &fixture.artifact).is_err());
    Ok(())
}

/// A complete record is insufficient if stream coverage, offsets, source
/// identity, causal input spans or capture cleanup do not match the contract.
#[test]
fn test_capture_rejects_partial_duplicate_and_misidentified_records() -> Result<()> {
    let original = complete_rows()?;
    assert!(parse_metadata(&original).is_ok());
    let mut cases = Vec::new();
    cases.push(original[..original.len() - 1].to_vec());
    let mut missing = original.clone();
    missing.remove(1);
    cases.push(missing);
    let mut duplicate = original.clone();
    duplicate.insert(2, duplicate[1].clone());
    cases.push(duplicate);
    for (row, pointer, value) in [
        (0, "/model/sha256", json!("1".repeat(64))),
        (0, "/source_sha256", json!("1".repeat(64))),
        (0, "/inputs/1/source_token_start", json!(0)),
        (0, "/inputs/0/tokens/0", json!(151936)),
        (0, "/inputs/0/token_sha256", json!("1".repeat(64))),
        (1, "/raw_offset", json!(2)),
        (1, "/raw_bytes", json!(STREAM_BYTES - 2)),
        (1, "/case/head", json!(2)),
        (1, "/raw_sha256", json!("bad")),
        (original.len() - 1, "/tracked_device_bytes_after", json!(1)),
    ] {
        let mut changed = original.clone();
        *changed[row].pointer_mut(pointer).unwrap() = value;
        cases.push(changed);
    }
    for (index, rows) in cases.iter().enumerate() {
        assert!(
            parse_metadata(rows).is_err(),
            "bad capture case {index} admitted"
        );
    }
    Ok(())
}

/// Stream-level identities remain mandatory after the full-file hash passes.
#[test]
fn test_capture_rejects_a_misbound_stream_digest() -> Result<()> {
    let mut fixture = Fixture::new()?;
    fixture.rows[1]["raw_sha256"] = json!("1".repeat(64));
    fixture.write_report()?;
    assert!(Capture::open(&fixture.report, &fixture.artifact).is_err());
    Ok(())
}

/// Attention fixtures reject NaN/Inf and malformed streams while keeping
/// queries deterministic, finite, full-width and distinct across sequences.
#[test]
fn test_fixture_rejects_nonfinite_kv_and_keeps_queries_distinct() -> Result<()> {
    let mut heads = vec![vec![0; SOURCE_HEAD_DIM * 2]; KV_HEADS];
    for special in [0x7c00_u16, 0xfc00, 0x7e01] {
        heads[1][..2].copy_from_slice(&special.to_le_bytes());
        assert!(interleave_heads(&heads, 1, 128).is_err());
    }
    heads[1][..2].copy_from_slice(&0_u16.to_le_bytes());
    assert_eq!(interleave_heads(&heads, 1, 64)?.len(), KV_HEADS * 64);
    heads[0].pop();
    assert!(interleave_heads(&heads, 1, 128).is_err());
    for width in [64, 128] {
        let queries: Vec<_> = (0..SEQUENCES)
            .map(|sequence| build_queries(17, sequence, width))
            .collect();
        assert_eq!(queries[0], build_queries(17, 0, width));
        assert_ne!(queries[0], build_queries(35, 0, width));
        for (index, query) in queries.iter().enumerate() {
            assert_eq!(query.len(), QUERY_HEADS * width);
            assert!(query
                .iter()
                .all(|value| value.is_finite() && value.abs() <= 0.125));
            assert!(queries[..index].iter().all(|earlier| earlier != query));
        }
    }
    Ok(())
}

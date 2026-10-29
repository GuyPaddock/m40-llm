//! Necessary source-capacity check, never a sufficient R1 admission decision.
//! Uses the real GGUF tokenizer and CLI ChatML formatter, without loading CUDA.

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use m40_llm::{
    generate::{prepare_prompt, PromptFormat},
    gguf::load_gguf,
    tokenizer::Tokenizer,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

const SPEC_SHA256: &str = "6ba7fc6606525bb142de110c7a7cd78cbe26d01a6f1edb8f03646871095d4bcd";

#[derive(Parser)]
struct Args {
    #[arg(long)]
    spec: PathBuf,
    /// Directory containing original train/validation/test.parquet and converted JSON.
    #[arg(long)]
    mbpp: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Local evidence containing prompt bytes and complete IDs; do not publish as code.
    #[arg(long)]
    row_ledger: PathBuf,
}

#[derive(Deserialize)]
struct Task {
    task_id: u32,
    text: String,
}

#[derive(Deserialize)]
struct Partition {
    schema_version: u32,
    source_sha256: String,
    source_file: String,
    source_rows: usize,
    rows: Vec<Task>,
}

fn hash_file(path: &Path) -> Result<String> {
    let output = Command::new("sha256sum").arg(path).output()?;
    ensure!(
        output.status.success(),
        "hash failed for {}",
        path.display()
    );
    Ok(String::from_utf8(output.stdout)?
        .split_whitespace()
        .next()
        .context("missing digest")?
        .to_owned())
}

fn hash_bytes(bytes: &[u8]) -> Result<String> {
    let mut child = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    child.stdin.take().context("hash stdin")?.write_all(bytes)?;
    let output = child.wait_with_output()?;
    ensure!(output.status.success(), "byte hash failed");
    Ok(String::from_utf8(output.stdout)?
        .split_whitespace()
        .next()
        .context("missing byte digest")?
        .to_owned())
}

fn write_new(path: &Path, value: &Value) -> Result<()> {
    let mut out = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut out, value)?;
    writeln!(out)?;
    Ok(())
}

fn check_ids(partition: &Partition, first: u32, last: u32) -> Result<()> {
    ensure!(partition.schema_version == 1, "unknown conversion schema");
    ensure!(
        partition.source_rows == partition.rows.len(),
        "row count differs"
    );
    let ids: BTreeSet<_> = partition.rows.iter().map(|row| row.task_id).collect();
    ensure!(ids.len() == partition.rows.len(), "duplicate task IDs");
    ensure!(
        ids == (first..=last).collect(),
        "source partition ID range differs"
    );
    Ok(())
}

fn classify_length(length: usize, buckets: &[[usize; 2]]) -> &'static str {
    if buckets.iter().any(|[lo, hi]| (*lo..=*hi).contains(&length)) {
        "length_eligible_before_other_filters"
    } else {
        "outside_frozen_length_buckets"
    }
}

fn summarize_counts(lengths: &[usize], buckets: &[[usize; 2]], quota: usize) -> Value {
    let eligible = lengths
        .iter()
        .filter(|&&n| classify_length(n, buckets) == "length_eligible_before_other_filters")
        .count();
    json!({
        "source_rows": lengths.len(), "required_quota": quota,
        "length_eligible_upper_bound": eligible,
        "has_sufficient_upper_bound": eligible >= quota,
        "minimum_prompt_tokens": lengths.iter().min(),
        "maximum_prompt_tokens": lengths.iter().max(),
        "bucket_counts": buckets.iter().map(|[lo,hi]| json!({
            "minimum":lo,"maximum":hi,
            "count":lengths.iter().filter(|&&n|(*lo..=*hi).contains(&n)).count()
        })).collect::<Vec<_>>()
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !cfg!(feature = "cuda"),
        "build this census with --no-default-features"
    );
    ensure!(
        hash_file(&args.spec)? == SPEC_SHA256,
        "frozen data specification changed"
    );
    ensure!(
        !args.output.exists() && !args.row_ledger.exists(),
        "refusing to overwrite evidence"
    );
    ensure!(
        args.output != args.row_ledger,
        "report and ledger paths must differ"
    );
    let spec: Value = serde_json::from_slice(&fs::read(&args.spec)?)?;
    let model_path = Path::new(
        spec["target"]["local_path"]
            .as_str()
            .context("model path")?,
    );
    let model_sha = hash_file(model_path)?;
    ensure!(
        Some(model_sha.as_str()) == spec["target"]["sha256"].as_str(),
        "target hash differs"
    );
    let gguf = load_gguf(model_path)?;
    let tokenizer = Tokenizer::from_gguf_metadata(&gguf.metadata)?;
    ensure!(
        tokenizer.is_qwen2() && tokenizer.has_chat_template(),
        "Qwen2 ChatML required"
    );
    ensure!(
        tokenizer.token_id("<|im_start|>") == Some(151644)
            && tokenizer.token_id("<|im_end|>") == Some(151645),
        "ChatML IDs differ"
    );
    let metadata: BTreeMap<_, _> = gguf
        .metadata
        .iter()
        .filter(|(name, _)| name.starts_with("tokenizer."))
        .collect();
    let tokenizer_sha = hash_bytes(&serde_json::to_vec(&metadata)?)?;
    let buckets: Vec<[usize; 2]> =
        serde_json::from_value(spec["selection"]["prompt_token_buckets_inclusive"].clone())?;
    let quotas = spec["selection"]["strata"]
        .as_array()
        .context("strata")?
        .iter()
        .find(|row| row["name"] == "mbpp_code")
        .context("MBPP quota")?;
    let ranges = &spec["sources"][1]["source_task_id_ranges"];
    let mut summary = Vec::new();
    let mut ledger = Vec::new();
    for (file, split) in [
        ("train", "train"),
        ("validation", "calibration"),
        ("test", "held_out"),
    ] {
        let input_path = args.mbpp.join(format!("{file}.json"));
        let partition: Partition = serde_json::from_slice(&fs::read(&input_path)?)?;
        let range: [u32; 2] = serde_json::from_value(ranges[split].clone())?;
        check_ids(&partition, range[0], range[1])?;
        ensure!(
            partition.source_file == format!("{file}.parquet"),
            "source filename differs"
        );
        let parquet_sha = hash_file(&args.mbpp.join(&partition.source_file))?;
        ensure!(
            parquet_sha == partition.source_sha256,
            "Parquet hash differs from conversion"
        );
        let mut lengths = Vec::new();
        let mut counts = Vec::new();
        for row in &partition.rows {
            ensure!(
                !row.text.contains("<|"),
                "embedded special-token delimiter at task {}",
                row.task_id
            );
            let (prompt, add_bos) = prepare_prompt(&tokenizer, &row.text, PromptFormat::QwenChat);
            ensure!(!add_bos, "unexpected BOS insertion");
            let ids = tokenizer.encode_with_specials(&prompt, add_bos, false)?;
            ensure!(
                tokenizer.decode(&ids)? == prompt,
                "token roundtrip at task {}",
                row.task_id
            );
            ensure!(
                ids.iter().all(|&id| id < 151936),
                "token outside target vocabulary"
            );
            lengths.push(ids.len());
            let status = classify_length(ids.len(), &buckets);
            counts.push(json!({"task_id":row.task_id,"prompt_tokens":ids.len(),"status":status}));
            ledger.push(json!({"split":split,"task_id":row.task_id,
                "source_text":row.text,"prompt":prompt,"prompt_token_ids":ids,"status":status}));
        }
        let quota = quotas[split].as_u64().context("split quota")? as usize;
        let mut row = summarize_counts(&lengths, &buckets, quota);
        row["split"] = json!(split);
        row["parquet_sha256"] = json!(parquet_sha);
        row["conversion_sha256"] = json!(hash_file(&input_path)?);
        row["row_counts"] = json!(counts);
        summary.push(row);
    }
    write_new(&args.row_ledger, &json!(ledger))?;
    let has_capacity = summary
        .iter()
        .all(|row| row["has_sufficient_upper_bound"] == true);
    let report = json!({
        "schema_version":1, "kind":"mbpp_source_capacity_necessary_gate",
        "spec_sha256":SPEC_SHA256, "model_sha256":model_sha,
        "tokenizer_metadata_sha256":tokenizer_sha,
        "tokenizer_hash_encoding":"serde_json of lexicographically ordered tokenizer.* GGUF metadata",
        "prompt_format":"runtime QwenChat; text only; no extra BOS or EOS",
        "has_source_capacity":has_capacity, "is_r1_admitted":false,
        "is_realized_training_manifest":false, "additional_gpu_hours":0,
        "checks_not_performed":["grouping","document_provenance","expanded_exclusions",
            "template_isolation","training_toolchain","causal_export"],
        "row_ledger_sha256":hash_file(&args.row_ledger)?,
        "partitions":summary
    });
    write_new(&args.output, &report)?;
    if !has_capacity {
        bail!("MBPP length-only upper bound cannot satisfy frozen quotas; stop before R1");
    }
    eprintln!("Capacity alone is not R1 admission; continue the remaining CPU audits.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inclusive bucket boundaries retain exactly the authorized prompt lengths.
    #[test]
    fn test_capacity_boundaries_do_not_pad_short_prompts() {
        let result = summarize_counts(
            &[64, 65, 127, 128, 255, 256, 447, 448],
            &[[65, 127], [128, 255], [256, 447]],
            7,
        );
        assert_eq!(result["length_eligible_upper_bound"], 6);
        assert_eq!(result["has_sufficient_upper_bound"], false);
        assert_eq!(
            summarize_counts(&[65, 447], &[[65, 447]], 2)["has_sufficient_upper_bound"],
            true
        );
    }

    /// No rows means a failed capacity gate, never an implicit passing split.
    #[test]
    fn test_empty_partition_fails_quota() {
        let result = summarize_counts(&[], &[[65, 447]], 48);
        assert_eq!(result["has_sufficient_upper_bound"], false);
        assert_eq!(result["minimum_prompt_tokens"], Value::Null);
    }

    /// Duplicate or foreign partition IDs cannot inflate the available data.
    #[test]
    fn test_partition_ids_are_complete_and_unique() {
        let mut partition = Partition {
            schema_version: 1,
            source_sha256: String::new(),
            source_file: String::new(),
            source_rows: 2,
            rows: vec![
                Task {
                    task_id: 601,
                    text: String::new(),
                },
                Task {
                    task_id: 602,
                    text: String::new(),
                },
            ],
        };
        check_ids(&partition, 601, 602).unwrap();
        partition.rows[1].task_id = 601;
        assert!(check_ids(&partition, 601, 602).is_err());
        partition.rows[1].task_id = 603;
        assert!(check_ids(&partition, 601, 602).is_err());
    }

    /// Hashes identify raw bytes, not whitespace-normalized content.
    #[test]
    fn test_byte_hash_preserves_exact_identity() {
        assert_eq!(
            hash_bytes(b"abc").unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_ne!(hash_bytes(b"abc ").unwrap(), hash_bytes(b"abc").unwrap());
    }
}

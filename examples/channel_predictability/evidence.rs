use anyhow::{ensure, Context, Result};
use m40_llm::{
    evaluation::{
        corpus::Corpus,
        ffn_capture::FfnCaptureArtifact,
        identity::{file_sha256, sha256},
    },
    gguf::load_gguf,
};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    process::{Command, Stdio},
};

pub const MODEL_SHA256: &str = "4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d";
pub const CORPUS_SHA256: &str = "2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38";

pub struct Evidence {
    pub artifact: FfnCaptureArtifact,
    pub down_norms: BTreeMap<usize, Vec<f64>>,
    pub artifact_sha256: String,
    pub report_sha256: String,
    pub manifest: Value,
}

fn read_bounded(path: &Path, limit: usize) -> Result<String> {
    let mut text = String::new();
    File::open(path)?
        .take(limit as u64 + 1)
        .read_to_string(&mut text)?;
    ensure!(text.len() <= limit, "evidence exceeds byte limit");
    Ok(text)
}

pub fn validate_report(
    artifact: &FfnCaptureArtifact,
    records: &[Value],
    artifact_sha256: &str,
) -> Result<()> {
    artifact.validate_complete_generations()?;
    ensure!(
        records.len() == artifact.tasks.len() + 2,
        "missing/extra report records"
    );
    let manifest = &records[0];
    let complete = records.last().context("empty report")?;
    ensure!(
        manifest["kind"] == "ffn_capture_manifest"
            && manifest["schema_version"] == 1
            && manifest["build_profile"] == "release"
            && manifest["worktree"] == ""
            && manifest["should_capture_complete_generations"] == true
            && manifest["revision"] == "d6f6ae9898db16b31956a3a07a6511de4b7ace95"
            && complete["kind"] == "ffn_capture_complete"
            && complete["is_complete"] == true
            && complete["artifact_sha256"] == artifact_sha256
            && complete["device_bytes_before"] == complete["device_bytes_after"]
            && complete["device_bytes_after"] == 0,
        "require completed clean release capture and owner cleanup"
    );
    ensure!(
        manifest["model"]["sha256"] == artifact.plan.model_sha256
            && manifest["corpus_sha256"] == artifact.plan.corpus_sha256
            && manifest["execution_sha256"] == artifact.plan.execution_sha256
            && sha256(&serde_json::to_vec(&manifest["execution_configuration"])?)?
                == artifact.plan.execution_sha256,
        "report identities do not match captured inputs"
    );
    for ((record, task), expected_id) in records[1..records.len() - 1]
        .iter()
        .zip(&artifact.tasks)
        .zip(&artifact.plan.calibration_task_ids)
    {
        let prompt: Vec<u32> = serde_json::from_value(record["prompt_token_ids"].clone())?;
        let baseline: Vec<u32> =
            serde_json::from_value(record["baseline"]["generated_token_ids"].clone())?;
        let observed: Vec<u32> =
            serde_json::from_value(record["captured"]["generated_token_ids"].clone())?;
        ensure!(
            record["kind"] == "ffn_capture_task"
                && record["id"] == *expected_id
                && task.id == *expected_id
                && record["has_identical_generation"] == true
                && prompt == task.prompt_token_ids
                && baseline == task.generated_token_ids
                && observed == baseline
                && record["captured_rows_per_layer"].as_u64()
                    == Some(observed.len().saturating_sub(1) as u64)
                && record["baseline"]["metrics"]["generated_tokens"].as_u64()
                    == Some(observed.len() as u64)
                && record["captured"]["metrics"]["generated_tokens"].as_u64()
                    == Some(observed.len() as u64)
                && record["baseline"]["output"] == record["captured"]["output"]
                && record["baseline"]["output"].is_string()
                && record["baseline"]["metrics"]["stop_reason"].is_string()
                && record["baseline"]["metrics"]["stop_reason"]
                    == record["captured"]["metrics"]["stop_reason"],
            "paired generation mismatch for {}",
            task.id
        );
    }
    Ok(())
}

/// Read original Q8 blocks by GGUF output/input coordinates; include both
/// payload and scale in each channel's down-column squared norm.
pub fn calculate_down_norms(bytes: &[u8], input: usize, output: usize) -> Result<Vec<f64>> {
    ensure!(
        input > 0
            && input.is_multiple_of(32)
            && output > 0
            && input
                .checked_div(32)
                .and_then(|blocks| blocks.checked_mul(output))
                .and_then(|blocks| blocks.checked_mul(34))
                == Some(bytes.len()),
        "invalid Q8 geometry"
    );
    let mut norms = vec![0.0; input];
    for (block_index, block) in bytes.chunks_exact(34).enumerate() {
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f64();
        ensure!(scale.is_finite(), "nonfinite source scale");
        let first = (block_index % (input / 32)) * 32;
        for channel in 0..32 {
            let weight = scale * f64::from(block[2 + channel] as i8);
            norms[first + channel] += weight * weight;
        }
    }
    for norm in &mut norms {
        *norm = norm.sqrt();
    }
    Ok(norms)
}

impl Evidence {
    pub fn load(model: &Path, inputs: &Path, report: &Path, corpus_path: &Path) -> Result<Self> {
        ensure!(file_sha256(model)? == MODEL_SHA256, "wrong original model");
        let corpus_text = read_bounded(corpus_path, 4 * 1024 * 1024)?;
        ensure!(
            sha256(corpus_text.as_bytes())? == CORPUS_SHA256,
            "wrong frozen corpus"
        );
        let corpus = Corpus::parse(&corpus_text)?;
        let text = read_bounded(inputs, 512 * 1024 * 1024)?;
        let artifact = FfnCaptureArtifact::parse(&text)?;
        let artifact_sha256 = sha256(text.as_bytes())?;
        drop(text);
        artifact.plan.validate_corpus(&corpus, CORPUS_SHA256)?;
        ensure!(
            artifact.plan.model_sha256 == MODEL_SHA256
                && artifact.plan.source_layer_count == 36
                && artifact.plan.selected_layer_ids == [0, 16, 17, 34, 35]
                && artifact.plan.d_model == 2048
                && artifact.plan.hidden_dim == 11008
                && artifact.plan.max_generated_rows_per_task == 160,
            "capture does not match frozen study dimensions"
        );
        let report_text = read_bounded(report, 8 * 1024 * 1024)?;
        let records = report_text
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        validate_report(&artifact, &records, &artifact_sha256)?;
        let manifest = records[0].clone();
        let configuration = &manifest["execution_configuration"];
        ensure!(
            configuration["source_layer_ids"] == serde_json::json!([0, 16, 17, 34, 35])
                && configuration["kv_mode"] == "off"
                && configuration["max_generated_input_rows_per_task"] == 160
                && configuration["capture_host_payload_budget_bytes"] == 256 * 1024 * 1024,
            "wrong capture configuration"
        );
        for (record, task) in records[1..records.len() - 1]
            .iter()
            .zip(&corpus.calibration.tasks)
        {
            ensure!(
                record["max_tokens"] == task.max_tokens && record["id"] == task.id,
                "generation limit changed"
            );
        }
        let metadata = load_gguf(model)?;
        let mut file = File::open(model)?;
        file.seek(SeekFrom::Start(metadata.data_offset))?;
        let mut hash_process = Command::new("sha256sum")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        {
            let mut input = hash_process.stdin.take().context("missing hash stdin")?;
            std::io::copy(&mut file, &mut input)?;
        }
        let hash_output = hash_process.wait_with_output()?;
        ensure!(hash_output.status.success(), "weight-region hash failed");
        let source_digest = String::from_utf8(hash_output.stdout)?;
        ensure!(
            source_digest.split_whitespace().next()
                == Some(artifact.source_weights_sha256.as_str()),
            "source weight-region hash differs"
        );
        let mut down_norms = BTreeMap::new();
        for layer in &artifact.layers {
            for source in [&layer.source.gate, &layer.source.up, &layer.source.down] {
                let tensor = metadata
                    .tensors
                    .iter()
                    .find(|tensor| tensor.name == source.name)
                    .context("missing original tensor")?;
                ensure!(
                    tensor.offset == source.byte_offset
                        && tensor.shape == source.shape.map(|value| value as u64)
                        && tensor.dtype == source.dtype,
                    "source tensor layout differs"
                );
                let mut bytes = vec![0; source.byte_len()?];
                file.seek(SeekFrom::Start(
                    metadata
                        .data_offset
                        .checked_add(source.byte_offset)
                        .context("tensor offset overflow")?,
                ))?;
                file.read_exact(&mut bytes)?;
                ensure!(
                    sha256(&bytes)? == source.source_sha256,
                    "source tensor digest differs"
                );
                if source.name == layer.source.down.name
                    && [0, 17, 35].contains(&layer.source.layer_id)
                {
                    down_norms.insert(
                        layer.source.layer_id,
                        calculate_down_norms(&bytes, source.shape[0], source.shape[1])?,
                    );
                }
            }
        }
        Ok(Self {
            artifact,
            down_norms,
            artifact_sha256,
            report_sha256: sha256(report_text.as_bytes())?,
            manifest,
        })
    }
}

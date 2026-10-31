//! Admit the frozen real-KV capture and construct explicitly synthetic queries.

use anyhow::{ensure, Context, Result};
use half::f16;
use m40_llm::evaluation::identity::{file_sha256, sha256};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::Path,
};

pub const LAYERS: [usize; 3] = [0, 17, 35];
pub const CONTEXTS: [usize; 4] = [1024, 2048, 4096, 8192];
pub const SEQUENCES: usize = 8;
pub const KV_HEADS: usize = 2;
pub const QUERY_HEADS: usize = 16;
const TOKENS: usize = 8192;
const SOURCE_HEAD_DIM: usize = 128;
const STREAM_BYTES: usize = TOKENS * SOURCE_HEAD_DIM * 2;
const STREAMS: usize = SEQUENCES * LAYERS.len() * KV_HEADS * 2;
const ARTIFACT_BYTES: usize = STREAMS * STREAM_BYTES;
const MODEL_SHA: &str = "4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d";
const SOURCE_SHA: &str = "a918722b18973bb2621a0e2f0fd9ccde0d01ff610456904773c1a975a9b10ade";

type StreamKey = (usize, usize, usize, String);

struct Stream {
    offset: u64,
    sha256: String,
}

struct Metadata {
    manifest: Value,
    streams: BTreeMap<StreamKey, Stream>,
    artifact_sha256: String,
}

pub struct Capture {
    metadata: Metadata,
    file: File,
}

/// Each sequence contains token-major, KV-head-major, channel-minor FP32
/// expansions of original FP16 words. Query rows are head-major FP32 vectors.
pub struct LayerSample {
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
}

impl LayerSample {
    pub fn bytes(&self) -> usize {
        self.keys
            .iter()
            .chain(&self.values)
            .chain(&self.queries)
            .map(|row| row.len() * 4)
            .sum()
    }

    pub fn sha256(&self) -> Result<String> {
        let mut bytes = Vec::with_capacity(self.bytes());
        for value in self
            .keys
            .iter()
            .chain(&self.values)
            .chain(&self.queries)
            .flatten()
        {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        sha256(&bytes)
    }
}

impl Capture {
    pub fn open(report: &Path, artifact: &Path) -> Result<Self> {
        ensure!(
            fs::metadata(report)?.len() <= 8 * 1024 * 1024,
            "capture report too large"
        );
        let rows = fs::read_to_string(report)?
            .lines()
            .map(serde_json::from_str)
            .collect::<serde_json::Result<Vec<Value>>>()?;
        let metadata = parse_metadata(&rows)?;
        ensure!(
            fs::metadata(artifact)?.len() == ARTIFACT_BYTES as u64,
            "raw capture length mismatch"
        );
        ensure!(
            file_sha256(artifact)? == metadata.artifact_sha256,
            "raw capture hash mismatch"
        );
        let mut capture = Self {
            metadata,
            file: File::open(artifact)?,
        };
        // Validate every stream, even if later callers use only an early prefix.
        let mut bytes = vec![0; STREAM_BYTES];
        for stream in capture.metadata.streams.values() {
            capture.file.seek(SeekFrom::Start(stream.offset))?;
            capture.file.read_exact(&mut bytes)?;
            ensure!(
                sha256(&bytes)? == stream.sha256,
                "stream hash mismatch at {}",
                stream.offset
            );
        }
        Ok(capture)
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.metadata.artifact_sha256
    }

    pub fn manifest(&self) -> &Value {
        &self.metadata.manifest
    }

    pub fn read_layer(
        &mut self,
        layer: usize,
        length: usize,
        head_dim: usize,
    ) -> Result<LayerSample> {
        ensure!(
            LAYERS.contains(&layer) && CONTEXTS.contains(&length) && [64, 128].contains(&head_dim),
            "unsupported fixture shape"
        );
        let mut sample = LayerSample {
            keys: Vec::new(),
            values: Vec::new(),
            queries: Vec::new(),
        };
        for sequence in 0..SEQUENCES {
            let mut roles = Vec::new();
            for role in ["key", "value"] {
                let mut heads = Vec::new();
                for head in 0..KV_HEADS {
                    let stream = &self.metadata.streams[&(sequence, layer, head, role.to_owned())];
                    self.file.seek(SeekFrom::Start(stream.offset))?;
                    let mut bytes = vec![0; length * SOURCE_HEAD_DIM * 2];
                    self.file.read_exact(&mut bytes)?;
                    heads.push(bytes);
                }
                roles.push(interleave_heads(&heads, length, head_dim)?);
            }
            sample.keys.push(roles.remove(0));
            sample.values.push(roles.remove(0));
            sample
                .queries
                .push(build_queries(layer, sequence, head_dim));
        }
        Ok(sample)
    }
}

fn get_usize(value: &Value, field: &str) -> Result<usize> {
    usize::try_from(
        value[field]
            .as_u64()
            .with_context(|| format!("missing {field}"))?,
    )
    .with_context(|| format!("{field} is too large"))
}

fn get_digest(value: &Value, field: &str) -> Result<String> {
    let digest = value[field]
        .as_str()
        .with_context(|| format!("missing {field}"))?;
    ensure!(
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "invalid {field}"
    );
    Ok(digest.to_ascii_lowercase())
}

fn parse_metadata(rows: &[Value]) -> Result<Metadata> {
    let manifest = rows.first().context("empty capture")?;
    ensure!(
        manifest["kind"] == "lossless_kv_capture_manifest"
            && manifest["schema_version"] == 1
            && manifest["sequences"] == SEQUENCES
            && manifest["layers"] == json!(LAYERS)
            && manifest["contexts"] == json!(CONTEXTS),
        "unexpected capture manifest"
    );
    ensure!(
        manifest["model"]["sha256"] == MODEL_SHA
            && manifest["model"]["head_dim"] == SOURCE_HEAD_DIM
            && manifest["model"]["kv_heads"] == KV_HEADS
            && manifest["model"]["query_heads"] == QUERY_HEADS
            && manifest["source_sha256"] == SOURCE_SHA
            && manifest["source"] == "0eee578:docs/perf_baselines.md"
            && manifest["kv_mode"] == "explicit dense FP16 off"
            && manifest["graphs"] == false,
        "capture model/source/config mismatch"
    );
    validate_inputs(manifest)?;
    let complete = rows.last().context("missing completion")?;
    ensure!(
        complete["kind"] == "lossless_kv_capture_complete"
            && complete["is_complete"] == true
            && complete["artifact_bytes"] == ARTIFACT_BYTES
            && complete["tracked_device_bytes_before"].is_u64()
            && complete["tracked_device_bytes_after"] == complete["tracked_device_bytes_before"],
        "capture incomplete or missing cleanup"
    );
    let mut streams = BTreeMap::new();
    let mut sequences = BTreeSet::new();
    for (index, row) in rows.iter().enumerate() {
        match row["kind"].as_str().context("missing record kind")? {
            "lossless_kv_capture_manifest" => ensure!(index == 0, "duplicate manifest"),
            "lossless_kv_capture_complete" => ensure!(index == rows.len() - 1, "early completion"),
            "lossless_kv_stream" => {
                let case = &row["case"];
                let sequence = get_usize(case, "sequence")?;
                let layer = get_usize(case, "layer")?;
                let head = get_usize(case, "head")?;
                let role = case["role"].as_str().context("missing role")?;
                ensure!(
                    sequence < SEQUENCES
                        && LAYERS.contains(&layer)
                        && head < KV_HEADS
                        && ["key", "value"].contains(&role)
                        && row["tokens"] == TOKENS
                        && row["head_dim"] == SOURCE_HEAD_DIM
                        && row["raw_bytes"] == STREAM_BYTES,
                    "invalid raw stream shape"
                );
                let expected = (((sequence * 3
                    + LAYERS.iter().position(|&v| v == layer).unwrap())
                    * KV_HEADS
                    + head)
                    * 2
                    + usize::from(role == "value"))
                    * STREAM_BYTES;
                ensure!(
                    get_usize(row, "raw_offset")? == expected,
                    "raw stream offset mismatch"
                );
                ensure!(
                    streams
                        .insert(
                            (sequence, layer, head, role.to_owned()),
                            Stream {
                                offset: expected as u64,
                                sha256: get_digest(row, "raw_sha256")?,
                            }
                        )
                        .is_none(),
                    "duplicate raw stream"
                );
            }
            "lossless_kv_sequence" => {
                let sequence = get_usize(row, "sequence")?;
                ensure!(
                    sequence < SEQUENCES
                        && sequences.insert(sequence)
                        && row["processed_input_tokens"] == TOKENS
                        && row["generated_token_count"] == 0,
                    "invalid completed sequence"
                );
            }
            "lossless_kv_compression" | "lossless_kv_readback" => {}
            other => anyhow::bail!("unexpected capture record {other}"),
        }
    }
    ensure!(
        streams.len() == STREAMS && sequences.len() == SEQUENCES,
        "missing capture coverage"
    );
    Ok(Metadata {
        manifest: manifest.clone(),
        streams,
        artifact_sha256: get_digest(complete, "artifact_sha256")?,
    })
}

fn validate_inputs(manifest: &Value) -> Result<()> {
    let inputs = manifest["inputs"]
        .as_array()
        .context("missing capture inputs")?;
    ensure!(inputs.len() == SEQUENCES, "wrong input count");
    let mut digests = BTreeSet::new();
    for (sequence, input) in inputs.iter().enumerate() {
        ensure!(
            input["sequence"] == sequence && input["source_token_start"] == sequence * TOKENS,
            "non-disjoint input layout"
        );
        let tokens: Vec<u32> = serde_json::from_value(input["tokens"].clone())?;
        ensure!(
            tokens.len() == TOKENS && tokens.iter().all(|&id| id < 151936),
            "invalid input IDs"
        );
        let bytes: Vec<u8> = tokens.iter().flat_map(|id| id.to_le_bytes()).collect();
        let digest = sha256(&bytes)?;
        ensure!(
            digest == get_digest(input, "token_sha256")? && digests.insert(digest),
            "input digest mismatch or duplicate input"
        );
    }
    Ok(())
}

fn interleave_heads(heads: &[Vec<u8>], length: usize, head_dim: usize) -> Result<Vec<f32>> {
    ensure!(
        heads.len() == KV_HEADS
            && length > 0
            && [64, 128].contains(&head_dim)
            && heads
                .iter()
                .all(|head| head.len() == length * SOURCE_HEAD_DIM * 2),
        "invalid head streams"
    );
    let mut output = Vec::with_capacity(length * KV_HEADS * head_dim);
    for token in 0..length {
        for head in heads {
            for channel in 0..head_dim {
                let offset = (token * SOURCE_HEAD_DIM + channel) * 2;
                let bits = u16::from_le_bytes([head[offset], head[offset + 1]]);
                let value = f16::from_bits(bits).to_f32();
                ensure!(
                    value.is_finite(),
                    "non-finite KV value cannot enter attention timing"
                );
                output.push(value);
            }
        }
    }
    Ok(output)
}

fn build_queries(layer: usize, sequence: usize, head_dim: usize) -> Vec<f32> {
    (0..QUERY_HEADS * head_dim)
        .map(|index| {
            // Integer mixing freezes the probe across host math libraries. Each
            // sequence/head is distinct; no future token or held-out answer enters it.
            let mut value = (index as u32)
                .wrapping_add((sequence as u32 + 1) * 104729)
                .wrapping_add((layer as u32 + 1) * 1009);
            value ^= value >> 16;
            value = value.wrapping_mul(0x7feb352d);
            value ^= value >> 15;
            value = value.wrapping_mul(0x846ca68b);
            value ^= value >> 16;
            ((value & 65535) as f32 - 32768.0) / 262144.0
        })
        .collect()
}

#[cfg(test)]
#[path = "capture_tests.rs"]
mod tests;

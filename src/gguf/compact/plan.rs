use crate::{
    gguf::{tensor_payload_end, GgmlDType, GgufModel, GgufScalar, GgufValue},
    infer::LayerMask,
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::collections::{BTreeSet, HashSet};

/// A byte-preserving tensor copy. Offsets are absolute in the source file and
/// relative to the tensor data section in the separate compact destination.
#[derive(Debug, Clone, Serialize)]
pub struct CompactTensorCopy {
    pub(super) source_name: String,
    pub(super) destination_name: String,
    pub(super) dtype: GgmlDType,
    pub(super) shape: Vec<u64>,
    pub(super) source_offset: u64,
    pub(super) destination_offset: u64,
    pub(super) payload_bytes: u64,
}

impl CompactTensorCopy {
    pub fn source_name(&self) -> &str {
        &self.source_name
    }

    pub fn destination_name(&self) -> &str {
        &self.destination_name
    }

    pub fn source_offset(&self) -> u64 {
        self.source_offset
    }

    pub fn destination_offset(&self) -> u64 {
        self.destination_offset
    }

    pub fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
}

/// Validated layer/tensor mapping, not proof of model quality. The export
/// caller must separately bind a frozen ranking and completed quality report.
#[derive(Debug, Clone, Serialize)]
pub struct CompactGgufPlan {
    pub(super) architecture: String,
    pub(super) original_layer_count: u32,
    pub(super) retained_original_layers: Vec<usize>,
    pub(super) removed_original_layers: Vec<usize>,
    pub(super) alignment: u64,
    pub(super) source_file_bytes: u64,
    pub(super) source_data_offset: u64,
    pub(super) source_weight_payload_bytes: u64,
    pub(super) retained_weight_payload_bytes: u64,
    pub(super) destination_data_bytes: u64,
    pub(super) tensors: Vec<CompactTensorCopy>,
}

fn checked_align(offset: u64, alignment: u64) -> Result<u64> {
    offset
        .checked_add((alignment - offset % alignment) % alignment)
        .context("compact tensor offset overflow")
}

fn metadata_integer(model: &GgufModel, key: &str) -> Result<u64> {
    match model.metadata.get(key) {
        Some(GgufValue::Scalar(GgufScalar::U32(value))) => Ok(u64::from(*value)),
        Some(GgufValue::Scalar(GgufScalar::U64(value))) => Ok(*value),
        _ => anyhow::bail!("compact export requires unsigned scalar {key}"),
    }
}

fn parse_layer_name(name: &str, layer_count: u32) -> Result<Option<(usize, &str)>> {
    let Some(rest) = name.strip_prefix("blk.") else {
        ensure!(
            [
                "token_embd.weight",
                "output.weight",
                "output.bias",
                "output_norm.weight",
                "output_norm.bias",
                "rope_freqs.weight",
            ]
            .contains(&name),
            "unsupported nonstandard tensor {name}; do not infer layer routing"
        );
        return Ok(None);
    };
    let (id, suffix) = rest.split_once('.').context("malformed blk tensor name")?;
    let layer: usize = id.parse().context("invalid original layer ID")?;
    ensure!(
        layer < layer_count as usize && id == layer.to_string() && !suffix.is_empty(),
        "noncanonical/out-of-range original layer in tensor {name}"
    );
    Ok(Some((layer, suffix)))
}

impl CompactGgufPlan {
    /// Validate a standard dense Llama/Qwen2 GGUF before creating any output.
    /// Unsupported metadata/layouts fail rather than guessing how to prune them.
    pub fn build(model: &GgufModel, source_file_bytes: u64, mask: &LayerMask) -> Result<Self> {
        ensure!(
            model.version == 3,
            "compact export initially requires GGUF v3"
        );
        let architecture = model
            .metadata
            .get("general.architecture")
            .and_then(GgufValue::as_str)
            .context("missing compact export architecture")?;
        ensure!(
            matches!(architecture, "llama" | "qwen2"),
            "compact export supports only standard dense llama/qwen2"
        );
        if let Some(kind) = model.metadata.get("general.type") {
            ensure!(
                kind.as_str() == Some("model"),
                "compact export requires a model, not an adapter or auxiliary artifact"
            );
        }
        for (key, value) in &model.metadata {
            ensure!(
                !key.starts_with("split.") && !key.starts_with("m40llm.pruning."),
                "compact export rejects sharded or previously compacted source metadata"
            );
            ensure!(
                !(key.starts_with(&format!("{architecture}."))
                    && (matches!(value, GgufValue::Array(_)) || key.contains("expert"))),
                "unsupported per-layer/expert metadata {key}; explicit remapping required"
            );
        }
        let original_layer_count = u32::try_from(metadata_integer(
            model,
            &format!("{architecture}.block_count"),
        )?)?;
        mask.validate(original_layer_count)?;
        ensure!(
            mask.has_skipped_layers(),
            "compact export requires a nonempty mask"
        );
        ensure!(
            original_layer_count as usize <= model.tensors.len(),
            "compact source has fewer tensors than declared layers"
        );
        let alignment = if model.metadata.contains_key("general.alignment") {
            metadata_integer(model, "general.alignment")?
        } else {
            32
        };
        ensure!(
            alignment.is_multiple_of(8) && (8..=1_048_576).contains(&alignment),
            "unsupported compact GGUF alignment"
        );
        ensure!(
            model.data_offset >= 24
                && model.data_offset.is_multiple_of(alignment)
                && model.data_offset <= source_file_bytes,
            "source tensor section must be aligned and within the file"
        );
        let retained_original_layers = mask
            .active_layers(original_layer_count)?
            .collect::<Vec<_>>();
        let mut layer_tensors = vec![BTreeSet::new(); original_layer_count as usize];
        let mut names = HashSet::new();
        let mut ranges = Vec::new();
        let mut source_weight_payload_bytes = 0u64;
        let mut retained_weight_payload_bytes = 0u64;
        let mut destination_data_bytes = 0u64;
        let mut tensors = Vec::new();
        for tensor in &model.tensors {
            ensure!(
                !tensor.name.is_empty() && tensor.name.len() <= 64 && names.insert(&tensor.name),
                "invalid/duplicate compact source tensor name"
            );
            ensure!(
                matches!(
                    tensor.dtype,
                    GgmlDType::F32 | GgmlDType::F16 | GgmlDType::Q8_0
                ),
                "compact export initially supports only F32/F16/Q8_0 tensor payloads"
            );
            ensure!(
                (1..=4).contains(&tensor.shape.len()) && tensor.shape.iter().all(|&dim| dim > 0),
                "invalid compact source tensor dimensions"
            );
            ensure!(
                tensor.dtype != GgmlDType::Q8_0 || tensor.shape[0].is_multiple_of(32),
                "Q8_0 compact export requires complete quantization blocks per row"
            );
            let end = tensor_payload_end(tensor).context("compact tensor payload size overflow")?;
            let source_offset = model
                .data_offset
                .checked_add(tensor.offset)
                .context("source offset overflow")?;
            let source_end = model
                .data_offset
                .checked_add(end)
                .context("source tensor end overflow")?;
            ensure!(
                tensor.offset.is_multiple_of(alignment) && source_end <= source_file_bytes,
                "misaligned or truncated source tensor {}",
                tensor.name
            );
            ranges.push((source_offset, source_end));
            let payload_bytes = end - tensor.offset;
            source_weight_payload_bytes = source_weight_payload_bytes
                .checked_add(payload_bytes)
                .context("source payload total overflow")?;
            let destination_name = match parse_layer_name(&tensor.name, original_layer_count)? {
                Some((original, suffix)) => {
                    layer_tensors[original].insert(suffix.to_owned());
                    let Some(new_id) = retained_original_layers
                        .iter()
                        .position(|&id| id == original)
                    else {
                        continue;
                    };
                    format!("blk.{new_id}.{suffix}")
                }
                None => tensor.name.clone(),
            };
            let destination_offset = checked_align(destination_data_bytes, alignment)?;
            destination_data_bytes = destination_offset
                .checked_add(payload_bytes)
                .context("compact data size overflow")?;
            retained_weight_payload_bytes = retained_weight_payload_bytes
                .checked_add(payload_bytes)
                .context("retained payload total overflow")?;
            tensors.push(CompactTensorCopy {
                source_name: tensor.name.clone(),
                destination_name,
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                source_offset,
                destination_offset,
                payload_bytes,
            });
        }
        ensure!(
            !layer_tensors[0].is_empty()
                && layer_tensors.iter().all(|names| names == &layer_tensors[0]),
            "compact source requires complete matching tensor families in every original layer"
        );
        ranges.sort_unstable();
        ensure!(
            ranges.windows(2).all(|pair| pair[0].1 <= pair[1].0),
            "overlapping/aliased compact source tensor payloads are unsupported"
        );
        ensure!(
            source_weight_payload_bytes > retained_weight_payload_bytes,
            "compact mask removed no stored weight bytes"
        );
        Ok(Self {
            architecture: architecture.to_owned(),
            original_layer_count,
            retained_original_layers,
            removed_original_layers: mask.skipped_layers().to_vec(),
            alignment,
            source_file_bytes,
            source_data_offset: model.data_offset,
            source_weight_payload_bytes,
            retained_weight_payload_bytes,
            destination_data_bytes,
            tensors,
        })
    }

    pub fn retained_original_layers(&self) -> &[usize] {
        &self.retained_original_layers
    }

    pub fn tensors(&self) -> &[CompactTensorCopy] {
        &self.tensors
    }

    pub fn retained_weight_payload_bytes(&self) -> u64 {
        self.retained_weight_payload_bytes
    }

    pub fn source_weight_payload_bytes(&self) -> u64 {
        self.source_weight_payload_bytes
    }

    pub fn destination_data_bytes(&self) -> u64 {
        self.destination_data_bytes
    }
}

use super::{CompactExportEvidence, CompactGgufPlan};
use crate::gguf::{
    read_scalar, read_string, read_u32, read_u64, GgmlDType, GgufModel, GgufScalar, GgufValue,
    GgufValueType,
};
use anyhow::{ensure, Context, Result};
use std::{
    collections::BTreeMap,
    io::{Read, Seek, SeekFrom, Write},
};

const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

struct MetadataRange {
    key: String,
    start: u64,
    end: u64,
}

fn scan_metadata(source: &mut (impl Read + Seek), model: &GgufModel) -> Result<Vec<MetadataRange>> {
    ensure!(
        model.data_offset <= MAX_HEADER_BYTES,
        "compact source header exceeds 64 MiB"
    );
    source.seek(SeekFrom::Start(8))?;
    ensure!(
        read_u64(source)? == model.tensors.len() as u64,
        "source tensor count changed"
    );
    ensure!(
        read_u64(source)? == model.metadata.len() as u64,
        "duplicate/changed source metadata"
    );
    let mut entries = Vec::new();
    let mut keys = std::collections::HashSet::new();
    for _ in 0..model.metadata.len() {
        let start = source.stream_position()?;
        let key = read_string(source)?;
        ensure!(
            keys.insert(key.clone()) && model.metadata.contains_key(&key),
            "duplicate/changed source metadata key"
        );
        let kind = GgufValueType::from_u32(read_u32(source)?).context("source metadata type")?;
        if matches!(kind, GgufValueType::Array) {
            let element =
                GgufValueType::from_u32(read_u32(source)?).context("source array type")?;
            let count = read_u64(source)?;
            ensure!(
                count <= MAX_HEADER_BYTES && !matches!(element, GgufValueType::Array),
                "unsupported source metadata array"
            );
            for _ in 0..count {
                read_scalar(source, element)?;
            }
        } else {
            read_scalar(source, kind)?;
        }
        let end = source.stream_position()?;
        ensure!(
            end <= model.data_offset,
            "source metadata overlaps tensor data"
        );
        entries.push(MetadataRange { key, start, end });
    }
    Ok(entries)
}

fn write_string(out: &mut Vec<u8>, text: &str) -> Result<()> {
    out.write_all(&(text.len() as u64).to_le_bytes())?;
    out.write_all(text.as_bytes())?;
    Ok(())
}

fn write_new_entry(out: &mut Vec<u8>, key: &str, value: &GgufValue) -> Result<()> {
    write_string(out, key)?;
    match value {
        GgufValue::Scalar(GgufScalar::U32(value)) => {
            out.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
            out.extend_from_slice(&value.to_le_bytes());
        }
        GgufValue::Scalar(GgufScalar::U64(value)) => {
            out.extend_from_slice(&(GgufValueType::Uint64 as u32).to_le_bytes());
            out.extend_from_slice(&value.to_le_bytes());
        }
        GgufValue::Scalar(GgufScalar::Str(value)) => {
            out.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
            write_string(out, value)?;
        }
        GgufValue::Scalar(GgufScalar::Bool(value)) => {
            out.extend_from_slice(&(GgufValueType::Bool as u32).to_le_bytes());
            out.push(u8::from(*value));
        }
        GgufValue::Array(values) => {
            out.extend_from_slice(&(GgufValueType::Array as u32).to_le_bytes());
            out.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
            out.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                let GgufScalar::U32(value) = value else {
                    anyhow::bail!("new compact metadata array must be U32");
                };
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        _ => anyhow::bail!("unsupported new compact metadata type"),
    }
    Ok(())
}

fn replacement_metadata(
    model: &GgufModel,
    plan: &CompactGgufPlan,
    evidence: &CompactExportEvidence,
) -> Result<BTreeMap<String, GgufValue>> {
    let scalar = GgufValue::Scalar;
    let mut values = BTreeMap::new();
    let block_key = format!("{}.block_count", plan.architecture);
    let block_count = plan.retained_original_layers.len() as u32;
    let count = match model.metadata.get(&block_key) {
        Some(GgufValue::Scalar(GgufScalar::U32(_))) => GgufScalar::U32(block_count),
        Some(GgufValue::Scalar(GgufScalar::U64(_))) => GgufScalar::U64(u64::from(block_count)),
        _ => anyhow::bail!("unsupported source block count metadata"),
    };
    values.insert(block_key, scalar(count));
    let original_name = match model.metadata.get("general.name") {
        Some(value) => value
            .as_str()
            .context("source model name must be a string")?,
        None => &plan.architecture,
    };
    values.insert(
        "general.name".into(),
        scalar(GgufScalar::Str(format!(
            "{original_name} m40-compact-{}of{}L",
            plan.retained_original_layers.len(),
            plan.original_layer_count
        ))),
    );
    let parameters = plan
        .tensors
        .iter()
        .try_fold(0u64, |total, tensor| {
            let elements = tensor
                .shape
                .iter()
                .try_fold(1u64, |count, dim| count.checked_mul(*dim))?;
            total.checked_add(elements)
        })
        .context("compact parameter count overflow")?;
    values.insert(
        "general.parameter_count".into(),
        scalar(GgufScalar::U64(parameters)),
    );
    for (key, hash) in [
        ("source_sha256", &evidence.source_sha256),
        ("ranking_sha256", &evidence.ranking_sha256),
        ("quality_report_sha256", &evidence.quality_report_sha256),
        ("corpus_sha256", &evidence.corpus_sha256),
    ] {
        values.insert(
            format!("m40llm.pruning.{key}"),
            scalar(GgufScalar::Str(hash.clone())),
        );
    }
    values.insert(
        "m40llm.pruning.schema_version".into(),
        scalar(GgufScalar::U32(1)),
    );
    values.insert(
        "m40llm.pruning.original_layer_count".into(),
        scalar(GgufScalar::U32(plan.original_layer_count)),
    );
    values.insert(
        "m40llm.pruning.model_semantics_changed".into(),
        scalar(GgufScalar::Bool(true)),
    );
    values.insert(
        "m40llm.pruning.quality_summary".into(),
        scalar(GgufScalar::Str(serde_json::to_string(&evidence.quality)?)),
    );
    for (key, ids) in [
        ("retained_original_layers", &plan.retained_original_layers),
        ("removed_original_layers", &plan.removed_original_layers),
    ] {
        values.insert(
            format!("m40llm.pruning.{key}"),
            GgufValue::Array(ids.iter().map(|&id| GgufScalar::U32(id as u32)).collect()),
        );
    }
    Ok(values)
}

pub(super) fn encode_header(
    source: &mut (impl Read + Seek),
    model: &GgufModel,
    plan: &CompactGgufPlan,
    evidence: &CompactExportEvidence,
) -> Result<Vec<u8>> {
    let entries = scan_metadata(source, model)?;
    let replacements = replacement_metadata(model, plan, evidence)?;
    // The source size label describes the unpruned parameter count. Preserve
    // authorship/license/base lineage, but do not export that stale size claim.
    let retained: Vec<_> = entries
        .iter()
        .filter(|entry| !replacements.contains_key(&entry.key) && entry.key != "general.size_label")
        .collect();
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(plan.tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&((retained.len() + replacements.len()) as u64).to_le_bytes());
    for entry in retained {
        source.seek(SeekFrom::Start(entry.start))?;
        let count = entry.end - entry.start;
        ensure!(
            std::io::copy(&mut source.take(count), &mut out)? == count,
            "truncated source metadata"
        );
    }
    for (key, value) in replacements {
        write_new_entry(&mut out, &key, &value)?;
    }
    for tensor in &plan.tensors {
        write_string(&mut out, &tensor.destination_name)?;
        out.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        for dim in &tensor.shape {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        let dtype: u32 = match tensor.dtype {
            GgmlDType::F32 => 0,
            GgmlDType::F16 => 1,
            GgmlDType::Q8_0 => 8,
            _ => anyhow::bail!("unvalidated compact tensor dtype"),
        };
        out.extend_from_slice(&dtype.to_le_bytes());
        out.extend_from_slice(&tensor.destination_offset.to_le_bytes());
    }
    ensure!(
        out.len() as u64 <= MAX_HEADER_BYTES,
        "compact output header exceeds 64 MiB"
    );
    Ok(out)
}

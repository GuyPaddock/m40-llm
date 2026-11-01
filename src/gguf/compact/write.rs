use super::{metadata::encode_header, CompactExportEvidence, CompactGgufPlan};
use crate::{
    evaluation::identity::file_sha256,
    gguf::{load_gguf, GgufScalar, GgufValue},
    infer::LayerMask,
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

struct PendingFile(PathBuf);

impl PendingFile {
    fn create(destination: &Path) -> Result<(Self, File)> {
        let parent = destination
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let name = destination
            .file_name()
            .context("compact export requires an output filename")?;
        let mut temporary = std::ffi::OsString::from(".");
        temporary.push(name);
        temporary.push(format!(
            ".{}-{}.partial",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        let path = parent.join(temporary);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok((Self(path), file))
    }
}

impl Drop for PendingFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[derive(Debug, Serialize)]
pub struct CompactExportResult {
    pub schema_version: u32,
    pub source: PathBuf,
    pub destination: PathBuf,
    pub output_sha256: String,
    pub output_file_bytes: u64,
    pub source_tensor_data_bytes: u64,
    pub output_tensor_data_bytes: u64,
    /// Export verification/publication only, not prior evidence loading or GPU inference.
    pub export_elapsed_ms: f64,
    pub evidence: CompactExportEvidence,
    pub plan: CompactGgufPlan,
}

fn pad(out: &mut impl Write, count: u64) -> Result<()> {
    ensure!(
        std::io::copy(&mut std::io::repeat(0).take(count), out)? == count,
        "short compact padding write"
    );
    Ok(())
}

fn validate_round_trip(path: &Path, plan: &CompactGgufPlan, data_offset: u64) -> Result<()> {
    let model = load_gguf(path)?;
    ensure!(
        model.version == 3
            && model.data_offset == data_offset
            && model.tensors.len() == plan.tensors.len(),
        "compact GGUF round-trip header mismatch"
    );
    let count = match model
        .metadata
        .get(&format!("{}.block_count", plan.architecture))
    {
        Some(GgufValue::Scalar(GgufScalar::U32(count))) => u64::from(*count),
        Some(GgufValue::Scalar(GgufScalar::U64(count))) => *count,
        _ => anyhow::bail!("compact GGUF block count missing"),
    };
    ensure!(
        count == plan.retained_original_layers.len() as u64,
        "compact GGUF layer count mismatch"
    );
    for (actual, expected) in model.tensors.iter().zip(&plan.tensors) {
        ensure!(
            actual.name == expected.destination_name
                && actual.dtype == expected.dtype
                && actual.shape == expected.shape
                && actual.offset == expected.destination_offset,
            "compact GGUF tensor descriptor mismatch"
        );
    }
    Ok(())
}

/// Create a separate, atomically published GGUF from an evaluated frozen mask.
/// Existing files, hard links and symlinks are never overwritten. Source must
/// remain unchanged; SHA-256 is checked both before and after byte copying.
pub fn export_compact_gguf(
    source_path: &Path,
    destination: &Path,
    evidence: &CompactExportEvidence,
) -> Result<CompactExportResult> {
    let started = Instant::now();
    match fs::symlink_metadata(destination) {
        Ok(_) => anyhow::bail!("refuse to overwrite compact export destination {destination:?}"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let source_path = fs::canonicalize(source_path)?;
    ensure!(
        file_sha256(&source_path)? == evidence.source_sha256,
        "compact export source differs from evaluated model"
    );
    let model = load_gguf(&source_path)?;
    let mut source = BufReader::with_capacity(1024 * 1024, File::open(&source_path)?);
    let source_bytes = source.get_ref().metadata()?.len();
    let mask = LayerMask::new(
        evidence.original_layer_count,
        &evidence.removed_original_layers,
    )?;
    let plan = CompactGgufPlan::build(&model, source_bytes, &mask)?;
    let hidden = match model
        .metadata
        .get(&format!("{}.embedding_length", plan.architecture))
    {
        Some(GgufValue::Scalar(GgufScalar::U32(value))) => u64::from(*value),
        Some(GgufValue::Scalar(GgufScalar::U64(value))) => *value,
        _ => anyhow::bail!("compact source embedding length must be unsigned scalar"),
    };
    ensure!(
        hidden == evidence.hidden_dim as u64,
        "compact source dimensions differ from ranking"
    );
    let header = encode_header(&mut source, &model, &plan, evidence)?;
    let header_len = header.len() as u64;
    let data_offset = header_len
        .checked_add((plan.alignment - header_len % plan.alignment) % plan.alignment)
        .context("compact header alignment overflow")?;
    let (pending, file) = PendingFile::create(destination)?;
    {
        let mut out = BufWriter::with_capacity(1024 * 1024, file);
        out.write_all(&header)?;
        pad(&mut out, data_offset - header_len)?;
        let mut written = 0;
        for tensor in &plan.tensors {
            pad(
                &mut out,
                tensor
                    .destination_offset
                    .checked_sub(written)
                    .context("unordered compact destination tensors")?,
            )?;
            source.seek(SeekFrom::Start(tensor.source_offset))?;
            ensure!(
                std::io::copy(&mut source.by_ref().take(tensor.payload_bytes), &mut out)?
                    == tensor.payload_bytes,
                "truncated compact source tensor"
            );
            written = tensor.destination_offset + tensor.payload_bytes;
        }
        ensure!(
            written == plan.destination_data_bytes,
            "compact data size differs from plan"
        );
        out.flush()?;
        out.get_ref().sync_all()?;
    }
    ensure!(
        source.get_ref().metadata()?.len() == source_bytes
            && file_sha256(&source_path)? == evidence.source_sha256,
        "compact source changed during export; destination not published"
    );
    validate_round_trip(&pending.0, &plan, data_offset)?;
    let output_file_bytes = fs::metadata(&pending.0)?.len();
    ensure!(
        Some(output_file_bytes) == data_offset.checked_add(plan.destination_data_bytes),
        "compact file length mismatch"
    );
    let output_sha256 = file_sha256(&pending.0)?;
    // A same-directory hard link publishes the completed file with create-new
    // semantics, including when another writer wins the destination-name race.
    fs::hard_link(&pending.0, destination)
        .context("publish compact GGUF without overwriting; same-directory hard links required")?;
    let result = CompactExportResult {
        schema_version: 1,
        source: source_path,
        destination: fs::canonicalize(destination)?,
        output_sha256,
        output_file_bytes,
        source_tensor_data_bytes: source_bytes - model.data_offset,
        output_tensor_data_bytes: plan.destination_data_bytes,
        export_elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        evidence: evidence.clone(),
        plan,
    };
    drop(pending);
    Ok(result)
}

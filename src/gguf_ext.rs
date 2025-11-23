// src/gguf_ext.rs
//! Optional integration with gguf-llms + gguf-rs-lib when feature `gguf_ext` is enabled.

#![allow(dead_code)]

/// High-level overview extracted via gguf-llms without loading tensor data.
#[derive(Debug)]
pub struct LlmsOverview {
    pub config: gguf_llms::model::ModelConfig,
    pub n_tensors: u64,
    pub kv_len: usize,
}

use anyhow::Result;
use std::path::Path;

use std::fs::File;
use std::io::Read;

/// Perform a lightweight inspection using gguf-rs-lib (optional).
pub fn inspect_with_gguf_rs_lib(path: &Path) -> Result<()> {
    use gguf_rs_lib::reader::file_reader::open_gguf_file;
    let reader = open_gguf_file(path)?;
    let hdr = reader.header();
    let _summary = reader.summary();
    let _tensor_count = reader.tensor_count();
    let _kv_len = reader.metadata().len();
    let _ = (hdr.version, _summary, _tensor_count, _kv_len);
    Ok(())
}

/// Extract typed model configuration using gguf-llms directly from file metadata.
/// This reads just the header counters and uses gguf-llms to parse the key/value section.
pub fn extract_model_config_with_gguf_llms(path: &Path) -> Result<gguf_llms::model::ModelConfig> {
    // Minimal header parse to get n_kv and advance file cursor to metadata start.
    let mut f = File::open(path)?;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        anyhow::bail!("Not a GGUF file (magic != GGUF)");
    }
    // version (u32 LE)
    let _version = read_u32(&mut f)?;
    // n_tensors (u64), n_kv (u64)
    let _n_tensors = read_u64(&mut f)?;
    let n_kv = read_u64(&mut f)?;

    /// Quick overview by combining gguf-llms metadata parsing and gguf-rs-lib counts.
    pub fn overview(path: &Path) -> Result<LlmsOverview> {
        use gguf_rs_lib::reader::file_reader::open_gguf_file;
        let reader = open_gguf_file(path)?;
        let n_tensors = reader.tensor_count() as u64;
        let kv_len = reader.metadata().len();
        let config = extract_model_config_with_gguf_llms(path)?;
        Ok(LlmsOverview {
            config,
            n_tensors,
            kv_len,
        })
    }

    // Use gguf-llms reader to parse exactly n_kv metadata entries from current position.
    let meta = gguf_llms::metadata::GgufReader::read_metadata(&mut f, n_kv)
        .map_err(|e| anyhow::anyhow!("gguf-llms: read_metadata failed: {e}"))?;

    // Extract typed model config from metadata
    let cfg = gguf_llms::config::extract_model_config(&meta)
        .map_err(|e| anyhow::anyhow!("gguf-llms: extract_model_config failed: {e}"))?;
    Ok(cfg)
}

/// Load all tensors using gguf-llms (reads tensor infos and data into memory).
pub fn load_all_tensors_with_llms(
    path: &Path,
) -> Result<std::collections::HashMap<String, gguf_llms::tensors::Tensor>> {
    use std::io::{Read, Seek};
    // Open file and minimally parse header/metadata to reach tensor infos
    let mut reader = std::fs::File::open(path)?;

    // Parse header to obtain n_tensors and n_kv so we can skip metadata
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        anyhow::bail!("Not a GGUF file (magic != GGUF)");
    }
    let _version = read_u32(&mut reader)?;
    let n_tensors = read_u64(&mut reader)?;
    let n_kv = read_u64(&mut reader)?;

    // Skip metadata section (we can reuse our simple parsers to advance)
    for _ in 0..n_kv {
        // key
        let _ = read_string_len_and_skip(&mut reader)?;
        // value type
        let vt = read_u32(&mut reader)?;
        if vt == 9
        /* Array */
        {
            let _elem_ty = read_u32(&mut reader)?;
            let len = read_u64(&mut reader)?;
            // Skip len elements according to elem type sizes by reusing read_scalar-sized skips
            // We just iterate to advance; cost negligible for metadata
            for _ in 0..len {
                let _ = read_scalar_generic_skip(&mut reader, _elem_ty)?;
            }
        } else {
            let _ = read_scalar_generic_skip(&mut reader, vt)?;
        }
    }

    // Now read tensor infos using gguf-llms helper to ensure identical interpretation
    let tensor_infos = gguf_llms::tensors::TensorLoader::read_tensor_info(&mut reader, n_tensors)
        .map_err(|e| anyhow::anyhow!("gguf-llms: read_tensor_info failed: {e}"))?;

    // The current position is the tensor data start
    let tensor_data_start = reader.stream_position()?;

    // Load all tensors
    let map = gguf_llms::tensors::TensorLoader::load_all_tensors(
        &mut reader,
        &tensor_infos,
        tensor_data_start,
    )
    .map_err(|e| anyhow::anyhow!("gguf-llms: load_all_tensors failed: {e}"))?;

    Ok(map)
}

// Local helpers (duplicated minimal readers) kept private to this module.
fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}
fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

// Helpers to skip over metadata quickly without allocating
fn read_string_len_and_skip<R: Read>(r: &mut R) -> Result<()> {
    let len = read_u64(r)?;
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    Ok(())
}

fn read_scalar_generic_skip<R: Read>(r: &mut R, vt_u32: u32) -> Result<()> {
    // Matches GgufValueType numeric codes used in src/gguf.rs
    match vt_u32 {
        0 | 1 => {
            // u8 / i8
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
        }
        2 | 3 => {
            // u16 / i16
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
        }
        4 | 5 | 6 | 7 => {
            // u32 / i32 / f32 / bool
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
        }
        8 => {
            // string
            read_string_len_and_skip(r)?;
        }
        10 | 11 | 12 => {
            // u64 / i64 / f64
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
        }
        9 => unreachable!("ARRAY handled by caller"),
        _ => anyhow::bail!("unknown GGUF value type: {vt_u32}"),
    }
    Ok(())
}

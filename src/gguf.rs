// src/gguf.rs
use anyhow::{Result, Context};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub enum GgufDType {
    F32,
    F16,
    // plus quantized dtypes later
}

#[derive(Debug)]
pub struct GgufTensor {
    pub name: String,
    pub dtype: GgufDType,
    pub shape: Vec<u64>,
    pub offset: u64,  // file offset of raw data
}

#[derive(Debug)]
pub struct GgufModel {
    pub tensors: Vec<GgufTensor>,
    pub metadata: HashMap<String, String>,
}

pub fn load_gguf(path: &Path) -> Result<GgufModel> {
    let mut f = File::open(path).with_context(|| format!("open gguf {:?}", path))?;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        anyhow::bail!("Not a GGUF file");
    }

    let mut version_bytes = [0u8; 4];
    f.read_exact(&mut version_bytes)?;
    let _version = u32::from_le_bytes(version_bytes);

    // For brevity: skip a *lot* of correct GGUF parsing here.
    // Real code needs to follow the spec: read n_kv, kv entries, tensor count,
    // then for each tensor: name, dtype, n_dims, dims, offset.

    // Placeholder: treat whole file after header as a single F16 blob
    // and call it "W".
    let offset = f.seek(SeekFrom::End(0))?;
    let data_size = offset - 8; // we read 8 bytes already
    let tensor = GgufTensor {
        name: "W".to_string(),
        dtype: GgufDType::F16,
        shape: vec![data_size / 2], // "vector of f16"
        offset: 8,
    };

    Ok(GgufModel {
        tensors: vec![tensor],
        metadata: HashMap::new(),
    })
}

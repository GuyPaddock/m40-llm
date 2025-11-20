// src/gguf.rs
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// GGUF "value type" enum (for metadata / kv pairs)
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum GgufValueType {
  Uint8   = 0,
  Int8    = 1,
  Uint16  = 2,
  Int16   = 3,
  Uint32  = 4,
  Int32   = 5,
  Float32 = 6,
  Bool    = 7,
  String  = 8,
  Array   = 9,
  Uint64  = 10,
  Int64   = 11,
  Float64 = 12,
}

impl GgufValueType {
  fn from_u32(x: u32) -> Option<Self> {
    use GgufValueType::*;
    Some(match x {
      0 => Uint8,
      1 => Int8,
      2 => Uint16,
      3 => Int16,
      4 => Uint32,
      5 => Int32,
      6 => Float32,
      7 => Bool,
      8 => String,
      9 => Array,
      10 => Uint64,
      11 => Int64,
      12 => Float64,
      _ => return None,
    })
  }
}

/// Scalar value used in GGUF metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GgufScalar {
  U8(u8),
  I8(i8),
  U16(u16),
  I16(i16),
  U32(u32),
  I32(i32),
  U64(u64),
  I64(i64),
  F32(f32),
  F64(f64),
  Bool(bool),
  Str(String),
}

/// A GGUF metadata value: either a single scalar or an array of scalars.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GgufValue {
  Scalar(GgufScalar),
  Array(Vec<GgufScalar>),
}

impl GgufValue {
  /// Convenience: get scalar string value if it is one.
  pub fn as_str(&self) -> Option<&str> {
    match self {
      GgufValue::Scalar(GgufScalar::Str(s)) => Some(s),
      _ => None,
    }
  }
}

/// GGML tensor (weights) dtypes as found in GGUF
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GgmlDType {
  F32,
  F16,
  Q4_0,
  Q4_1,
  Q5_0,
  Q5_1,
  Q8_0,
  Q8_1,
  Q2K,
  Q3K,
  Q4K,
  Q5K,
  Q6K,
  Q8K,
  Unknown(u32),
}

impl GgmlDType {
  pub fn from_u32(x: u32) -> Self {
    use GgmlDType::*;
    match x {
      0  => F32,
      1  => F16,
      2  => Q4_0,
      3  => Q4_1,
      6  => Q5_0,
      7  => Q5_1,
      8  => Q8_0,
      9  => Q8_1,
      10 => Q2K,
      11 => Q3K,
      12 => Q4K,
      13 => Q5K,
      14 => Q6K,
      15 => Q8K,
      other => Unknown(other),
    }
  }

  pub fn is_f16(&self) -> bool {
    matches!(self, GgmlDType::F16)
  }

  pub fn is_f32(&self) -> bool {
    matches!(self, GgmlDType::F32)
  }
}

/// A single tensor entry in GGUF
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufTensor {
  pub name: String,
  pub dtype: GgmlDType,
  pub shape: Vec<u64>,
  /// Offset (bytes) from the **start of the tensor data section** in the file.
  pub offset: u64,
}

/// Parsed GGUF model
#[derive(Debug, Serialize, Deserialize)]
pub struct GgufModel {
  pub version: u32,
  pub metadata: HashMap<String, GgufValue>,
  pub tensors: Vec<GgufTensor>,
  /// File offset where the tensor data region begins.
  pub data_offset: u64,
}

impl GgufModel {
    /// Create a new dummy GgufModel for testing
    pub fn new(_dummy: u32) -> Self {
        GgufModel {
            version: 1,
            metadata: HashMap::new(),
            tensors: Vec::new(),
            data_offset: 0,
        }
    }

/// Public entry point: parse GGUF headers & tensor descriptors from file.
pub fn load_gguf(path: &Path) -> Result<GgufModel> {
  let mut f = File::open(path)
    .with_context(|| format!("failed to open GGUF file {:?}", path))?;

  // 1. Magic
  let mut magic = [0u8; 4];
  f.read_exact(&mut magic)?;
  if &magic != b"GGUF" {
    anyhow::bail!("Not a GGUF file (magic != GGUF)");
  }

  // 2. Version
  let version = read_u32(&mut f)?;
  // (You can add version checks here if you want.)

  // 3. n_tensors, n_kv
  // NOTE: GGUF uses little-endian u64 for these.
  let n_tensors = read_u64(&mut f)?;
  let n_kv = read_u64(&mut f)?;

  // 4. Key-value metadata
  let mut metadata = HashMap::new();
  for _ in 0..n_kv {
    let key = read_string(&mut f)?;
    let vt_raw = read_u32(&mut f)?;
    let vt = GgufValueType::from_u32(vt_raw)
      .ok_or_else(|| anyhow::anyhow!("unknown GGUF value type: {}", vt_raw))?;

    let val = match vt {
      GgufValueType::Array => {
        // Arrays have their own element type + length
        let elem_type_raw = read_u32(&mut f)?;
        let elem_type = GgufValueType::from_u32(elem_type_raw)
          .ok_or_else(|| anyhow::anyhow!("unknown GGUF array element type: {}", elem_type_raw))?;
        let len = read_u64(&mut f)?;
        let mut elems = Vec::with_capacity(len as usize);
        for _ in 0..len {
          let scalar = read_scalar(&mut f, elem_type)?;
          elems.push(scalar);
        }
        GgufValue::Array(elems)
      }
      other => {
        let scalar = read_scalar(&mut f, other)?;
        GgufValue::Scalar(scalar)
      }
    };

    metadata.insert(key, val);
  }

  // 5. Tensor descriptors
  let mut tensors = Vec::with_capacity(n_tensors as usize);
  for _ in 0..n_tensors {
    let name = read_string(&mut f)?;

    // number of dimensions (u32), then dims as u64[]
    let n_dims = read_u32(&mut f)?;
    let mut shape = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
      shape.push(read_u64(&mut f)?);
    }

    let dtype_raw = read_u32(&mut f)?;
    let dtype = GgmlDType::from_u32(dtype_raw);

    // The offset here is relative to the **start of the tensor data section**
    let offset = read_u64(&mut f)?;

    tensors.push(GgufTensor {
      name,
      dtype,
      shape,
      offset,
    });
  }

  // The current file position is the start of the tensor data block.
  let data_offset = f.seek(SeekFrom::Current(0))?;

  Ok(GgufModel {
    version,
    metadata,
    tensors,
    data_offset,
  })
}

// ---------- helpers ----------

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

fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
  let mut buf = [0u8; 4];
  r.read_exact(&mut buf)?;
  Ok(i32::from_le_bytes(buf))
}

fn read_i64<R: Read>(r: &mut R) -> Result<i64> {
  let mut buf = [0u8; 8];
  r.read_exact(&mut buf)?;
  Ok(i64::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
  let mut buf = [0u8; 4];
  r.read_exact(&mut buf)?;
  Ok(f32::from_le_bytes(buf))
}

fn read_f64<R: Read>(r: &mut R) -> Result<f64> {
  let mut buf = [0u8; 8];
  r.read_exact(&mut buf)?;
  Ok(f64::from_le_bytes(buf))
}

fn read_bool<R: Read>(r: &mut R) -> Result<bool> {
  let mut b = [0u8; 1];
  r.read_exact(&mut b)?;
  Ok(b[0] != 0)
}

fn read_string<R: Read>(r: &mut R) -> Result<String> {
  // GGUF strings: length (u64), then that many raw bytes (no null terminator)
  let len = read_u64(r)?;
  let mut buf = vec![0u8; len as usize];
  r.read_exact(&mut buf)?;
  let s = String::from_utf8(buf)
    .context("GGUF string is not valid UTF-8")?;
  Ok(s)
}

fn read_scalar<R: Read>(r: &mut R, vt: GgufValueType) -> Result<GgufScalar> {
  use GgufScalar::*;
  use GgufValueType::*;
  let scalar = match vt {
    Uint8 => {
      let mut b = [0u8; 1];
      r.read_exact(&mut b)?;
      U8(b[0])
    }
    Int8 => {
      let mut b = [0u8; 1];
      r.read_exact(&mut b)?;
      I8(b[0] as i8)
    }
    Uint16 => {
      let mut buf = [0u8; 2];
      r.read_exact(&mut buf)?;
      U16(u16::from_le_bytes(buf))
    }
    Int16 => {
      let mut buf = [0u8; 2];
      r.read_exact(&mut buf)?;
      I16(i16::from_le_bytes(buf))
    }
    Uint32 => U32(read_u32(r)?),
    Int32 => I32(read_i32(r)?),
    Uint64 => U64(read_u64(r)?),
    Int64 => I64(read_i64(r)?),
    Float32 => F32(read_f32(r)?),
    Float64 => F64(read_f64(r)?),
    Bool => Bool(read_bool(r)?),
    String => {
      let s = read_string(r)?;
      Str(s)
    }
    Array => {
      // The ARRAY case should be handled at a higher level.
      anyhow::bail!("read_scalar called with ARRAY type")
    }
  };
  Ok(scalar)
}

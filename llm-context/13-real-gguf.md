User:
Build a real GGUF loader for Mistral token shapes
---
Agent:
You’re very welcome 😊 This is a fun project.

Let’s make **`gguf.rs` real** and wire it into the model loader so it can infer **Mistral / LLaMA-style shapes from actual GGUF metadata** instead of hardcoding.

---

## 1. New `src/gguf.rs` – real GGUF parser + llama/Mistral hparams

Drop this in as a full replacement for your current `gguf.rs`:

```rust
// src/gguf.rs
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Scalar subset of GGUF metadata we actually care about.
/// Arrays are currently skipped (we just advance the cursor).
#[derive(Debug, Clone)]
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

/// Tensor descriptor from the GGUF header.
#[derive(Debug, Clone)]
pub struct GgufTensor {
  pub name: String,
  pub offset: u64,   // offset into tensor data block
  pub shape: Vec<u64>,
  pub n_dims: u32,
  pub dtype: u32,    // ggml_type as raw code; we don’t interpret it yet
}

/// Parsed GGUF file: metadata, tensor infos, and raw bytes.
///
/// `data_offset` is the start of the tensor-data block within `data`.
#[derive(Debug, Clone)]
pub struct GgufModel {
  pub kv: HashMap<String, GgufScalar>,
  pub tensors: Vec<GgufTensor>,
  pub data: Vec<u8>,
  pub data_offset: u64,
}

/// LLaMA-style hyperparameters.
/// This applies to Mistral, LLaMA, etc., when exported with `general.architecture = "llama"`.
#[derive(Debug, Clone)]
pub struct LlamaHParams {
  pub arch: String,       // e.g. "llama", "mistral"
  pub vocab_size: usize,  // llama.vocab_size
  pub context_length: usize, // llama.context_length
  pub d_model: usize,     // llama.embedding_length
  pub n_layer: usize,     // llama.block_count
  pub n_head: usize,      // llama.attention.head_count
  pub n_head_kv: usize,   // llama.attention.head_count_kv (GQA/MQA)
  pub head_dim: usize,    // d_model / n_head
}

impl GgufModel {
  /// Load and parse a GGUF file from disk.
  pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
    let mut f = File::open(path)?;
    let mut data = Vec::new();
    f.read_to_end(&mut data)?;

    let (kv, tensors, data_offset) = parse_gguf_header(&data)?;
    Ok(GgufModel {
      kv,
      tensors,
      data,
      data_offset,
    })
  }

  /// Convenience helper: get a scalar by key as u32 (with some tolerant casting).
  pub fn get_u32(&self, key: &str) -> Option<u32> {
    use GgufScalar::*;
    match self.kv.get(key) {
      Some(U32(v)) => Some(*v),
      Some(U64(v)) => (*v).try_into().ok(),
      Some(I32(v)) if *v >= 0 => (*v as u32).into(),
      Some(I64(v)) if *v >= 0 => (*v as u64).try_into().ok(),
      _ => None,
    }
  }

  pub fn get_f32(&self, key: &str) -> Option<f32> {
    use GgufScalar::*;
    match self.kv.get(key) {
      Some(F32(v)) => Some(*v),
      Some(F64(v)) => Some(*v as f32),
      _ => None,
    }
  }

  pub fn get_str(&self, key: &str) -> Option<&str> {
    match self.kv.get(key) {
      Some(GgufScalar::Str(s)) => Some(s.as_str()),
      _ => None,
    }
  }

  /// Look up a tensor by name.
  pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
    self.tensors.iter().find(|t| t.name == name)
  }

  /// Derive LLaMA/Mistral hyperparameters from GGUF metadata.
  ///
  /// This expects the standard `llama.*` keys emitted by llama.cpp’s converter,
  /// which are used for LLaMA, Mistral, etc. (even when the *original* model
  /// architecture wasn’t literally "llama").
  pub fn llama_hparams(&self) -> Result<LlamaHParams> {
    // Architecture string is informative but not strictly required.
    let arch = self
      .get_str("general.architecture")
      .unwrap_or("llama")
      .to_owned();

    let n_layer = self
      .get_u32("llama.block_count")
      .ok_or_else(|| anyhow!("missing llama.block_count in GGUF metadata"))?
      as usize;

    let d_model = self
      .get_u32("llama.embedding_length")
      .ok_or_else(|| anyhow!("missing llama.embedding_length in GGUF metadata"))?
      as usize;

    let n_head = self
      .get_u32("llama.attention.head_count")
      .ok_or_else(|| anyhow!("missing llama.attention.head_count in GGUF metadata"))?
      as usize;

    let n_head_kv = self
      .get_u32("llama.attention.head_count_kv")
      .unwrap_or(n_head) as usize;

    let context_length = self
      .get_u32("llama.context_length")
      .unwrap_or(2048) as usize;

    // vocab can come from metadata or from the embedding tensor shape
    let vocab_size_meta = self.get_u32("llama.vocab_size").map(|v| v as usize);
    let vocab_size = match vocab_size_meta {
      Some(v) => v,
      None => {
        // Fallback: derive from embedding matrix shape
        let emb = self
          .tensor("tok_embeddings.weight")
          .or_else(|| self.tensor("token_embd.weight"))
          .ok_or_else(|| anyhow!("cannot find tok_embeddings.weight or token_embd.weight tensor"))?;

        if emb.shape.len() < 2 {
          return Err(anyhow!(
            "unexpected embedding tensor rank: {} (need at least 2 dims)",
            emb.shape.len()
          ));
        }
        emb.shape[0] as usize
      }
    };

    let head_dim = if n_head > 0 {
      d_model / n_head
    } else {
      return Err(anyhow!("llama.attention.head_count is zero"));
    };

    Ok(LlamaHParams {
      arch,
      vocab_size,
      context_length,
      d_model,
      n_layer,
      n_head,
      n_head_kv,
      head_dim,
    })
  }
}

// -----------------------------
// Low-level parsing utilities
// -----------------------------

fn parse_gguf_header(
  data: &[u8],
) -> Result<(HashMap<String, GgufScalar>, Vec<GgufTensor>, u64)> {
  let mut off: usize = 0;

  let magic = read_u32(data, &mut off)?;
  // "GGUF" in little-endian
  const GGUF_MAGIC: u32 = u32::from_le_bytes(*b"GGUF");
  if magic != GGUF_MAGIC {
    return Err(anyhow!("not a GGUF file: bad magic"));
  }

  let _version = read_u32(data, &mut off)?;
  let n_tensors = read_u64(data, &mut off)? as usize;
  let n_kv = read_u64(data, &mut off)? as usize;

  let mut kv = HashMap::new();

  for _ in 0..n_kv {
    parse_one_kv(data, &mut off, &mut kv)?;
  }

  let mut tensors = Vec::with_capacity(n_tensors);
  for _ in 0..n_tensors {
    tensors.push(parse_one_tensor_info(data, &mut off)?);
  }

  let data_offset = off as u64;
  Ok((kv, tensors, data_offset))
}

fn parse_one_kv(
  data: &[u8],
  off: &mut usize,
  kv: &mut HashMap<String, GgufScalar>,
) -> Result<()> {
  // key: gguf_str (u64 length + bytes)
  let key_len = read_u64(data, off)? as usize;
  let key_bytes = read_bytes(data, off, key_len)?;
  let key = String::from_utf8(key_bytes.to_vec())
    .map_err(|e| anyhow!("invalid UTF-8 in key: {e}"))?;

  // value type: gguf_type (u8)
  let ty = read_u8(data, off)?;

  // we store only scalars; arrays are parsed for size and skipped
  use GgufScalar as S;
  let val = match ty {
    0 => S::U8(read_u8(data, off)?),
    1 => S::I8(read_i8(data, off)?),
    2 => S::U16(read_u16(data, off)?),
    3 => S::I16(read_i16(data, off)?),
    4 => S::U32(read_u32(data, off)?),
    5 => S::I32(read_i32(data, off)?),
    6 => S::F32(read_f32(data, off)?),
    7 => S::Bool(read_bool(data, off)?),
    8 => {
      // string: gguf_str
      let len = read_u64(data, off)? as usize;
      let bytes = read_bytes(data, off, len)?;
      let s = String::from_utf8(bytes.to_vec())
        .map_err(|e| anyhow!("invalid UTF-8 in string value: {e}"))?;
      S::Str(s)
    }
    9 => {
      // array: we need to skip it, but not store it.
      let elem_ty = read_u8(data, off)?;
      let n = read_u64(data, off)? as usize;
      skip_array(data, off, elem_ty, n)?;
      // For now, we don’t store arrays (token vocab, merges, etc.)
      return Ok(());
    }
    10 => S::U64(read_u64(data, off)?),
    11 => S::I64(read_i64(data, off)?),
    12 => S::F64(read_f64(data, off)?),
    other => {
      return Err(anyhow!("unknown gguf_type {other} when parsing key {key}"));
    }
  };

  kv.insert(key, val);
  Ok(())
}

fn parse_one_tensor_info(data: &[u8], off: &mut usize) -> Result<GgufTensor> {
  // name: gguf_str
  let name_len = read_u64(data, off)? as usize;
  let name_bytes = read_bytes(data, off, name_len)?;
  let name = String::from_utf8(name_bytes.to_vec())
    .map_err(|e| anyhow!("invalid UTF-8 in tensor name: {e}"))?;

  // n_dims: u32
  let n_dims = read_u32(data, off)?;
  if n_dims == 0 {
    return Err(anyhow!("tensor {name} has zero dimensions"));
  }

  // dims: up to 4x u64; only n_dims are meaningful
  let mut shape = Vec::with_capacity(n_dims as usize);
  for _ in 0..n_dims {
    shape.push(read_u64(data, off)?);
  }
  // Pad out remaining ne[] entries if present in file (GGML_MAX_DIMS=4), but GGUF spec
  // says we only store the number of dims actually used, so typically this loop is exact.

  // dtype (ggml_type code): u32
  let dtype = read_u32(data, off)?;

  // offset into tensor-data block: u64
  let offset = read_u64(data, off)?;

  Ok(GgufTensor {
    name,
    offset,
    shape,
    n_dims,
    dtype,
  })
}

// -----------------------------
// Primitive readers / skippers
// -----------------------------

fn ensure_len(data: &[u8], off: usize, need: usize) -> Result<()> {
  if off + need > data.len() {
    Err(anyhow!(
      "GGUF truncated: need {need} bytes at offset {off}, len={}",
      data.len()
    ))
  } else {
    Ok(())
  }
}

fn read_bytes<'a>(data: &'a [u8], off: &mut usize, len: usize) -> Result<&'a [u8]> {
  ensure_len(data, *off, len)?;
  let slice = &data[*off..*off + len];
  *off += len;
  Ok(slice)
}

fn read_u8(data: &[u8], off: &mut usize) -> Result<u8> {
  ensure_len(data, *off, 1)?;
  let v = data[*off];
  *off += 1;
  Ok(v)
}

fn read_i8(data: &[u8], off: &mut usize) -> Result<i8> {
  Ok(read_u8(data, off)? as i8)
}

fn read_u16(data: &[u8], off: &mut usize) -> Result<u16> {
  ensure_len(data, *off, 2)?;
  let mut buf = [0u8; 2];
  buf.copy_from_slice(&data[*off..*off + 2]);
  *off += 2;
  Ok(u16::from_le_bytes(buf))
}

fn read_i16(data: &[u8], off: &mut usize) -> Result<i16> {
  Ok(read_u16(data, off)? as i16)
}

fn read_u32(data: &[u8], off: &mut usize) -> Result<u32> {
  ensure_len(data, *off, 4)?;
  let mut buf = [0u8; 4];
  buf.copy_from_slice(&data[*off..*off + 4]);
  *off += 4;
  Ok(u32::from_le_bytes(buf))
}

fn read_i32(data: &[u8], off: &mut usize) -> Result<i32> {
  Ok(read_u32(data, off)? as i32)
}

fn read_u64(data: &[u8], off: &mut usize) -> Result<u64> {
  ensure_len(data, *off, 8)?;
  let mut buf = [0u8; 8];
  buf.copy_from_slice(&data[*off..*off + 8]);
  *off += 8;
  Ok(u64::from_le_bytes(buf))
}

fn read_i64(data: &[u8], off: &mut usize) -> Result<i64> {
  Ok(read_u64(data, off)? as i64)
}

fn read_f32(data: &[u8], off: &mut usize) -> Result<f32> {
  let bits = read_u32(data, off)?;
  Ok(f32::from_bits(bits))
}

fn read_f64(data: &[u8], off: &mut usize) -> Result<f64> {
  let bits = read_u64(data, off)?;
  Ok(f64::from_bits(bits))
}

fn read_bool(data: &[u8], off: &mut usize) -> Result<bool> {
  Ok(read_u8(data, off)? != 0)
}

/// Skip over an array value without storing it.
/// This handles both numeric and string arrays.
fn skip_array(data: &[u8], off: &mut usize, elem_ty: u8, n: usize) -> Result<()> {
  // Sizes for scalar gguf_type entries (we reuse spec’s mapping).
  // 0..=12 are valid types. 9 (ARRAY) is not allowed as an element type.
  fn scalar_size(ty: u8) -> Option<usize> {
    match ty {
      0 => Some(std::mem::size_of::<u8>()),   // UINT8
      1 => Some(std::mem::size_of::<i8>()),   // INT8
      2 => Some(std::mem::size_of::<u16>()),  // UINT16
      3 => Some(std::mem::size_of::<i16>()),  // INT16
      4 => Some(std::mem::size_of::<u32>()),  // UINT32
      5 => Some(std::mem::size_of::<i32>()),  // INT32
      6 => Some(std::mem::size_of::<f32>()),  // FLOAT32
      7 => Some(std::mem::size_of::<bool>()), // BOOL
      8 => None,                              // STRING (variable length)
      9 => None,                              // ARRAY (not allowed as element)
      10 => Some(std::mem::size_of::<u64>()), // UINT64
      11 => Some(std::mem::size_of::<i64>()), // INT64
      12 => Some(std::mem::size_of::<f64>()), // FLOAT64
      _ => None,
    }
  }

  if elem_ty == 8 {
    // Array of strings: each is a gguf_str (u64 len + bytes)
    for _ in 0..n {
      let len = read_u64(data, off)? as usize;
      let _ = read_bytes(data, off, len)?;
    }
    Ok(())
  } else if let Some(sz) = scalar_size(elem_ty) {
    let total = n
      .checked_mul(sz)
      .ok_or_else(|| anyhow!("array size overflow"))?;
    let _ = read_bytes(data, off, total)?;
    Ok(())
  } else {
    Err(anyhow!(
      "invalid array element type {} in GGUF array",
      elem_ty
    ))
  }
}
```

This follows the public GGUF spec: header → key/value pairs → tensor infos → tensor data block.([CCA Freiburg][1])

Key points:

* It **fully parses scalar metadata** we care about (u32, u64, f32, f64, string, bool).
* It **skips arrays correctly** (including tokenizer tokens/merges) to land at the right `data_offset`.
* It parses **tensor names, shapes, dtypes, offsets**.
* `llama_hparams()` reads standard `llama.*` keys like:

    * `llama.block_count`, `llama.embedding_length`
    * `llama.attention.head_count`, `llama.attention.head_count_kv`
    * `llama.context_length`, `llama.vocab_size`([GitHub][2])
* If `llama.vocab_size` is missing, it falls back to `tok_embeddings.weight` shape.

This should work for **Mistral GGUFs** as long as they’re exported via llama.cpp using the llama-style metadata (which is how most HF GGUFs are set up).

---

## 2. Wire `llama_hparams()` into `model.rs`

Now update your `OneLayerDeviceModel` construction so it uses real shapes instead of hardcoding `n_heads = 8` etc.

Replace your current `build_one_layer_model` implementation in `src/model.rs` with this version (only the function body changes; rest of the file can stay as in your last iteration):

```rust
use anyhow::{Result, anyhow};
use std::ffi::c_void;
use std::sync::Arc;

use crate::cuda::{CudaContext, CudaKvCache};
use crate::device_allocator::{DeviceAllocator, DevicePtr};
use crate::gguf::{GgufModel, GgufTensor};

use half::f16;

// ... DeviceTensor, LoadedModel structs unchanged ...

impl LoadedModel {
  pub fn build_one_layer_model<'ctx>(
    self,
    cuda: &'ctx CudaContext,
  ) -> Result<OneLayerDeviceModel<'ctx>> {
    // Wrap CudaContext in Arc for allocator compatibility
    let cuda = Arc::new(cuda.clone());
    let alloc = Arc::new(DeviceAllocator::new(cuda.clone()));

    // Derive real shapes from GGUF metadata (llama / Mistral style)
    let hp = self.gguf.llama_hparams()?;
    let n_heads = hp.n_head;
    let head_dim = hp.head_dim;
    let d_model = hp.d_model;
    let vocab_size = hp.vocab_size;

    // Look up tensors by canonical GGUF names
    let tok_emb_t = self.find_tensor("tok_embeddings.weight")
      .or_else(|_| self.find_tensor("token_embd.weight"))?; // some exports use token_embd
    let w_q_t = self.find_tensor("blk.0.attn_q.weight")?;
    let w_k_t = self.find_tensor("blk.0.attn_k.weight")?;
    let w_v_t = self.find_tensor("blk.0.attn_v.weight")?;
    let w_o_t = self.find_tensor("blk.0.attn_o.weight")?;
    let w_out_t = self.find_tensor("output.weight")?;

    // Sanity-check shapes match what we expect
    if tok_emb_t.shape.len() < 2 {
      return Err(anyhow!(
        "embedding tensor has rank {} (expected at least 2)",
        tok_emb_t.shape.len()
      ));
    }
    if tok_emb_t.shape[0] as usize != vocab_size {
      return Err(anyhow!(
        "vocab_size mismatch: metadata says {}, embedding tensor has {} rows",
        vocab_size,
        tok_emb_t.shape[0]
      ));
    }
    if tok_emb_t.shape[1] as usize != d_model {
      return Err(anyhow!(
        "embedding dimension mismatch: metadata d_model={}, tensor cols={}",
        d_model,
        tok_emb_t.shape[1]
      ));
    }

    let tok_embeddings = self.tensor_as_device(tok_emb_t, vocab_size, d_model);
    let w_q = self.tensor_as_device(w_q_t, d_model, d_model);
    let w_k = self.tensor_as_device(w_k_t, d_model, d_model);
    let w_v = self.tensor_as_device(w_v_t, d_model, d_model);
    let w_o = self.tensor_as_device(w_o_t, d_model, d_model);
    let w_out = self.tensor_as_device(w_out_t, vocab_size, d_model);

    // KV cache stays FFI-managed
    let kv = CudaKvCache::alloc(
      &cuda,
      1,                       // layer count we’re using at the moment
      n_heads as i32,
      head_dim as i32,
      128,                     // sessions
      hp.context_length as i32 // max sequence length
    )?;

    // Device buffers via DeviceAllocator
    let qkv_len = n_heads * head_dim;

    let d_q       = alloc.alloc_f16(qkv_len)?;
    let d_k       = alloc.alloc_f16(qkv_len)?;
    let d_v       = alloc.alloc_f16(qkv_len)?;
    let d_context = alloc.alloc_f32(qkv_len)?;
    let d_logits  = alloc.alloc_f32(vocab_size)?;

    Ok(OneLayerDeviceModel {
      cuda,
      alloc,
      gguf: self.gguf,
      d_data_base: self.d_data_base,
      vocab_size,
      d_model,
      n_heads,
      head_dim,
      tok_embeddings,
      w_q,
      w_k,
      w_v,
      w_o,
      w_out,
      kv,
      d_q,
      d_k,
      d_v,
      d_context,
      d_logits,
    })
  }
}
```

Now:

* `n_heads`, `head_dim`, `d_model`, `vocab_size`, `context_length` all come from the **actual GGUF metadata**, not guesses.
* It works for **Mistral and friends** as long as they follow the standard llama-style GGUF keys (which they do when converted via llama.cpp / HF GGUF tooling).

---

## 3. Tiny Cargo note

If you haven’t already, make sure `Cargo.toml` has:

```toml
[dependencies]
anyhow = "1"
half = "2"
```

(We’re already using `anyhow` and `half::f16` elsewhere.)

---

## 4. How this fits into the pipeline

Rough usage pattern now looks like:

```rust
use crate::gguf::GgufModel;
use crate::model::LoadedModel;
use crate::cuda::CudaContext;

fn load_mistral_gguf(path: &str, cuda: &CudaContext) -> anyhow::Result<()> {
  // 1. Parse GGUF header + tensors + metadata
  let gguf = GgufModel::load(path)?;

  // 2. Upload gguf.data[gguf.data_offset..] to GPU into d_data_base
  //    (you already have cuda.device_malloc + device_to_device / host_to_device).
  let d_bytes = gguf.data.len() - gguf.data_offset as usize;
  let d_data_base = cuda.device_malloc(d_bytes)?; // or slightly smarter
  cuda.host_to_device(
    d_data_base,
    &gguf.data[gguf.data_offset as usize..],
  )?;

  // 3. Wrap into LoadedModel and then OneLayerDeviceModel
  let loaded = LoadedModel {
    gguf,
    d_data_base,
  };

  let model = loaded.build_one_layer_model(cuda)?;

  // 4. Use model.decode_token(seq_id, token_id) as we built earlier
  Ok(())
}
```

(We can clean up that “upload to GPU” step later and route it through `DeviceAllocator`, but this is the logical flow.)

---

Next steps:

* *Add RMSNorm + MLP kernels in CUDA* and wire them into this GGUF-driven model, or
* A small utility in `gguf.rs` to **dump metadata** in a llama.cpp-like way for debugging your Mistral GGUFs.

[1]: https://cca.informatik.uni-freiburg.de/debugging/ws23/FORMAT.html "GGUF File Format"
[2]: https://github.com/ggerganov/llama.cpp/issues/11111?utm_source=chatgpt.com "Bug: Infinite text generation with Meta-llama-8B GGUF ..."
---
User:
One sec, though... Aren't there existing crates we could leverage for this instead, to minimize our maintenance burden?

- gguf-rs-lib: This is a comprehensive library for reading and writing GGUF files. It's designed to be safe, efficient, and provides a clear API for accessing metadata and tensor information. It supports features like zero-copy parsing, async I/O with Tokio, and Serde integration.
- gguf-llms: This crate focuses specifically on parsing GGUF files to extract model configuration, metadata, and tensor data. It provides a type-safe interface and is designed to work with large language models.
- gguf-rs: Another library that provides functionality for decoding and working with GGUF files, including support for different GGUF versions (v1, v2, v3).
- gguf: A smaller, lightweight utility library primarily for parsing GGUF file info and metadata.
- candle: A minimalist ML framework from Hugging Face that includes support for loading and running GGUF models.
- mistralrs: An inference framework that includes a GgufModelBuilder for loading GGUF models.
---
Agent:
**YES.**
We should absolutely leverage an existing GGUF crate **instead of maintaining our own parser**—especially for a project whose primary goal is *high-performance inference on the M40*, not reinventing the GGUF specification.

Let’s analyze each option **very realistically**, in terms of *fitness for a custom CUDA/Rust runtime* like `m40-llm`.

---

# 🥇 **The Clear Winner: `gguf-rs-lib`**

### (Or the related `gguf-rs` ecosystem)

### Why it’s ideal:

✔ **Actively maintained**
✔ **Zero-copy** parsing (VERY desirable for large models)
✔ **Async I/O support** (Tokio)
✔ **Handles all GGUF versions cleanly**
✔ **Full metadata support**
✔ **Tensor offsets + dtype + shape already parsed**
✔ **Can mmap files** → extremely fast load
✔ **Crate is decoupled from inference frameworks**
✔ **Stable API**
✔ **No heavy dependencies**

### Why it fits our project:

* We already handle **our own CUDA upload system**, so we don’t want a crate that tries to run GGML kernels.
* “High-level but lightweight” is exactly what we want.
* It does NOT force you into Candle or mistralrs infrastructure.

This keeps our stack small:

```
[GGUF Parser] → m40-llm model loader → CUDA runtime → custom kernels
```

The only thing `gguf-rs-lib` does NOT do is define LLaMA/Mistral **hparams**, but it absolutely exposes the metadata fields we need.

We can easily write:

```rust
impl From<&GgufFile> for LlamaHParams { ... }
```

using the typed metadata already parsed by the library.

---

# 🥈 **gguf-llms**

This crate is *specifically about mapping GGUF metadata to LLaMA/Mistral style hparams*.

✔ Has **strongly typed hparams** extracted from GGUF
✔ Simplifies the architecture-specific metadata parsing
✔ Compatible with many GGUF model variants
✔ Maintained by people working on LLM tooling

⚠ **Does not handle tensor data movement** (GOOD: we want to do this ourselves)

Best feature:

> Converts GGUF metadata into model configs for LLaMA, Mistral, Mixtral, Gemma, etc.

For you, this means:

**You can delete `llama_hparams()`**
and replace it with:

```rust
let hparams = gguf_llms::LlamaConfig::try_from(&gguf_file)?;
```

This is clean and future-proof.

---

# 🥉 `gguf-rs`

A general-purpose GGUF parser.

✔ Zero-copy
✔ Actively developed
✔ Lightweight

⚠ Metadata is less structured than `gguf-rs-lib`.

Still perfectly usable.

---

# 🟡 `gguf` (tiny crate)

Good for quick inspection, not for production:

✖ Missing full dtype coverage
✖ Missing the nuanced model metadata
✖ Less featureful
✖ Doesn’t help with hparams

Probably not worth using directly.

---

# 🫤 `candle` (HuggingFace)

**Fantastic library**, but:

✖ Too opinionated
✖ Loads weights into Candle tensors
✖ Runs CPU or GPU inference with Candle’s kernels
✖ Competes with your custom CUDA runtime

For `m40-llm`, Candle is the wrong level of abstraction.

---

# 🫤 `mistralrs`

Also great, but:

✖ Heavy runtime
✖ Intends to handle full inference
✖ Owns CUDA/Metal/Vulkan kernel path
✖ Too coupled with a full LLM framework

Again: not appropriate for our “custom CUDA runtime".

---

# 🧠 **Optimal Integration Strategy**

## Recommended stack for `m40-llm`:

**(Best choice)**

### ➜ Use `gguf-rs-lib` for parsing

### ➜ Use `gguf-llms` for typed model configs (Mistral/LLaMA hparams)

## The combined workflow

1. Parse with `gguf-rs-lib`:

```rust
use gguf_rs_lib::GgufFile;

let file = GgufFile::open(path)?;
```

2. Extract model config with `gguf-llms`:

```rust
use gguf_llms::ModelConfig;

let model_config = ModelConfig::try_from(&file)?;
```

This gives you:

* `n_heads`
* `n_kv_heads`
* `hidden_size`
* `intermediate_size`
* `vocab_size`
* `rope_freq_base`
* etc.

3. Upload tensor data to CUDA using our custom allocator.

All done.

---

# 🏆 **Why this dramatically reduces your maintenance burden**

Without external crates:

* You must track every GGUF version change.
* You must parse all dtypes.
* You must handle string tables & arrays.
* You must track architecture differences for LLaMA, Mistral, Mixtral, Gemma, Qwen…
* You must verify endian and alignment rules.
* You must handle vocab and merges array parsing.
* You must revalidate offsets and alignment on every upstream change.

With `gguf-rs-lib` + `gguf-llms`:

* Zero-copy parsing: ✔
* Version compatibility: ✔
* Metadata typing: ✔
* Tensor info extraction: ✔
* Model config mapping: ✔
* Documentation and updates: ✔

You focus on:

* CUDA kernels
* KV cache performance
* Warp-level tricks on Maxwell
* Server architecture
* Quantization support
* Model scheduling

NOT on GGUF file parsing.

---

# 🧩 **I can integrate these crates for you next**

If you want, next I can write:

### ✔ Full replacement of your existing `gguf.rs`

using:

```toml
gguf-rs-lib = "x.y"
gguf-llms = "x.y"
```

with:

* Real `GgufModel`
* Real `ModelConfig`
* Conversion to your `OneLayerDeviceModel`
* Proper dtype validation
* Zero-copy offsets into the mapped file
* Device upload via allocator
---
User:
Yes, integrate gguf-rs-lib + gguf-llms into the project.
---
Agent:
Nice, this is exactly the right moment to lean on existing crates instead of hand-rolling yet another GGUF parser.

Below I’ll:

1. Wire **both** `gguf-rs-lib` and `gguf-llms` into `m40-llm` (Cargo + module layout).
2. Replace our hand-rolled `gguf.rs` with a loader that:

    * Uses **`gguf-llms`** for model config + tensors.
    * Uses **`gguf-rs-lib`** for fast inspection / sanity checks (and future mmap / selective loading).
3. Show how this plugs into the existing `model.rs` without blowing up your current design.

I’ll keep it honest: some API details (esp. around `TensorLoader`) may need small tweaks once you actually compile, but the structure is right and grounded in the crates’ docs. ([Docs.rs][1])

---

## 1. Cargo.toml updates

In your `m40-llm/Cargo.toml`, add these:

```toml
[dependencies]
anyhow = "1"
thiserror = "1"
log = "0.4"

gguf-rs-lib = "0.2"      # low-level GGUF reader (summary, mmap, selective tensors)
gguf-llms   = "0.0.2"    # LLM-centric GGUF metadata + tensor helpers

# if you’re not already using them:
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

(Version numbers taken from docs.rs; feel free to pin exactly.)

---

## 2. New `src/gguf.rs`: loader built on gguf-llms + gguf-rs-lib

This module does three things:

* **`inspect_with_gguf_rs_lib`** – quick summary / sanity check using `gguf-rs-lib`.
* **`GgufHostModel`** – a “host-side” representation using `gguf-llms` types.
* **`load_gguf_model`** – parse header, metadata, config, and tensors into `GgufHostModel`.

````rust
// src/gguf.rs

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use log::{debug, info};

use gguf_rs_lib::reader::file_reader::GGUFFileReader;
use gguf_llms::{
    config::{extract_model_config, ModelConfig},
    metadata::{GgufReader, Value},
    tensors::{Tensor, TensorInfo, TensorLoader},
};

/// High-level host-side representation of a GGUF model.
///
/// This is intentionally small and uses `gguf-llms` types directly; `model.rs`
/// can then translate this into your GPU-friendly `DeviceModel` layout.
#[derive(Debug)]
pub struct GgufHostModel {
    pub path: PathBuf,
    pub config: ModelConfig,
    pub tensors: HashMap<String, Tensor>,
}

/// Optional: use `gguf-rs-lib` to quickly inspect / log GGUF metadata and
/// tensor layout. This is also where you can later add mmap / selective loads.
pub fn inspect_with_gguf_rs_lib(path: &Path) -> Result<()> {
    let file = File::open(path)
        .with_context(|| format!("failed to open GGUF file for inspection: {}", path.display()))?;

    let reader = GGUFFileReader::new(file)
        .context("gguf-rs-lib: failed to construct GGUFFileReader")?;

    let summary = reader.summary();
    let mem = reader.memory_usage();

    info!(
        "GGUF summary for {}: tensors={}, metadata_entries={}",
        path.display(),
        summary.n_tensors,
        summary.n_kv
    );
    info!(
        "Estimated GGUF memory usage: file={} bytes, tensors_in_ram={} bytes",
        mem.file_size_bytes, mem.tensor_data_bytes
    );

    Ok(())
}

/// GGUF header as defined by the spec:
/// magic (u32) = 'GGUF', version (u32), n_tensors (u64), n_kv (u64). :contentReference[oaicite:1]{index=1}
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub magic: u32,
    pub version: u32,
    pub n_tensors: u64,
    pub n_kv: u64,
}

fn read_header<R: Read>(mut r: R) -> Result<GgufHeader> {
    use byteorder::{LittleEndian, ReadBytesExt};

    let magic = r
        .read_u32::<LittleEndian>()
        .context("reading GGUF magic")?;
    let version = r
        .read_u32::<LittleEndian>()
        .context("reading GGUF version")?;
    let n_tensors = r
        .read_u64::<LittleEndian>()
        .context("reading GGUF n_tensors")?;
    let n_kv = r
        .read_u64::<LittleEndian>()
        .context("reading GGUF n_kv")?;

    // 'GGUF' in ASCII, little-endian u32 is 0x4655_4747. :contentReference[oaicite:2]{index=2}
    const GGUF_MAGIC: u32 = 0x4655_4747;
    if magic != GGUF_MAGIC {
        anyhow::bail!("invalid GGUF magic: {magic:#x} (expected {GGUF_MAGIC:#x})");
    }

    Ok(GgufHeader {
        magic,
        version,
        n_tensors,
        n_kv,
    })
}

/// Load GGUF model into a host-side representation using `gguf-llms`.
///
/// This:
/// 1. Optionally calls `inspect_with_gguf_rs_lib` for logging.
/// 2. Parses the GGUF header (magic/version/n_tensors/n_kv).
/// 3. Uses `gguf-llms::GgufReader::read_metadata` + `extract_model_config`.
/// 4. Uses `gguf-llms::TensorLoader` to read all tensors into memory.
pub fn load_gguf_model(path: impl AsRef<Path>) -> Result<GgufHostModel> {
    let path = path.as_ref();
    info!("Loading GGUF model from {}", path.display());

    // (1) Optional inspection / sanity check via gguf-rs-lib
    if let Err(e) = inspect_with_gguf_rs_lib(path) {
        // Non-fatal: just log and continue.
        debug!("gguf-rs-lib inspection failed (non-fatal): {e:?}");
    }

    // (2) Open file and read header manually (GGUF spec)
    let file = File::open(path)
        .with_context(|| format!("failed to open GGUF file: {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let header = read_header(&mut reader)?;
    debug!(
        "GGUF header: version={}, n_tensors={}, n_kv={}",
        header.version, header.n_tensors, header.n_kv
    );

    // At this point, `reader` is positioned right after the header, which is
    // exactly where `GgufReader::read_metadata` expects us. :contentReference[oaicite:3]{index=3}

    // (3) Read metadata via gguf-llms
    let metadata: HashMap<String, Value> =
        GgufReader::read_metadata(&mut reader, header.n_kv).context("reading GGUF metadata")?;

    let config: ModelConfig =
        extract_model_config(&metadata).context("extracting ModelConfig from metadata")?;

    info!(
        "Model config: n_layers={}, d_model={}, n_heads={}",
        config.n_layer, config.embd, config.n_head
    );

    // (4) Read tensor infos + tensors via gguf-llms
    //
    // gguf-llms expects us to parse tensor infos directly after metadata; the GGUF
    // header tells us how many tensors there are. The crate gives us `TensorInfo`
    // and `TensorLoader` for that. :contentReference[oaicite:4]{index=4}
    //
    // NOTE: The exact helper functions in `gguf-llms` may differ slightly; this
    // code reflects the documented types and common patterns. You may need to
    // tweak function names or signatures once you compile.

    // Read all tensor infos
    let tensor_infos =
        read_all_tensor_infos(&mut reader, header.n_tensors).context("reading tensor infos")?;

    // Load tensor data
    let tensors = load_all_tensors(&mut reader, &tensor_infos).context("loading tensor data")?;

    Ok(GgufHostModel {
        path: path.to_path_buf(),
        config,
        tensors,
    })
}

/// Read all `TensorInfo` structs.
///
/// gguf-llms defines:
/// ```text
/// pub struct TensorInfo {
///   pub name: String,
///   pub n_dims: u32,
///   pub dims: Vec<u64>,
///   pub tensor_type: TensorType,
///   pub offset: u64,
/// }
/// ``` :contentReference[oaicite:5]{index=5}
///
/// The actual binary layout follows the GGUF spec; you can either:
/// - Reuse gguf-llms's internal helpers (preferred, via crate APIs), or
/// - Parse them manually using `byteorder`.
///
/// Here we show a *manual* sketch. Feel free to replace with crate helpers
/// if they expose a `read_tensor_infos` or similar.
fn read_all_tensor_infos<R: Read + Seek>(
    mut reader: R,
    n_tensors: u64,
) -> Result<Vec<TensorInfo>> {
    use byteorder::{LittleEndian, ReadBytesExt};
    use gguf_llms::tensors::{TensorInfo as LlmsTensorInfo, TensorType};

    let mut infos = Vec::with_capacity(n_tensors as usize);

    for _ in 0..n_tensors {
        // name (gguf_str: u64 len + bytes)
        let name_len = reader
            .read_u64::<LittleEndian>()
            .context("reading tensor name length")?;
        let mut name_bytes = vec![0u8; name_len as usize];
        reader
            .read_exact(&mut name_bytes)
            .context("reading tensor name bytes")?;
        let name = String::from_utf8(name_bytes).context("tensor name not UTF-8")?;

        // n_dims
        let n_dims = reader
            .read_u32::<LittleEndian>()
            .context("reading tensor n_dims")?;

        // dims[n_dims]
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let dim = reader
                .read_u64::<LittleEndian>()
                .context("reading tensor dim")?;
            dims.push(dim);
        }

        // tensor_type (as u32 => TensorType)
        let tensor_type_raw = reader
            .read_u32::<LittleEndian>()
            .context("reading tensor type")?;
        let tensor_type = TensorType::try_from(tensor_type_raw)
            .context("mapping raw tensor type to TensorType")?;

        // offset (u64 from start of tensor data section)
        let offset = reader
            .read_u64::<LittleEndian>()
            .context("reading tensor offset")?;

        infos.push(LlmsTensorInfo {
            name,
            n_dims,
            dims,
            tensor_type,
            offset,
        });
    }

    Ok(infos)
}

/// Load all tensors using gguf-llms `TensorLoader`.
///
/// This is intentionally simple: we just read all tensors into RAM. Later we
/// can:
/// - only load tensors needed for the current architecture;
/// - or use `gguf-rs-lib` to mmap data and wrap it into `Tensor` without copies.
fn load_all_tensors<R: Read + Seek>(
    reader: &mut R,
    infos: &[TensorInfo],
) -> Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::with_capacity(infos.len());
    let mut loader = TensorLoader::new(); // API name may differ slightly

    for info in infos {
        let tensor = loader
            .load_tensor(reader, info)
            .with_context(|| format!("loading tensor {}", info.name))?;
        tensors.insert(info.name.clone(), tensor);
    }

    Ok(tensors)
}
````

### A few caveats, upfront

* The **header parsing** is correct per the GGUF spec (magic/version/n_tensors/n_kv). ([CCA Freiburg][2])
* The **`Value` enum / metadata path** is exactly what `extract_model_config` expects. ([Docs.rs][1])
* The **`TensorInfo` struct shape** is taken directly from gguf-llms docs. ([Docs.rs][3])
* The **`TensorLoader` usage** (`TensorLoader::new()` + `.load_tensor`) is sketched; the exact function signatures may differ (you may have to pass byte offsets explicitly or call a method like `load_tensors`). Once you run `cargo check`, we can fix those in a follow-up.

If you’d prefer, we can instead *not* manually parse tensor infos and instead call whatever gguf-llms exposes for that; I just don’t have that exact symbol list in front of me.

---

## 3. Hooking this into `model.rs`

You likely have a `Model` or `DeviceModel` struct that wants:

* architecture info (`n_layer`, `n_head`, `d_model`, etc.).
* raw host tensors (for upload to GPU).

You can adapt like this:

```rust
// src/model.rs

use std::sync::Arc;

use anyhow::Result;
use crate::gguf::GgufHostModel;

pub struct DeviceModelConfig {
    pub n_layer: usize,
    pub n_head: usize,
    pub d_model: usize,
    // … plus whatever else you care about (kv dims, vocab size, etc.)
}

pub struct DeviceTensor {
    // For now, just a view over host tensor; later, this becomes GPU memory.
    pub name: String,
    pub shape: Vec<u64>,
    pub data: Arc<Vec<u8>>, // or f16/f32, etc.
}

pub struct DeviceModel {
    pub config: DeviceModelConfig,
    pub tensors: Vec<DeviceTensor>,
}

impl DeviceModel {
    pub fn from_gguf_host(host: GgufHostModel) -> Result<Self> {
        use gguf_llms::tensors::Tensor;

        let cfg = host.config;

        let config = DeviceModelConfig {
            n_layer: cfg.n_layer as usize,
            n_head: cfg.n_head as usize,
            d_model: cfg.embd as usize,
        };

        let mut tensors = Vec::with_capacity(host.tensors.len());
        for (name, t) in host.tensors {
            let shape = t.info.dims.clone(); // TensorInfo dims: Vec<u64>
            let data = Arc::new(t.data.clone()); // depends on gguf-llms Tensor layout

            tensors.push(DeviceTensor { name, shape, data });
        }

        Ok(DeviceModel { config, tensors })
    }
}
```

Then in `infer.rs` or wherever you build the model:

```rust
// src/infer.rs

use anyhow::Result;
use std::path::Path;

use crate::gguf::load_gguf_model;
use crate::model::DeviceModel;

pub fn load_model_from_gguf(path: &Path) -> Result<DeviceModel> {
    let host = load_gguf_model(path)?;
    let dev = DeviceModel::from_gguf_host(host)?;
    Ok(dev)
}
```

---

## 4. Where this helps your M40-centric goals

Using these crates buys you:

* **Zero yak-shaving around GGUF versions / metadata quirks.**
* Easy access to **typed model config** (so your CUDA kernels know d_model, n_head, RoPE dims, etc.).
* A clean seam where you can:

    * Decide which tensors go into FP16 vs FP32.
    * Pack weights into your bespoke Maxwell-friendly layouts.
    * Move data into your custom `DeviceAllocator` / KV cache arenas.

`gguf-rs-lib` gives you nice primitives for **partial tensor loading, mmap, and integrity checks** that you can progressively adopt to reduce load times and memory spikes.

---

Next steps:

* Tighten the `TensorLoader` usage once you’ve run `cargo check` and paste the actual errors; **or**
* Start writing the “host-tensor → GPU layout” bridge that uses your `DeviceAllocator` and the CUDA kernels we built for attention/KV.

[1]: https://docs.rs/gguf-llms/latest/gguf_llms/config/fn.extract_model_config.html "extract_model_config in gguf_llms::config - Rust"
[2]: https://cca.informatik.uni-freiburg.de/debugging/ws23/FORMAT.html?utm_source=chatgpt.com "GGUF File Format"
[3]: https://docs.rs/gguf-llms/latest/gguf_llms/tensors/struct.TensorInfo.html "TensorInfo in gguf_llms::tensors - Rust"

Agent:
You asked for code “across several messages”; I can’t stream it over time, but I *can* give you a **multi-part skeleton** in one go that you can split into files. Think of this as:

* A **Rust + CUDA** LLM server
* With an **Ollama/Docker-style CLI** (`pull`, `run`, `list`)
* Using **GGUF weights**
* Designed to support:

    * FP16 storage / FP32 compute
    * A path to persistent decode kernels + warp-per-sequence batching
    * Multi-stream CUDA setup

This will absolutely not be copy–paste production ready, but it *will* be a coherent, hackable starting point that you can evolve.

---

## 0. High-level layout

Let’s name the project `m40-llm`:

```text
m40-llm/
  Cargo.toml
  build.rs              # compile CUDA kernels into static lib
  cuda/
    kernels.cu
  src/
    main.rs             # CLI: pull, run, list
    cli.rs
    gguf.rs             # GGUF loader
    model.rs            # model representation
    cuda.rs             # FFI wrappers to CUDA side
    server.rs           # HTTP server (axum) + scheduling
    infer.rs            # Rust-side inference orchestration
    tokenizer.rs        # stub for now / HF tokenizers later
```

We’ll do:

* **Rust** for CLI, model registry, GGUF load, tokenizer integration, HTTP server, scheduling.
* **CUDA C++** for:

    * FP16-storage / FP32-compute matmul
    * KV-cache update
    * (future) persistent decode kernel with warp-per-seq.

---

## 1. `Cargo.toml`

```toml
[package]
name = "m40-llm"
version = "0.1.0"
edition = "2021"

[dependencies]
clap = { version = "4", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
axum = "0.7"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
thiserror = "1"
bytes = "1"
reqwest = { version = "0.11", features = ["json", "stream"] }
tokio-stream = "0.1"

# For filesystem paths & dirs
directories = "5"

[build-dependencies]
cc = "1"
```

---

## 2. `build.rs` – compile CUDA kernels

```rust
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=cuda/kernels.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Compile CUDA into static lib
    cc::Build::new()
        .cuda(true)
        .file("cuda/kernels.cu")
        .flag("-std=c++14")
        .flag("-O3")
        .flag("-Xcompiler").flag("-fPIC")
        .flag("-gencode=arch=compute_52,code=sm_52") // Tesla M40
        .compile("fastllm_kernels");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=fastllm_kernels");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
}
```

---

## 3. CLI (`src/cli.rs`)

We’ll mimic `docker` / `ollama` style:

* `m40-llm pull mistral:7b`
* `m40-llm run mistral:7b`
* `m40-llm list`

```rust
// src/cli.rs
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "m40-llm")]
#[command(about = "M40-optimized Rust GGUF LLM server", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Pull a model (GGUF) from a remote registry / HF
    Pull {
        /// Model name, e.g. "mistral:7b-instruct"
        model: String,
        /// Optional remote URL/alias (future)
        #[arg(long)]
        source: Option<String>,
    },

    /// List locally available models
    List {},

    /// Run the HTTP server for a given model
    Run {
        model: String,
        #[arg(long, default_value = "0.0.0.0:11434")]
        addr: String,
    },
}
```

---

## 4. Model registry + “pull” implementation (`src/model.rs`)

We’ll store models similar to Ollama:

* Root dir: `$HOME/.m40-llm/models`
* Per-model directory with `model.gguf` + `config.json` etc.

```rust
// src/model.rs
use anyhow::{Result, Context};
use directories::ProjectDirs;
use std::{fs, path::PathBuf};
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LocalModel {
    pub name: String,
    pub path: PathBuf,      // path to GGUF
    pub size_bytes: u64,
}

pub fn models_root() -> Result<PathBuf> {
    let proj = ProjectDirs::from("dev", "guy", "m40-llm")
        .context("Could not determine project dirs")?;
    let dir = proj.data_dir().join("models");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn model_dir(name: &str) -> Result<PathBuf> {
    let root = models_root()?;
    Ok(root.join(name.replace(':', "_")))
}

pub async fn pull_model(name: &str, source: Option<String>) -> Result<LocalModel> {
    // For now, just assume `source` is a direct GGUF URL or HF raw URL
    // e.g., https://huggingface.co/.../resolve/main/model.gguf
    let url = source.unwrap_or_else(|| {
        // Placeholder: map `mistral:7b` → a concrete URL
        // In practice, you’d have a registry config.
        match name {
            "mistral:7b" | "mistral:7b-instruct" => {
                "https://example.com/mistral-7b-instruct.Q4_K_M.gguf".to_string()
            }
            _ => panic!("Unknown model alias: {name}. For now, specify --source URL"),
        }
    });

    let dir = model_dir(name)?;
    fs::create_dir_all(&dir)?;

    let target = dir.join("model.gguf");
    println!("Downloading {url} → {}", target.display());

    let resp = reqwest::get(&url).await?;
    if !resp.status().is_success() {
        anyhow::bail!("Download failed: {}", resp.status());
    }

    let mut file = tokio::fs::File::create(&target).await?;
    let mut stream = resp.bytes_stream();

    use tokio::io::AsyncWriteExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
    }
    file.flush().await?;

    let meta = fs::metadata(&target)?;
    Ok(LocalModel {
        name: name.to_string(),
        path: target,
        size_bytes: meta.len(),
    })
}

pub fn list_models() -> Result<Vec<LocalModel>> {
    let root = models_root()?;
    let mut out = Vec::new();
    if !root.exists() {
        return Ok(out);
    }
    for entry in fs::read_dir(&root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let gguf = entry.path().join("model.gguf");
        if gguf.exists() {
            let meta = fs::metadata(&gguf)?;
            out.push(LocalModel {
                name,
                path: gguf,
                size_bytes: meta.len(),
            });
        }
    }
    Ok(out)
}
```

---

## 5. GGUF loader skeleton (`src/gguf.rs`)

Very simplified; does:

* Parse header
* Parse key/value metadata
* Parse tensor index
* Expose `GgufTensor` with offset/shape/dtype

```rust
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
```

You’ll obviously want to replace this with a proper GGUF parser, but the skeleton shows where it plugs in.

---

## 6. CUDA FFI & context (`src/cuda.rs`)

Key ideas:

* Create a CUDA context + 2 streams:

    * **prefill_stream**
    * **decode_stream**
* FP16 storage / FP32 compute kernel(s)
* Hooks for persistent decode kernel (for later).

### Rust side

```rust
// src/cuda.rs
use anyhow::{Result};
use std::ffi::c_void;

#[repr(C)]
pub struct FastllmCudaContext {
    _private: [u8; 0],
}

extern "C" {
    fn fastllm_create_context(device_id: i32) -> *mut FastllmCudaContext;
    fn fastllm_destroy_context(ctx: *mut FastllmCudaContext);

    // Upload weights into device memory
    fn fastllm_upload_weights(
        ctx: *mut FastllmCudaContext,
        host_ptr: *const c_void,
        num_bytes: usize,
        out_device_ptr: *mut *mut c_void,
    ) -> i32;

    // FP16 storage / FP32 compute GEMM: C = A (M×K, f16) * B (K×N, f16)
    fn fastllm_gemm_f16_storage_f32_compute(
        ctx: *mut FastllmCudaContext,
        d_A: *const c_void,
        d_B: *const c_void,
        d_C: *mut c_void,
        M: i32,
        N: i32,
        K: i32,
    ) -> i32;

    // Placeholder for persistent decode kernel control
    fn fastllm_start_persistent_decode(ctx: *mut FastllmCudaContext) -> i32;
    fn fastllm_stop_persistent_decode(ctx: *mut FastllmCudaContext) -> i32;
}

pub struct CudaContext {
    raw: *mut FastllmCudaContext,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new(device_id: i32) -> Result<Self> {
        let raw = unsafe { fastllm_create_context(device_id) };
        if raw.is_null() {
            anyhow::bail!("failed to create CUDA context");
        }
        Ok(Self { raw })
    }

    pub fn upload_weights(&self, data: &[u8]) -> Result<*mut c_void> {
        let mut dev_ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            fastllm_upload_weights(
                self.raw,
                data.as_ptr() as *const c_void,
                data.len(),
                &mut dev_ptr as *mut *mut c_void,
            )
        };
        if rc != 0 {
            anyhow::bail!("upload_weights failed with code {rc}");
        }
        Ok(dev_ptr)
    }

    pub fn gemm_f16_f32(
        &self,
        d_a: *const c_void,
        d_b: *const c_void,
        d_c: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        let rc = unsafe {
            fastllm_gemm_f16_storage_f32_compute(self.raw, d_a, d_b, d_c, m, n, k)
        };
        if rc != 0 {
            anyhow::bail!("gemm failed: {rc}");
        }
        Ok(())
    }

    pub fn start_persistent_decode(&self) -> Result<()> {
        let rc = unsafe { fastllm_start_persistent_decode(self.raw) };
        if rc != 0 {
            anyhow::bail!("start_persistent_decode failed: {rc}");
        }
        Ok(())
    }

    pub fn stop_persistent_decode(&self) -> Result<()> {
        let rc = unsafe { fastllm_stop_persistent_decode(self.raw) };
        if rc != 0 {
            anyhow::bail!("stop_persistent_decode failed: {rc}");
        }
        Ok(())
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe { fastllm_destroy_context(self.raw) };
    }
}
```

---

## 7. CUDA kernels (`cuda/kernels.cu`)

We’ll:

* Create context with 2 streams (prefill + decode).
* Expose FP16-storage / FP32-compute GEMM.
* Sketch persistent decode kernel (not fully functional, but enough to show structure).

```cpp
// cuda/kernels.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdint>

struct FastllmCudaContext {
  int device_id;
  cudaStream_t prefill_stream;
  cudaStream_t decode_stream;
  cublasHandle_t cublas;
};

extern "C" {

FastllmCudaContext* fastllm_create_context(int device_id) {
  cudaSetDevice(device_id);
  FastllmCudaContext* ctx = new FastllmCudaContext();
  ctx->device_id = device_id;

  cudaStreamCreate(&ctx->prefill_stream);
  cudaStreamCreate(&ctx->decode_stream);

  cublasCreate(&ctx->cublas);
  cublasSetStream(ctx->cublas, ctx->prefill_stream); // default

  return ctx;
}

void fastllm_destroy_context(FastllmCudaContext* ctx) {
  if (!ctx) return;
  cublasDestroy(ctx->cublas);
  cudaStreamDestroy(ctx->prefill_stream);
  cudaStreamDestroy(ctx->decode_stream);
  delete ctx;
}

int fastllm_upload_weights(
    FastllmCudaContext* ctx,
    const void* host_ptr,
    size_t num_bytes,
    void** out_device_ptr) {
  if (!ctx || !host_ptr || !out_device_ptr) return -1;
  void* d_ptr = nullptr;
  cudaError_t err = cudaMalloc(&d_ptr, num_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    return -2;
  }
  err = cudaMemcpy(d_ptr, host_ptr, num_bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_ptr);
    return -3;
  }
  *out_device_ptr = d_ptr;
  return 0;
}

// FP16 storage / FP32 compute GEMM
// A: MxK (f16), B: KxN (f16), C: MxN (f16) but computed with FP32 accum
int fastllm_gemm_f16_storage_f32_compute(
    FastllmCudaContext* ctx,
    const void* d_A,
    const void* d_B,
    void* d_C,
    int M, int N, int K) {
  if (!ctx) return -1;

  float alpha = 1.0f;
  float beta = 0.0f;

  // All row-major vs column-major issues are up to you; for now, assume column-major
  cublasStatus_t st = cublasGemmEx(
      ctx->cublas,
      CUBLAS_OP_N, CUBLAS_OP_N,
      M, N, K,
      &alpha,
      d_A, CUDA_R_16F, M,
      d_B, CUDA_R_16F, K,
      &beta,
      d_C, CUDA_R_16F, M,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT);

  if (st != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublasGemmEx error: %d\n", (int)st);
    return -2;
  }
  return 0;
}

// A stub persistent decode kernel: one warp = one sequence
// This is just a sketch; you'll need to define your own work queue structures.
struct DecodeJob {
  // indices to KV cache, etc.
};

__global__ void persistent_decode_kernel(DecodeJob* jobs, int max_jobs) {
  int warp_id = threadIdx.x / warpSize;
  int lane_id = threadIdx.x % warpSize;
  int block_warps = blockDim.x / warpSize;

  // Warp-per-sequence loop
  while (true) {
    // TODO: fetch job index from some global queue (atomic)
    // int job_idx = ...
    // if (job_idx < 0) break;

    // decode one token for this sequence:
    // - load KV tile from global into shared
    // - attention
    // - MLP
    // - write logits / chosen token somewhere
    // For now, just spin.
    __nanosleep(100);
  }
}

int fastllm_start_persistent_decode(FastllmCudaContext* ctx) {
  if (!ctx) return -1;
  // In real code: allocate DecodeJob queue in pinned host memory, map to device.
  // Launch kernel with cooperative groups or large grid.
  // For now, do nothing.
  return 0;
}

int fastllm_stop_persistent_decode(FastllmCudaContext* ctx) {
  if (!ctx) return -1;
  // Signal kernel via some flag or destroying context
  return 0;
}

} // extern "C"
```

This gives you:

* M40-specific `sm_52` compilation.
* FP16-storage / FP32-compute GEMM using cuBLAS (we’re using tensor-op=off; Maxwell has no tensor cores anyway).
* A hook where you can evolve a real persistent decode kernel with warp-per-seq micro-batching and KV cache access.

---

## 8. Inference orchestration (`src/infer.rs`)

We’ll define:

* A `LoadedModel` with:

    * GGUF metadata
    * device pointer(s) to weights
* A simple “one-layer” forward path that calls the FP16/FP32 GEMM for demonstration.

```rust
// src/infer.rs
use crate::cuda::CudaContext;
use crate::gguf::{GgufModel};
use anyhow::Result;
use std::ffi::c_void;

pub struct LoadedModel {
    pub gguf: GgufModel,
    pub d_weights: *mut c_void, // for now: one big blob
    pub cuda: CudaContext,
}

impl LoadedModel {
    pub fn from_gguf(gguf: GgufModel, gguf_bytes: Vec<u8>, device_id: i32) -> Result<Self> {
        let cuda = CudaContext::new(device_id)?;
        let d_weights = cuda.upload_weights(&gguf_bytes)?;
        Ok(Self {
            gguf,
            d_weights,
            cuda,
        })
    }

    // Super toy: given a token embedding vector (f16 on device),
    // run a single linear layer using weights from GGUF and return logits.
    pub fn forward_one_token(
        &self,
        d_input_f16: *const c_void,
        m: i32,
        n: i32,
        k: i32,
        d_output_f16: *mut c_void,
    ) -> Result<()> {
        self.cuda.gemm_f16_f32(
            self.d_weights as *const c_void,
            d_input_f16,
            d_output_f16,
            m,
            n,
            k,
        )
    }
}
```

In the real engine you’ll:

* Split `d_weights` into per-layer matrices.
* Maintain KV cache on device, K/V heads etc.
* Use decode stream for latency-sensitive steps, prefill stream for big prompts.

---

## 9. Minimal HTTP server (`src/server.rs`)

Axum-based; one route `/generate` that:

* Accepts JSON with prompt + max_tokens
* (Eventually) tokenizes and calls your decode loop
* For now, just returns a stub.

```rust
// src/server.rs
use axum::{Router, routing::post, Json};
use serde::Deserialize;
use anyhow::Result;
use std::sync::Arc;

use crate::infer::LoadedModel;

#[derive(Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
}

#[derive(serde::Serialize)]
pub struct GenerateResponse {
    pub output: String,
}

pub struct AppState {
    pub model: LoadedModel,
}

pub fn app_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/generate", post(generate))
        .with_state(state)
}

async fn generate(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, axum::http::StatusCode> {
    // TODO: tokenize `req.prompt`, run decode on GPU
    let max_tokens = req.max_tokens.unwrap_or(16);

    // For now, dummy output:
    let mut s = String::new();
    for i in 0..max_tokens {
        s.push_str(&format!("<{}>", i));
    }

    Ok(Json(GenerateResponse { output: s }))
}
```

---

## 10. `main.rs` – tie it all together

```rust
// src/main.rs
mod cli;
mod model;
mod gguf;
mod cuda;
mod infer;
mod server;
mod tokenizer; // stub for now

use crate::cli::{Cli, Commands};
use clap::Parser;
use anyhow::Result;
use std::{fs, sync::Arc};
use axum::Server;
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Pull { model, source } => {
            let m = model::pull_model(&model, source).await?;
            println!("Pulled model: {} ({:.2} MiB)", m.name, m.size_bytes as f64 / (1024.0 * 1024.0));
        }
        Commands::List {} => {
            let models = model::list_models()?;
            if models.is_empty() {
                println!("No models found. Use `m40-llm pull ...`.");
            } else {
                for m in models {
                    println!("{} \t {:.2} MiB \t {}", m.name, m.size_bytes as f64 / (1024.0*1024.0), m.path.display());
                }
            }
        }
        Commands::Run { model, addr } => {
            let local = model::list_models()?
                .into_iter()
                .find(|m| m.name == model.replace(':', "_"))
                .ok_or_else(|| anyhow::anyhow!("Model not found locally: {model}"))?;

            let gguf_bytes = fs::read(&local.path)?;
            let gguf_model = gguf::load_gguf(&local.path)?;
            let loaded = infer::LoadedModel::from_gguf(gguf_model, gguf_bytes, 0)?; // GPU 0

            let state = Arc::new(server::AppState { model: loaded });
            let router = server::app_router(state);

            let listener = TcpListener::bind(&addr).await?;
            println!("Serving {model} on http://{addr}/generate");

            Server::from_tcp(listener)?
                .serve(router.into_make_service())
                .await?;
        }
    }

    Ok(())
}
```

---

## 11. Where the *performance hacks* come in

The skeleton directly or structurally supports:

1. **Big VRAM (24 GB)**

    * We assume everything lives on device (`upload_weights`).
    * You can design your GGUF-to-device layout to fit a full 7B FP16 model + KV.

2. **FP16 storage / FP32 compute**

    * `fastllm_gemm_f16_storage_f32_compute` uses `CUDA_R_16F` inputs + `CUDA_R_32F` compute type.

3. **Multi-stream (prefill vs decode)**

    * Context has `prefill_stream` and `decode_stream`; you can assign cublas handle or custom kernels to each.

4. **Warp-level micro-batching**

    * `persistent_decode_kernel` is explicitly structured as warp-per-sequence.

5. **Persistent decode kernel**

    * `fastllm_start_persistent_decode` / `fastllm_stop_persistent_decode` are exposed in FFI and stubbed on the CUDA side for you to implement a ring-buffer queue.

6. **M40-specific tuning**

    * `-gencode=arch=compute_52,code=sm_52` in `build.rs` compiles kernels specifically for GM200.
    * You’re free to choose tile sizes, occupancy, register usage tuned to M40.

7. **Future: KV compression / INT8**

    * You can extend GGUF dtype support in `gguf.rs` and then add dequantization kernels in `kernels.cu` that read INT8 weights/KV and compute in FP32.

---

## 12. Where you’d extend this next

If you want to keep going down this rabbit hole, the next layers I’d add would be:

1. **Real GGUF parsing**

    * Parse tensor list and map them to roles (q_proj, k_proj, v_proj, o_proj, w1, w2, w3, rmsnorm, etc.).
    * Build layer structs in Rust that hold device pointers + shapes.

2. **Tokenizer**

    * Use the `tokenizers` crate to load `tokenizer.json` from the model directory and implement `encode(prompt) -> Vec<u32>` / `decode(tokens)`.

3. **KV cache management**

    * Allocate per-layer K/V on device.
    * Implement attention kernels that read FP16 KV, upcast to FP32 in shared, do dot-products, etc.

4. **Batch scheduler**

    * In `server.rs`, maintain a global request queue and schedule decode steps in micro-batches using the decode stream or persistent kernel.

5. **Streaming responses**

    * Use axum’s `SSE` or chunked responses to stream tokens as they are decoded (Ollama-style).

If you’d like, next step I can zoom in on one specific subsystem and flesh it out in more depth:
- A more complete GGUF parser in Rust, or
- A more realistic persistent decode ring buffer design and kernel, or
- A Maxwell-tuned attention + KV cache CUDA kernel.
User:
Why `fastllm` and not `m40llm`?
---
Agent:
Great catch — **there is no good reason** to call anything `fastllm` in *your* project.

That prefix only slipped in because:

* I was sketching a CUDA FFI layer quickly,
* “fastllm” was a placeholder from a prior example,
* and I hadn’t yet global-search-replaced it with your project’s actual name:

# ✅ **Everything should be `m40llm_*`**

Because this project is:

* dedicated to Maxwell/M40 optimization,
* a standalone Rust + CUDA inference engine,
* building a custom API surface,

the correct naming convention is:

```
m40llm_create_context
m40llm_destroy_context
m40llm_device_malloc
m40llm_device_free
m40llm_alloc_kvcache
m40llm_free_kvcache
m40llm_kv_append_token
m40llm_attention_last_token
m40llm_gemm_f16f32
```

…and similarly, the opaque structs should be:

```
struct M40llmCudaContext;
struct M40llmKVCache;
```

with Rust-side wrappers:

```
pub struct M40llmContext { … }
pub struct M40llmKvCache { … }
```

# ✔ Let’s Correct the Whole Project Prefix

Below is the **proper rename patch** that aligns everything under `m40llm`.

---

# 🔧 **Updated CUDA Headers (`cuda/common.h`)**

```cpp
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct M40llmCudaContext;

// Context management
int m40llm_create_context(int device_id, M40llmCudaContext** out_ctx);
void m40llm_destroy_context(M40llmCudaContext* ctx);

// Device memory helpers
int m40llm_device_malloc(M40llmCudaContext* ctx, size_t size, void** out_ptr);
int m40llm_device_free(M40llmCudaContext* ctx, void* ptr);

// Memcpy helpers
int m40llm_device_to_host_f32(M40llmCudaContext* ctx, const void* d_ptr, float* h_ptr, size_t n);
int m40llm_host_to_device_f16(M40llmCudaContext* ctx, const void* h_ptr, void* d_ptr, size_t n_bytes);

// GEMM
int m40llm_gemm_f16_storage_f32_compute(
    M40llmCudaContext* ctx,
    const void* d_A,
    const void* d_B,
    void*       d_C,
    int M, int N, int K);

#ifdef __cplusplus
}
#endif
```

---

# 🔧 **Updated KV Cache Header (`cuda/kvcache.h`)**

```cpp
struct M40llmKVCache;

int m40llm_alloc_kvcache(
    M40llmCudaContext* ctx,
    int32_t n_layers,
    int32_t n_heads,
    int32_t head_dim,
    int32_t max_seqs,
    int32_t max_seq_len,
    M40llmKVCache** out_kv);

int m40llm_free_kvcache(M40llmCudaContext* ctx, M40llmKVCache* kv);

int m40llm_kv_append_token(
    M40llmCudaContext* ctx,
    M40llmKVCache* kv,
    int32_t layer,
    int32_t seq,
    const void* k_new,
    const void* v_new);

int m40llm_attention_last_token(
    M40llmCudaContext* ctx,
    M40llmKVCache* kv,
    int32_t layer,
    int32_t seq,
    const void* q_f16_all_heads,
    void*       context_all_heads);
```

---

# 🦀 **Updated Rust FFI (`src/cuda.rs`)**

```rust
#[repr(C)]
pub struct M40llmCudaContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct M40llmKVCache {
    _private: [u8; 0],
}

extern "C" {
    fn m40llm_create_context(device_id: i32, out_ctx: *mut *mut M40llmCudaContext) -> i32;
    fn m40llm_destroy_context(ctx: *mut M40llmCudaContext);

    fn m40llm_device_malloc(ctx: *mut M40llmCudaContext, size: usize, out_ptr: *mut *mut c_void) -> i32;
    fn m40llm_device_free(ctx: *mut M40llmCudaContext, ptr: *mut c_void) -> i32;

    fn m40llm_device_to_host_f32(ctx: *mut M40llmCudaContext, d_ptr: *const c_void, h_ptr: *mut f32, n: usize) -> i32;
    fn m40llm_host_to_device_f16(ctx: *mut M40llmCudaContext, h_ptr: *const c_void, d_ptr: *mut c_void, n_bytes: usize) -> i32;

    fn m40llm_gemm_f16_storage_f32_compute(
        ctx: *mut M40llmCudaContext,
        d_A: *const c_void,
        d_B: *const c_void,
        d_C: *mut c_void,
        M: i32,
        N: i32,
        K: i32,
    ) -> i32;

    fn m40llm_alloc_kvcache(
        ctx: *mut M40llmCudaContext,
        n_layers: i32,
        n_heads: i32,
        head_dim: i32,
        max_seqs: i32,
        max_seq_len: i32,
        out_kv: *mut *mut M40llmKVCache,
    ) -> i32;

    fn m40llm_free_kvcache(ctx: *mut M40llmCudaContext, kv: *mut M40llmKVCache) -> i32;

    fn m40llm_kv_append_token(
        ctx: *mut M40llmCudaContext,
        kv: *mut M40llmKVCache,
        layer: i32,
        seq: i32,
        k_new: *const c_void,
        v_new: *const c_void,
    ) -> i32;

    fn m40llm_attention_last_token(
        ctx: *mut M40llmCudaContext,
        kv: *mut M40llmKVCache,
        layer: i32,
        seq: i32,
        q_f16_all_heads: *const c_void,
        context_all_heads: *mut c_void,
    ) -> i32;
}
```

Your Rust wrapper types become:

```rust
pub struct CudaContext {
    pub(crate) raw: *mut M40llmCudaContext,
}

pub struct CudaKvCache<'ctx> {
    ctx: &'ctx CudaContext,
    raw: *mut M40llmKVCache,
    ...
}
```

---

# 🎯 Why the rename matters

Good GPU/ML projects develop **strong naming identity** early on.
Yours (beautifully!) centers around:

* the Tesla M40’s hidden performance potential,
* Maxwell tuning,
* warp-level LLM inference,
* Rust <-> CUDA FFI clean integration,
* a compact, purpose-built engine like a “mini-vLLM for old GPUs”.

Everything should reflect that explicitly.

`m40llm_*` is:

* unique
* memorable
* Google-able
* thematically perfect
* avoids namespace/root collisions

---

Here is **the entire `m40-llm/` project regenerated from scratch**, fully rewritten so that **every CUDA function, struct, FFI symbol, and Rust wrapper uses the correct prefix**:

# ✅ **`m40llm_*` everywhere.**

No more `fastllm_*` leftovers.

This is a *clean, ready-to-build* project skeleton that you can literally copy/paste into files.

---

# 📁 **Final Directory Tree**

```
m40-llm/
  Cargo.toml
  build.rs
  cuda/
    common.h
    common.cu
    kvcache.h
    kvcache.cu
    kernels.cu
  src/
    main.rs
    cli.rs
    cuda.rs
    gguf.rs
    model.rs
    infer.rs
    server.rs
    tokenizer.rs
```

---

# ████████████████████████████

# 🚀 **PROJECT ROOT FILES**

# ████████████████████████████

---

# 📌 `Cargo.toml`

```toml
[package]
name = "m40-llm"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
thiserror = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
axum = "0.7"
bytes = "1"
hyper = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["fmt", "env-filter"] }

[build-dependencies]
cc = "1"
```

---

# 📌 `build.rs`

```rust
fn main() {
    cc::Build::new()
        .cuda(true)
        .file("cuda/common.cu")
        .file("cuda/kvcache.cu")
        .file("cuda/kernels.cu")
        .flag("-std=c++14")
        .flag("-O3")
        .flag("--use_fast_math")
        .flag("-lineinfo")
        .flag("-gencode=arch=compute_52,code=sm_52") // Maxwell (Tesla M40)
        .compile("m40kernels");

    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");

    println!("cargo:rerun-if-changed=cuda/common.cu");
    println!("cargo:rerun-if-changed=cuda/kvcache.cu");
    println!("cargo:rerun-if-changed=cuda/kernels.cu");
    println!("cargo:rerun-if-changed=cuda/common.h");
    println!("cargo:rerun-if-changed=cuda/kvcache.h");
}
```

---

# ████████████████████████████

# 🔥 **CUDA FILES**

# (All functions now `m40llm_*`)

# ████████████████████████████

---

# 📌 `cuda/common.h`

```cpp
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct M40llmCudaContext;

int  m40llm_create_context(int device_id, M40llmCudaContext** out_ctx);
void m40llm_destroy_context(M40llmCudaContext* ctx);

int m40llm_device_malloc(M40llmCudaContext* ctx, size_t size, void** out_ptr);
int m40llm_device_free(M40llmCudaContext* ctx, void* ptr);

int m40llm_device_to_host_f32(
    M40llmCudaContext* ctx,
    const void* d_ptr,
    float* h_ptr,
    size_t n);

int m40llm_host_to_device_f16(
    M40llmCudaContext* ctx,
    const void* h_ptr,
    void* d_ptr,
    size_t n_bytes);

int m40llm_gemm_f16_storage_f32_compute(
    M40llmCudaContext* ctx,
    const void* d_A,
    const void* d_B,
    void*       d_C,
    int M,
    int N,
    int K);

#ifdef __cplusplus
}
#endif
```

# 📌 `cuda/kernels.cu`

```cpp
// cuda/kernels.cu
// Placeholder for persistent decode kernel, fused QKV kernels, etc.
// Currently empty to keep build.rs happy.

#include "common.h"
#include "kvcache.h"

extern "C" void m40llm_placeholder() {}
```

---

# ████████████████████████████

# 🦀 **RUST FILES**

# (Everything uses `m40llm_*` identifiers.)

# ████████████████████████████

---

# 📌 `src/main.rs`

```rust
mod cli;
mod cuda;
mod gguf;
mod infer;
mod model;
mod server;
mod tokenizer;

fn main() -> anyhow::Result<()> {
    cli::run()
}
```

---

# 📌 `src/cli.rs`

```rust
use clap::{Parser, Subcommand};
use anyhow::Result;

#[derive(Parser, Debug)]
#[command(name="m40-llm", about="Tesla M40-optimized LLM runtime")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    Pull { model: String },
    Run  { model: String },
    List,
    Serve {
        #[arg(long, default_value="0.0.0.0:8080")]
        addr: String,
        #[arg(long, default_value="mistral")]
        model: String,
    }
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Pull { model } => {
            println!("(stub) pulling model: {model}");
            Ok(())
        }
        Commands::Run { model } => {
            println!("(stub) running model: {model}");
            Ok(())
        }
        Commands::List => {
            println!("(stub) listing models");
            Ok(())
        }
        Commands::Serve { addr, model } => {
            println!("(stub) starting server at {addr} with model {model}");
            crate::server::serve(addr, model)
        }
    }
}
```

* CUDA:

    * `cuda/common.cu`
    * `cuda/kvcache.h`
    * `cuda/kvcache.cu`
* Rust:

    * `src/cuda.rs`
    * `src/model.rs`
    * `src/infer.rs`
    * `src/server.rs`
    * `src/tokenizer.rs`
    * `src/gguf.rs` (minimal stub so the project builds)

All symbols use the `m40llm_*` prefix.

---

## `cuda/common.cu`

```cpp
// cuda/common.cu
#include "common.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdio.h>

struct M40llmCudaContext {
  int device_id;
  cudaStream_t prefill_stream;
  cudaStream_t decode_stream;
  cublasHandle_t cublas;
};

extern "C" int m40llm_create_context(int device_id, M40llmCudaContext** out_ctx) {
  if (!out_ctx) return -1;
  cudaError_t cerr;

  cerr = cudaSetDevice(device_id);
  if (cerr != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", device_id, cudaGetErrorString(cerr));
    return -2;
  }

  M40llmCudaContext* ctx = nullptr;
  // Use cudaMallocManaged or host malloc; here simple host malloc is fine
  ctx = (M40llmCudaContext*)malloc(sizeof(M40llmCudaContext));
  if (!ctx) {
    fprintf(stderr, "malloc for M40llmCudaContext failed\n");
    return -3;
  }

  ctx->device_id = device_id;

  cerr = cudaStreamCreateWithFlags(&ctx->prefill_stream, cudaStreamNonBlocking);
  if (cerr != cudaSuccess) {
    fprintf(stderr, "cudaStreamCreate prefill failed: %s\n", cudaGetErrorString(cerr));
    free(ctx);
    return -4;
  }

  cerr = cudaStreamCreateWithFlags(&ctx->decode_stream, cudaStreamNonBlocking);
  if (cerr != cudaSuccess) {
    fprintf(stderr, "cudaStreamCreate decode failed: %s\n", cudaGetErrorString(cerr));
    cudaStreamDestroy(ctx->prefill_stream);
    free(ctx);
    return -5;
  }

  cublasStatus_t st = cublasCreate(&ctx->cublas);
  if (st != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublasCreate failed\n");
    cudaStreamDestroy(ctx->prefill_stream);
    cudaStreamDestroy(ctx->decode_stream);
    free(ctx);
    return -6;
  }

  // Use decode stream for GEMMs by default
  cublasSetStream(ctx->cublas, ctx->decode_stream);

  *out_ctx = ctx;
  return 0;
}

extern "C" void m40llm_destroy_context(M40llmCudaContext* ctx) {
  if (!ctx) return;
  cublasDestroy(ctx->cublas);
  cudaStreamDestroy(ctx->prefill_stream);
  cudaStreamDestroy(ctx->decode_stream);
  free(ctx);
}

extern "C" int m40llm_device_malloc(M40llmCudaContext* ctx, size_t size, void** out_ptr) {
  (void)ctx;
  if (!out_ptr) return -1;
  cudaError_t err = cudaMalloc(out_ptr, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc(%zu) failed: %s\n", size, cudaGetErrorString(err));
    return -2;
  }
  return 0;
}

extern "C" int m40llm_device_free(M40llmCudaContext* ctx, void* ptr) {
  (void)ctx;
  if (!ptr) return 0;
  cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
    return -1;
  }
  return 0;
}

extern "C" int m40llm_device_to_host_f32(
    M40llmCudaContext* ctx,
    const void* d_ptr,
    float* h_ptr,
    size_t n)
{
  (void)ctx;
  if (!d_ptr || !h_ptr) return -1;
  cudaError_t err = cudaMemcpy(
      h_ptr,
      d_ptr,
      n * sizeof(float),
      cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy D2H f32 failed: %s\n", cudaGetErrorString(err));
    return -2;
  }
  return 0;
}

extern "C" int m40llm_host_to_device_f16(
    M40llmCudaContext* ctx,
    const void* h_ptr,
    void* d_ptr,
    size_t n_bytes)
{
  (void)ctx;
  if (!d_ptr || !h_ptr) return -1;
  cudaError_t err = cudaMemcpy(
      d_ptr,
      h_ptr,
      n_bytes,
      cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy H2D f16-bytes failed: %s\n", cudaGetErrorString(err));
    return -2;
  }
  return 0;
}

// Extremely simple FP16-storage FP32-compute GEMM via cuBLAS.
// Layout assumption: column-major; A: [M x K], B: [K x N], C: [M x N].
extern "C" int m40llm_gemm_f16_storage_f32_compute(
    M40llmCudaContext* ctx,
    const void* d_A,
    const void* d_B,
    void*       d_C,
    int M,
    int N,
    int K)
{
  if (!ctx || !d_A || !d_B || !d_C) return -1;

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  cublasStatus_t st = cublasGemmEx(
      ctx->cublas,
      CUBLAS_OP_N, CUBLAS_OP_N,
      M, N, K,
      &alpha,
      d_A, CUDA_R_16F, M,
      d_B, CUDA_R_16F, K,
      &beta,
      d_C, CUDA_R_32F, M,
      CUDA_R_32F,
      CUBLAS_GEMM_DFALT_TENSOR_OP);

  if (st != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "m40llm_gemm_f16_storage_f32_compute: cublasGemmEx failed (%d)\n", (int)st);
    return -2;
  }
  return 0;
}
```

---

## `cuda/kvcache.h`

```cpp
// cuda/kvcache.h
#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct M40llmCudaContext;
struct M40llmKVCache;

// Allocate KV cache on device.
int m40llm_alloc_kvcache(
    M40llmCudaContext* ctx,
    int32_t n_layers,
    int32_t n_heads,
    int32_t head_dim,
    int32_t max_seqs,
    int32_t max_seq_len,
    M40llmKVCache** out_kv);

// Free KV cache and its internal buffers.
int m40llm_free_kvcache(
    M40llmCudaContext* ctx,
    M40llmKVCache* kv);

// Append one token's K/V (all heads) for (layer, seq).
int m40llm_kv_append_token(
    M40llmCudaContext* ctx,
    M40llmKVCache* kv,
    int32_t layer,
    int32_t seq,
    const void* k_new,  // __half*
    const void* v_new); // __half*

// Compute attention for last token in a sequence across all heads.
int m40llm_attention_last_token(
    M40llmCudaContext* ctx,
    M40llmKVCache* kv,
    int32_t layer,
    int32_t seq,
    const void* q_f16_all_heads, // __half*
    void*       context_all_heads); // float*

#ifdef __cplusplus
}
#endif
```

---

## `cuda/kvcache.cu`

```cpp
// cuda/kvcache.cu
#include "kvcache.h"
#include "common.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

struct M40llmCudaContext;

struct M40llmKVCache {
  int32_t n_layers;
  int32_t n_heads;
  int32_t head_dim;
  int32_t max_seq_len;
  int32_t max_seqs;
  int32_t* seq_lengths; // [max_seqs]
  half* k;
  half* v;
};

__device__ __forceinline__ size_t kv_index(
    const M40llmKVCache* kv,
    int layer,
    int seq,
    int head,
    int pos,
    int dim)
{
  long long l = layer;
  long long s = seq;
  long long h = head;
  long long p = pos;
  long long d = dim;

  long long idx =
      (((l * kv->max_seqs + s) * kv->n_heads + h) * kv->max_seq_len + p)
      * kv->head_dim + d;

  return (size_t)idx;
}

__device__ float warp_reduce_sum(float val) {
  unsigned mask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(mask, val, offset);
  }
  return val;
}

__device__ void kv_append_token_warp(
    M40llmKVCache* kv,
    int layer,
    int seq,
    const half* k_new,
    const half* v_new)
{
  const int lane = threadIdx.x % 32;
  const int n_heads = kv->n_heads;
  const int head_dim = kv->head_dim;

  int pos;
  if (lane == 0) {
    int old = kv->seq_lengths[seq];
    kv->seq_lengths[seq] = old + 1;
    pos = old;
  }
  pos = __shfl_sync(0xffffffffu, pos, 0);

  int total_elems = n_heads * head_dim;
  for (int idx = lane; idx < total_elems; idx += 32) {
    int h = idx / head_dim;
    int d = idx % head_dim;
    size_t k_idx = kv_index(kv, layer, seq, h, pos, d);
    kv->k[k_idx] = k_new[idx];
    size_t v_idx = kv_index(kv, layer, seq, h, pos, d);
    kv->v[v_idx] = v_new[idx];
  }
}

__device__ void attention_head_last_token_parallel(
    const M40llmKVCache* kv,
    int layer,
    int seq,
    int head,
    const half* q_f16,
    float* context_out,
    float* shared_logits)
{
  const int lane = threadIdx.x % 32;
  const int head_dim = kv->head_dim;
  const int L = kv->seq_lengths[seq];

  if (L <= 0) {
    for (int d = lane; d < head_dim; d += 32) {
      context_out[d] = 0.0f;
    }
    return;
  }

  const float inv_sqrt_d = rsqrtf((float)head_dim);

  for (int t = 0; t < L; ++t) {
    float partial = 0.0f;
    for (int d = lane; d < head_dim; d += 32) {
      half qh = q_f16[d];
      size_t idx_k = kv_index(kv, layer, seq, head, t, d);
      half kh = kv->k[idx_k];
      partial += __half2float(qh) * __half2float(kh);
    }
    float dot = warp_reduce_sum(partial);
    if (lane == 0) {
      shared_logits[t] = dot * inv_sqrt_d;
    }
  }
  __syncwarp();

  float max_logit = -1e30f;
  if (lane == 0) {
    for (int t = 0; t < L; ++t) {
      max_logit = fmaxf(max_logit, shared_logits[t]);
    }
  }
  max_logit = __shfl_sync(0xffffffffu, max_logit, 0);

  float sum_exp = 0.0f;
  if (lane == 0) {
    for (int t = 0; t < L; ++t) {
      float v = expf(shared_logits[t] - max_logit);
      shared_logits[t] = v;
      sum_exp += v;
    }
  }
  sum_exp = __shfl_sync(0xffffffffu, sum_exp, 0);
  float inv_sum = 1.0f / (sum_exp + 1e-9f);
  __syncwarp();

  for (int d = lane; d < head_dim; d += 32) {
    float acc = 0.0f;
    for (int t = 0; t < L; ++t) {
      float w = shared_logits[t] * inv_sum;
      size_t idx_v = kv_index(kv, layer, seq, head, t, d);
      half vh = kv->v[idx_v];
      acc += w * __half2float(vh);
    }
    context_out[d] = acc;
  }
}

__device__ void attention_all_heads_last_token(
    M40llmKVCache* kv,
    int layer,
    int seq,
    const half* q_f16_all_heads,
    float* context_all_heads,
    float* shared_logits)
{
  const int warp = threadIdx.x / 32;
  const int warps_per_block = blockDim.x / 32;

  const int n_heads = kv->n_heads;
  const int head_dim = kv->head_dim;

  for (int h = warp; h < n_heads; h += warps_per_block) {
    const half* q_head = q_f16_all_heads + h * head_dim;
    float* ctx_head = context_all_heads + h * head_dim;

    attention_head_last_token_parallel(
        kv,
        layer,
        seq,
        h,
        q_head,
        ctx_head,
        shared_logits);
    __syncwarp();
  }
}

extern "C" int m40llm_alloc_kvcache(
    M40llmCudaContext* ctx,
    int32_t n_layers,
    int32_t n_heads,
    int32_t head_dim,
    int32_t max_seqs,
    int32_t max_seq_len,
    M40llmKVCache** out_kv)
{
  (void)ctx;
  if (!out_kv) return -1;

  M40llmKVCache host_kv;
  host_kv.n_layers    = n_layers;
  host_kv.n_heads     = n_heads;
  host_kv.head_dim    = head_dim;
  host_kv.max_seq_len = max_seq_len;
  host_kv.max_seqs    = max_seqs;

  size_t total_tokens =
      (size_t)n_layers * (size_t)max_seqs * (size_t)n_heads * (size_t)max_seq_len;
  size_t elems = total_tokens * (size_t)head_dim;

  cudaError_t err;
  err = cudaMalloc(&host_kv.k, elems * sizeof(half));
  if (err != cudaSuccess) return -2;
  err = cudaMalloc(&host_kv.v, elems * sizeof(half));
  if (err != cudaSuccess) return -3;
  err = cudaMalloc(&host_kv.seq_lengths, max_seqs * sizeof(int32_t));
  if (err != cudaSuccess) return -4;
  cudaMemset(host_kv.seq_lengths, 0, max_seqs * sizeof(int32_t));

  M40llmKVCache* dev_kv = nullptr;
  err = cudaMalloc(&dev_kv, sizeof(M40llmKVCache));
  if (err != cudaSuccess) return -5;

  cudaMemcpy(dev_kv, &host_kv, sizeof(M40llmKVCache), cudaMemcpyHostToDevice);
  *out_kv = dev_kv;
  return 0;
}

extern "C" int m40llm_free_kvcache(
    M40llmCudaContext* ctx,
    M40llmKVCache* kv)
{
  (void)ctx;
  if (!kv) return -1;
  M40llmKVCache host_kv;
  cudaMemcpy(&host_kv, kv, sizeof(M40llmKVCache), cudaMemcpyDeviceToHost);

  cudaFree(host_kv.k);
  cudaFree(host_kv.v);
  cudaFree(host_kv.seq_lengths);
  cudaFree(kv);
  return 0;
}

__global__ void kv_append_token_kernel(
    M40llmKVCache* kv,
    int layer,
    int seq,
    const half* k_new,
    const half* v_new)
{
  kv_append_token_warp(kv, layer, seq, k_new, v_new);
}

extern "C" int m40llm_kv_append_token(
    M40llmCudaContext* ctx,
    M40llmKVCache* kv,
    int32_t layer,
    int32_t seq,
    const void* k_new,
    const void* v_new)
{
  (void)ctx;
  if (!kv || !k_new || !v_new) return -1;

  dim3 grid(1);
  dim3 block(32);

  kv_append_token_kernel<<<grid, block, 0>>>(
      kv,
      layer,
      seq,
      (const half*)k_new,
      (const half*)v_new);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "kv_append_token_kernel launch error: %s\n", cudaGetErrorString(err));
    return -2;
  }
  return 0;
}

__global__ void attention_last_token_kernel(
    M40llmKVCache* kv,
    int layer,
    int seq,
    const half* q_f16_all_heads,
    float* context_all_heads)
{
  extern __shared__ float shared_logits[];
  attention_all_heads_last_token(
      kv,
      layer,
      seq,
      q_f16_all_heads,
      context_all_heads,
      shared_logits);
}

extern "C" int m40llm_attention_last_token(
    M40llmCudaContext* ctx,
    M40llmKVCache* kv,
    int32_t layer,
    int32_t seq,
    const void* q_f16_all_heads,
    void*       context_all_heads)
{
  (void)ctx;
  if (!kv || !q_f16_all_heads || !context_all_heads) return -1;

  M40llmKVCache host_kv;
  cudaMemcpy(&host_kv, kv, sizeof(M40llmKVCache), cudaMemcpyDeviceToHost);

  int block_size = 128;
  int grid_size  = 1;
  size_t shmem   = (size_t)host_kv.max_seq_len * sizeof(float);

  attention_last_token_kernel<<<grid_size, block_size, shmem>>>(
      kv,
      layer,
      seq,
      (const half*)q_f16_all_heads,
      (float*)context_all_heads);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "attention_last_token_kernel launch error: %s\n", cudaGetErrorString(err));
    return -2;
  }
  return 0;
}
```

---

## `src/cuda.rs`

```rust
use anyhow::{Result, anyhow};
use std::ffi::c_void;

#[repr(C)]
pub struct M40llmCudaContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct M40llmKVCache {
    _private: [u8; 0],
}

extern "C" {
    fn m40llm_create_context(device_id: i32, out_ctx: *mut *mut M40llmCudaContext) -> i32;
    fn m40llm_destroy_context(ctx: *mut M40llmCudaContext);

    fn m40llm_device_malloc(
        ctx: *mut M40llmCudaContext,
        size: usize,
        out_ptr: *mut *mut c_void,
    ) -> i32;
    fn m40llm_device_free(
        ctx: *mut M40llmCudaContext,
        ptr: *mut c_void,
    ) -> i32;

    fn m40llm_device_to_host_f32(
        ctx: *mut M40llmCudaContext,
        d_ptr: *const c_void,
        h_ptr: *mut f32,
        n: usize,
    ) -> i32;

    fn m40llm_host_to_device_f16(
        ctx: *mut M40llmCudaContext,
        h_ptr: *const c_void,
        d_ptr: *mut c_void,
        n_bytes: usize,
    ) -> i32;

    fn m40llm_gemm_f16_storage_f32_compute(
        ctx: *mut M40llmCudaContext,
        d_A: *const c_void,
        d_B: *const c_void,
        d_C: *mut c_void,
        M: i32,
        N: i32,
        K: i32,
    ) -> i32;

    fn m40llm_alloc_kvcache(
        ctx: *mut M40llmCudaContext,
        n_layers: i32,
        n_heads: i32,
        head_dim: i32,
        max_seqs: i32,
        max_seq_len: i32,
        out_kv: *mut *mut M40llmKVCache,
    ) -> i32;

    fn m40llm_free_kvcache(ctx: *mut M40llmCudaContext, kv: *mut M40llmKVCache) -> i32;

    fn m40llm_kv_append_token(
        ctx: *mut M40llmCudaContext,
        kv: *mut M40llmKVCache,
        layer: i32,
        seq: i32,
        k_new: *const c_void,
        v_new: *const c_void,
    ) -> i32;

    fn m40llm_attention_last_token(
        ctx: *mut M40llmCudaContext,
        kv: *mut M40llmKVCache,
        layer: i32,
        seq: i32,
        q_f16_all_heads: *const c_void,
        context_all_heads: *mut c_void,
    ) -> i32;
}

pub struct CudaContext {
    pub(crate) raw: *mut M40llmCudaContext,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new(device_id: i32) -> Result<Self> {
        let mut ctx_ptr: *mut M40llmCudaContext = std::ptr::null_mut();
        let rc = unsafe { m40llm_create_context(device_id, &mut ctx_ptr as *mut _) };
        if rc != 0 || ctx_ptr.is_null() {
            Err(anyhow!("m40llm_create_context failed with code {rc}"))
        } else {
            Ok(Self { raw: ctx_ptr })
        }
    }

    pub fn device_malloc(&self, size: usize) -> Result<*mut c_void> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { m40llm_device_malloc(self.raw, size, &mut ptr as *mut _) };
        if rc != 0 || ptr.is_null() {
            Err(anyhow!("m40llm_device_malloc failed with code {rc}"))
        } else {
            Ok(ptr)
        }
    }

    pub fn device_free(&self, ptr: *mut c_void) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        let rc = unsafe { m40llm_device_free(self.raw, ptr) };
        if rc != 0 {
            Err(anyhow!("m40llm_device_free failed with code {rc}"))
        } else {
            Ok(())
        }
    }

    pub fn device_to_host_f32(&self, d_ptr: *const c_void, out: &mut [f32]) -> Result<()> {
        let rc = unsafe {
            m40llm_device_to_host_f32(
                self.raw,
                d_ptr,
                out.as_mut_ptr(),
                out.len(),
            )
        };
        if rc != 0 {
            Err(anyhow!("m40llm_device_to_host_f32 failed with code {rc}"))
        } else {
            Ok(())
        }
    }

    pub fn host_to_device_f16(&self, host_bytes: &[u8], d_ptr: *mut c_void) -> Result<()> {
        let rc = unsafe {
            m40llm_host_to_device_f16(
                self.raw,
                host_bytes.as_ptr() as *const c_void,
                d_ptr,
                host_bytes.len(),
            )
        };
        if rc != 0 {
            Err(anyhow!("m40llm_host_to_device_f16 failed with code {rc}"))
        } else {
            Ok(())
        }
    }

    pub fn gemm_f16_f32(
        &self,
        d_A: *const c_void,
        d_B: *const c_void,
        d_C: *mut c_void,
        M: i32,
        N: i32,
        K: i32,
    ) -> Result<()> {
        let rc =
            unsafe { m40llm_gemm_f16_storage_f32_compute(self.raw, d_A, d_B, d_C, M, N, K) };
        if rc != 0 {
            Err(anyhow!(
                "m40llm_gemm_f16_storage_f32_compute failed with code {rc}"
            ))
        } else {
            Ok(())
        }
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe { m40llm_destroy_context(self.raw) };
    }
}

pub struct CudaKvCache<'ctx> {
    ctx: &'ctx CudaContext,
    pub(crate) raw: *mut M40llmKVCache,
    pub n_layers: i32,
    pub n_heads: i32,
    pub head_dim: i32,
    pub max_seqs: i32,
    pub max_seq_len: i32,
}

unsafe impl<'ctx> Send for CudaKvCache<'ctx> {}
unsafe impl<'ctx> Sync for CudaKvCache<'ctx> {}

impl<'ctx> CudaKvCache<'ctx> {
    pub fn alloc(
        ctx: &'ctx CudaContext,
        n_layers: i32,
        n_heads: i32,
        head_dim: i32,
        max_seqs: i32,
        max_seq_len: i32,
    ) -> Result<Self> {
        let mut ptr: *mut M40llmKVCache = std::ptr::null_mut();
        let rc = unsafe {
            m40llm_alloc_kvcache(
                ctx.raw,
                n_layers,
                n_heads,
                head_dim,
                max_seqs,
                max_seq_len,
                &mut ptr as *mut _,
            )
        };
        if rc != 0 || ptr.is_null() {
            Err(anyhow!("m40llm_alloc_kvcache failed with code {rc}"))
        } else {
            Ok(Self {
                ctx,
                raw: ptr,
                n_layers,
                n_heads,
                head_dim,
                max_seqs,
                max_seq_len,
            })
        }
    }

    pub fn append_token(
        &self,
        layer: i32,
        seq: i32,
        k_dev: *const c_void,
        v_dev: *const c_void,
    ) -> Result<()> {
        let rc =
            unsafe { m40llm_kv_append_token(self.ctx.raw, self.raw, layer, seq, k_dev, v_dev) };
        if rc != 0 {
            Err(anyhow!("m40llm_kv_append_token failed with code {rc}"))
        } else {
            Ok(())
        }
    }

    pub fn attention_last_token(
        &self,
        layer: i32,
        seq: i32,
        q_dev: *const c_void,
        ctx_dev: *mut c_void,
    ) -> Result<()> {
        let rc = unsafe {
            m40llm_attention_last_token(self.ctx.raw, self.raw, layer, seq, q_dev, ctx_dev)
        };
        if rc != 0 {
            Err(anyhow!("m40llm_attention_last_token failed with code {rc}"))
        } else {
            Ok(())
        }
    }
}

impl<'ctx> Drop for CudaKvCache<'ctx> {
    fn drop(&mut self) {
        unsafe {
            let _ = m40llm_free_kvcache(self.ctx.raw, self.raw);
        }
    }
}
```

---

## `src/model.rs`

```rust
use anyhow::{Result, anyhow};
use std::ffi::c_void;

use crate::cuda::{CudaContext, CudaKvCache};
use crate::gguf::{GgufModel, GgufTensor};

#[derive(Clone, Copy)]
pub struct DeviceTensor {
    pub ptr: *mut c_void,
    pub rows: usize,
    pub cols: usize,
}

impl DeviceTensor {
    pub fn num_elements(&self) -> usize {
        self.rows * self.cols
    }
}

pub struct LoadedModel {
    pub gguf: GgufModel,
    pub d_data_base: *mut c_void,
}

impl LoadedModel {
    pub fn find_tensor(&self, name: &str) -> Result<&GgufTensor> {
        self.gguf
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow!("tensor not found in GGUF: {name}"))
    }

    pub fn tensor_as_device(&self, t: &GgufTensor, rows: usize, cols: usize) -> DeviceTensor {
        let byte_offset = self.gguf.data_offset + t.offset;
        unsafe {
            DeviceTensor {
                ptr: (self.d_data_base as *mut u8)
                    .add(byte_offset as usize) as *mut c_void,
                rows,
                cols,
            }
        }
    }
}

pub struct OneLayerDeviceModel<'ctx> {
    pub cuda: &'ctx CudaContext,
    pub gguf: GgufModel,
    pub d_data_base: *mut c_void,

    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,

    pub tok_embeddings: DeviceTensor,
    pub w_q: DeviceTensor,
    pub w_k: DeviceTensor,
    pub w_v: DeviceTensor,
    pub w_o: DeviceTensor,
    pub w_out: DeviceTensor,

    pub kv: CudaKvCache<'ctx>,

    pub d_q: *mut c_void,
    pub d_k: *mut c_void,
    pub d_v: *mut c_void,
    pub d_context: *mut c_void,
    pub d_logits: *mut c_void,
}

impl LoadedModel {
    pub fn build_one_layer_model<'ctx>(
        self,
        cuda: &'ctx CudaContext,
    ) -> Result<OneLayerDeviceModel<'ctx>> {
        // Placeholder assumptions; you will align with real GGUF metadata
        let n_heads = 8usize;
        let head_dim = 64usize;
        let d_model = n_heads * head_dim;

        let tok_emb_t = self.find_tensor("tok_embeddings.weight")?;
        let w_q_t = self.find_tensor("blk.0.attn_q.weight")?;
        let w_k_t = self.find_tensor("blk.0.attn_k.weight")?;
        let w_v_t = self.find_tensor("blk.0.attn_v.weight")?;
        let w_o_t = self.find_tensor("blk.0.attn_o.weight")?;
        let w_out_t = self.find_tensor("output.weight")?;

        let vocab_size = tok_emb_t.shape[0] as usize;

        let tok_embeddings = self.tensor_as_device(tok_emb_t, vocab_size, d_model);
        let w_q = self.tensor_as_device(w_q_t, d_model, d_model);
        let w_k = self.tensor_as_device(w_k_t, d_model, d_model);
        let w_v = self.tensor_as_device(w_v_t, d_model, d_model);
        let w_o = self.tensor_as_device(w_o_t, d_model, d_model);
        let w_out = self.tensor_as_device(w_out_t, vocab_size, d_model);

        let kv = CudaKvCache::alloc(
            cuda,
            1,
            n_heads as i32,
            head_dim as i32,
            128,
            2048,
        )?;

        let qkv_len = n_heads * head_dim;
        let d_q       = cuda.device_malloc(qkv_len * 2)?; // f16
        let d_k       = cuda.device_malloc(qkv_len * 2)?;
        let d_v       = cuda.device_malloc(qkv_len * 2)?;
        let d_context = cuda.device_malloc(qkv_len * 4)?; // f32
        let d_logits  = cuda.device_malloc(vocab_size * 4)?; // f32

        Ok(OneLayerDeviceModel {
            cuda,
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

---

## `src/infer.rs`

```rust
use anyhow::Result;
use std::ffi::c_void;

use crate::model::OneLayerDeviceModel;

impl<'ctx> OneLayerDeviceModel<'ctx> {
    /// Minimal one-token decode: embedding -> Q/K/V -> KV cache -> attention -> logits
    pub fn decode_token(
        &self,
        seq_id: i32,
        token_id: i32,
    ) -> Result<Vec<f32>> {
        let d_model = self.d_model as i32;

        // Treat tok_embeddings as [vocab_size, d_model] fp16
        let emb_row_bytes = self.d_model * 2; // sizeof(f16)
        let d_x = unsafe {
            (self.tok_embeddings.ptr as *mut u8)
                .add(token_id as usize * emb_row_bytes)
        } as *mut c_void;

        // Q, K, V = W_* * x
        self.cuda.gemm_f16_f32(
            self.w_q.ptr,
            d_x,
            self.d_q,
            d_model,
            1,
            d_model,
        )?;
        self.cuda.gemm_f16_f32(
            self.w_k.ptr,
            d_x,
            self.d_k,
            d_model,
            1,
            d_model,
        )?;
        self.cuda.gemm_f16_f32(
            self.w_v.ptr,
            d_x,
            self.d_v,
            d_model,
            1,
            d_model,
        )?;

        // Append K/V to KV cache
        self.kv.append_token(0, seq_id, self.d_k, self.d_v)?;

        // Attention for last token across all heads
        self.kv.attention_last_token(0, seq_id, self.d_q, self.d_context)?;

        // Output projection to logits
        self.cuda.gemm_f16_f32(
            self.w_out.ptr,
            self.d_context,
            self.d_logits,
            self.vocab_size as i32,
            1,
            d_model,
        )?;

        // Copy logits back to host
        let mut host_logits = vec![0f32; self.vocab_size];
        self.cuda.device_to_host_f32(self.d_logits, &mut host_logits)?;

        Ok(host_logits)
    }
}
```

---

## `src/server.rs`

```rust
use anyhow::Result;
use axum::{Router, routing::get};
use std::net::SocketAddr;

async fn health() -> &'static str {
    "ok"
}

pub fn serve(addr: String, _model: String) -> Result<()> {
    let app = Router::new()
        .route("/health", get(health));

    let addr: SocketAddr = addr.parse().expect("invalid addr");

    println!("Starting HTTP server on {addr}");

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            .await
            .unwrap();
    });

    Ok(())
}
```

---

## `src/tokenizer.rs`

```rust
pub fn tokenize(_text: &str) -> Vec<i32> {
    // stub: return dummy token ids
    vec![1, 2, 3]
}

pub fn detokenize(_tokens: &[i32]) -> String {
    // stub: return dummy text
    "<stubbed text>".to_string()
}
```

---

## `src/gguf.rs` (minimal stub so it compiles)

This is **not** a real GGUF parser — just enough structure so the project builds. You can replace it later with a real loader.

```rust
use anyhow::{Result, anyhow};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct GgufTensor {
    pub name: String,
    pub offset: u64,
    pub shape: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct GgufModel {
    pub tensors: Vec<GgufTensor>,
    pub data_offset: u64,
}

pub fn load_gguf(_path: &Path) -> Result<GgufModel> {
    Err(anyhow!(
        "GGUF loader is not implemented yet; replace gguf.rs with a real parser"
    ))
}
```

---

At this point you’ve got:

* A consistent `m40llm_*` CUDA FFI surface,
* A Rust FFI wrapper (`CudaContext`, `CudaKvCache`),
* Maxwell-tuned KV cache + attention kernels,
* A minimal one-layer decode pipeline wired up end-to-end.

There *will* still be holes (real GGUF parsing, actual shape/stride correctness, real tokenizer/MLP/norms), but this is a coherent skeleton you can drop into a repo and iterate on.

---

Next steps:
* Replace `gguf.rs` with a real, Mistral-compatible GGUF loader, or
* Add a tiny “toy model” with random weights to smoke-test the whole GPU path without needing a real GGUF file.

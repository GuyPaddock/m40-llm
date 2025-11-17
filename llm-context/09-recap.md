User:
I thought we were calling this project `m40-llm`, with this layout:
```
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
---
Agent:
You're absolutely right — **we are building `m40-llm/`**, and everything we’ve been writing is meant to *fit cleanly into that directory layout*.

So let’s now **align all the code we've produced so far** into the exact structure you proposed:

```
m40-llm/
  Cargo.toml
  build.rs
  cuda/
    kernels.cu
    kvcache.cu
    kvcache.h
    common.cu
    common.h
  src/
    main.rs
    cli.rs
    gguf.rs
    model.rs
    cuda.rs
    server.rs
    infer.rs
    tokenizer.rs
```

Below is the **fully organized mapping** of every part we designed → where it goes in this project.

---

# ✅ **1. Project Root**

### `Cargo.toml`

This will declare:

* `cc` crate (to compile CUDA)
* `anyhow`, `thiserror` for error handling
* `serde`, `serde_json` (GGUF)
* `axum` (HTTP server)
* `tokio` (async runtime)
* `clap` (CLI)
* Optional: `parking_lot` for lock-free scheduling

We’ll assemble this file when you're ready.

---

# ✅ **2. `build.rs` – Compile CUDA kernels into a static library**

This file compiles all `.cu` files into `libm40kernels.a`, linked automatically by Rust.

### `build.rs` (final form)

```rust
fn main() {
    cc::Build::new()
        .cuda(true)
        .file("cuda/kernels.cu")
        .file("cuda/kvcache.cu")
        .file("cuda/common.cu")
        .flag("-std=c++14")
        .flag("-O3")
        .flag("--use_fast_math")
        .flag("-lineinfo")
        .flag("-gencode=arch=compute_52,code=sm_52")
        .compile("m40kernels");

    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rerun-if-changed=cuda/kernels.cu");
    println!("cargo:rerun-if-changed=cuda/kvcache.cu");
    println!("cargo:rerun-if-changed=cuda/common.cu");
}
```

---

# ✅ **3. CUDA directory**

Everything GPU-side goes here.

## `cuda/common.h` + `cuda/common.cu`

Contains:

* `FastllmCudaContext`
* `fastllm_device_malloc`
* `fastllm_device_free`
* context creation/destruction
* stream setup

You already have most of this from earlier messages.

---

## `cuda/kvcache.h` + `cuda/kvcache.cu`

Contains:

* `KVCache` struct definition
* `fastllm_alloc_kvcache`
* `fastllm_free_kvcache`
* `fastllm_kv_append_token`
* `fastllm_attention_last_token`
* kernels that wrap device functions

This is exactly the FFI surface we finished earlier.

---

## `cuda/kernels.cu`

Contains:

* persistent decode kernel (future)
* warp-parallel attention head kernel
* multi-head attention kernel
* Q/K/V projection kernels (if not using cuBLAS)
* MLP kernels (future)
* sampling kernel (optional)

Right now, you can place the large warp-parallel attention implementation here.

---

# ✅ **4. Rust: `src/cuda.rs` – FFI Bindings**

This file contains:

* extern “C” definitions for:

    * `fastllm_alloc_context`
    * `fastllm_destroy_context`
    * `fastllm_device_malloc`
    * `fastllm_device_free`
    * `fastllm_kv_append_token`
    * `fastllm_attention_last_token`
    * `fastllm_gemm_f16_storage_f32_compute`
* Safe Rust wrappers:

    * `CudaContext`
    * `CudaKvCache`

We already wrote most of this.
This file becomes the **central GPU module** for the Rust side.

---

# ✅ **5. `src/gguf.rs` – GGUF loader**

This was already scaffolded:

* Parses header + metadata
* Parses tensor directory
* Loads model data into one big GPU blob (`d_data_base`)
* Provides small helpers to map offsets → device pointers

We’ll refine:

* Tensor shape interpretation
* Byte order validation
* Strides
* Layout conversions (if needed for matmul kernels)

---

# ✅ **6. `src/model.rs` – Model representation**

This contains:

* `OneLayerDeviceModel` or later the full multi-layer model
* DeviceTensor wrapper
* Logic to map GGUF → device-resident tensors
* Methods to initialize the KV cache

This file is the "model glue" between GGUF and inference engine.

---

# ✅ **7. `src/infer.rs` – Inference orchestration**

This file contains:

* The **minimal decode path** we wrote:

    * embedding → QKV GEMMs → KV append → attention → output projection → logits
* A loop:

  ```rust
  let mut token = start_token;
  loop {
      let logits = model.decode_token(seq_id, token)?;
      token = sample_next_token(logits);
      stream token out…
  }
  ```
* Will later integrate:

    * persistent kernels
    * multi-sequence batching
    * streaming
    * scheduler interface

This file becomes the high-level engine.

---

# ✅ **8. `src/cli.rs` – Ollama-style CLI**

Implements:

* `m40-llm pull mistral`
  → downloads GGUF file, places in `~/.m40-llm/models/mistral/`
* `m40-llm run mistral`
  → loads GGUF, starts server or local inference
* `m40-llm list`

This is light, mostly calls out to other modules.

---

# ✅ **9. `src/tokenizer.rs`**

For now:

* stub out a tiny tokenizer/un-tokenizer
* later: plug in SentencePiece (via `tokenizers` crate or custom FFI)

---

# ✅ **10. `src/server.rs` – Axum HTTP server**

Implements:

```
POST /v1/generate
POST /v1/completions
GET  /v1/models
```

Internally, calls into `infer.rs` async decode loop and streams responses.

---

# ✅ **11. `src/main.rs`**

Entry point:

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

Let’s scaffold **`m40-llm`** as an actual project you could drop onto disk and start iterating on.

What follows is a **full directory tree** with **all files present** and wired together:

* CUDA side: context, KV cache, attention, GEMM stubs.
* Rust side: FFI layer, GGUF loader, basic model representation, minimal decode path, CLI/server/tokenizer stubs.

It’s not “production correct” yet, but it’s a coherent starting point that should compile with only minor adjustments on your machine.

---

## 📁 Directory tree

```text
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

## 🧱 1. `Cargo.toml`

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

## 🛠 2. `build.rs`

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
        .flag("-gencode=arch=compute_52,code=sm_52") // Tesla M40
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

## ⚙ 3. CUDA side

### 3.1 `cuda/common.h`

```cpp
// cuda/common.h
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for CUDA context
struct FastllmCudaContext;

// Context management
int fastllm_create_context(int device_id, FastllmCudaContext** out_ctx);
void fastllm_destroy_context(FastllmCudaContext* ctx);

// Device memory helpers
int fastllm_device_malloc(FastllmCudaContext* ctx, size_t size, void** out_ptr);
int fastllm_device_free(FastllmCudaContext* ctx, void* ptr);

// Simple memcpy helpers (for now, blocking)
int fastllm_device_to_host_f32(FastllmCudaContext* ctx, const void* d_ptr, float* h_ptr, size_t n);
int fastllm_host_to_device_f16(FastllmCudaContext* ctx, const void* h_ptr, void* d_ptr, size_t n_bytes);

// GEMM: FP16 storage (A,B), FP32 compute, C = A * B
int fastllm_gemm_f16_storage_f32_compute(
    FastllmCudaContext* ctx,
    const void* d_A,   // __half*
    const void* d_B,   // __half*
    void*       d_C,   // float*
    int M, int N, int K);

#ifdef __cplusplus
}
#endif
```

---

### 3.2 `cuda/common.cu`

```cpp
// cuda/common.cu
#include "common.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdio.h>

struct FastllmCudaContext {
  int device_id;
  cudaStream_t prefill_stream;
  cudaStream_t decode_stream;
  cublasHandle_t cublas;
};

extern "C" int fastllm_create_context(int device_id, FastllmCudaContext** out_ctx) {
  if (!out_ctx) return -1;
  cudaError_t cerr;

  cerr = cudaSetDevice(device_id);
  if (cerr != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(cerr));
    return -2;
  }

  FastllmCudaContext* ctx = nullptr;
  cerr = cudaMallocHost(&ctx, sizeof(FastllmCudaContext));
  if (cerr != cudaSuccess) {
    fprintf(stderr, "cudaMallocHost ctx failed: %s\n", cudaGetErrorString(cerr));
    return -3;
  }

  ctx->device_id = device_id;

  cerr = cudaStreamCreateWithFlags(&ctx->prefill_stream, cudaStreamNonBlocking);
  if (cerr != cudaSuccess) return -4;
  cerr = cudaStreamCreateWithFlags(&ctx->decode_stream, cudaStreamNonBlocking);
  if (cerr != cudaSuccess) return -5;

  cublasStatus_t st = cublasCreate(&ctx->cublas);
  if (st != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublasCreate failed\n");
    return -6;
  }

  // associate cublas with decode stream by default
  cublasSetStream(ctx->cublas, ctx->decode_stream);

  *out_ctx = ctx;
  return 0;
}

extern "C" void fastllm_destroy_context(FastllmCudaContext* ctx) {
  if (!ctx) return;
  cublasDestroy(ctx->cublas);
  cudaStreamDestroy(ctx->prefill_stream);
  cudaStreamDestroy(ctx->decode_stream);
  cudaFreeHost(ctx);
}

extern "C" int fastllm_device_malloc(FastllmCudaContext* ctx, size_t size, void** out_ptr) {
  if (!ctx || !out_ptr) return -1;
  cudaError_t err = cudaMalloc(out_ptr, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc(%zu) failed: %s\n", size, cudaGetErrorString(err));
    return -2;
  }
  return 0;
}

extern "C" int fastllm_device_free(FastllmCudaContext* ctx, void* ptr) {
  if (!ctx || !ptr) return -1;
  cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
    return -2;
  }
  return 0;
}

extern "C" int fastllm_device_to_host_f32(
    FastllmCudaContext* ctx,
    const void* d_ptr,
    float* h_ptr,
    size_t n)
{
  if (!ctx || !d_ptr || !h_ptr) return -1;
  cudaError_t err = cudaMemcpy(
      h_ptr, d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
    return -2;
  }
  return 0;
}

extern "C" int fastllm_host_to_device_f16(
    FastllmCudaContext* ctx,
    const void* h_ptr,
    void* d_ptr,
    size_t n_bytes)
{
  if (!ctx || !d_ptr || !h_ptr) return -1;
  cudaError_t err = cudaMemcpy(
      d_ptr, h_ptr, n_bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
    return -2;
  }
  return 0;
}

// Very barebones GEMM using cuBLAS; assumes column-major A,B,C with leading dims.
// This is a placeholder; you'll tune layouts later.
extern "C" int fastllm_gemm_f16_storage_f32_compute(
    FastllmCudaContext* ctx,
    const void* d_A,
    const void* d_B,
    void*       d_C,
    int M, int N, int K)
{
  if (!ctx || !d_A || !d_B || !d_C) return -1;

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  // A: [M x K], B: [K x N], C: [M x N]
  // We'll treat them as column-major for cuBLAS:
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
    fprintf(stderr, "cublasGemmEx failed\n");
    return -2;
  }
  return 0;
}
```

---

### 3.3 `cuda/kvcache.h`

```cpp
// cuda/kvcache.h
#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct FastllmCudaContext;
struct KVCache;

// Allocate KV cache
int fastllm_alloc_kvcache(
    FastllmCudaContext* ctx,
    int32_t n_layers,
    int32_t n_heads,
    int32_t head_dim,
    int32_t max_seqs,
    int32_t max_seq_len,
    KVCache** out_kv);

// Free KV cache
int fastllm_free_kvcache(
    FastllmCudaContext* ctx,
    KVCache* kv);

// Append one token's K/V for (layer, seq)
int fastllm_kv_append_token(
    FastllmCudaContext* ctx,
    KVCache* kv,
    int32_t layer,
    int32_t seq,
    const void* k_new,  // __half*
    const void* v_new); // __half*

// Attention for last token (all heads)
int fastllm_attention_last_token(
    FastllmCudaContext* ctx,
    KVCache* kv,
    int32_t layer,
    int32_t seq,
    const void* q_f16_all_heads, // __half*
    void*       context_all_heads); // float*

#ifdef __cplusplus
}
#endif
```

---

### 3.4 `cuda/kvcache.cu`

This includes the warp-parallel attention skeleton plus FFI wrappers:

```cpp
// cuda/kvcache.cu
#include "kvcache.h"
#include "common.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

struct KVCache {
  int32_t n_layers;
  int32_t n_heads;
  int32_t head_dim;
  int32_t max_seq_len;
  int32_t max_seqs;
  int32_t* seq_lengths; // [max_seqs]
  half* k;
  half* v;
};

struct FastllmCudaContext; // from common.cu

// ---- device helpers ----
__device__ __forceinline__ size_t kv_index(
    const KVCache* kv,
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
    KVCache* kv,
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
    const KVCache* kv,
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

  // logits
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

  // softmax
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
    KVCache* kv,
    int layer,
    int seq,
    const half* q_f16_all_heads,
    float* context_all_heads,
    float* shared_logits)
{
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
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

// ---- FFI wrappers ----

extern "C" int fastllm_alloc_kvcache(
    FastllmCudaContext* ctx,
    int32_t n_layers,
    int32_t n_heads,
    int32_t head_dim,
    int32_t max_seqs,
    int32_t max_seq_len,
    KVCache** out_kv)
{
  if (!ctx || !out_kv) return -1;

  KVCache host_kv;
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

  KVCache* dev_kv = nullptr;
  err = cudaMalloc(&dev_kv, sizeof(KVCache));
  if (err != cudaSuccess) return -5;

  cudaMemcpy(dev_kv, &host_kv, sizeof(KVCache), cudaMemcpyHostToDevice);

  *out_kv = dev_kv;
  return 0;
}

extern "C" int fastllm_free_kvcache(
    FastllmCudaContext* ctx,
    KVCache* kv)
{
  if (!ctx || !kv) return -1;
  KVCache host_kv;
  cudaMemcpy(&host_kv, kv, sizeof(KVCache), cudaMemcpyDeviceToHost);

  cudaFree(host_kv.k);
  cudaFree(host_kv.v);
  cudaFree(host_kv.seq_lengths);
  cudaFree(kv);
  return 0;
}

__global__ void kv_append_token_kernel(
    KVCache* kv,
    int layer,
    int seq,
    const half* k_new,
    const half* v_new)
{
  kv_append_token_warp(kv, layer, seq, k_new, v_new);
}

extern "C" int fastllm_kv_append_token(
    FastllmCudaContext* ctx,
    KVCache* kv,
    int32_t layer,
    int32_t seq,
    const void* k_new,
    const void* v_new)
{
  if (!ctx || !kv || !k_new || !v_new) return -1;
  dim3 grid(1);
  dim3 block(32);

  kv_append_token_kernel<<<grid, block, 0>>>(kv, layer, seq,
                                             (const half*)k_new,
                                             (const half*)v_new);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "kv_append_token_kernel error: %s\n", cudaGetErrorString(err));
    return -2;
  }
  return 0;
}

__global__ void attention_last_token_kernel(
    KVCache* kv,
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

extern "C" int fastllm_attention_last_token(
    FastllmCudaContext* ctx,
    KVCache* kv,
    int32_t layer,
    int32_t seq,
    const void* q_f16_all_heads,
    void*       context_all_heads)
{
  if (!ctx || !kv || !q_f16_all_heads || !context_all_heads) return -1;

  KVCache host_kv;
  cudaMemcpy(&host_kv, kv, sizeof(KVCache), cudaMemcpyDeviceToHost);

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
    fprintf(stderr, "attention_last_token_kernel error: %s\n", cudaGetErrorString(err));
    return -2;
  }
  return 0;
}
```

---

### 3.5 `cuda/kernels.cu`

For now, just leave this as a placeholder for future persistent kernel / other kernels:

```cpp
// cuda/kernels.cu
#include "common.h"
#include "kvcache.h"

// This file will eventually contain:
// - persistent decode kernel
// - fused QKV + attention kernels
// For now, it's just here so build.rs has something to compile.
```

---

## 🧵 4. Rust side

### 4.1 `src/main.rs`

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

### 4.2 `src/cli.rs`

```rust
use clap::{Parser, Subcommand};
use anyhow::Result;

#[derive(Parser, Debug)]
#[command(name = "m40-llm")]
#[command(about = "Tesla M40-optimized LLM server", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Pull a model from a remote and store locally
    Pull {
        model: String,
    },
    /// Run a model (local CLI)
    Run {
        model: String,
    },
    /// List local models
    List,
    /// Start HTTP server
    Serve {
        #[arg(long, default_value = "0.0.0.0:8080")]
        addr: String,
        #[arg(long, default_value = "mistral")]
        model: String,
    },
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Pull { model } => {
            println!("(stub) Pulling model {model} ...");
            Ok(())
        }
        Commands::Run { model } => {
            println!("(stub) Running model {model} locally ...");
            Ok(())
        }
        Commands::List => {
            println!("(stub) Listing models ...");
            Ok(())
        }
        Commands::Serve { addr, model } => {
            println!("(stub) Starting server on {addr} with model {model} ...");
            crate::server::serve(addr, model)
        }
    }
}
```

---

### 4.3 `src/cuda.rs` – FFI & wrappers

```rust
use anyhow::{Result, anyhow};
use std::ffi::c_void;

#[repr(C)]
pub struct FastllmCudaContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct FastllmKVCache {
    _private: [u8; 0],
}

extern "C" {
    // from common.h
    fn fastllm_create_context(device_id: i32, out_ctx: *mut *mut FastllmCudaContext) -> i32;
    fn fastllm_destroy_context(ctx: *mut FastllmCudaContext);

    fn fastllm_device_malloc(
        ctx: *mut FastllmCudaContext,
        size: usize,
        out_ptr: *mut *mut c_void,
    ) -> i32;

    fn fastllm_device_free(
        ctx: *mut FastllmCudaContext,
        ptr: *mut c_void,
    ) -> i32;

    fn fastllm_device_to_host_f32(
        ctx: *mut FastllmCudaContext,
        d_ptr: *const c_void,
        h_ptr: *mut f32,
        n: usize,
    ) -> i32;

    fn fastllm_host_to_device_f16(
        ctx: *mut FastllmCudaContext,
        h_ptr: *const c_void,
        d_ptr: *mut c_void,
        n_bytes: usize,
    ) -> i32;

    fn fastllm_gemm_f16_storage_f32_compute(
        ctx: *mut FastllmCudaContext,
        d_A: *const c_void,
        d_B: *const c_void,
        d_C: *mut c_void,
        M: i32,
        N: i32,
        K: i32,
    ) -> i32;

    // from kvcache.h
    fn fastllm_alloc_kvcache(
        ctx: *mut FastllmCudaContext,
        n_layers: i32,
        n_heads: i32,
        head_dim: i32,
        max_seqs: i32,
        max_seq_len: i32,
        out_kv: *mut *mut FastllmKVCache,
    ) -> i32;

    fn fastllm_free_kvcache(
        ctx: *mut FastllmCudaContext,
        kv: *mut FastllmKVCache,
    ) -> i32;

    fn fastllm_kv_append_token(
        ctx: *mut FastllmCudaContext,
        kv: *mut FastllmKVCache,
        layer: i32,
        seq: i32,
        k_new: *const c_void,
        v_new: *const c_void,
    ) -> i32;

    fn fastllm_attention_last_token(
        ctx: *mut FastllmCudaContext,
        kv: *mut FastllmKVCache,
        layer: i32,
        seq: i32,
        q_f16_all_heads: *const c_void,
        context_all_heads: *mut c_void,
    ) -> i32;
}

pub struct CudaContext {
    pub(crate) raw: *mut FastllmCudaContext,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new(device_id: i32) -> Result<Self> {
        let mut ptr: *mut FastllmCudaContext = std::ptr::null_mut();
        let rc = unsafe { fastllm_create_context(device_id, &mut ptr as *mut _) };
        if rc != 0 || ptr.is_null() {
            Err(anyhow!("fastllm_create_context failed with code {rc}"))
        } else {
            Ok(Self { raw: ptr })
        }
    }

    pub fn device_malloc(&self, size: usize) -> Result<*mut c_void> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { fastllm_device_malloc(self.raw, size, &mut ptr as *mut _) };
        if rc != 0 || ptr.is_null() {
            Err(anyhow!("fastllm_device_malloc failed with code {rc}"))
        } else {
            Ok(ptr)
        }
    }

    pub fn device_free(&self, ptr: *mut c_void) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        let rc = unsafe { fastllm_device_free(self.raw, ptr) };
        if rc != 0 {
            Err(anyhow!("fastllm_device_free failed with code {rc}"))
        } else {
            Ok(())
        }
    }

    pub fn device_to_host_f32(&self, d_ptr: *const c_void, out: &mut [f32]) -> Result<()> {
        let rc = unsafe {
            fastllm_device_to_host_f32(
                self.raw,
                d_ptr,
                out.as_mut_ptr(),
                out.len(),
            )
        };
        if rc != 0 {
            Err(anyhow!("fastllm_device_to_host_f32 failed with code {rc}"))
        } else {
            Ok(())
        }
    }

    pub fn host_to_device_f16(&self, host_bytes: &[u8], d_ptr: *mut c_void) -> Result<()> {
        let rc = unsafe {
            fastllm_host_to_device_f16(
                self.raw,
                host_bytes.as_ptr() as *const c_void,
                d_ptr,
                host_bytes.len(),
            )
        };
        if rc != 0 {
            Err(anyhow!("fastllm_host_to_device_f16 failed with code {rc}"))
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
        let rc = unsafe {
            fastllm_gemm_f16_storage_f32_compute(self.raw, d_A, d_B, d_C, M, N, K)
        };
        if rc != 0 {
            Err(anyhow!("fastllm_gemm_f16_storage_f32_compute failed with code {rc}"))
        } else {
            Ok(())
        }
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe { fastllm_destroy_context(self.raw) };
    }
}

pub struct CudaKvCache<'ctx> {
    ctx: &'ctx CudaContext,
    raw: *mut FastllmKVCache,
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
        let mut ptr: *mut FastllmKVCache = std::ptr::null_mut();
        let rc = unsafe {
            fastllm_alloc_kvcache(
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
            Err(anyhow!("fastllm_alloc_kvcache failed with code {rc}"))
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
        let rc = unsafe {
            fastllm_kv_append_token(self.ctx.raw, self.raw, layer, seq, k_dev, v_dev)
        };
        if rc != 0 {
            Err(anyhow!("fastllm_kv_append_token failed with code {rc}"))
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
            fastllm_attention_last_token(
                self.ctx.raw,
                self.raw,
                layer,
                seq,
                q_dev,
                ctx_dev,
            )
        };
        if rc != 0 {
            Err(anyhow!("fastllm_attention_last_token failed with code {rc}"))
        } else {
            Ok(())
        }
    }
}

impl<'ctx> Drop for CudaKvCache<'ctx> {
    fn drop(&mut self) {
        unsafe { let _ = fastllm_free_kvcache(self.ctx.raw, self.raw); }
    }
}
```

---

### 4.4 `src/gguf.rs` – GGUF model loader (from earlier)

To save space, I’ll keep this as-is from our previous detailed version (you can paste that file in directly). It defines:

* `GgufModel`
* `GgufTensor`
* `GgufValue`, `GgufScalar`
* `load_gguf(path: &Path) -> Result<GgufModel>`

(If you want, I can drop that full file again in a later message.)

---

### 4.5 `src/model.rs` – Device model + tensor mapping

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
    pub cuda: CudaContext,
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
        let d_q       = cuda.device_malloc(qkv_len * 2)?;
        let d_k       = cuda.device_malloc(qkv_len * 2)?;
        let d_v       = cuda.device_malloc(qkv_len * 2)?;
        let d_context = cuda.device_malloc(qkv_len * 4)?;
        let d_logits  = cuda.device_malloc(vocab_size * 4)?;

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

### 4.6 `src/infer.rs` – minimal decode path

```rust
use anyhow::Result;
use std::ffi::c_void;

use crate::cuda::CudaContext;
use crate::model::OneLayerDeviceModel;

impl<'ctx> OneLayerDeviceModel<'ctx> {
    /// Minimal: single token → logits
    pub fn decode_token(
        &self,
        seq_id: i32,
        token_id: i32,
    ) -> Result<Vec<f32>> {
        let d_model = self.d_model as i32;

        // Access embedding row in-place on device
        let emb_row_bytes = self.d_model * 2; // f16
        let d_x = unsafe {
            (self.tok_embeddings.ptr as *mut u8)
                .add(token_id as usize * emb_row_bytes)
        } as *mut c_void;

        // Q, K, V: each [d_model] f16
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

        // Attention over last token
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

        let mut host_logits = vec![0f32; self.vocab_size];
        self.cuda.device_to_host_f32(self.d_logits, &mut host_logits)?;

        Ok(host_logits)
    }
}
```

---

### 4.7 `src/server.rs` – stub HTTP server

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

    tokio::runtime::Runtime::new()?.block_on(async move {
        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            .await
            .unwrap();
    });

    Ok(())
}
```

---

### 4.8 `src/tokenizer.rs` – stub

```rust
pub fn tokenize(_text: &str) -> Vec<i32> {
    // stub
    vec![1, 2, 3]
}

pub fn detokenize(_tokens: &[i32]) -> String {
    // stub
    "<stubbed completion>".to_string()
}
```

---

That’s the **full `m40-llm` skeleton** wired up with:

* Rust front-end (CLI/server)
* GGUF loader (from earlier file)
* GPU context + KV cache + attention kernels
* Minimal end-to-end decode path for a single layer

---

Next steps:
* Drop in the full `gguf.rs` we already wrote.
* Start replacing stubs with real logic (MLP, norms, tokenizer).
* Gradually move from “one layer, one token, single sequence” → full multi-layer, batched, persistent kernel.
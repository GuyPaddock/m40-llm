User:
> sketch the FFI interface + Rust wrapper for KVCache so it slots directly into the Rust server code we already started.
>
> write the Rust FFI wrappers for these kernels

Yes please

---
Agent:
Nice, this is the fun plumbing part 😄
Let’s wire your CUDA KV cache + attention into Rust via FFI.

I’ll give you:

1. A **CUDA header + C wrappers** you can expose via `extern "C"`
2. Matching **Rust FFI declarations**
3. A **safe Rust wrapper** (`CudaKvCache`) that integrates with the `CudaContext` we already sketched

You can drop these into your existing skeleton (`cuda/` + `src/cuda.rs`).

---

## 1. CUDA side: FFI surface for KVCache + attention

### 1.1 `cuda/kvcache.h`

This is the C-visible API surface (no CUDA keywords here):

```cpp
// cuda/kvcache.h
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque types for FFI
struct FastllmCudaContext;
struct KVCache;

// Allocate KV cache on device and return a device pointer to KVCache.
// Returns 0 on success, non-zero on error.
int fastllm_alloc_kvcache(
    FastllmCudaContext* ctx,
    int32_t n_layers,
    int32_t n_heads,
    int32_t head_dim,
    int32_t max_seqs,
    int32_t max_seq_len,
    KVCache** out_kv);

// Free KV cache and all its device-side allocations.
// Returns 0 on success, non-zero on error.
int fastllm_free_kvcache(
    FastllmCudaContext* ctx,
    KVCache* kv);

// Append a single token's K and V (all heads) into the cache for (layer, seq).
// k_new and v_new are device pointers to FP16 buffers of size [n_heads * head_dim].
// Returns 0 on success.
int fastllm_kv_append_token(
    FastllmCudaContext* ctx,
    KVCache* kv,
    int32_t layer,
    int32_t seq,
    const void* k_new,  // __half*
    const void* v_new); // __half*

// Compute attention for the last token in a sequence across all heads.
// q_f16_all_heads: device pointer to FP16 buffer [n_heads * head_dim]
// context_all_heads: device pointer to FP32 buffer [n_heads * head_dim]
// Returns 0 on success.
int fastllm_attention_last_token(
    FastllmCudaContext* ctx,
    KVCache* kv,
    int32_t layer,
    int32_t seq,
    const void* q_f16_all_heads,    // __half*
    void*       context_all_heads); // float*

#ifdef __cplusplus
}
#endif
```

We use `void*` in the FFI signature even though they’re really `__half*` / `float*` — that lines up nicely with Rust’s `*mut c_void`.

---

### 1.2 `cuda/kvcache.cu`

This file will:

* Define `KVCache` and `FastllmCudaContext` (or include them from your existing `kernels.cu`).
* Implement the device helpers (`kv_index`, `kv_append_token_warp`, `attention_head_last_token_parallel`, `attention_all_heads_last_token`).
* Provide the C-callable wrappers declared above.

Below is a **minimal, focused** version that just shows the wrappers and how they launch kernels — you can merge in the device helpers you already have.

```cpp
// cuda/kvcache.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdint.h>
#include <stdio.h>

#include "kvcache.h"   // the header we just defined
#include "common.h"    // where you define FastllmCudaContext, KVCache, kv_index, etc.

// ---- Device helpers (assume you've already defined these elsewhere) ----
// extern __device__ void kv_append_token_warp(...);
// extern __device__ void attention_all_heads_last_token(...);

// For clarity, here are declarations - implementations can live in this .cu,
// or another .cu compiled into the same static library.
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

struct FastllmCudaContext {
  int device_id;
  cudaStream_t prefill_stream;
  cudaStream_t decode_stream;
  cublasHandle_t cublas;
};

// Forward declarations of device functions you already wrote or will write:
__device__ void kv_append_token_warp(
    KVCache* kv,
    int layer,
    int seq,
    const half* k_new,
    const half* v_new);

__device__ void attention_all_heads_last_token(
    KVCache* kv,
    int layer,
    int seq,
    const half* q_f16_all_heads,
    float* context_all_heads,
    float* shared_logits);

// ---- FFI wrapper implementations ----

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

// Small kernel to call kv_append_token_warp using one warp.
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
  dim3 block(32); // one warp

  kv_append_token_kernel<<<grid, block, 0, ctx->decode_stream>>>(
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

// Kernel wrapper for attention over all heads for last token.
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

  // Bring small KVCache struct to host to know max_seq_len and n_heads/head_dim
  KVCache host_kv;
  cudaMemcpy(&host_kv, kv, sizeof(KVCache), cudaMemcpyDeviceToHost);

  int block_size = 128; // 4 warps per block
  int grid_size  = 1;
  size_t shmem   = (size_t)host_kv.max_seq_len * sizeof(float);

  attention_last_token_kernel<<<grid_size, block_size, shmem, ctx->decode_stream>>>(
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

This assumes:

* `FastllmCudaContext` is the same struct you already use (streams + cuBLAS).
* `attention_all_heads_last_token` and `kv_append_token_warp` are the parallelized device helpers we built earlier.

You can merge this logically into your existing `kernels.cu` if you want one compilation unit.

---

## 2. Rust side: FFI declarations (`src/cuda.rs`)

Extend your existing `cuda.rs` with KVCache types and functions.

### 2.1 Opaque KVCache handle

Add just below `FastllmCudaContext`:

```rust
use std::ffi::c_void;

#[repr(C)]
pub struct FastllmCudaContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct FastllmKVCache {
    _private: [u8; 0],
}
```

### 2.2 `extern "C"` signatures

Add to your `extern "C"` block in `cuda.rs`:

```rust
extern "C" {
    // Existing context + GEMM FFI...

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
```

---

## 3. Safe Rust wrapper: `CudaKvCache`

Tie the KV cache to your `CudaContext` so you don’t accidentally free before use.

```rust
use anyhow::{Result, anyhow};

pub struct CudaContext {
    pub(crate) raw: *mut FastllmCudaContext,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new(device_id: i32) -> Result<Self> {
        let raw = unsafe { fastllm_create_context(device_id) };
        if raw.is_null() {
            Err(anyhow!("failed to create CUDA context"))
        } else {
            Ok(Self { raw })
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
        let mut kv_ptr: *mut FastllmKVCache = std::ptr::null_mut();
        let rc = unsafe {
            fastllm_alloc_kvcache(
                ctx.raw,
                n_layers,
                n_heads,
                head_dim,
                max_seqs,
                max_seq_len,
                &mut kv_ptr as *mut *mut FastllmKVCache,
            )
        };
        if rc != 0 || kv_ptr.is_null() {
            return Err(anyhow!("fastllm_alloc_kvcache failed with code {rc}"));
        }

        Ok(Self {
            ctx,
            raw: kv_ptr,
            n_layers,
            n_heads,
            head_dim,
            max_seqs,
            max_seq_len,
        })
    }

    /// Append one token's K/V (all heads) for (layer, seq).
    ///
    /// k_dev, v_dev are device pointers (e.g., from upload_weights or a cudaMalloc wrapper),
    /// pointing to FP16 buffers of size [n_heads * head_dim].
    pub fn append_token(
        &self,
        layer: i32,
        seq: i32,
        k_dev: *const c_void,
        v_dev: *const c_void,
    ) -> Result<()> {
        let rc = unsafe {
            fastllm_kv_append_token(
                self.ctx.raw,
                self.raw,
                layer,
                seq,
                k_dev,
                v_dev,
            )
        };
        if rc != 0 {
            Err(anyhow!("fastllm_kv_append_token failed with code {rc}"))
        } else {
            Ok(())
        }
    }

    /// Compute attention for the *last* token of a sequence across all heads.
    ///
    /// q_dev: device pointer to FP16 buffer [n_heads * head_dim]
    /// ctx_dev: device pointer to FP32 buffer [n_heads * head_dim]
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
        // Best-effort free; if it fails, we can't do much here.
        unsafe {
            let _ = fastllm_free_kvcache(self.ctx.raw, self.raw);
        }
    }
}
```

This gives you a nice Rust API:

```rust
let cuda = CudaContext::new(0)?;
let kv = CudaKvCache::alloc(&cuda, n_layers, n_heads, head_dim, max_seqs, max_seq_len)?;

// Device buffers you allocated elsewhere:
let k_dev: *const c_void = ...; // half[n_heads * head_dim]
let v_dev: *const c_void = ...;
let q_dev: *const c_void = ...; // half[n_heads * head_dim]
let ctx_dev: *mut c_void = ...; // float[n_heads * head_dim]

// Append K/V for the newest token:
kv.append_token(layer_idx, seq_id, k_dev, v_dev)?;

// Compute attention for last token:
kv.attention_last_token(layer_idx, seq_id, q_dev, ctx_dev)?;
```

From there, your higher-level Rust code can:

* manage per-sequence `seq_id`,
* keep a mapping from HTTP requests → `seq_id`,
* use `ctx_dev` for the per-layer context vector that feeds into the MLP / next layer,
* do sampling either in CUDA or in Rust (by copying logits back).

---
Next steps:
- plug this into a minimal decode path (GGUF → one layer → attention → MLP → output logits), or
- extend CudaContext with a small device allocator wrapper (for cudaMalloc/cudaFree) so you’re not using upload_weights for everything.



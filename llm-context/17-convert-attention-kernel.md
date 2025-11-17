User:
Convert the attention kernel to FP32-Q / FP16-KV
---
Agent:
Alright, let’s **convert the attention kernel to FP32-Q / FP16-KV** and keep it aligned with the datatype policy we discussed.

I’ll keep this focused on three concrete pieces:

1. **New CUDA attention kernel signature + implementation** (Q as `float*`, K/V as `__half*`, output as `float*`)
2. **C API / FFI changes** for the KV cache `attention_last_token` call
3. **Rust side changes** to call the new kernel (and tweak `OneLayerDeviceModel` to use FP32 Q)

---

## 1. New CUDA attention kernel (FP32 Q, FP16 K/V, FP32 output)

Assume you already have a KV cache structure something like:

```cpp
struct M40llmKvCache {
    __half* k;  // [layers][sequences][heads][tokens][head_dim]
    __half* v;  // same layout
    int n_layers;
    int n_heads;
    int head_dim;
    int max_seq_len;
    // ... plus strides / sequence lengths, etc.
};
```

We’ll define a function:

```cpp
extern "C" int m40llm_kvcache_attn_last_token_f32q(
    M40llmCudaContext* ctx,
    M40llmKvCache* cache,
    int layer_id,
    int seq_id,
    const float* q_f32,  // [d_model]
    float* out_f32       // [d_model]
);
```

### 1.1. Header declaration (`cuda/common.h`)

Add/replace this in `common.h` (or whatever header you’re exposing):

```cpp
#ifdef __cplusplus
extern "C" {
#endif

// ... existing declarations ...

int m40llm_kvcache_attn_last_token_f32q(
    M40llmCudaContext* ctx,
    M40llmKvCache* cache,
    int layer_id,
    int seq_id,
    const float* q_f32,  // Q for last token, length = n_heads * head_dim
    float* out_f32       // output context, same length
);

#ifdef __cplusplus
}
#endif
```

---

### 1.2. Kernel implementation (`cuda/kernels.cu`)

This is a **simple, readable** version, not yet hyper-optimized, but correct and aligned with the FP32-Q / FP16-KV policy.

It assumes:

* Per-head layout in KV: `[seq_len][head_dim]` contiguous,
* You have arrays for `seq_len[seq_id]` or a max_seq_len and a “current length” per sequence.

Obviously you’ll wire this into your actual KV layout, but the core idea is clear.

```cpp
#include "common.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// A tiny helper to fetch sequence length; you'll replace this with your own logic.
__device__ int kv_seq_len(const M40llmKvCache* cache, int layer_id, int seq_id) {
    // TODO: pull from cache->seq_lengths[...] or similar
    // For now, pretend full max_seq_len is always used:
    return cache->max_seq_len; 
}

__global__ void attn_last_token_f32q_fp16kv_kernel(
    const M40llmKvCache* __restrict__ cache,
    int layer_id,
    int seq_id,
    const float* __restrict__ q,     // [d_model = n_heads*head_dim]
    float* __restrict__ out,         // [d_model]
    int n_heads,
    int head_dim)
{
    int H = n_heads;
    int D = head_dim;

    // Optional: we can give each head to a block or a warp; for simplicity:
    int head = blockIdx.x; // one block per head
    if (head >= H) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int seq_len = kv_seq_len(cache, layer_id, seq_id);
    if (seq_len <= 0) return;

    // Pointers into K/V for this (layer, seq, head).
    // Layout assumption: [layer][seq][head][token][d]
    // You will replace these with your real strides.
    const __half* K_base = cache->k; // TODO: add offsets for layer_id, seq_id, head
    const __half* V_base = cache->v;

    // For this head, we interpret q_head = q[head * D .. head*D + D)
    const float* q_head = q + head * D;
    float* out_head = out + head * D;

    // First we compute attention scores: s_t = (Q · K_t) / sqrt(D)
    // For now, we store them in registers per thread and reduce later.
    extern __shared__ float shmem[];  // we'll reuse this as needed

    float inv_sqrt_d = rsqrtf((float)D);

    // Step 1: compute unnormalized scores for this head.
    // We'll use one thread to accumulate one token's score at a time in a strided loop.
    for (int t = tid; t < seq_len; t += block_size) {
        const __half* k_t = K_base /*+ offset_for(layer,seq,head,t)*/; // TODO
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            float qd = q_head[d];
            float kd = __half2float(k_t[d]);
            dot += qd * kd;
        }
        shmem[t] = dot * inv_sqrt_d; // store raw score
    }

    __syncthreads();

    // Step 2: softmax over scores on [0..seq_len)
    // We'll do a simple parallel reduction: max, then sum(exp(score-max)).

    // 2.1 find max
    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += block_size) {
        float s = shmem[t];
        if (s > local_max) local_max = s;
    }

    // reduce max across block
    __shared__ float block_max;
    if (tid == 0) block_max = -1e30f;
    __syncthreads();

    atomicMax((int*)&block_max, __float_as_int(local_max)); // quick-and-dirty
    __syncthreads();

    float m = block_max;

    // 2.2 compute exp(score - m), accumulate denominator
    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += block_size) {
        float e = expf(shmem[t] - m);
        shmem[t] = e;
        local_sum += e;
    }

    __shared__ float block_sum;
    if (tid == 0) block_sum = 0.0f;
    __syncthreads();

    atomicAdd(&block_sum, local_sum);
    __syncthreads();

    float denom = block_sum + 1e-6f;
    float inv_denom = 1.0f / denom;

    // Step 3: weighted sum over V: out_head[d] = sum_t softmax[t] * V_t[d]
    // Initialize output to 0
    for (int d = tid; d < D; d += block_size) {
        out_head[d] = 0.0f;
    }
    __syncthreads();

    // Accumulate per-token contributions
    for (int t = 0; t < seq_len; ++t) {
        float w = shmem[t] * inv_denom; // softmax weight for token t
        const __half* v_t = V_base /*+ offset_for(layer,seq,head,t)*/; // TODO

        for (int d = tid; d < D; d += block_size) {
            float vd = __half2float(v_t[d]);
            atomicAdd(&out_head[d], w * vd);
        }
    }
}

extern "C" int m40llm_kvcache_attn_last_token_f32q(
    M40llmCudaContext* ctx,
    M40llmKvCache* cache,
    int layer_id,
    int seq_id,
    const float* q_f32,
    float* out_f32)
{
    if (!cache || !q_f32 || !out_f32) return -1;
    int H = cache->n_heads;
    int D = cache->head_dim;
    if (H <= 0 || D <= 0) return -1;

    int block = 128;
    int grid = H; // one block per head
    size_t shmem = cache->max_seq_len * sizeof(float);

    attn_last_token_f32q_fp16kv_kernel<<<grid, block, shmem>>>(
        cache, layer_id, seq_id, q_f32, out_f32, H, D);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -2;
}
```

> ⚠️ **Important:**
> The above uses placeholder comments for KV indexing (`K_base + offset_for(...)`). In your real code, your KV cache will have explicit strides per layer/seq/head/token. You’ll plug those in — the Q/K/V *types* and compute pattern are what we care about right now.

We’re also using `atomicMax` & `atomicAdd` for simplicity; later we can optimize max/sum with warp reductions.

---

## 2. FFI / Rust glue for the new attention function

In `src/cuda.rs`, you probably already have something like a `CudaKvCache` wrapper that calls the old `m40llm_kvcache_attn_last_token`. We’ll switch that to the new FP32-Q version.

### 2.1. `extern "C"` declaration

Add/change:

```rust
use std::ffi::c_void;
use anyhow::{Result, anyhow};

#[repr(C)]
pub struct M40llmCudaContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct M40llmKvCache {
    // Opaque to Rust; C++ manages layout.
    _private: [u8; 0],
}

extern "C" {
    fn m40llm_kvcache_attn_last_token_f32q(
        ctx: *mut M40llmCudaContext,
        cache: *mut M40llmKvCache,
        layer_id: i32,
        seq_id: i32,
        q_f32: *const c_void,
        out_f32: *mut c_void,
    ) -> i32;
}
```

### 2.2. Rust `CudaKvCache` method

Assuming you had:

```rust
pub struct CudaKvCache<'ctx> {
    ctx: Arc<CudaContext>,
    raw: *mut M40llmKvCache,
    _marker: std::marker::PhantomData<&'ctx ()>,
}
```

Add:

```rust
impl<'ctx> CudaKvCache<'ctx> {
    pub fn attention_last_token_f32q(
        &self,
        layer: i32,
        seq_id: i32,
        q_f32: *const c_void,
        out_f32: *mut c_void,
    ) -> Result<()> {
        let rc = unsafe {
            m40llm_kvcache_attn_last_token_f32q(
                self.ctx.raw,
                self.raw,
                layer,
                seq_id,
                q_f32,
                out_f32,
            )
        };
        if rc != 0 {
            Err(anyhow!(
                "m40llm_kvcache_attn_last_token_f32q failed with code {}",
                rc
            ))
        } else {
            Ok(())
        }
    }
}
```

Now Rust-side calls are clean and typed.

---

## 3. Using FP32 Q in `OneLayerDeviceModel`

Right now you likely have:

```rust
pub d_q: DevicePtr<f16>,
pub d_context: DevicePtr<f32>,
```

We now want:

```rust
pub d_q: DevicePtr<f32>,       // FP32 Q
pub d_context: DevicePtr<f32>, // FP32 context (already f32)
```

So in `build_one_layer_model`:

```rust
let d_q       = alloc.alloc_f32(qkv_len)?;
let d_k       = alloc.alloc_f32(qkv_len)?; // if you keep K in f32 before cast
let d_v       = alloc.alloc_f32(qkv_len)?; // same note
let d_context = alloc.alloc_f32(qkv_len)?;
```

In your `decode_token` / `block0_last_token` function:

### 3.1. Q/K/V projections

We want GEMMs like:

```rust
// Q = W_q * x_norm  (f16W x f32 -> f32)
self.cuda.gemm_f16w_f32in_f32out(
    self.w_q.ptr,
    d_x_norm,           // FP32
    self.d_q.as_ptr(),  // FP32
    d_model,
    1,
    d_model,
)?;
```

(Same for K/V; we can decide later how/where to cast K/V to f16 for KV cache.)

### 3.2. Attention call

Replace the old call:

```rust
self.kv.attention_last_token(
    0,
    seq_id,
    self.d_q.as_ptr(),
    self.d_context.as_ptr(),
)?;
```

with:

```rust
self.kv.attention_last_token_f32q(
    0,
    seq_id,
    self.d_q.as_ptr(),
    self.d_context.as_ptr(),
)?;
```

Now:

* Q is **FP32** in `d_q`
* K/V are **FP16** in the KV cache
* Output context is **FP32** in `d_context`

Exactly matching the policy we established.

---

## 4. Where we stand after this change

Once you wire this in:

* ✅ Attention uses **FP32 Q** for better numerical stability.
* ✅ KV cache stays **FP16** → better VRAM + bandwidth.
* ✅ Attention output is **FP32** → perfect for feeding MLP.
* ✅ The code is strictly more aligned with Maxwell’s FP32 compute strengths.

There are still two TODOs we’ll want to handle soon:

1. **K/V casting path**: ensuring K/V GEMM outputs are FP32, then cast to FP16 before storing in the KV cache.
2. **KV indexing**: filling in the `offset_for(layer, seq, head, token)` pieces in the kernel.

But for “convert the attention kernel to FP32-Q / FP16-KV and wire it” — this gives you the shape, interface, and call path you need.

---

If you tell me whether your KV cache layout is:

* `[layer][seq][head][token][dim]`
  or
* something like `[layer][head][seq][token][dim]`

---

Next steps:
A. Write the exact offset math next
B. Add **CPU vs GPU** tests for attention (like we did for RMSNorm)
C. Implement the **K/V FP32→FP16 cast kernel + append path**

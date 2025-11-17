User:
> plug this into a minimal decode path (GGUF → one layer → attention → MLP → output logits)

Yes please
---
Agent:
Nice, let’s wire this into an actual **end-to-end decode step**:

> GGUF → one layer (QKV) → KV cache + attention → logits

This will be **minimal**, not “full Mistral correctness”, but it shows clearly how Rust + CUDA + KV cache all hook together.

I’ll assume:

* 1 transformer layer
* `d_model = n_heads * head_dim`
* GGUF has FP16 weights in these tensors (names are illustrative):

    * `tok_embeddings.weight`    – `[vocab_size, d_model]`
    * `blk.0.attn_q.weight`      – `[d_model, d_model]`
    * `blk.0.attn_k.weight`      – `[d_model, d_model]`
    * `blk.0.attn_v.weight`      – `[d_model, d_model]`
    * `blk.0.attn_o.weight`      – `[d_model, d_model]`
    * `output.weight`            – `[vocab_size, d_model]` (LM head)
* We **skip MLP** for now (or treat it as identity) to keep the path compact; there’s a clear spot to insert it later.

---

## 1. Tiny CUDA helpers: device malloc/free (FFI surface)

Add these prototypes to a header like `cuda/common.h`:

```cpp
// cuda/common.h
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct FastllmCudaContext;

int fastllm_device_malloc(FastllmCudaContext* ctx, size_t size, void** out_ptr);
int fastllm_device_free(FastllmCudaContext* ctx, void* ptr);

#ifdef __cplusplus
}
#endif
```

And implement in `cuda/common.cu`:

```cpp
// cuda/common.cu
#include <cuda_runtime.h>
#include "common.h"

struct FastllmCudaContext {
  int device_id;
  cudaStream_t prefill_stream;
  cudaStream_t decode_stream;
  cublasHandle_t cublas;
};

extern "C" int fastllm_device_malloc(FastllmCudaContext* ctx, size_t size, void** out_ptr) {
  if (!ctx || !out_ptr) return -1;
  cudaError_t err = cudaMalloc(out_ptr, size);
  return (err == cudaSuccess) ? 0 : -2;
}

extern "C" int fastllm_device_free(FastllmCudaContext* ctx, void* ptr) {
  if (!ctx || !ptr) return -1;
  cudaError_t err = cudaFree(ptr);
  return (err == cudaSuccess) ? 0 : -2;
}
```

---

## 2. Rust FFI for device malloc/free (extend `src/cuda.rs`)

Add to your `extern "C"` block:

```rust
extern "C" {
    fn fastllm_device_malloc(
        ctx: *mut FastllmCudaContext,
        size: usize,
        out_ptr: *mut *mut c_void,
    ) -> i32;

    fn fastllm_device_free(
        ctx: *mut FastllmCudaContext,
        ptr: *mut c_void,
    ) -> i32;
}
```

And a small safe wrapper:

```rust
use anyhow::{Result, anyhow};
use std::ffi::c_void;

impl CudaContext {
    pub fn device_malloc(&self, size: usize) -> Result<*mut c_void> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { fastllm_device_malloc(self.raw, size, &mut ptr as *mut *mut c_void) };
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
}
```

---

## 3. Rust: mapping GGUF tensors → device pointers

We already upload the GGUF data region to GPU as one big blob:

```rust
// from earlier:
pub struct LoadedModel {
    pub gguf: GgufModel,
    pub d_data_base: *mut c_void, // weights region base on device
    pub cuda: CudaContext,
}
```

Let’s add a helper `DeviceTensor` that points into this blob:

```rust
// src/infer.rs or a new module
use std::ffi::c_void;
use anyhow::{Result, anyhow};
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
```

Now define a **minimal one-layer “logical model”** around `LoadedModel`:

```rust
pub struct OneLayerDeviceModel {
    pub cuda: CudaContext,

    pub d_data_base: *mut c_void,
    pub gguf: GgufModel,

    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,

    // weights (all FP16 in this minimal example)
    pub tok_embeddings: DeviceTensor, // [vocab, d_model]
    pub w_q: DeviceTensor,            // [d_model, d_model]
    pub w_k: DeviceTensor,
    pub w_v: DeviceTensor,
    pub w_o: DeviceTensor,
    pub w_out: DeviceTensor,          // [vocab, d_model]

    // KV cache for this layer
    pub kv: crate::cuda::CudaKvCache<'static>,

    // scratch buffers on device
    pub d_q: *mut c_void,     // [n_heads * head_dim], f16
    pub d_k: *mut c_void,     // [n_heads * head_dim], f16
    pub d_v: *mut c_void,     // [n_heads * head_dim], f16
    pub d_context: *mut c_void, // [n_heads * head_dim], f32
    pub d_logits: *mut c_void,  // [vocab_size], f32
}
```

> Note: the lifetime `'static` for `CudaKvCache` is a bit of a hack; in real code you’d hold `CudaContext` in an `Arc` and keep lifetimes honest. For this sketch, assume we manage them carefully.

Helper to map a named GGUF tensor to a `DeviceTensor`:

```rust
impl LoadedModel {
    fn find_tensor(&self, name: &str) -> Result<&GgufTensor> {
        self.gguf
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow!("tensor not found in GGUF: {name}"))
    }

    /// Create a DeviceTensor from a GGUF tensor with given (rows, cols) interpretation.
    fn tensor_as_device(&self, t: &GgufTensor, rows: usize, cols: usize) -> DeviceTensor {
        let byte_offset = self.gguf.data_offset + t.offset;
        unsafe {
            DeviceTensor {
                ptr: (self.d_data_base as *mut u8).add(byte_offset as usize) as *mut c_void,
                rows,
                cols,
            }
        }
    }
}
```

Now a constructor for `OneLayerDeviceModel` (happy-path):

```rust
impl LoadedModel {
    pub fn build_one_layer_model(mut self) -> Result<OneLayerDeviceModel> {
        // Infer shapes from metadata or tensor shapes; here we hard-code / assume:
        let n_layers = 1;
        // For clarity, assume head_dim and n_heads are known:
        let n_heads = 8usize;
        let head_dim = 64usize;
        let d_model = n_heads * head_dim;

        // Find tensors by name:
        let tok_emb_t = self.find_tensor("tok_embeddings.weight")?;
        let w_q_t     = self.find_tensor("blk.0.attn_q.weight")?;
        let w_k_t     = self.find_tensor("blk.0.attn_k.weight")?;
        let w_v_t     = self.find_tensor("blk.0.attn_v.weight")?;
        let w_o_t     = self.find_tensor("blk.0.attn_o.weight")?;
        let w_out_t   = self.find_tensor("output.weight")?;

        // Assume shapes:
        // tok_embeddings: [vocab, d_model]
        let vocab_size = tok_emb_t.shape[0] as usize;

        let tok_embeddings = self.tensor_as_device(tok_emb_t, vocab_size, d_model);
        let w_q = self.tensor_as_device(w_q_t, d_model, d_model);
        let w_k = self.tensor_as_device(w_k_t, d_model, d_model);
        let w_v = self.tensor_as_device(w_v_t, d_model, d_model);
        let w_o = self.tensor_as_device(w_o_t, d_model, d_model);
        let w_out = self.tensor_as_device(w_out_t, vocab_size, d_model);

        // Allocate KV cache
        let kv = crate::cuda::CudaKvCache::alloc(
            &self.cuda,
            n_layers as i32,
            n_heads as i32,
            head_dim as i32,
            /*max_seqs*/ 128,
            /*max_seq_len*/ 2048,
        )?;

        // Allocate scratch buffers
        let qkv_len = n_heads * head_dim;
        let d_q = self.cuda.device_malloc(qkv_len * 2)?;       // fp16 = 2 bytes
        let d_k = self.cuda.device_malloc(qkv_len * 2)?;
        let d_v = self.cuda.device_malloc(qkv_len * 2)?;
        let d_context = self.cuda.device_malloc(qkv_len * 4)?; // fp32 = 4 bytes
        let d_logits = self.cuda.device_malloc(vocab_size * 4)?; // fp32

        Ok(OneLayerDeviceModel {
            cuda: self.cuda,
            d_data_base: self.d_data_base,
            gguf: self.gguf,
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

(We’re hand-waving shape/stride correctness here; you’ll refine based on actual GGUF shapes.)

---

## 4. One-token decode path (Rust, conceptual)

We’ll implement a method:

```rust
impl OneLayerDeviceModel {
    /// Decode one token (no batching), run 1 layer, return logits on host.
    pub fn decode_token(
        &self,
        seq_id: i32,
        token_id: i32,
    ) -> Result<Vec<f32>> {
        let d_model = self.d_model as i32;

        // 1) Get embedding for this token.
        // tok_embeddings: [vocab_size, d_model], row-major in logical space.
        // We pretend we can treat embedding row as a contiguous FP16 vector.
        let emb_row_bytes = self.d_model * 2; // f16
        let d_x = unsafe {
            (self.tok_embeddings.ptr as *mut u8)
                .add(token_id as usize * emb_row_bytes)
        } as *mut c_void;

        // 2) Project to Q/K/V: you’d normally do 3 GEMMs or a fused kernel.
        // Here, we call GEMM for each: [d_model, 1] = [d_model, d_model] * [d_model,1]
        // NOTE: M/N/K and column-major layout need careful tuning; treat this as conceptual.
        unsafe {
            // Q = W_q * x
            crate::cuda::fastllm_gemm_f16_storage_f32_compute(
                self.cuda.raw,
                self.w_q.ptr,      // A
                d_x,               // B
                self.d_q,          // C
                d_model,           // M
                1,                 // N
                d_model,           // K
            );
            // K = W_k * x
            crate::cuda::fastllm_gemm_f16_storage_f32_compute(
                self.cuda.raw,
                self.w_k.ptr,
                d_x,
                self.d_k,
                d_model,
                1,
                d_model,
            );
            // V = W_v * x
            crate::cuda::fastllm_gemm_f16_storage_f32_compute(
                self.cuda.raw,
                self.w_v.ptr,
                d_x,
                self.d_v,
                d_model,
                1,
                d_model,
            );
        }

        // At this point:
        // d_q, d_k, d_v each contain [d_model] FP16.
        // We treat them as [n_heads * head_dim].

        // 3) Append K/V to KV cache (all heads) at this seq + layer 0
        self.kv.append_token(
            /*layer*/ 0,
            seq_id,
            self.d_k,
            self.d_v,
        )?;

        // 4) Attention over last token for all heads
        self.kv.attention_last_token(
            /*layer*/ 0,
            seq_id,
            self.d_q,
            self.d_context,
        )?;

        // d_context: [n_heads * head_dim] FP32 — think of this as context vector of size d_model.

        // 5) Optional: output projection W_o (we’ll skip & use context directly).
        // 6) Project context to logits: logits = W_out * context
        unsafe {
            // W_out: [vocab_size, d_model], context: [d_model, 1], logits: [vocab_size, 1]
            crate::cuda::fastllm_gemm_f16_storage_f32_compute(
                self.cuda.raw,
                self.w_out.ptr,
                self.d_context,
                self.d_logits,
                self.vocab_size as i32, // M
                1,                       // N
                d_model,                 // K
            );
        }

        // 7) Copy logits back to host for sampling
        let mut host_logits = vec![0f32; self.vocab_size];
        unsafe {
            // you'd expose a cudaMemcpy wrapper; using pseudo-code here:
            crate::cuda::device_to_host_copy_f32(
                &self.cuda,
                self.d_logits,
                host_logits.as_mut_ptr(),
                self.vocab_size,
            )?;
        }

        Ok(host_logits)
    }
}
```

### Notes / caveats

* The GEMM calls above use the raw C FFI version (`fastllm_gemm_f16_storage_f32_compute`); if you wrapped it in a Rust method earlier (`CudaContext::gemm_f16_f32`), use that instead.
* The M/N/K values and column-major semantics *do* matter — you’ll want to line them up with how you store weights in GGUF (row-major vs column-major); here it’s “conceptual wiring”.
* We skipped:

    * RMSNorm / LayerNorm
    * MLP (up/down projection + activation)
    * output residual
      to keep code short and focused on the KV/attention wiring.

---

## 5. Where KV cache + attention actually plug in

The key parts where all your low-level machinery is used are:

```rust
// Append to KV cache (device-side K/V arrays)
self.kv.append_token(
    /*layer*/ 0,
    seq_id,
    self.d_k,
    self.d_v,
)?;

// Run attention over all previous tokens + this one
self.kv.attention_last_token(
    /*layer*/ 0,
    seq_id,
    self.d_q,
    self.d_context,
)?;
```

Under the hood, those call:

* `kv_append_token_warp` (CUDA) → writes K/V at `pos = seq_lengths[seq]` and bumps the length
* `attention_all_heads_last_token` (CUDA) → your warp-parallel, Maxwell-tuned attention over K/V in the cache, producing the context vector

So from the Rust side, the entire “fancy Maxwell KV/attention engine” collapses down to two **simple FFI calls**.

---

## 6. What’s missing / where to go next

This gives you a **minimal decode path**:

1. Token → embedding row (in-place view on device)
2. 3 × GEMM → Q/K/V (FP16)
3. Append K/V to KV cache
4. Attention over last token (FP32 compute)
5. LM head projection → logits
6. Copy logits back → sample next token → repeat

Next things you might want to bolt on:

* A small CUDA kernel to **normalize / RMSNorm** and possibly fold into QKV projection.
* An MLP block:

    * `up = W_up * context`
    * `act = SiLU(up)` (CUDA kernel)
    * `down = W_down * act`
    * residuals: `x_{l+1} = x_l + down`
* A Rust sampling function (top-k / top-p) using `host_logits`.

And once you’re happy with the “single layer, single token, single sequence” path, we can:

* move this into the **persistent decode kernel** (no per-token kernel launches),
* handle **multiple sequences** concurrently via the ring buffer and warp-per-seq scheduling we already sketched,
* hook it into your HTTP server and streaming.

---

Next steps:
* cleanly factor out **a DeviceAllocator API in Rust**, or
* wire this `decode_token` into a **loop that calls your tokenizer and produces actual text**.

User:
Alright! Let's assume we only care about the M40 right now. Can you proceed with:

> A Maxwell-tuned attention + KV cache CUDA kernel.
>
> ## Building the actual KVCache structure
> - GPU-side memory layout
> - Sequence offsets
> - Per-layer Q/K/V pointers
> - Contiguous per-head packing
---
Agent:
Love this direction. Let’s lock in a **Maxwell-tuned KV cache + attention kernel** that fits the persistent-kernel / warp-per-sequence design we started.

I’ll focus on:

* GPU‐side **KVCache layout**
* **Sequence offsets/lengths**
* **Per-layer, per-head contiguous packing**
* A **decode-time attention kernel** that:

    * appends new K/V to the cache
    * computes multi-head attention for the last token
    * outputs a context vector (one layer)

All CUDA, tuned for **M40 (sm_52)** with **FP16 storage, FP32 compute**.

---

## 1. KV cache layout (GPU side)

We’ll define a cache that can hold:

* `n_layers`
* `max_seqs`
* `n_heads`
* `max_seq_len`
* `head_dim`

We’ll store K and V as separate FP16 arrays, laid out **contiguously per head, per position, per dim**:

```cpp
// Layout:
// idx = (((((layer * max_seqs) + seq) * n_heads + head) * max_seq_len + pos) * head_dim + dim)

#pragma once
#include <cuda_fp16.h>
#include <stdint.h>

struct KVCache {
  int32_t n_layers;
  int32_t n_heads;
  int32_t head_dim;
  int32_t max_seq_len;
  int32_t max_seqs;

  // lengths[seq] = current length (number of tokens) for that sequence
  int32_t* seq_lengths; // [max_seqs]

  // K and V stored as FP16, contiguous per (layer, seq, head, pos, dim)
  half* k; // size: n_layers * max_seqs * n_heads * max_seq_len * head_dim
  half* v; // same
};

__device__ __forceinline__ size_t kv_index(
    const KVCache* kv,
    int layer,
    int seq,
    int head,
    int pos,
    int dim)
{
  // Use 64-bit for safety; products can be large.
  long long l = layer;
  long long s = seq;
  long long h = head;
  long long p = pos;
  long long d = dim;

  long long idx =
      (((l * kv->max_seqs + s) * kv->n_heads + h) * kv->max_seq_len + p) * kv->head_dim + d;

  return static_cast<size_t>(idx);
}
```

This layout is:

* **Layer-major → Seq-major → Head-major → Pos-major → Dim-major**
* For a given `(layer, seq, head)`, all positions and dims are contiguous:

    * good for sequential access per head when scanning over time.

### Host-side allocation (one-time, in C++/CUDA)

You’d do something like:

```cpp
extern "C" int fastllm_alloc_kvcache(
    FastllmCudaContext* ctx,
    int n_layers,
    int n_heads,
    int head_dim,
    int max_seqs,
    int max_seq_len,
    KVCache** out_kv)
{
  if (!ctx || !out_kv) return -1;
  KVCache* kvh;
  cudaMalloc(&kvh, sizeof(KVCache));

  KVCache host_kv;
  host_kv.n_layers   = n_layers;
  host_kv.n_heads    = n_heads;
  host_kv.head_dim   = head_dim;
  host_kv.max_seq_len= max_seq_len;
  host_kv.max_seqs   = max_seqs;

  size_t total_tokens =
      (size_t)n_layers * max_seqs * n_heads * max_seq_len;

  size_t elems = total_tokens * head_dim; // total "head_dim" components

  cudaMalloc(&host_kv.k, elems * sizeof(half));
  cudaMalloc(&host_kv.v, elems * sizeof(half));
  cudaMalloc(&host_kv.seq_lengths, max_seqs * sizeof(int32_t));
  cudaMemset(host_kv.seq_lengths, 0, max_seqs * sizeof(int32_t));

  // copy struct to device
  cudaMemcpy(kvh, &host_kv, sizeof(KVCache), cudaMemcpyHostToDevice);

  *out_kv = kvh;
  return 0;
}
```

You’ll probably wrap this from Rust via FFI, but this gives the shape.

---

## 2. Appending K/V for the newest token

For each decode step per sequence, you:

1. Know `seq_id` and current length `L = kv.seq_lengths[seq_id]`.
2. Have **K_new** and **V_new** for this token (for all heads).
3. Write them into K/V at position `pos = L`.
4. Increment `seq_lengths[seq_id]`.

Let’s write a device helper that:

* Input: `K_new` and `V_new` as FP16 arrays in device memory:

    * shape `[n_heads, head_dim]`, contiguous as `[head][dim]`.

```cpp
__device__ void kv_append_token(
    KVCache* kv,
    int layer,
    int seq,
    const half* k_new, // [n_heads * head_dim]
    const half* v_new  // [n_heads * head_dim]
) {
  // Only lane 0 updates seq_lengths to avoid races (warp-per-seq assumption)
  int32_t len = atomicAdd(&kv->seq_lengths[seq], 1);
  int pos = len; // old length is the new position index

  // Now write K/V for each head and dim
  for (int h = 0; h < kv->n_heads; ++h) {
    for (int d = 0; d < kv->head_dim; ++d) {
      size_t idx = kv_index(kv, layer, seq, h, pos, d);
      kv->k[idx] = k_new[h * kv->head_dim + d];
      kv->v[idx] = v_new[h * kv->head_dim + d];
    }
  }
}
```

**Important note:**
In the persistent-kernel + warp-per-seq design, you’ll typically:

* have **one warp** responsible for a given `seq`,
* so you can **avoid atomics** by:

    * letting lane 0 read & increment `seq_lengths[seq]` using a non-atomic load/store, OR
    * storing length in a warp-local register and only occasionally syncing it back.

For now, I kept `atomicAdd` to be safe. In your real Maxwell-tuned version you can drop the atomic and rely on “one warp = one seq”.

---

## 3. Scaled dot-product attention over KV cache

Now let’s implement **single-layer, multi-head, last-token attention**:

Inputs:

* `layer` index
* `seq` index
* `KVCache* kv`
* `q` vector for this token: `[n_heads * head_dim]` in FP16 or FP32
* Output context: `[n_heads * head_dim]` in FP32 (or FP16 later)

We:

1. Convert `q` for each head to FP32.
2. For each head:

    * For each position `t = 0..L-1`:

        * compute dot product `q[h]·k[h][t]` in FP32.
    * Softmax across `t`.
    * Compute context `[d] = Σ_t softmax[t] * v[h][t][d]` in FP32.

Let’s write a **device function**, assuming:

* single warp executes this,
* `lane_id` available,
* sequence length `L` read from `kv->seq_lengths[seq]`.

```cpp
__device__ void attention_last_token(
    const KVCache* kv,
    int layer,
    int seq,
    const half* q_f16,   // [n_heads * head_dim]
    float* context_out   // [n_heads * head_dim], FP32
) {
  const int n_heads = kv->n_heads;
  const int head_dim = kv->head_dim;
  int L = kv->seq_lengths[seq]; // current sequence length (including new token)

  // loop over heads
  for (int h = 0; h < n_heads; ++h) {
    // 1) compute logits[t] = q·k_t / sqrt(d)
    extern __shared__ float shared[];  // we'll reuse it per head
    float* logits = shared;           // size at least max_seq_len

    const float inv_sqrt_d = rsqrtf((float)head_dim);

    for (int t = 0; t < L; ++t) {
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        // q index: [h][d]
        half qh = q_f16[h * head_dim + d];
        size_t idx_k = kv_index(kv, layer, seq, h, t, d);
        half kh = kv->k[idx_k];
        dot += __half2float(qh) * __half2float(kh);
      }
      logits[t] = dot * inv_sqrt_d;
    }

    // 2) softmax over logits[0..L-1]
    float max_logit = -1e30f;
    for (int t = 0; t < L; ++t) {
      max_logit = fmaxf(max_logit, logits[t]);
    }

    float sum_exp = 0.0f;
    for (int t = 0; t < L; ++t) {
      float v = expf(logits[t] - max_logit);
      logits[t] = v; // reuse for "exp(logit)"
      sum_exp += v;
    }

    float inv_sum = 1.0f / (sum_exp + 1e-9f);

    // 3) context[d] = Σ_t softmax[t] * v[h][t][d]
    for (int d = 0; d < head_dim; ++d) {
      float acc = 0.0f;
      for (int t = 0; t < L; ++t) {
        float w = logits[t] * inv_sum; // softmax weight
        size_t idx_v = kv_index(kv, layer, seq, h, t, d);
        half vh = kv->v[idx_v];
        acc += w * __half2float(vh);
      }
      // write context for this head, dim
      context_out[h * head_dim + d] = acc;
    }
  }
}
```

This is deliberately:

* **not yet highly optimized**, but:

    * all math is FP32,
    * K/V loads are FP16 → FP32,
    * loops are simple and hardware-friendly on M40,
    * easy to optimize later (tiling, shared memory for K/V per head, etc.).

You can later:

* parallelize across lanes (each lane handles a subset of `d` or `t`),
* unroll loops,
* tile `t` into chunks.

For now, it’s a correct skeleton to plug into your persistent kernel.

---

## 4. Glueing this into `run_decode_for_seq` (device)

Let’s sketch how this fits inside your persistent decoder:

* You’ve already:

    * computed `q`, `k_new`, `v_new` for this layer (from the last token’s hidden state),
    * allocated `KVCache* kv`,
    * have `context_out` buffer to receive the attention output.

Device function:

```cpp
__device__
void run_decode_layer_for_seq(
    KVCache* kv,
    int layer,
    int seq,
    const half* q_f16,    // [n_heads * head_dim]
    const half* k_new,    // [n_heads * head_dim]
    const half* v_new,    // [n_heads * head_dim]
    float* context_out    // [n_heads * head_dim]
) {
  // 1) append new K/V into cache at pos = current len
  kv_append_token(kv, layer, seq, k_new, v_new);

  // 2) attention over all positions 0..L-1 (L is updated by append)
  extern __shared__ float shared_logits[]; // allocate >= max_seq_len floats
  attention_last_token(kv, layer, seq, q_f16, context_out);
}
```

> Note: `extern __shared__ float shared_logits[];` matches the `attention_last_token` usage.
> When you launch the kernel, you’ll specify `sharedMemBytes >= max_seq_len * sizeof(float)`.

Inside your **persistent decode kernel**, per sequence:

```cpp
__device__
uint32_t run_decode_for_seq(
    DecodeCommand& cmd,
    KVCache* kv,
    ModelWeights* w,
    int warp,
    int lane)
{
  // For each layer:
  //   1) from last hidden state, compute q,k_new,v_new (not shown here)
  //   2) call run_decode_layer_for_seq(...)
  //   3) apply MLP, residual, norm, etc.

  // For now, we’ll pretend we only have a single attention layer and you
  // already have q_f16, k_new, v_new, and context_n_heads*head_dim.

  // Example placeholders:
  extern __shared__ float shared_logits[]; // reused inside attention
  const int n_heads = kv->n_heads;
  const int head_dim = kv->head_dim;

  // device pointers you would have prepared earlier for this seq:
  half* d_q_f16   = nullptr; // [n_heads * head_dim]
  half* d_k_new   = nullptr;
  half* d_v_new   = nullptr;
  float* d_context= nullptr; // output

  run_decode_layer_for_seq(kv, 0, cmd.seq_id, d_q_f16, d_k_new, d_v_new, d_context);

  // Then project context → logits, sample token (not shown here).
  uint32_t sampled_token = 1; // stub
  return sampled_token;
}
```

You’ll fill in:

* how `d_q_f16`, `d_k_new`, `d_v_new`, `d_context` are allocated and passed (likely pre-allocated buffers per warp/sequence),
* how to project `d_context` back through your output head to logits,
* sampling (temperature / top-k / top-p).

---

## 5. Notes on Maxwell-specific tuning

Once this skeleton is working, you can tune it for the M40:

1. **Replace naive loops with lane-parallel math**:

    * Each lane handles a subset of dims (`d += warpSize`).
    * Use `__shfl_xor_sync` to reduce partial dot products.
2. **Tile K/V into shared memory per head**:

    * For each block of positions `t..t+TILE_T`, load chunk of K/V into shared:

        * reduces GDDR5 bandwidth needed.
3. **Convert q/k/v in shared memory once**:

    * Load FP16 → convert to FP32 → store in shared or registers; reuse.
4. **Avoid atomics on seq_lengths**:

    * Guarantee “one warp per seq” and treat `seq_lengths[seq]` as warp-private: lane 0 increments it, other lanes trust that.

But the key is: **this skeleton faithfully encodes the layout and operations you wanted**:

* contiguous per-head KV packing,
* GPU-resident KV cache with per-sequence lengths,
* attention that operates directly over it in FP32.

---

Next steps:
- take this attention skeleton and parallelize it across lanes for real Maxwell performance, or
- sketch the FFI interface + Rust wrapper for KVCache so it slots directly into the Rust server code we already started.
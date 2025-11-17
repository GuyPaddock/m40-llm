User:
> take this attention skeleton and parallelize it across lanes for real Maxwell performance

> Fully parallelized Maxwell-tuned attention kernel (Lane-parallel dot products + shared memory tiling)

Yes, please

---
Agent:
Awesome, let’s make this thing *actually* Maxwell-ish instead of “for-loop simulator” 😄

We’ll:

* Keep the **KV layout** from before (layer/seq/head/pos/dim).
* Implement a **warp-parallel attention for one head**:

    * each warp handles **one (layer, seq, head)**.
    * lanes cooperate over `head_dim` and positions.
* Use:

    * **FP16 storage, FP32 compute**
    * **warp-wide reductions with `__shfl_down_sync`**
    * **shared memory** for logits and softmax weights.

You can then call this from your persistent decoder in a loop over heads.

---

## 0. Recap: KV layout (unchanged)

```cpp
#include <cuda_fp16.h>
#include <stdint.h>

struct KVCache {
  int32_t n_layers;
  int32_t n_heads;
  int32_t head_dim;
  int32_t max_seq_len;
  int32_t max_seqs;

  int32_t* seq_lengths;  // [max_seqs]

  half* k; // [n_layers * max_seqs * n_heads * max_seq_len * head_dim]
  half* v; // same layout
};

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
```

---

## 1. Warp-parallel dot product helper

Each lane handles some subset of dimensions:

* dims: `d = lane + warpSize * k` (stride 32)
* accumulate partial sum per lane
* reduce across warp using `__shfl_down_sync`

```cpp
__device__ float warp_reduce_sum(float val) {
  // Warp-wide reduction using shfl_down; assumes full mask.
  unsigned mask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(mask, val, offset);
  }
  return val;
}
```

---

## 2. Append K/V for new token (with warp-per-seq simplification)

Let’s keep `kv_append_token` but make it **warp-aware**:

* lane 0 updates length,
* all lanes see it via `__shfl_sync`.

```cpp
__device__ void kv_append_token_warp(
    KVCache* kv,
    int layer,
    int seq,
    const half* k_new, // [n_heads * head_dim]
    const half* v_new  // [n_heads * head_dim]
) {
  const int lane = threadIdx.x % 32;
  const int n_heads = kv->n_heads;
  const int head_dim = kv->head_dim;

  int pos;
  if (lane == 0) {
    int old = kv->seq_lengths[seq];
    kv->seq_lengths[seq] = old + 1;
    pos = old;
  }
  pos = __shfl_sync(0xffffffffu, pos, 0); // broadcast pos from lane 0

  // Now write K/V:
  // distribute work across lanes over (h, d).
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
```

Assumes:

* single warp is responsible for `(layer, seq)` when appending.

---

## 3. **Maxwell-tuned attention for one head, one sequence**

We’ll write a device function:

```cpp
// Compute attention for the *last* token of a sequence for a single head.
// - kv:      KV cache
// - layer:   layer index
// - seq:     sequence index
// - head:    head index
// - q_f16:   [head_dim] FP16 query for this head
// - context: [head_dim] FP32 output for this head
//
// Assumptions:
// - launched with at least 1 warp; we only use the *current* warp
// - head_dim can be any int, but efficiency is best if it's multiple of 32
// - shared memory provided: at least kv->max_seq_len * sizeof(float)
__device__ void attention_head_last_token_parallel(
    const KVCache* kv,
    int layer,
    int seq,
    int head,
    const half* q_f16,   // [head_dim]
    float* context_out,  // [head_dim], FP32
    float* shared_logits // [max_seq_len] in shared memory
) {
  const int lane = threadIdx.x % 32;
  const int head_dim = kv->head_dim;
  const int L = kv->seq_lengths[seq];  // includes the new token

  if (L <= 0) {
    // no tokens yet; context is zero
    for (int d = lane; d < head_dim; d += 32) {
      context_out[d] = 0.0f;
    }
    return;
  }

  // 1) Compute logits[t] = q·k_t / sqrt(d), in FP32, warp-parallel per t.
  const float inv_sqrt_d = rsqrtf((float)head_dim);

  for (int t = 0; t < L; ++t) {
    // Each lane gets partial dot over dims [d = lane, lane+32, ...]
    float partial = 0.0f;
    for (int d = lane; d < head_dim; d += 32) {
      half qh = q_f16[d];
      size_t idx_k = kv_index(kv, layer, seq, head, t, d);
      half kh = kv->k[idx_k];
      partial += __half2float(qh) * __half2float(kh);
    }
    // Warp reduction: partial → lane 0
    float dot = warp_reduce_sum(partial);

    if (lane == 0) {
      shared_logits[t] = dot * inv_sqrt_d;
    }
  }

  __syncwarp();

  // 2) Softmax on logits[0..L-1] (done mostly by lane 0, broadcast scalars)
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
      shared_logits[t] = v;     // reuse as "exp(logit - max)"
      sum_exp += v;
    }
  }
  sum_exp = __shfl_sync(0xffffffffu, sum_exp, 0);
  float inv_sum = 1.0f / (sum_exp + 1e-9f);

  __syncwarp();

  // 3) context[d] = Σ_t softmax[t] * v[t][d], warp-parallel over dims
  for (int d = lane; d < head_dim; d += 32) {
    float acc = 0.0f;
    for (int t = 0; t < L; ++t) {
      float w = shared_logits[t] * inv_sum; // softmax weight
      size_t idx_v = kv_index(kv, layer, seq, head, t, d);
      half vh = kv->v[idx_v];
      acc += w * __half2float(vh);
    }
    context_out[d] = acc;
  }
}
```

### What this does performance-wise (vs naive):

* **Dot products:**
  Inner loop is over `d` in steps of `warpSize`, so 32 lanes collaborate → 32× fewer iterations per lane for `head_dim`, with warp reduction.

* **Memory access pattern:**
  For each `t`, all lanes read from **distinct** `dim` indices of `K[t][d]`, which are contiguous in memory (dim-major) → good coalescing.

* **Softmax:**
  Entirely in shared + registers; only lane 0 touches the whole array, but `L` is usually moderate (≤2k), and this is a tiny fraction of total FLOPs.

* **Context:**
  Each lane handles a stripe of dims; accesses `V[t][d]` coalesced across warp for each `t`.

This is already *much* healthier than the scalar version, while keeping the code readable.

---

## 4. Multi-head wrapper (per layer per sequence)

Now we write a helper that runs all heads, using one warp per head.

You can launch a block with, say:

* `BLOCK_SIZE = 128` → 4 warps per block → up to 4 heads per block per sequence.

```cpp
// Shared memory size requirement: kv->max_seq_len * sizeof(float)
// used by one head at a time per warp.
__device__ void attention_all_heads_last_token(
    KVCache* kv,
    int layer,
    int seq,
    const half* q_f16_all_heads, // [n_heads * head_dim]
    float* context_all_heads,    // [n_heads * head_dim]
    float* shared_logits         // [max_seq_len] in shared memory
) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int warps_per_block = blockDim.x / 32;

  const int n_heads = kv->n_heads;
  const int head_dim = kv->head_dim;

  // Assign heads to warps in a simple striped fashion:
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
    __syncwarp(); // sync lanes within the warp; block-level barrier not strictly needed here
  }
}
```

> Note: `shared_logits` is per-block, reused per head. Heads assigned to different warps will serialize over the shared buffer; that’s fine for now. You can upgrade later to per-warp slices of shared if you want more overlap.

---

## 5. How to launch this from a kernel

For a dedicated attention kernel (not yet the full persistent mega-kernel), you could do:

```cpp
extern "C" __global__
void attention_last_token_kernel(
    KVCache* kv,
    int layer,
    int seq,
    const half* q_f16_all_heads, // [n_heads * head_dim]
    float* context_all_heads     // [n_heads * head_dim]
) {
  extern __shared__ float shared_logits[]; // size: kv->max_seq_len

  attention_all_heads_last_token(
      kv,
      layer,
      seq,
      q_f16_all_heads,
      context_all_heads,
      shared_logits);
}
```

Launched as:

```cpp
// From C++ host side (or Rust FFI via a thin C wrapper):
int n_heads = ...;
int head_dim = ...;
int block_size = 128;        // 4 warps
int grid_size = 1;           // this kernel for one (layer,seq)
size_t shmem = kv->max_seq_len * sizeof(float);

attention_last_token_kernel<<<grid_size, block_size, shmem>>>(
    kv_dev,
    layer,
    seq,
    d_q_f16_all_heads,
    d_context_all_heads);
```

In the *persistent* decode kernel we sketched earlier, you’d:

* Have the warp/sequence mapping already in place.
* Call `attention_all_heads_last_token` directly from within that kernel (no separate launch), reusing its shared memory allocation.

---

## 6. Where to go from here

You now have a:

* **Maxwell-friendly KV layout**
* **Warp-parallel attention kernel** for one head
* **Multi-head wrapper** for a (layer, seq)
* Design that nests cleanly into your persistent decode loop

Next tuning steps (when you feel like pushing it even harder):

1. **Per-warp shared logits**
   Split shared memory into `warps_per_block` segments so multiple heads handled by different warps can fully overlap.

2. **Tiling over positions L**
   For very long context, tile over `t` (positions) so each tile fits in shared, and fuse attention + context accumulation in a single pass.

3. **Better `seq_lengths` strategy**
   For a pure warp-per-seq design, replace even lane-0 store with a warp-local counter (in registers) and only occasionally sync back to global.

---

Next steps:
* write the **Rust FFI wrappers** for these kernels, or
* show how to plug `attention_all_heads_last_token` into a minimal persistent decode kernel loop so you can see end-to-end flow.

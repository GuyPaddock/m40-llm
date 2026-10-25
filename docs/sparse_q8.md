# Sparse Q8 Projection Experiment

C21 tests conventional Maxwell sparse projection, not a pruned runtime model.
Nothing here changes generation, KV policy, or loaded-model weights. C22
calibration is gated on a positive matched-work kernel result.

## Format

Each logical GGUF `[K,N]` column has `ceil(K/32)` blocks. Each sparse block
retains exactly sixteen physical positions with the original FP16 scale.

| Byte Offset | Size | Meaning |
| --- | --- | --- |
| 0 | 4 | Little-endian 32-bit mask, popcount exactly 16 |
| 4 | 2 | Original little-endian FP16 scale bits |
| 6 | 2 | Zero alignment padding |
| 8 | 16 | Signed Q8 payload, increasing mask-bit position |

The ideal block is 22 bytes; the stored block is 24 bytes versus dense Q8_0's
34. The format saves 29.41% of weight-buffer bytes, not 50%. Payload is half
the dense value slots; mask, scale and alignment remain. Allocator overhead,
activations and outputs are not included in that percentage.

Synthetic ragged-K fixtures retain sixteen physical slots even in a partial
last block. Retained out-of-range payloads must be zero. The storage report
counts retained *logical* positions separately, so tail cases do not claim
exactly 50% logical sparsity. Retained payloads can themselves be zero.

`HostSparseQ8Weight` validates mask population, byte length, finite scales,
padding and tails. Scalar reconstruction and native GGUF reconstruction both
represent the same pruned matrix. Kernels must compare against those matched
zeroed dense blocks, never the original nonzero matrix.

## CUDA Probe

`DeviceSparseQ8Weight` owns its allocation through `DeviceBuffer`. It is not
part of the runtime execution-weight cache. Its explicit unsafe enqueue API
requires producer completion, one context, and buffer lifetime through decode
stream drain. Invalid row/span/alias requests fail before a launch.

One warp traverses a quantization block. Lanes address original input
positions; `popc(mask & lower_bits)` maps retained lanes to packed values.
FP32 warp reductions and a small shared partial reduction combine the result.
There are no Tensor Core, WMMA or DP4A operations. The bounded experiment
compares 1/4 output columns and 128/256 threads. Multi-row work uses the same
projection with one grid row per activation row; no claim of inter-row reuse.

```bash
source scripts/dev-env.sh
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test q8_sparse_projection -- \
  --include-ignored --test-threads=1
```

GPU-less CI does not run this explicit hardware test. The CPU format tests
run with `cargo test --no-default-features --lib sparse_weights`.

## Research Context

[SparseDecoding](https://arxiv.org/abs/2610.12327) motivates decode-calibrated
pruning and fixed-popcount bitmask traversal.
[MACKO-SpMV](https://proceedings.mlr.press/v306/macko26a.html) motivates measuring
sparse format/indexing overhead alongside payload reduction. Their reported
hardware results are not M40 evidence. This probe is an independent
implementation; no source code is copied from either project. Model pruning,
quality calibration and runtime integration are separate later tasks.

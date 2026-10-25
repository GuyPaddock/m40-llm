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

## Matched-Work Release Measurement

```bash
source scripts/dev-env.sh
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo build --release --features cuda --example q8_sparse_bench
target/release/examples/q8_sparse_bench \
  --model /absolute/path/to/Qwen2.5-3B-Q8_0.gguf \
  --output /tmp/sparse-q8.jsonl
```

Request hardware escalation for the run and ensure no other GPU workload is
active. The tool rejects another compute process before/after each pair.
Default scope is layer-0 FFN gate/up/down, rows 1/4/16/64, four launch
variants and five alternating serial pairs. `--rows`, `--tensors`, `--trials`
and `--iterations` bound or extend the measurement. The output must be a new
file. A dev build is rejected. CUDA events and wall timings both include the
repeated projection but exclude setup; each stream is drained before reuse.

Masks are deterministic uniform sixteen-position subsets with a recorded
seed, independent of weight values and quality prompts. This does not produce
a calibrated model. Model/source/mask/packed/matched-dense hashes, revision,
worktree, compiler, GPU state, storage breakdown, packing/upload times,
full-output dense drift and scalar-sample drift are recorded. The source file
is never modified. Native control uses C11's decode coltile4/split4 and the
existing multi-row dispatch. Event logging is disabled for the control.

Byte reduction is measured allocation size, not measured DRAM transactions.
GPU counters are not collected by this probe; `measured_dram_bytes` is null.
An isolated win can justify investigating C22, but cannot promote a pruned
runtime or count toward unchanged-model HTTP throughput targets. Borderline
timing gains require additional pairs rather than relying on five samples.

## Research Context

[SparseDecoding](https://arxiv.org/abs/2610.12327) motivates decode-calibrated
pruning and fixed-popcount bitmask traversal.
[MACKO-SpMV](https://proceedings.mlr.press/v306/macko26a.html) motivates measuring
sparse format/indexing overhead alongside payload reduction. Their reported
hardware results are not M40 evidence. This probe is an independent
implementation; no source code is copied from either project. Model pruning,
quality calibration and runtime integration are separate later tasks.

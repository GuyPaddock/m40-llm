# Sparse Q8 Projection Experiment

C21's conventional Maxwell sparse projection failed the decode speed gate.
See [the measured screen](experiments/2026-10-25-sparse-q8-screen.md).
The tested CUDA probe/tool is preserved in `6d02feb`, with results committed
in `51fa76b`, before separate removal of active code. Only the checked CPU
format/reconstruction remains. Nothing changes generation, KV policy or
loaded-model weights. C22 calibration and C27 sparse-Q4 are postponed.

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

## Archived CUDA Probe

This section describes the preservation revision `6d02feb`. The device API,
test target and benchmark below are not available after active-code removal;
reproduction requires an independent checkout of that revision.

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

GPU-less CI did not run this explicit hardware test. The retained CPU format
tests run with `cargo test --no-default-features --lib sparse_weights`.

## Archived Release Measurement

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

The first release attempt stopped on a down-projection `M=16` control mismatch.
An isolated `M=16` run passed, but running `M=1,4,16` reproduced it: sparse
`0.5484369` and scalar `0.54843678` versus native dense `0.5027627`, row 5,
column 0. The probe had not fenced its last pageable H2D input copy before
launching the nonblocking prefill stream. It now reads back and validates
the full input before numerical checks, outside timing. According to
[CUDA's copy synchronization contract](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/api-sync-behavior.html),
pageable H2D can return after staging but before final DMA; a D2H on the same
default stream completes after that upload. This is not a kernel change or a
claim that all existing runtime copy sites have been audited. Aborted runs
must not be combined with a complete qualified matrix.

## Research Context

[SparseDecoding](https://arxiv.org/abs/2610.12327) motivates decode-calibrated
pruning and fixed-popcount bitmask traversal.
[MACKO-SpMV](https://proceedings.mlr.press/v306/macko26a.html) motivates measuring
sparse format/indexing overhead alongside payload reduction. Their reported
hardware results are not M40 evidence. This probe is an independent
implementation; no source code is copied from either project. Model pruning,
quality calibration and runtime integration are separate later tasks.

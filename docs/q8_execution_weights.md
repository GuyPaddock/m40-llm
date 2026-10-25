# Q8 Execution-Weight Experiments

The API in `infer::execution_weights` owns lossless execution representations,
separate from `DeviceTensorView`, uploaded GGUF weights, and the model's FP32
materialization cache. C08-C10 established ownership and isolated measurements;
C10A.1 adds opt-in model-owned preparation and standard projection dispatch.
Preparation remains disabled by default. No generation speedup is claimed yet.

C09 now adds a lossless separate-scale/value repacker, bounded CUDA consumer,
and real-tensor release microbenchmark. See the
[measured results](experiments/2026-10-24-q8-separate-projections.md).
C10 independently implements lossless input-tiled storage and two bounded
CUDA mappings. See the [tiled experiment](experiments/2026-10-24-q8-input-tiled.md).
Its vector mapping is a per-family candidate; the coalesced-lane mapping was
slower. Those microbenchmarks alone do not qualify a runtime configuration.

## Opt-In Runtime Preparation

Set these before loading a model, not during generation:

```bash
M40LLM_Q8_EXECUTION_WEIGHTS=split
M40LLM_Q8_EXECUTION_BUDGET_MB=6144
```

`off` is the default and skips hashing, packing, and prepared allocations.
`split` prepares separate original FP16 scales and signed Q8 values for each
recognized Q/K/V/O, gate/up/down, and vocabulary projection. Tied output
embeddings are prepared once under their chosen tensor identity. The measured
single-token launch uses four columns/128 threads, except K/V use one column.
`mixed` adds width16 input-tiled Q/O/down copies for rows >=64; smaller rows
still use split planes. It does not globally replace the layout for every role.

`LoadedModel::from_gguf` reads this configuration once;
`from_gguf_with_q8_execution` accepts an explicit `Q8ExecutionConfig` so two
models can use different immutable caches in the same process. There is no
mutable global prepared-weight registry or on-demand decode repack.

Preparation currently requires an sm_52 CUDA device, uploaded source weights,
and at least one recognized rank-two Q8_0 projection with K divisible by 32.
Invalid modes, malformed eligible tensors, aliased source offsets, insufficient
budget, skipped upload, and allocation/upload errors fail clearly rather than
silently bypassing the requested cache. Non-Q8/unprepared projections retain
their existing implementation. The cache budget limits extra prepared payload,
not total process VRAM; the default 6144 MiB does not reserve that amount.

The cache remains separate from native GGUF storage. `split` adds another copy
of each prepared tensor's payload; `mixed` adds selected tiled copies too.
Model SHA-256 is calculated over the supplied original GGUF bytes with `ring`.
Keys retain that hash, tensor name, source offset, dtype, logical shape, and
execution layout. A model-local native-pointer lookup finds those immutable
owners; pointers are not used as cross-model cache identities.

`q8_execution_stats()` exposes estimated/actual prepared bytes, budget,
source uploaded bytes, total source+prepared device payload, per-tensor/layout
keys and bytes, and hash/preparation milliseconds. `peak_setup_host_payload_bytes`
estimates the original file + retained tensor-data blob + largest temporary
packed tensor. These payload estimates exclude allocator/page overhead,
metadata, CUDA contexts, workspace, and KV. Source upload remains part of cold
model loading, outside the separately reported preparation/hash time.

Standard sync/async projection APIs consume the cache without per-call
allocation, upload, or packing. Enqueue explicitly uses the existing producer
stream policy: multi-row work uses prefill; single-row uses decode only when
`M40LLM_Q8_DECODE_STREAM=decode`, otherwise prefill. The sync API waits on that
actual producer; the async API does not synchronize.

`M40LLM_FUSED_Q8_QKV=1` now lets split/mixed caches supply a single fused
Q/K/V launch. Query tiles use four columns, K/V tiles one, all with 128 threads
and the original optional FP32 biases. The launch preserves the existing Q8
producer-stream policy and reuses the split projection's dot/reduction code.
All three prepared tensors must be present and shape-compatible; missing cache
entries retain native fused QKV, while invalid cached shapes fail before launch.
Multi-row calls still use the standard projection path.

`M40LLM_FUSED_Q8_MLP_SWIGLU=1` also lets prepared caches supply fused gate/up
and SwiGLU. This keeps one output column per 128-thread block, reuses each
activation vector across both operands, and reads aligned split value planes.
It deliberately does not retry the earlier failed four-column fused MLP
experiment. Split accumulation/reduction is shared with standard projections
and QKV; no intermediate gate/up allocation or extra activation launch is added.
Both weights must be cached and shape-compatible; missing cache entries retain
the native fused path.

The specialized greedy vocabulary path remains native until its C10A.2
integration is ready. Use of the cache is experimental, not a claim that the
current selected fused HTTP benchmark is faster.

## Source and Identity

`Q8SourceIdentity` includes the caller-supplied 32-byte model SHA-256, tensor
name, original byte offset, Q8_0 dtype, and logical `[K,N]` shape. It does not
contain a transient device pointer. `Q8ExecutionKey` adds the execution layout
and tile width, so different models, tensors, shapes, offsets, or layouts do
not alias. Callers compute/verify real model hashes; synthetic fixtures use
explicit synthetic identities rather than claiming a file hash was measured.

`Q8SourceView` borrows exactly one tensor from a **tensor-data-relative weights
blob**, not the complete GGUF file. Checked offset, element, block, and byte
arithmetic rejects overflow/out-of-range views before allocation. Source bytes
are immutable and never replaced by an execution representation.

For real GGUF tensors, `from_gguf_tensor` requires rank two, Q8_0, nonzero
dimensions, and complete 32-element input blocks. Synthetic ragged-K tests
must explicitly use `from_column_blocks`; each output column then stores
`ceil(K/32)` complete 34-byte blocks, including all inactive tail bytes.
Do not size that synthetic format as `ceil(N*K/32) * 34`, which can undercount
per-column padding and mix adjacent columns.

The original source offset is identity/bounds metadata. A copied execution
buffer starts at its own allocation base; do **not** add that source offset
again when invoking a projection.

## Layout and Memory Contracts

Each Q8_0 block has the original two FP16 scale bytes and 32 signed-value
bytes. C08 copies existing column blocks verbatim. C09 implements separate
planes, and C10 implements the input-tiled representation:

| Layout | Owned Planes | Ordering |
| --- | --- | --- |
| `ColumnBlocks` | One block plane | Original `[column][input_block][34 bytes]` |
| `SeparateScalesValues` | FP16 scale bytes; signed-value bytes | Both column/block major; values retain all 32 lanes |
| `InputTiled { columns }` | One record plane | `[input_block][output_tile]`, then scale header and `[quant_index][output_column]` values |

Input-tile widths must be powers of two from 1 through 32. Each record places
`columns * 2` scale bytes before a 16-byte-aligned value region, followed by
`columns * 32` value bytes. Output tiles pad to the chosen width. The allocation
plan exposes tile count, value offset, record bytes, per-plane bytes, original
source bytes, and total/padding bytes; all size arithmetic is checked.

For `K=35,N=3`, original padded column blocks occupy 204 bytes, split planes
12+192 bytes, and four-column tiled records 288 bytes. For a Qwen gate/up
matrix `K=2048,N=11008`, original and eight-column-tiled payloads both require
23,953,408 bytes. These are format calculations, not measured memory traffic.
`resident_bytes` reports requested device payload bytes, not allocator/page
granularity, CUDA context overhead, or total process VRAM.

`HostQ8ExecutionWeight::tile_inputs` copies all original payload bytes and scale
bits into those records, zeroing only alignment and padded output columns.
`Q8TiledLaunch` permits widths4/8/16, threads128/256, and vector/coalesced-lane
access. Storage widths1/2/32 remain valid descriptors/host packing choices but
are not CUDA launch choices in this experiment. The requested launch width
must match the owning execution layout. `project_tiled_async` enqueues on the
decode stream after validating dimensions, buffer sizes, and non-aliasing.

## Ownership and Use

`HostQ8ExecutionWeight::new` verifies plane counts/sizes against its key before
GPU allocation. It does not validate a plane's encoded values; C09/C10 packers
must prove lossless scale/value mappings. C09's `separate_scales_values` copies
all scale/payload bytes unchanged, and `from_tensor_bytes` accepts an already
sliced tensor while preserving its original source offset as identity.
`DeviceQ8ExecutionWeight::upload`
synchronously uploads planes into immediately RAII-owned `DeviceBuffer`s.
Partial allocation or upload failures release both prior and uncommitted
buffers. The device owner outlives the source/host planes, and its borrowed
view distinguishes original, split, and tiled buffers by enum variant.

```rust
use m40_llm::infer::execution_weights::{
    DeviceQ8ExecutionView, DeviceQ8ExecutionWeight, HostQ8ExecutionWeight,
    Q8SourceIdentity, Q8SourceView,
};

// model_sha256, tensor, weights_blob, and ctx are supplied by the benchmark.
let identity = Q8SourceIdentity::from_gguf_tensor(model_sha256, tensor)?;
let source = Q8SourceView::new(&identity, weights_blob)?;
let host = HostQ8ExecutionWeight::copy_column_blocks(source)?;
let device = DeviceQ8ExecutionWeight::upload(&ctx, &host)?;
match device.view() {
    DeviceQ8ExecutionView::ColumnBlocks { blocks, .. } => {
        // Use blocks.as_ptr() with the existing unsafe Q8 projection API.
        // Keep device alive until the consuming stream completes.
    }
    _ => unreachable!(),
}
```

The standalone owner API above is also used by the opt-in runtime cache.
It does not reuse FP32 materialization buffers. Preparation time and duplicate
resident payload must be included in any later runtime report.

## Validation and Next Work

C10A.1 runtime tests cover all eight projection roles at rows 1/4/64 against
native GPU and CPU references, per-family split/tiled selection, both producer
streams, sync/async behavior, distinct model hashes/lifetimes, unmodified source
bytes, and budget/unsupported/skipped-upload failures. A later-tensor allocation
failure releases prior prepared planes while preserving uploaded source data.
Invalid aliased/out-of-bounds/shape plans never reach the upload boundary.
The `81948df` prerequisite corrects native-weight accounting on final context
drop; the native buffer was already physically freed before that correction.

The real Qwen2.5-3B Q8_0 split-cache canary answers `4`, captures finite prompt
logits, and records 5,313 prepared projection launches. It prepares 253 tensor
layouts: 3,278,553,088 extra bytes alongside 3,279,519,744 source-weight bytes,
for 6,558,072,832 weight payload bytes before KV/workspace. Hash plus preparation
takes 30.512 s in this **debug** correctness run, not a release performance
claim. See the [raw canary log](experiments/2026-10-24-q8-prepared-split-canary.txt).
CUDA layout/attention/forward/projection/MLP checks pass, including cuBLAS-off
grid/wrappers; non-CUDA warning-as-error tests and both Clippy feature lanes pass.

Prepared fused QKV tests cover equal/GQA-shaped/tail widths, no/partial/all
biases, split and mixed caches, both producer streams, CPU/native GPU parity,
and output guards. Each invocation remains one launch with no internal sync,
allocation, free, H2D, or D2H. Invalid cached dimensions, multi-row requests,
and a disabled fusion flag do not launch this consumer.

The real Qwen2.5-3B split-cache canary with prepared QKV and native fused MLP
also answers `4`, with finite prompt logits and **756 prepared fused-QKV
launches** over 21 forward tokens and 36 layers. The canary asserts that
requested prepared fusion actually executes. This debug correctness run is
not a controlled timing comparison; see its [raw log](experiments/2026-10-24-q8-prepared-qkv-canary.txt).
It uses `M40LLM_Q8_EXECUTION_WEIGHTS=split`, decode-stream Q8 scheduling,
`M40LLM_FUSED_Q8_QKV=1`, `M40LLM_FUSED_Q8_MLP_SWIGLU=1`, and the native
Q8 decode/MLP split-qblock settings of 4. The KV cache is explicitly dense,
the context bound 256, and the Rust test profile is unoptimized.

Prepared MLP tests use distinct gate/up payloads and cover hidden widths
1/31/65/128, positive/negative input amplitudes, both streams, both cache modes,
CPU/native parity, output guards, and rejected invalid/non-decode calls.
Synthetic tensor payloads are now distinct by tensor so accidental operand
swaps cannot pass QKV/MLP tests merely because fixture weights were identical.

The full prepared-QKV/MLP Qwen canary answers `4`, with 756 launches of each
prepared fused consumer and finite prompt logits. See the
[raw MLP canary](experiments/2026-10-24-q8-prepared-mlp-canary.txt). Its debug
generation total is 15.637 s versus 10.698 s in the earlier native-MLP canary.
These are unpaired debug correctness runs, not a release promotion comparison,
but the increase is a reason to keep native fused MLP as an alternative in
C10A.3. Do not infer that preparation improves every fused consumer. The later
release gate must measure per-family choices without another broad tuning round.

C08 validation passes eight CPU tests for semantic identities, source bounds,
real/synthetic shapes, plan overflow/alignment, byte copies, and failing
allocation/upload boundaries using a working RAII fake. A real CUDA test
forces the second plane's allocation to fail after uploading the first, then
checks one allocation/free and restoration of tracked bytes. The prerequisite
`0750d85` fixes a pre-existing double-lock in allocator error reporting so
that failure can return instead of hanging.

Two CUDA integration tests round-trip every layout's planes, drop host storage,
check tracked cleanup, and run a real `M=2,N=3,K=35` projection from an owned
column-block view with a nonzero original source offset. Non-CUDA warning-as-error
tests, CUDA/server and non-CUDA/server Clippy, and formatting pass. No model
inference timing is claimed for this ownership-only change.

Run through the project environment with hardware access for CUDA tests:

```bash
source scripts/dev-env.sh
RUSTFLAGS='-D warnings' cargo test --no-default-features --locked
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --lib \
  infer::execution_weights -- --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test q8_execution_weights --test cuda_allocation_errors \
  -- --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test q8_runtime_weights -- --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test q8_prepared_qkv -- --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test q8_prepared_mlp -- --nocapture --test-threads=1
```

Next: finish prepared greedy vocabulary integration, model-level parity, and repeated HTTP
gates. C09/C10 isolated results do not prove whole-model wins or justify
replacing useful fused operations. Keep setup,
duplicate resident weights, and peak payload accounting explicit.
The selected exact-attention
benchmark configuration remains split8/256 from C07; global runtime/KV defaults
are unchanged, and neither unchanged-model throughput target has been reached.

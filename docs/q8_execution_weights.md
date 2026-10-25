# Q8 Execution-Weight Experiments

The C08 API in `infer::execution_weights` owns **benchmark-only** execution
representations. It is separate from `DeviceTensorView`, uploaded GGUF weights,
and the model's FP32 materialization cache. Ordinary generation does not call
this API, and no runtime/default layout selection or model cache is added.

C09 now adds a lossless separate-scale/value repacker, bounded CUDA consumer,
and real-tensor release microbenchmark. See the
[measured results](experiments/2026-10-24-q8-separate-projections.md).
C10 will independently test input-tiled storage; its descriptor alone is not
a kernel speed or quality result.

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
bytes. C08 copies existing column blocks verbatim. C09 implements the separate
planes; input-tiled storage is still a descriptor for the independent experiment:

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

There is no process-global prepared-weight registry, on-demand decode repack,
loader hook, or automatic FP32 cache reuse. Preparation time, peak/resident
memory, and projection timings belong in the later experiment reports.

## Validation and Next Work

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
```

Next: C10 uses these owners independently for input-tiled lossless repacking,
projection parity, and real-shape isolated measurements. It does not depend
on C09 winning. Only measured winners proceed to C10A loader/
model-cache integration and repeated HTTP gates. The selected exact-attention
benchmark configuration remains split8/256 from C07; global runtime/KV defaults
are unchanged, and neither unchanged-model throughput target has been reached.

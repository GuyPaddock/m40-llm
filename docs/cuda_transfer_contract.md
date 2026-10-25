# CUDA Transfer Contract

## Host Uploads

`CudaContext::memcpy_h2d` copies through the CUDA default stream and waits
for that transfer to complete before returning. On success the destination
can be consumed immediately on either nonblocking project stream. The host
source may then be changed or released. The completion wait is recorded as
`memcpy_h2d_completion`, separate from the H2D byte/copy count.

This is not a device-wide ownership fence. Before overwriting a device buffer,
callers must finish its prior nonblocking producers and consumers. The upload
does not make concurrent reads/writes to the same destination safe. Likewise,
`memcpy_d2h` requires the source's producer to finish before readback; a copy
through the default stream does not order an unrelated nonblocking producer.
Async D2D, kernels and cuBLAS still require their normal same-stream ordering
or explicit event dependencies.

The [CUDA API synchronization contract](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/api-sync-behavior.html)
allows pageable H2D to return once staging finishes, before final DMA. The
[stream contract](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/stream-sync-behavior.html)
excludes nonblocking streams from implicit legacy-default synchronization.
Relying on the old unfenced copy therefore left both consumer streams at risk.
The explicit wait finishes only default-stream work, not all device work.

## Regression

At pre-fix runtime revision `3818104`, the new immediate-consumer test fails
on M40 at rows=16, K=11008, iteration=1, index=76032: decode reads 0 instead
of 0.022460938. No sparse projection is involved. Fixtures are fenced once
before the repeated-copy scope; each subsequent upload is consumed directly
by decode residual-add and native Q8 prefill, without a readback/allocation
barrier hiding the race. Repeated rows change values and reuse the same device
allocation. Small uploads, large uploads and both streams are covered.

```bash
source scripts/dev-env.sh
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test cuda_transfer_order -- \
  --include-ignored --test-threads=1
```

The hardware test is explicitly ignored by GPU-less CI; skipping it is not a
GPU correctness result. Tests use producer drains before readback and reuse.

## Audit Scope

The central upload covers Rust embedding fallbacks, packed/batched metadata,
prepared Q8 weights, graph parameters and host forward fallbacks. Its
completion rule does not authorize changing graph or KV defaults.

Native weight upload now uses the same completion contract, including the
pageable fallback when host registration is unavailable. It selects the
owning device, publishes a pointer only on success and releases successful
host registrations on allocation failure. The wait is counted as
`upload_weights_completion`.

Dense and compressed KV constructors initialize all allocated buffers through
one checked fill list and drain the default stream before publishing storage.
Optional dense shadows, q8/q4 backing, recent rings, summary accumulators,
counts and representatives retain their prior zero values; representative
positions retain the invalid `u32::MAX` sentinel. Initialization failure
destroys partial ownership rather than returning a usable cache. The setup
wait is counted as `kvcache_initialization`, not a hot attention-kernel wait.
Immediate first append/attention tests cover dense head64/head128, head64
summary/q8 caches, and head64/head128 mixed FP16-K/q4-V, without a reset or
readback before consumption. Metadata and an unused sequence stay unchanged.

Legacy length-aware append now drains its decode producer before copying the
current length to host. Plain FP32 and rotary FP32 kernels publish the next
length directly by-value with their K/V stores, removing asynchronous copies
from stack-owned counters. Plain FP32 uses scalar FP16 stores for odd widths
and 2-byte-aligned token slots. FP16 append finishes its default-stream
K/V/length copies through the central upload fence. Compressed caches reject
these legacy dense APIs; their explicit-position path remains unchanged.
The legacy read wait and four-byte D2H are counted explicitly. FP16 copy-only
append no longer reports a fictional kernel launch.

Production explicit-position append still publishes length on device with
no host copy, allocation, event wait or stream synchronization. Hardware
regressions verify that property at head64/head128 and test back-to-back
legacy appends behind a decode backlog, including odd widths, overflow and
rotary position distinct from the cache slot. Callers still must finish
input/cache producers on other streams and drain before readback or reuse.

The audited native H2D operations all go through the completed upload;
default-stream KV memsets share the checked constructor publication fence.
Runtime batch/prefill host metadata is uploaded through the central method.
Async reset/reconstruction is scheduled and drained on decode. Debug/selection
readbacks drain decode; async device-to-device copies and cross-stream kernel
consumers retain their caller-side dependency requirements. This is not a
claim that arbitrary external raw-pointer misuse is safe.

Benchmark timing impact must be measured separately; these correctness changes
are not throughput optimizations.

## Validation Fixture Follow-Up

The broad cuBLAS-off forward run exposed four pre-existing fast-fits graph
fixture failures. An isolated worktree at pre-fix `3818104` reproduces the
same `20 passed / 4 failed` result and synchronous F16 GEMM capture error.
The [baseline log](experiments/2026-10-25-upload-baseline-graph-off.txt) is
retained. Running the raw forward graph case alone also exposed its reliance
on prior tests selecting dense KV for head_dim=4.

These four fixtures now explicitly scope dense reference KV and require
compiled cuBLAS, because their materialized-FP32 graph uses async cuBLAS
GEMMs. They are hardware-ignored by default and fail if explicitly requested
without M40. This does not restrict all graph backends or claim CUDA graphs
generally require cuBLAS. Other async kernels can be captured independently.
Runtime graph defaults, fallback behavior and capture coverage are unchanged.

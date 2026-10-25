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

Legacy append metadata and explicit async producer dependencies remain under
audit. Benchmark timing impact must be measured; these correctness changes
are not throughput optimizations.

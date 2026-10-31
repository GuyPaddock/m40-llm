# Host Greedy-Token Adapter Cost

The qualified GPU-greedy path in `generate_text` reads a selected token,
allocates `token_id + 1` FP32 logits initialized to negative infinity,
sets the selected index to zero and invokes the CPU greedy sampler. This
recovers an already-selected token through an unnecessary dense array.
It is concrete host work, but not established scope for a useful E2E gain.

Measure this adapter before changing runtime behavior. Reuse the existing
Qwen throughput test target for one ignored CPU-only release diagnostic.
Use only the explicit, checksum-pinned qualified actual512 token trace;
do not load a model, create a CUDA context or scan/download model caches.
Run with CUDA devices hidden. Replay the real IDs rather than repeating
only worst-case vocabulary indices.

Compare the current synthetic-logits construction and real `Sampler::sample`
against direct token forwarding. Both must return every input ID unchanged,
including token zero and high vocabulary IDs in behavioral tests. Keep
five alternating pairs and 256 complete-trace repetitions per arm, fixed
before timing, with untimed warmup. Black-box inputs/results and verify
checksums to prevent removal of measured work. Record raw host wall times,
source/build/input identities, requested vector payload and tracked device
bytes. Requested payload is not measured allocator traffic or peak memory.

This is isolated CPU cost, not a GPU/HTTP benchmark or a rigorous inference
latency bound. Compare its observed per-trace cost with the qualified HTTP
interval only as screening context; CPU/GPU work can overlap. If observed
adapter cost is below the unchanged 3% E2E floor, do not advance this as a
performance candidate or combine rejected kernels to manufacture scope.
An unexpectedly large cost still needs typed runtime integration, unchanged
actual512/frozen quality and fresh paired HTTP qualification before retention.

No FFN/projection/prefill, sampling distribution, stop, scheduler, graph,
KV or runtime-default changes. No performance claim follows from fewer
allocations alone. Both original exact speed requirements remain unmet.

## Atomic Delivery

1. `docs(research): bound greedy adapter study`: this contract.
2. `test(bench): measure greedy adapter CPU cost`: explicit-trace diagnostic,
   observable behavioral tests and bounded raw reporting; depends on 1.
3. `docs(perf): record greedy adapter scope`: clean release measurements,
   independent audit and next-action decision; depends on tested 2.

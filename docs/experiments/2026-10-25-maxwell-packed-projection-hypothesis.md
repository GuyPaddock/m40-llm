# Packed Loads In Qualified Native QKV And Output

## Evidence And Fixed Scope

The [qualified runtime profile](2026-10-25-maxwell-projection-stage-profile.md)
places **9.24%** of middle-token GPU work in QKV and **6.64%** in output
projection. The [compiled-code follow-up](2026-10-25-maxwell-byte-ffn-projection-followup.sass.gz)
confirms eight U8 weight loads and eight scalar FP32 activation loads per
eight-value task. The original-tree warp reductions are already qualified;
they did not vectorize these source accesses. These are static instruction
observations, not cache, bandwidth, occupancy or stall measurements.

Run two independent fixed load-vectorization experiments:

- QKV: one output column per block, 256 threads, original split4/eight-value
  assignment, original-tree warp finish and original optional FP32 biases.
- Output: four columns per block, 256 threads, the same per-column task and
  reduction association, original odd-column guards and shared activations.

For each task load two aligned float4 activation vectors. Read its eight
raw Q8_0 weights through four uint16 pairs, then use the original signed-byte
conversion and scale/FMA order. Q8_0 blocks have 34-byte strides and only
two-byte payload alignment; do not reinterpret them as uint32 vectors.
Explicit APIs require 16-byte activation alignment and fail before launch
when invalid. Keep raw source storage, FP16 scales, streams, scratch and
qualified FFN/down/vocabulary unchanged. No additional prepared weights,
activation quantization, cache hints, conversion tricks or geometry search.

Inspect sm_52 code and register/spill reports before timing. Stop a stage
if the intended load instructions do not change. Preserve controls' SASS.
Behavioral tests must retain original control bits and CPU tolerance with
two-byte-only weight bases, zero scales, signed extremes, width/column tails,
optional biases, both streams, ordered producers, guards, invalid alignment,
scratch reuse, no enqueue side effects and RAII cleanup.

## Measurements And Gates

Extend the existing projection benchmark, with explicit comparison roles:
qualified warp control versus packed-load warp candidate. Retain old
shared-versus-warp comparisons as separate named modes, not an implicit
control switch. Use frozen M02 normalized FFN inputs as projection
surrogates, layers 0/17/35, one/four distinct requests, five alternating
pairs and ten iterations. Four requests remain independent M=1 launches,
not a batched GEMM or four copies of one input. Include the entire QKV or
output projection and required producer dependencies. Count actual dispatch
and verify no timed allocations/transfers/synchronizations. Setup and CPU
references stay outside event timing; serialize all GPU work.

Each stage needs >=5% throughput gain and a positive paired 95% interval
at all three one-request cases. Borderline results require twenty fresh
pairs, not a geometry rescue. Use current profile fractions, not the older
pre-warp profile, for optimistic scope. Only qualifying stages whose
combined optimistic E2E gain reaches 3% may proceed to runtime work. Do not
pool below-gate stages or rejected FFN variants to bypass this boundary.
Actual-512/frozen-quality, incremental HTTP and fresh official Ollama gates
remain mandatory before recommending any runtime change. No defaults change.

## Atomic Delivery

1. `docs(perf): scope packed projection loads`: this fixed experiment.
2. `feat(cuda): probe packed Q8 projection loads`: kernels, explicit
   alignment contracts, shared-runner comparison roles and behavioral tests.
3. `docs(perf): record packed projection screens`: clean release provenance,
   both independent screens, instructions, raw timing/quality/memory audits
   and the measured decision.

The current qualified **47.379 HTTP tok/s** still misses both original
goals, **53.06** and **56.026** for 1.30x the fresh Ollama median. Kernel-only
numbers, altered arithmetic, sparsity or scheduling cannot replace them.

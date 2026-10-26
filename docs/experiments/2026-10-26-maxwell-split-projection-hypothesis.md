# Split-Plane Attention Projection Hypothesis

## Evidence And Scope

Clean source `12ed46b` retains qualified raw-GGUF packed/warp QKV and
output projections. The latest selected-runtime profile assigns them
7.09% and 5.16% of sampled middle-token GPU time. FFN still dominates,
but its closed reuse, virtual-warp, joint-layout and other probes do not
justify more tuning of those candidates.

`cuda/q8_projection_packed.cuh::accumulate_eight` loads four U16 payloads
per eight-value task. Original GGUF Q8_0 records are 34 bytes and payload
starts two bytes after each record, so generic eight-byte alignment cannot
be assumed. The [qualified compiled entries](2026-10-25-maxwell-packed-projection-candidates.sass.gz)
confirm four `LDG.E.CI.U16` payload sites per task/column plus the scale
load. QKV retains two 128-bit activation loads. Output reuses those
activations across four columns but still has four payload loads per column.

The project's existing `SeparateScalesValues` execution layout preserves
the original FP16 scale bits and signed bytes in separately aligned planes.
Its eight-byte payload loads are already used by prepared down/vocabulary.
The missing experiment is that storage layout on the *current* qualified
QKV/output task mapping, not another change to FFN or the down reduction.
Fewer load/address instructions might help; there is no measured bandwidth
or occupancy attribution yet.

## One Fixed Candidate Per Projection

Prepare QKV/output using existing typed, owned split-plane weights. Preserve
all 109 qualified gate/up/down/vocabulary identities in any eventual runtime
integration. The screen holds its 37 prepared down/vocabulary owners fixed,
as in previous projection screens. Both timing arms retain the same extra
candidate planes. No new format or lossy conversion is introduced.

Replace four U16 payload loads with one aligned `uint2` per eight values,
and read the original scale from its separate plane. Keep two `float4`
activation loads, original signed conversion, per-weight FP32 scale/FMA
association, task stride 256, publication, original-tree warp finish and
bias addition. QKV keeps one column per block; output keeps four columns.
Both use 256 threads and the same 1024/4096 shared bytes. No other geometry,
cache hint, register limit, scale factoring or reduction variant is allowed.

The Qwen layer adds 5,570,560 prepared QKV bytes or 4,456,448 output bytes
in its respective screen. A full-model combination would add 360,972,288
bytes if original GGUF storage stays resident; this is an estimate, not a
measured runtime allocation. Include preparation time and actual allocations
outside timed projection work, with a 128 MiB per-case bound and RAII cleanup.

The consumer is benchmark-only initially. It must reject missing/misaligned
planes, incomplete Q8 blocks, invalid widths/streams and aliases before
launch. No per-call allocations, copies, synchronization or silent fallback.

## Measurements And Gates

Extend the existing Q8 projection benchmark with explicit split comparisons
against the qualified *packed/warp* controls, not the older scalar/shared
kernels. Use frozen M02 layer 0/17/35 tensors and normalized-input surrogates.
Report their surrogate limitation. Rows 1/4 are distinct independent decode
requests, one launch per row, not batched GEMM or repeated identical inputs.

Run five alternating pairs per shape with bounded warmup and ten iterations.
Require exact control bits, CPU tolerance, guards, immutable raw and prepared
weights, ordered producers on both streams and cleanup. Inspect sm_52 loads,
registers and stack/local storage before timing; preserve control entries.

Each independent projection requires at least 5% complete-projection gain
with a positive paired interval for every one-request layer case. Four-request
results are separate. Use the latest profile's actual stage fractions for
an optimistic feasibility estimate; passing stages together must clear 3%
before runtime integration. That estimate is not an HTTP gain. Borderline
findings follow the existing fresh-confirmation rule, not a parameter search.

Only qualified primitives may receive model-local runtime wiring, actual
512-ID/frozen quality checks, unchanged-identity verification, and matched
incremental/fresh official Ollama HTTP qualification. A losing candidate
stays committed with its measurements. No combination with rejected probes.

## Atomic Delivery

1. Commit this source-backed hypothesis before implementation.
2. Commit benchmark-only split consumers and behavioral CUDA tests after
   verification. No runtime selector or defaults change.
3. Commit the existing benchmark extension and provenance/dispatch/memory
   checks before release timing.
4. Commit paired results, compiled evidence, audit and measured decision.
5. Only if gates pass, use separate runtime, quality and HTTP commits.

Both original goals remain **53.06 HTTP tok/s** and **1.30x fresh official
Ollama**, currently **56.037**. No activation quantization, approximate
attention, sparsity, pruning, speculation or scheduling substitute is used.

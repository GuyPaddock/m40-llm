# Fixed Four-Row Joint-Q8 Operator Experiment

The user approved the fixed probe from the
[feasibility assessment](2026-10-27-direct-q8-feasibility.md). This is an
operator experiment, not authorization for a target verifier or drafter.

## Frozen Contract

- Four distinct captured FP32 rows, D=2048, H=11008, layers 0/17/35.
- Existing joint-packed gate/up sources; one 128-thread CTA per output column.
  Share payload loads and original FP32 scales across independent row
  accumulation chains. Preserve thread/task assignment, FP32 association,
  reduction tree and precise SwiGLU. No activation quantization.
- Three contemporaneous arms: four qualified one-row compound calls;
  existing independent-row four-row compound; fixed shared-read candidate.
  Every complete FFN uses four original one-row prepared-down calls, with the
  qualified four-column/256-thread launch, on the decode stream.
- Candidate limits: at most 64 registers/thread, 4608 shared bytes, no stack,
  local storage or spills. No hot-path allocation/free/copy/upload/sync.
- Require exact gate/up/SwiGLU/down bits, finite values, canaries, unchanged
  inputs, source/prepared identities, allocation accounting and zero teardown.
  All 137 original compiled CUDA controls must retain their instruction words.
- Preserve the immutable 50.910426 HTTP tok/s checkpoint, preferred joint-only
  compound ownership and runtime defaults. A benchmark-local pair is the
  only extra weight owner; no redundant split gate/up representation is needed.

## Measurement And Stop Rules

After correctness/resource gates, run a clean release screen with ten repeats
per stage and five rotating/reversed contemporaneous trials. Warm each arm for
at least 1000 ms of GPU work; record isolated gate/up, isolated original down,
and complete FFN with CUDA events and host enqueue/completion time. Snapshot
counters and GPU isolation/clock state outside intervals. No other GPU work,
builds, tests or MemPalace calls overlap primary measurement.

Use the existing 10,000-resample seed-zero paired bootstrap and throughput
MAD/median. Over 5% variation extends once to ten trials; continued instability
blocks a conclusion. The candidate must improve complete-FFN throughput by at
least 5%, with positive paired confidence bounds, at every layer against both
controls. A point estimate within one percentage point of the gate or a
qualifying estimate whose interval crosses zero requires exactly twenty fresh
confirmation trials, never pooled with screening. No geometry/parameter rescue.

Stop on a numerical, resource, ownership, isolation or performance failure.
Preserve failed code and results before any separately committed removal.
No automatic removal or new search is required by this task.

If all gates pass, update the separate full-verifier model with measured
gate/up cost. Charge QKV/output/down, any remaining conversion/SGEMM,
causal attention, vocabulary/argmax, transfer/dependency/workspace and KV
commit/rollback/correction costs. Include non-perfect acceptance, drafter
cost and finite serial tails. Passing this microbenchmark alone cannot
establish a fast full verifier. Stop after the revised assessment for review.

## Atomic Delivery

1. `docs(research): freeze four-row operator gate` records this contract,
   dependencies and MemPalace checkpoint before changing CUDA.
2. `feat(cuda): probe fixed four-row Q8 reuse` adds exactly one consumer,
   guarded research-only enqueue API, isolated benchmark entrypoint and
   behavioral correctness tests. Existing dispatch/defaults stay untouched.
3. `test(research): verify four-row Q8 contracts` records compiled-control,
   resource, numerical, ownership, cuBLAS-on/off and regression evidence.
   A failed hard gate closes the experiment before timing.
4. `docs(research): decide four-row Q8 reuse` preserves clean-source release
   measurements, independent audit and the gate decision. Only a passing
   operator proceeds to a measured-cost feasibility update in this unit.

Each unit depends on the preceding one. Do not push, rewrite measured history,
download models, change defaults or implement generation integration.

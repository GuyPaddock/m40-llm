# Shared-Scale Gate/Up FFN Study

## Independent Cost Hypothesis

Qualified split-f32x4 gate/up accounts for 41.11% of sampled middle-token
GPU time. Each Q8 scale is loaded and converted to FP32 eight times per
CTA, once per four-value task. Test loading and converting each scale once
into shared FP32 memory. This is not scale factoring: every weight retains
the original scale multiplication and activation FMA association.

Source searches found no prior shared-scale staging experiment. Existing
cache-hint, byte conversion, scale-factoring, task and column experiments
remain closed. Global caches may already serve repeated scale loads; no
DRAM saving, occupancy gain or explanation of Ollama performance is assumed.

## Fixed Scope

- One output per 128-thread CTA, N CTAs per row, original task order,
  float4 activations, u32 weight words and original split weight planes.
- Positive K divisible by 32, K at most 4096, arbitrary positive N.
  One thread loads one gate/up scale pair when tid is below K/32.
- Dynamic shared storage is 1024 + 8*(K/32) bytes: original gate/up
  partials followed by two FP32 scale arrays. This is 1536 bytes at K=2048.
- One barrier publishes scales before the unchanged accumulation loop.
  The original partial-publication barrier and 128-partial reduction remain.
  There are two block barriers instead of one.
- Preserve precise SiLU, prepared down/vocabulary, stream ownership, typed
  buffers and all runtime/KV defaults. No new global allocation, weight
  layout, launch, transfer or host synchronization in the FFN path.
- One fixed candidate only. No tile, geometry, staging-depth, register,
  cache-hint or unroll rescue; do not combine rejected experiments.

## Validation And Decision

Require bit-identical gate, up, SwiGLU and complete FFN outputs, plus the
existing independent CPU tolerances. Cover dynamic K, odd N, distinct rows,
signed extremes, zero/negative-zero scales, capture parity, guards,
producer ordering, ownership, reuse, cleanup and invalid spans/K rejection.
Run affected CUDA tests with cuBLAS on/off, attention/forward/projection/MLP
regressions, warning-as-error non-CUDA checks, Clippy and formatting.

Commit tested code before clean release timing. Inspect sm_52 instructions
and resources and verify original control/prepared instruction words.
Use frozen M02 real layers 0/17/35, genuine rows 1/4, five alternating
pairs and ten iterations per stage in the existing shared FFN benchmark.
Require at least 5% complete one-row FFN improvement with positive paired
lower bounds at all three layers. Apply existing fresh confirmation rules
without pooling. Four-row results cannot bypass the one-row gate.

Only a qualifying microbenchmark and independently supported E2E scope
authorize separate runtime selection, actual512/frozen dense-valid quality
and controlled HTTP qualification. Both original 53.06 HTTP tok/s and 1.30x
fresh official Ollama targets remain required. No default changes.

## Atomic Delivery

1. `docs(research): bound shared-scale FFN study`: this contract.
2. `feat(cuda): probe shared FFN scale reuse`: primitive, shared benchmark
   and behavioral tests, dependent on 1.
3. `docs(perf): record shared-scale FFN screen`: release evidence,
   resources, audited results and keep/close decision, dependent on 2.
4. Only if qualified, plan runtime, quality and HTTP delivery separately.
   Preserve failed code before any later removal.

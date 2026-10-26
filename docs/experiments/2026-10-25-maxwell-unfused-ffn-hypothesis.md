# Unfused Prepared Gate/Up Hypothesis

## Evidence Before Implementation

The qualified `gate_up_split_f32x4` still uses 32 registers with no reported
stack/local storage. Its release SASS dump from executable
`6aab5b02fcf77ba3efb259589526d23fe3ced73a53480a75eb7a7fbdb4ab9f75`
has SHA-256 `7c2d3c8e46c37367a7ce14d14d865ea5a6476b4ab5baf4e2578b45adfa182b22`.
The first task's five loads and conversion/scaling/FMA chain remain as in
the [earlier instruction audit](2026-10-25-maxwell-joint-ffn-audit.md).
Later code already overlaps next-task loads at `07f8`, `0818`, `0850`,
`0870` and `0888` with current-task FMAs at `0838`, `0868`, `0878` and
`08b0`. A generic software-prefetch claim therefore does not identify new
missing work. Static instruction counts are not stall or DRAM measurements.

Pinned llama.cpp `9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e`
`ggml-cuda.cu:1907` rejects MMVQ fusion on Pascal and older. The
[official launch observation](2026-10-25-maxwell-q8-dispatch.md) confirms
unfused MMVQ on M40, but that path also quantizes activations. It does not
prove that unfusing our FP32-activation arithmetic will help.

This motivates one **decomposition** comparison, not another fused-column
search. Test separate gate and up projections followed by SwiGLU, allowing
each projection to reuse one activation vector across four output columns.
Keep four-value tasks, 128 threads, existing separate scale/value planes and
the qualified per-output accumulation/reduction tree. Unlike the failed
two-column fused candidate, each projection traverses one weight matrix and
has no second projection's pointer/scaling dependency chain. Both strategies
can have four live accumulators; no register or occupancy advantage is
assumed. Extra launches and intermediate traffic may outweigh any benefit.

## One Fixed Candidate

Use exactly four output columns per projection block and 128 threads.
Run gate, then up, then SwiGLU on the existing decode stream. Reuse the
existing gate/up scratch in `FfnPipeline`; no new layout, weight copies,
device scratch or global runtime selection. Genuine rows 1/4 are separate
measurements. Do not vary column count, block size, task width or layout if
this candidate loses. Do not combine the below-gate joint packing, fixed-K,
two-column fused or prepared-down warp experiments.

Every per-output FP32 product/FMA and the reduction association must match
the qualified one-column split kernel. Preserve its SwiGLU expression
`g * (1 / (1 + exp(-g))) * u` as well. The general existing elementwise
kernel uses `g / (1 + exp(-g)) * u`, so naively reusing it could introduce
rounding drift. Use an experiment-local epilogue with the control expression;
do not change the general kernel or silently relax bit equality.

Compare optional gate/up capture, completed hidden vectors and unchanged
prepared-down output against control and CPU references. Include guards,
signed/zero/tail inputs, distinct rows, producer dependencies, scratch reuse,
invalid metadata and cleanup. Count all three candidate launches in gate/up
timing and all four in complete FFN, versus one/two control launches. Include
every dependency and intermediate write/read; no inner-loop-only claims.
Inspect sm_52 instructions, registers and spills without retuning geometry.

## Delivery And Gates

1. `docs(perf): scope unfused FFN comparison`: this instruction finding and
   fixed experiment boundary, before implementation.
2. `feat(cuda): probe unfused prepared FFN`: benchmark-only kernels, explicit
   owned scratch contract, behavioral tests and existing runner extension.
3. `docs(perf): record unfused FFN screen`: clean release provenance, five
   alternating pairs on frozen layers 0/17/35 and rows 1/4, ten iterations,
   actual dispatch, exactness, memory and decision.

Keep the current one-column split gate/up and qualified prepared down/head
as controls. Require >=5% complete one-row FFN gain with positive paired
95% interval. A borderline result needs twenty fresh pairs, not another
geometry. Only a qualifying candidate may proceed to original actual-512
quality, frozen dense-valid tasks, incremental HTTP and fresh Ollama gates.
Both original goals remain unmet: **53.06 HTTP tok/s** and **1.30x fresh
official Ollama**, currently **56.026 tok/s**. The qualified runtime remains
**47.379 HTTP tok/s** and all generation defaults are unchanged.

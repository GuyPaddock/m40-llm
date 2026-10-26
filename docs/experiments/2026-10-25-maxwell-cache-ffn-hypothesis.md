# L2-Only Weight Loads In The Qualified FFN

## Evidence And Scope

The [qualified stage profile](2026-10-25-maxwell-projection-stage-profile.md)
places 39.43% of sampled middle-token GPU time in gate/up/SwiGLU. The
[unfused screen](2026-10-25-maxwell-unfused-ffn-screen.md) loses decisively;
do not revisit decomposition or its geometry. Its archived control PTX
already uses `ld.global.nc.v4.f32` for activations and `ld.global.nc.u32` /
`ld.global.nc.u16` for weight values/scales. Adding `__ldg` is not a new
execution strategy. The control SASS dump has SHA-256
`7c2d3c8e46c37367a7ce14d14d865ea5a6476b4ab5baf4e2578b45adfa182b22`.

One independent hypothesis remains: streaming weights through the global
L2-only load path might reduce competition with reused activations in the
read-only path. This is not an observed cache bottleneck. CUPTI traffic
counters remain unavailable, and no hit-rate, bandwidth or occupancy
attribution follows from this experiment.

NVIDIA documents cache-hinted loads for compute capability 5.0 and higher
in the [CUDA 12.8 programming guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#load-functions-using-cache-hints).
The [PTX cache-operator definition](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)
defines `.cg` as L2-only, bypassing L1. Use `__ldcg` for each original
32-bit packed weight and each original FP16 scale, leaving activation
loads unchanged. Cache operators are hints, not correctness primitives.
Do not use newer eviction-priority instructions, textures or async copies.

## Fixed Candidate And Early Exit

Keep the fused one-column, four-value, 128-thread split-plane kernel,
runtime K, original FP32 association/reduction, optional intermediate
capture and original SwiGLU expression. No layout, task assignment,
weight preparation, scratch, stream, register cap or thread-count change.
Do not combine failed joint, fixed-K, prepared-down or unfused candidates.

First compile for sm_52 and inspect PTX, SASS, registers and local storage.
If the explicit policy does not produce distinct weight-load instructions,
stop without GPU timing or runtime integration. Otherwise extend the
existing explicit benchmark and behavioral parity cases only. Require
control-bit equality for gate/up, hidden and completed down outputs, guards,
signed/zero/tail inputs, distinct rows, validation, dependencies, reuse and
cleanup. The qualified down/vocabulary path stays unchanged.

Run five alternating pairs on frozen layers 0/17/35, genuine rows 1/4 and
ten iterations. Include complete gate/up/SwiGLU and complete FFN, not only
loads. Count actual dispatch and prove no timed allocations/transfers/syncs.
Use the frozen M02 inputs/environment and record memory and GPU conditions.
Require >=5% complete one-row FFN gain and positive paired 95% interval.
Only borderline results justify twenty fresh confirmation pairs; do not
rescue a loser with another cache operator or geometry search.

## Atomic Delivery

1. `docs(perf): scope FFN weight-cache test`: this predeclared hypothesis.
2. `feat(cuda): probe L2-only FFN weight loads`: distinct-code evidence,
   benchmark-only explicit path and behavioral tests, if justified.
3. `docs(perf): record FFN cache-policy screen`: clean-build provenance,
   raw observations, audit and decision, including negative findings.

Only a qualifying screen may proceed to actual-512/frozen-quality,
incremental HTTP and fresh Ollama evaluation. Do not alter defaults.
The qualified runtime remains 47.379 HTTP tok/s versus 43.097 Ollama;
both original goals remain unmet: 53.06 and 56.026 tok/s for this fresh
1.30x reference. Kernel timing cannot substitute for either goal.

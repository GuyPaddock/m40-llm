# Matched Short-Request Activity Calibration

## Question

The first official activity trace has about 21 ms of GPU execution in a
44.5 ms median decode interval. Per-launch callback logging is intrusive.
Determine whether callback-free, buffer-flushed concurrent activities can
support quantitative stage attribution before choosing another kernel.

## Fixed Scope

- Keep inference/kernel/model code and all runtime defaults unchanged.
  Extend only the external observer with explicit
  `M40LLM_CUDA_TRACE_CALLBACKS=0`, requiring activity collection. Existing
  launch-only and callback-plus-activity behavior remain unchanged.
- Activity-only mode enables no launch callbacks and flushes output once
  per completed CUPTI buffer, not per kernel. Keep the existing eight 1 MiB
  host-buffer and 100,000-activity bounds. Zero records/drops/errors are not
  passing model coverage; require actual expected dispatch and output.
- Extend the existing throughput test with one explicit ignored diagnostic
  for the qualified dense runtime. Four fresh-KV 16-token warmups precede
  one measured 16-token generation. Record actual prompt/generated IDs,
  operation counts, phase times, immutable prepared cache and cleanup.
- Use the same Qwen2.5-3B Q8_0 bytes and canonical 65 prompt IDs for the
  unchanged official CUDA 12 runner. Four warmups precede the measured
  request; replace/clear retained prompt KV and verify zero reused prompt
  tokens before accepting the measured request. Obtain actual native
  returned token IDs where supported. No forced continuation or retries.
- Run five contemporaneous observer-off/on process pairs per engine,
  reversing order on odd pairs. Models never execute concurrently. Resident
  request prefill/decode time excludes process/model loading. Preserve cold
  identity, GPU isolation, before/after state and all outputs separately.
- Official native-runner timing is diagnostic, not Ollama's public buffered
  HTTP route. M40 timing is resident generation, not its HTTP route. Neither
  replaces the original 512-token qualification or its paired comparison.
- Require output/ID and dispatch stability across each engine's trials.
  Cross-engine numerical/continuation equality is not assumed. Separate
  warmup and prefill activity from the fifteen measured decode forwards.
- Record per-engine observer-on/off phase overhead and variation. A median
  decode overhead above 3%, unstable intervals, missing work, unsupported
  collection or changed outputs blocks quantitative engine-stage conclusions.
  Do not tune another collector configuration until evidence is documented.

## Delivery And Interpretation

1. `docs(research): bound activity calibration`: this fixed contract.
2. `feat(tools): collect callback-free GPU activity`: opt-in mode and real
   output/coverage/invalid-setting checks.
3. `test(bench): calibrate resident GPU observation`: shared throughput
   entrypoint and bounded request orchestration, without runtime changes.
4. `docs(perf): record activity calibration`: all pairs, raw captures,
   independent audit, limits, current status and MemPalace checkpoint.

Use release Rust and unchanged NVCC flags. Validate only affected tools and
tests; do not claim unrelated CUDA suites were rerun. GPU work is serial,
escalated and never overlaps builds, tests or MemPalace. Commit each verified
unit before measurement. Preserve failed attempts instead of restarting an
observation merely because a tool call times out.

If calibration passes, compare matched decode-stage spans as diagnostic
execution evidence, not DRAM/occupancy attribution or a new speedup. If it
fails, retain explicit limits rather than choosing kernels from perturbed
intervals. No closed candidate, numerical Q8_1 path, attention approximation
or scheduler change substitutes for the unchanged exact-output goal.

Qualified status remains 49.146 HTTP tok/s versus 43.106 official Ollama;
53.06 and 56.037 HTTP tok/s remain unmet. The previous goal turn made
progress by implementing and validating actual GPU activity collection.

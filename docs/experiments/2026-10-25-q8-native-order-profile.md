# Qualified Prepared Q8 Stage Profile

## Scope and Perturbation

This profiles the [qualified exact configuration](2026-10-25-q8-native-order-http.md),
not another optimization. Clean revision `1a5388c` uses the same release
executable, SHA-256
`36df94961ca5fa042ebb2da4dc45ed91877086e675041be541867e5bcb807af9`.
Original Qwen2.5-3B Q8_0 bytes, all 36 layers, dense FP16 KV/context 1024,
canonical 65-ID prompt and 512 greedy generated tokens remain unchanged.
Only profiling switches between paired requests. Split preparation covers
down/vocabulary, with native-order 4-column/256-thread down. Other C11
backend flags and CUDA -O3 / sm_52 / compute_52 remain fixed.

Five alternating ordinary/profiled HTTP pairs follow separate warmups.
All ten rows evaluate all 65 prompt tokens with zero reuse, produce 512
tokens with length stops, and have identical output. Untimed companions
match all actual IDs, including the preceding qualified benchmark's trace.
Measured intervals do not overlap. Ollama is an untimed oracle only.

| Pair | Profile-Off HTTP tok/s | Profile-On HTTP tok/s |
| --- | ---: | ---: |
| 0 | 35.753444 | 35.458581 |
| 1 | 35.667019 | 35.424761 |
| 2 | 35.626551 | 35.390622 |
| 3 | 35.646179 | 35.371218 |
| 4 | 35.614564 | 35.373281 |
| Median | 35.646179 | 35.390622 |

Instrumentation reduces median throughput by 0.717%. MAD/median is
0.0585% off and 0.0548% on. Median HTTP wall is 14363.391 ms off versus
14467.110 ms on; internal prefill is 1577.997 versus 1613.207 ms and decode
12739.334 versus 12808.983 ms. These independent medians do not add exactly.
This comparison quantifies perturbation, not optimization promotion or a new
timed Ollama result. Do not replace the earlier paired native/prepared gain
with a ratio between separately measured runs.

## GPU Event Spans

Stage report sessions 2-6 are the five measured profiled requests. Exclude
session 1, the warmup, and session 7, the untimed actual-ID companion.
Each selected prediction step has absolute position 64, 319 or 575.
Layer stages sum across 36 layers within each token before taking medians.

| Median Stage ms | Step 0 | Step 255 | Step 511 |
| --- | ---: | ---: | ---: |
| Embedding | 0.001 | 0.027 | 0.029 |
| Attention norm | 0.390 | 0.387 | 0.388 |
| QKV | 2.502 | 2.508 | 2.511 |
| Q RoPE | 0.276 | 0.275 | 0.275 |
| K RoPE / KV append | 0.319 | 0.329 | 0.327 |
| Attention | 0.624 | 0.777 | 0.975 |
| Output projection | 1.742 | 1.739 | 1.741 |
| Attention residual | 0.159 | 0.162 | 0.161 |
| FFN norm | 0.401 | 0.394 | 0.393 |
| Gate/up/SwiGLU | 11.872 | 11.872 | 11.879 |
| Down projection | 4.788 | 4.774 | 4.772 |
| MLP residual | 0.159 | 0.164 | 0.164 |
| Output norm | 0.011 | 0.010 | 0.011 |
| Vocabulary projection | 1.563 | 1.563 | 1.565 |
| Greedy argmax | 0.259 | 0.259 | 0.259 |
| Union of GPU stage intervals | 25.061 | 25.267 | 25.446 |
| Token wall | 79.276 | 26.039 | 26.318 |

All recorded stages use the decode stream. These are event-delimited spans,
not isolated kernel service times or DRAM/occupancy counters. They can contain
enqueue gaps. Never add host completion waits to overlapping GPU work.
The middle token's median stream-completion wait is 20.019 ms, not another
20 ms to add to its 25.267 ms GPU interval union.

First-pool-use token wall remains inflated, as in the earlier profiler report.
The reported pool setup itself is only about 0.5 ms; the much larger first
sample gap has not been isolated and is not evidence of ordinary CPU cost.
Use reused middle/late samples for steady-state attribution. Profiling creates
592 reusable event pairs per session, without adding per-operation waits.

All fifteen measured samples report 435 counted kernel launches, zero cuBLAS
calls, zero device allocations/frees, two stream synchronizations and no
cross-stream waits. Each has one 8192-byte H2D copy and two D2H copies totaling
2180 bytes. These counters cover the selected token, not the whole request.

## Consequences for the Remaining Goal

At step 255, FFN gate/up/down accounts for a median 65.96% of GPU interval
union: gate/up/SwiGLU 47.02%, down about 19%. Attention is 3.07%, rising to
3.83% at step 511. This makes native fused gate/up the first remaining exact
kernel investigation, not another attention approximation or compressed-KV
policy sweep. Prepared down and vocabulary already reduce those portions;
the old native stage profile is not the current bottleneck breakdown.

About 1.58 s of ordinary request time remains in sequential prefill. Existing
packed-prefix behavior would need its own actual-output and matched timing
check before a speed claim. Reducing a single sampled stage cannot establish
the required HTTP gain. Any further implementation remains bounded, parity
checked and opt-in until the full-workload acceptance gate passes.

The exact result remains below both 53.06 tok/s and 1.30x the latest timed
Ollama baseline, 56.1115 tok/s. The negative offline n-gram and static sparse
decode gates are unchanged. Pruning and activation sparsity remain separate
quality-cost experiments, not a way to satisfy the unchanged-model goal.

## Reproduction and Evidence

Use [the existing profiler comparison](../controlled_benchmark.md#check-stage-profiler-perturbation)
with all C11 backend/attention flags, but explicitly load the qualified
single-model configuration:

```bash
export M40LLM_Q8_EXECUTION_WEIGHTS=split
export M40LLM_Q8_EXECUTION_FAMILIES=down,vocabulary
export M40LLM_Q8_EXECUTION_BUDGET_MB=6144
```

Use `--compare-stage-profile`, not `--compare-q8-execution`. Profiling mode
loads its one model from the environment; the prepared-family comparison
instead constructs two immutable configurations explicitly. Select steps
`0,255,511`, start with profiling unset, and use new report paths.

- [HTTP JSONL](2026-10-25-q8-native-order-profile-http.jsonl), SHA-256
  `6fbdf8e18365d2f2504f9608c5dd5b7f490863bd3c53e53cf7fcd8f672ebde24`.
- [Raw stage JSONL, gzip](2026-10-25-q8-native-order-stage-events.jsonl.gz).
  Uncompressed SHA-256
  `59d762509fc8b7de280a656dc7de191ad4d4d64a990be67df96ea66d91e7c42e`;
  gzip SHA-256
  `c340c183db2993aab8c0191e71533329df77777974bccc8722bb2dc05758b887`.
- [Derived audit and medians](2026-10-25-q8-native-order-profile-audit.json).
- [Build log](2026-10-25-q8-native-order-profile-build.txt),
  [run log](2026-10-25-q8-native-order-profile-run.txt) and
  [oracle log](2026-10-25-q8-native-order-profile-ollama.txt).

Text logs trim trailing whitespace and redundant final blank lines only.
GPU is the same M40 UUID/driver as the qualified screen, P0 at 1113/3004 MHz
in timed boundary samples, temperatures 46-71 C. Only this model and the idle
oracle are resident; inference, tests, builds and memory work never overlap
timing. Both tools exit and the isolated server is stopped afterward. Final
hardware check confirms 0 MiB allocated / 0% utilization. No runtime or model
defaults change.

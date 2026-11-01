# Native-Order Prepared Q8 Confirmation

## Decision

Split down/vocabulary preparation now qualifies as an opt-in exact benchmark
configuration on the original browser workload. Clean release `dd8c5f6`
measures 35.724705 versus 31.962281 HTTP E2E tok/s, a gain of 11.771% with
95% paired bootstrap interval 11.631-11.866%. All 512 actual generated IDs
match, all timed outputs match, and all three dense-valid smokes pass.

This does not change general runtime defaults. Prepared weights remain off
unless requested. The project KV default remains compressed top8; this
short-context comparison explicitly uses dense full attention. No layers,
weights or context are removed, and no activation thresholds are applied.

Both original speed goals remain unmet. This experiment does not time Ollama:
it is an oracle before the two m40-llm configurations execute. The latest
separate controlled Ollama baseline remains 43.162711 tok/s, giving a 56.1115
tok/s relative target in addition to the absolute 53.06 target. Do not present
the native-control gain as an Ollama speedup.

## Why the Old Candidate Branched

The [pre-fix probe](2026-10-25-q8-prepared-prefix-drift.md) isolated the
token-310 branch to forward hidden-state drift, not vocabulary head choice.
The selected native down kernel used 256 threads while prepared down used
128, changing FP32 accumulation order despite preserving quantized weights.
The existing 4-column/256-thread separate kernel matches native FP32 output
bits on the two regression shapes. Commit `dd8c5f6` selects it only for
single-row prepared down; other families and multi-row dispatch are unchanged.
No CUDA kernel or new geometry was added.

The corrected clean release probe forces the same historical native prefixes
through step 316. All 317 native/prepared actual greedy predictions match;
all native predictions still match the historical trace. All twelve sampled
hidden vectors and forward-induced generic logits match exactly. At the old
branch, both paths now choose native token 1008 instead of candidate 5815.
The one mismatch against the *old prepared* history is the intended correction,
not a new parity failure. All sampled top-ten sets overlap 10/10.

Same-hidden generic vocabulary projections still differ by at most 3.815e-6
at step 310. Generic projections are diagnostic and are not claimed to use
the native fused greedy head's arithmetic. Production greedy equality is
verified separately, including the full 512-token free-generation trace.
The probe completes with tracked allocation bytes returning 0 -> 0.

## Identity and Protocol

- Revision `dd8c5f60340100ff854bbc8593bb68f9b5ed053d`, clean worktree, Rust
  release profile, Rust 1.95.0, NVCC 12.8.93, g++ 14.3.1.
- CUDA `-O3`, embedded sm_52 SASS and compute_52 PTX, no fast math.
- Original Qwen2.5-3B Q8_0 GGUF SHA-256
  `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
  All 36 layers remain active.
- Original browser question and canonical raw Qwen ChatML prompt, 65 verified
  prompt IDs, context capacity 1024, greedy top1, natural stop handling.
- Native control uses C11's decode-stream Q8 coltile4/split4, native fused
  QKV/MLP, shared-activation native greedy head, and parallel head128 split8/256.
  Candidate differs only by immutable split down/vocabulary preparation with
  a 6144 MiB extra-payload budget. Effective prepared down is 4 columns/256
  threads, recorded in cache telemetry.
- One warmup/configuration, then five alternating serial pairs, no concurrent
  tests, builds, inference or MemPalace work. Both models stay resident.
  Every request resets KV, reuses zero prompt tokens and evaluates all 65.
  Every timed request produces 512 actual tokens with a length stop.
- Primary time is client send through complete buffered HTTP body, with weights
  resident. Cold loading, preparation, oracle work and trace companions are
  excluded. All measured HTTP intervals are non-overlapping.

The M40 UUID is `GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852`, driver 580.159.04.
Timed boundary snapshots show P0, 1113 MHz SM / 3004 MHz memory clocks,
50-72 C, about 67.9-77.4 W. The only compute processes are the benchmark and
its idle Ollama oracle runner; only one request executes at a time. Temperature
rises through the run, so pair order alternates rather than discarding trials.
After both tools exit and the isolated server is stopped, the GPU is idle at
0 MiB allocated. Boundary snapshots are not per-kernel utilization counters.

HTTP executable SHA-256
`36df94961ca5fa042ebb2da4dc45ed91877086e675041be541867e5bcb807af9`.
Prefix-probe executable SHA-256
`c7e634490d7a98cf13edea1a320ac89379b15dc4466c3b8838298aad5bf4fdec`.

## Five-Pair Screen

| Pair | First Configuration | Native HTTP tok/s | Prepared HTTP tok/s |
| --- | --- | ---: | ---: |
| 0 | Native | 32.040593 | 35.767182 |
| 1 | Prepared | 31.985830 | 35.751553 |
| 2 | Native | 31.962281 | 35.714194 |
| 3 | Prepared | 31.935269 | 35.724705 |
| 4 | Native | 31.932816 | 35.705022 |

| Median Metric | Native | Prepared |
| --- | ---: | ---: |
| Prompt/prefill ms | 1672.089 | 1576.200 |
| Generated-token decode ms | 14300.146 | 12714.413 |
| Server generation total ms | 15976.203 | 14289.755 |
| Client HTTP total ms | 16018.882 | 14331.819 |
| HTTP E2E tok/s | 31.962281 | 35.724705 |
| Throughput MAD / median | 0.0845% | 0.0551% |

Each row is an independent median; do not sum column medians to reconstruct
another median. The gain exceeds the predeclared 3% threshold and twice the
larger MAD. Bootstrap uses 10,000 paired resamples, seed 0. Neither the
variation-extension nor borderline twenty-pair confirmation rule triggers.
No slow observation is dropped. This is five-pair screening evidence, not
an uncertainty estimate for other workloads or hardware.

Both models output `4`, `OK` and `37` on the three raw ChatML smokes. Both
untimed actual-token companions reproduce measured output and match all
512 IDs. The native trace also equals the old experiment's native trace.
The prior +12.53% result used a different continuation and is retained as a
failed historical screen, not merged with this result or treated as a paired
estimate of the 128-to-256-thread correction's timing cost.

## Allocation and Preparation Cost

| Requested Storage Per Model | Native Bytes | Prepared Bytes |
| --- | ---: | ---: |
| Original uploaded GGUF weights | 3,279,519,744 | 3,279,519,744 |
| Extra prepared down/vocabulary | 0 | 1,192,935,424 |
| Total weight payload | 3,279,519,744 | 4,472,455,168 |
| Native dense FP16 KV capacity | 37,748,736 | 37,748,736 |

Prepared storage duplicates source payload; it is not model compression.
The Rust allocation tracker excludes native KV. Its combined post-warmup
7,752,523,792 bytes covers *both* models' tracked allocations. GPU snapshots
also include the idle oracle runner. Neither figure is standalone peak VRAM.
No DRAM-byte counter or measured bandwidth claim is made.

Observed cold-load time is 5069.920 ms native and 15172.293 ms candidate.
Candidate preparation is 9499.437 ms, **including** 8920.962 ms hashing;
do not add those nested durations. These one-shot observations are separate
from the warm HTTP comparison, not a repeated cold-cache performance result.
Estimated candidate setup host payload is 6,895,608,736 bytes; process peak
RSS is 10,275,388 KiB including both models. Neither is isolated host overhead.

## Validation and Reproduction

Before measurement, the corrected implementation passes 17 execution-weight
lib tests, prepared/cache/projection/greedy/MLP checks, attention grid and
forward checks with cuBLAS on, focused parity with it off, all 21 shared
quality-runner checks, both warning-as-error CI lanes, non-CUDA/CUDA-server
all-targets Clippy, and fmt. Logs below preserve the executed scope.
No tests ran concurrently with these measurements.

Use [the controlled comparison instructions](../controlled_benchmark.md#compare-prepared-q8-families),
all C11 base flags, and `--compare-q8-execution split --q8-families down,vocabulary`.
Keep ambient preparation off because this comparison constructs its candidate
explicitly. Ordinary single-model experiments instead request that subset
through the [load-time configuration](../q8_execution_weights.md#opt-in-runtime-preparation).
Never silently change frozen model-quality baselines to this new configuration.

## Artifacts and Next Work

- [HTTP JSONL](2026-10-25-q8-native-order-http.jsonl), SHA-256
  `b56b6200ca962a7f0a5fb3694071ba13255b46788f51c12c693cc9da62e57716`.
- [Corrected prefix JSONL](2026-10-25-q8-native-order-prefix.jsonl), SHA-256
  `95940d5fe872baccc9973882cc42b294ec75fe99ec441612516b7fd686b82ed2`.
- [Derived audit and median timings](2026-10-25-q8-native-order-audit.json).
- [HTTP run](2026-10-25-q8-native-order-http.txt),
  [oracle log](2026-10-25-q8-native-order-ollama.txt),
  [HTTP build](2026-10-25-q8-native-order-http-build.txt),
  [prefix run](2026-10-25-q8-native-order-prefix.txt) and
  [prefix build](2026-10-25-q8-native-order-release-build.txt).
- [Lib](2026-10-25-q8-native-order-lib.txt),
  [dispatch parity on](2026-10-25-q8-native-order-dispatch-parity-on.txt),
  [dispatch parity off](2026-10-25-q8-native-order-dispatch-parity-off.txt),
  [runner](2026-10-25-q8-native-order-runner.txt),
  [CI non-CUDA](2026-10-25-q8-native-order-ci-noncuda.txt),
  [CI NVCC/no device](2026-10-25-q8-native-order-ci-nvcc.txt),
  [Clippy non-CUDA](2026-10-25-q8-native-order-clippy-noncuda.txt),
  [Clippy CUDA/server](2026-10-25-q8-native-order-clippy-cuda.txt).

JSONL is unchanged from execution. Text logs trim trailing whitespace and
redundant final blank lines only.

Next remeasure stage attribution on the qualified exact configuration before
choosing further work. The original throughput goals remain active; model
modification cannot satisfy them. The negative n-gram/static-sparse gates
remain intact. This trace's restored equality supplies no new reason to
reopen their failed feasibility screens.

# Qualified Projection Runtime Stage Profile

The [fresh-Ollama-qualified combination](2026-10-25-maxwell-projection-ollama-http.md)
remains FFN dominated after the QKV/output improvements. This diagnostic
locates remaining work; it does not replace the authoritative **47.379
HTTP tok/s** result or satisfy either original throughput goal.

Clean revision `81b3b161e732711725d0ae6c7d03995290cfe6a8`, release comparison
SHA-256 `48db57beb05b4738d0ceb54fed3572d743403f534a7aabe244e93f569f5a2960`.
The executable is unchanged from corrected fresh timing. Model, canonical
65 ChatML IDs, greedy 512-token continuation, dense KV/context 1024,
`split-f32x4-128`, dequant-SGEMM prefix, `warp-256` QKV/output and all 109
prepared gate/up/down/vocabulary identities remain fixed. Official Ollama
0.40.11 is an **untimed oracle only** in this run.

## Instrumentation And Correctness

Five alternating profiling-off/on pairs measure **47.470170 / 46.990957
HTTP tok/s**, a **1.009504% throughput reduction** from instrumentation.
All ten measured outputs and both actual 512-ID replays match completed
quality. Do not subtract instrumentation cost to claim an inferred speedup.
Both paths preserve the actual QKV/output warp launch counts, prepared
projection counts, copies, allocations, waits and synchronizations.

Reproduce with the same frozen environment and runtime/quality arguments as
the fresh comparison, replacing `--time-ollama` with `--profile-q8-runtime`.
Set `M40LLM_STAGE_PROFILE_REPORT` to a new path. Default sampled steps are
0/255/511. No kernel, runtime policy, sampling or cache semantics changed.

Seven profiled sessions emit 21 event rows. Exclude the warmup and final
actual-ID replay, leaving five primary sessions. Every row contains 400
spans, complete 36-layer core-stage coverage, absolute positions 64/319/575
and the reusable event pool. The vocabulary projection occurs once per
token, not once per layer. Values below are medians of stage totals across
all layers; shares are medians of per-sample fractions.

| Stage | Step 0 ms | Step 255 ms | Step 511 ms | Middle share |
| --- | ---: | ---: | ---: | ---: |
| Gate/up/SwiGLU | 8.0325 | 8.0306 | 8.0340 | 39.43% |
| Prepared down | 4.7545 | 4.7502 | 4.7536 | 23.33% |
| QKV warp projection | 1.8918 | 1.8818 | 1.8818 | 9.24% |
| Vocabulary projection | 1.5632 | 1.5646 | 1.5627 | 7.68% |
| Output warp projection | 1.3497 | 1.3556 | 1.3541 | 6.64% |
| Attention | 0.6120 | 0.7752 | 0.9740 | 3.80% |
| All summed stages | 20.2172 | 20.3573 | 20.5801 | 100% |

Compared with the [previous native-projection profile](2026-10-25-maxwell-split-stage-profile.md),
the expected QKV/output stages are smaller while FFN remains dominant.
These are separately instrumented runs, not paired per-stage promotion
evidence. Event spans do not prove measured DRAM bandwidth, occupancy or
unprofiled latency. The first sampled token also includes setup outside the
summed GPU stages; its wall time is not steady-token latency.

## Next Bounded Hypothesis

Inspect the generated sm_52 instructions for the qualified prepared
projection before changing it. Source `cuda/q8_execution.cuh` still uses
one shared publication plus a block-wide barrier after every reduction
level in `q8_separate_reduce`. The current one-row down launch is fixed at
four columns / 256 threads; vocabulary uses four columns / 128 threads.
These are source properties, not measured bottleneck attribution.

Test one original-tree warp-finish candidate for the existing prepared
projection, retaining the qualified shared-tree path as control. Keep the
same FP32 arithmetic association, per-thread tasks, vector loads, separate
scale/value planes, column count, thread count and shared capacity. Change
only reduction execution, with one publication barrier and a final warp.
This is independent of the failed gate/up geometry/layout experiments;
do not revive those experiments or change their gates.

Start benchmark-only, with exact bit comparisons, CPU references, distinct
rows, tails/output guards, stream ordering and allocation cleanup. Measure
real down shapes on layers 0/17/35 and the real vocabulary projection.
For down, include complete FFN latency with the **unchanged split-plane
gate/up**; isolated down gains alone do not satisfy the original >=5%
complete-FFN gate. Treat vocabulary separately, with its own complete
projection >=5% gate and positive paired interval. Genuine four-row results
are separate from decode and preserve the control's row-specific launch.
Do not add geometry variants if this fixed candidate loses.

Before any runtime integration, combine only passing stage improvements in
a profile-weighted feasibility estimate against the original 3% E2E floor.
An estimate is not a runtime claim. Actual 512 IDs, frozen dense-valid
quality and original incremental HTTP/fresh-Ollama gates remain mandatory.
The qualified prepared down/head controls and all current defaults remain
untouched until those gates pass.

Atomic order: this profile checkpoint; one benchmark-only CUDA/API/test
commit for the fixed candidate; a measured screening decision; conditional
runtime/quality/HTTP commits only for qualifying work. Both original
objectives remain **53.06** and **1.30x fresh controlled Ollama**.

## Evidence And Ownership

- [Off/on requests and actual token traces](2026-10-25-maxwell-projection-stage-http.jsonl)
- [Raw CUDA event rows](2026-10-25-maxwell-projection-stage-events.jsonl)
- [Derived stage medians](2026-10-25-maxwell-projection-stage-audit.json), using the unchanged [stage aggregator](2026-10-25-maxwell-selected-stage-audit.jq)
- [HTTP/coverage audit](2026-10-25-maxwell-projection-stage-http-audit.json) and [frozen program](2026-10-25-maxwell-projection-stage-http-audit.jq)
- [Run log](2026-10-25-maxwell-projection-stage-profile.log.gz), [oracle log](2026-10-25-maxwell-projection-stage-ollama.log.gz) and [checksums](2026-10-25-maxwell-projection-stage-profile.sha256)

Benchmark PID 1535906 is the only active inference process, with idle oracle
runner PID 1536839 resident. All twenty primary boundaries show P0 and
1113/3004 MHz. No builds, tests or MemPalace work overlap timing. Benchmark
session and owned official server PID 1534988 both exit successfully; the
final GPU compute-process list is empty. Offline audit negative controls
reject missing warp dispatch and missing per-layer attention coverage.

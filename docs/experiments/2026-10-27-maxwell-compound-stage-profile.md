# Qualified Compound Runtime Stage Costs

## Decision

The unchanged compound runtime remains FFN-dominated. This is the single
[measurement-only follow-up](2026-10-27-maxwell-compound-profile-plan.md),
not a new optimization. Five alternating release profiling-off/on pairs
measure **50.877782 / 50.379423 HTTP tok/s**, a **0.9795% throughput loss**
or **0.9892% latency increase** from instrumentation, below the fixed 3%
attribution limit. Do not subtract this overhead from measured throughput.

All ten primary outputs and both actual 512-ID replays match the completed
[compound quality evidence](2026-10-27-maxwell-compound-runtime-quality.jsonl).
Ordinary enqueue counters, dispatch, copies, allocations, waits and stream
synchronizations remain unchanged. All original 109 prepared identities and
36 additional joint pairs remain fixed and fully accounted.

Keep **50.910426 versus 43.194730 HTTP tok/s** as the authoritative separate
[official checkpoint](2026-10-27-maxwell-compound-runtime.md). Both original
targets remain unmet. No kernel, geometry, arithmetic, scheduling, precision,
KV policy, profiler code or runtime default changed for this measurement.

## Sampled Costs

Seven profiled sessions emit 21 rows at generated steps 0/255/511. Exclude
warmup and the final actual-ID replay, leaving five primary sessions and
fifteen samples. Every sample contains 400 spans, all 36 layers for eleven
layer stages, plus one embedding, output norm, vocabulary and greedy argmax.
Absolute positions remain 64/319/575. The 592-pair event pool is reused after
the first sampled token in each session.

Values are medians of per-sample totals across layers, not one-layer costs.
Shares are medians of per-sample fractions. The vocabulary runs once per
token, not once per layer.

| Stage | Step 0 ms | Step 255 ms | Step 511 ms | Middle share |
| --- | ---: | ---: | ---: | ---: |
| Compound gate/up/SwiGLU | 7.4332 | 7.4215 | 7.4210 | 39.13% |
| Prepared down | 4.7686 | 4.7614 | 4.7596 | 25.12% |
| Vocabulary | 1.5632 | 1.5638 | 1.5634 | 8.24% |
| Packed QKV | 1.4079 | 1.3918 | 1.3877 | 7.34% |
| Packed output | 1.0151 | 1.0164 | 1.0088 | 5.35% |
| Dense attention | 0.6375 | 0.7802 | 0.9649 | 4.12% |
| All summed stages | 18.8664 | 18.9588 | 19.1313 | 100% |

Complete FFN accounts for about **64.3%** of middle sampled stage time.
The earlier [split profile](2026-10-26-maxwell-packed-stage-profile.md) is
historical, separately instrumented evidence. Do not subtract its stage
medians from these to claim another paired gain; the existing complete-FFN
and HTTP comparisons establish the compound's measured improvement.

| Independent interval medians | Step 0 | Step 255 | Step 511 |
| --- | ---: | ---: | ---: |
| GPU stage union, ms | 18.8664 | 18.9588 | 19.1313 |
| GPU timeline span, ms | 27.2274 | 19.7680 | 19.9378 |
| Token wall, ms | 27.2578 | 19.7954 | 19.9618 |
| Wall minus GPU union, ms | 8.3914 | 0.8324 | 0.8236 |

Separate medians do not necessarily add. First-token setup is not steady
decode. Event spans can include queue/host effects, and the residual is not
pure removable CPU time. These data do not measure physical DRAM traffic,
occupancy or warp stalls and do not establish Maxwell's hardware ceiling.

## Remaining Target Scope

Use the authoritative unprofiled 50.910426 rate, actual 512 tokens and fresh
official 43.194730 median for request-time arithmetic:

| Target | Required extra throughput | Required request-time saving | Saving per generated token |
| --- | ---: | ---: | ---: |
| 53.06 HTTP tok/s | 4.2223% | 407.426 ms | 0.7958 ms |
| 56.153149 HTTP tok/s, 1.30x official | 10.2979% | 938.958 ms | 1.8339 ms |

The stronger target requires 9.3365% less complete request time. If all that
saving came from sampled complete FFN at 12.1829 ms/token, it would require
about **15.05% FFN latency reduction**, or **17.72% FFN throughput gain**.
Gate/up alone would need 24.71% lower latency; down alone 38.52%. These are
optimistic equivalent budgets, not performance forecasts or critical-path
proofs, and do not change the existing screening/promotion gates.

At the middle sample, eliminating all attention 0.7802 ms or all vocabulary
1.5638 ms would not supply the stronger target's equivalent 1.8339 ms/token.
That does not rule out combinations or different workloads. It explains why
another small isolated arithmetic change is not a credible plan by itself.

No new implementation is admitted by this profile. A further exact-FFN or
dataflow proposal must name substantial avoided work, distinguish itself
from closed probes, account for memory/setup, and keep existing gates. The
single compound exception remains exhausted; do not retune its geometry or
combine more losing candidates. The negative perfect-drafter study remains
closed, with no EAGLE or verifier rescue. Runtime defaults stay unchanged.

## Provenance And Validation

Clean measured revision `e83d4613b72c2d38deb947f6b9962ef420b2d69e`.
The release comparison executable remains SHA-256
`eb6dc69f94a2e67ecd62bc07a02a12e47c26067236ac6f7df40e2c4f78b5b86f`,
identical to compound qualification. The intervening commits change only
documents/evidence; the archived source-code diff is empty. Reuse the
existing NVCC 12.8.93 `-O3` sm_52/compute_52 build, no fast math.

Same source Qwen2.5-3B Q8_0, canonical 65 prompt IDs, greedy actual 512 tokens,
natural length stop, explicit dense FP16 KV/context 1024, compound128 FFN,
packed-warp-256 QKV/output, bounded dequant-SGEMM prefix and prepared
down/vocabulary. Original preparation remains 2,917,580,800 bytes plus
1,724,645,376 joint bytes; all weight owners total 7,921,745,920 bytes.
Resident dense KV remains 37,748,736 bytes. No hidden alternative path or
additional allocation appeared; the existing 42 allocations/frees per
request remain unchanged.

Only benchmark PID 611360 and idle owned oracle runner PID 612193 appear in
all twenty primary boundaries: P0, 1113 MHz SM/3004 MHz memory. Inference is
serial. No tests, builds or MemPalace work overlaps primary timing. The
owned server is stopped and the final GPU process list is empty.

The independent audit binds completed quality, exact runtime/dispatch,
memory, paired timing, stage/position/pool coverage, streams, identities,
clocks and process isolation. Thirteen deliberately corrupted HTTP/event
controls are rejected. The existing stage aggregator is unchanged. No new
Rust/CUDA tests are claimed for this code-unchanged measurement; formatting,
diff, artifact and link checks pass.

Reproduce with the same compound runtime/quality arguments, replacing
`--time-ollama` with `--profile-q8-runtime`. Set only
`M40LLM_STAGE_PROFILE_REPORT` to a new path; use the default steps 0/255/511.
The archived orchestration includes the exact command and isolated oracle.

Artifacts: [off/on HTTP and traces](2026-10-27-maxwell-compound-stage-http.jsonl),
[event rows](2026-10-27-maxwell-compound-stage-events.jsonl),
[derived stages](2026-10-27-maxwell-compound-stage-stages.json),
[audit program](2026-10-27-maxwell-compound-stage-audit.jq),
[audit result](2026-10-27-maxwell-compound-stage-audit.json),
[remaining-scope program](2026-10-27-maxwell-compound-stage-scope.jq),
[scope result](2026-10-27-maxwell-compound-stage-scope.json),
[negative controls](2026-10-27-maxwell-compound-stage-negative-controls.sh),
[orchestration/logs](2026-10-27-maxwell-compound-stage-orchestration.tar.gz)
and [checksums](2026-10-27-maxwell-compound-stage.sha256).

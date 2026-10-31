# Activity-Only Collection Misses Joint Calibration

## Result

The bounded calibration **fails**. M40 preserves the frozen token prefix
and dispatch, but its five activity captures truncate. The official native
runner has complete captures, but observer-on decode costs **5.607% more**,
above the prespecified 3% ceiling. Do not use these data for a quantitative
engine-stage comparison or another kernel recommendation.

Five contemporaneous off/on process pairs per engine reverse order on odd
pairs. Each process loads one model, runs four warmups, then measures one
fresh-KV 65-prompt/16-generated-token request. All **100 generations** keep
the actual first sixteen IDs from the frozen browser continuation, with
natural length stops. That is bounded prefix preservation, not a new
512-token or task-quality qualification.

| Engine / phase | Off median ms | Activity-only median ms | Latency change | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| M40 decode | 288.019 | 292.570 | +1.580% | +0.610% to +2.471% |
| M40 prefill | 521.080 | 519.735 | -0.258% | -13.569% to +6.257% |
| M40 phase total | 809.114 | 812.327 | +0.397% | -8.417% to +4.224% |
| Official native decode | 336.464 | 355.330 | +5.607% | +4.622% to +7.430% |
| Official native prefill | 229.695 | 237.946 | +3.592% | +3.519% to +4.553% |
| Official native phase total | 566.377 | 593.276 | +4.749% | +4.177% to +6.262% |

Intervals exhaust all **3125 ordered paired bootstrap draws**, percentile
indices 78/3046, for the ratio of paired-sample medians. This is explicitly
not the ten-thousand-draw seeded optimization-promotion protocol. All five
pairs remain visible; there is no pooled rerun or borderline optimization
confirmation. Decode MAD/median is below 1% in all four cohorts.

M40's phase total is its generation-timer total; native phase total is
`prompt_ms + predicted_ms`. Outer M40 wall time and native client wall time
are retained separately. Neither includes loading in the measured request.
Native timing uses the **official runner's native completion route**, not
Ollama's public buffered HTTP API. Do not compare these totals as a new
HTTP speed result. M40's prefill interval includes first-token work; phase
boundaries are not assumed identical across engines.

## Two Independent Rejections

Every M40 on-process retains **100,000 activities** and reports **7520
dropped**, reconciling the expected 107,520 activities across its five
generations. The fixed output cap is reached. This version combines SDK
drops and observer-cap drops in one counter, so their individual shares are
not independently identified by this capture.
Timestamps are valid and no launch callbacks are enabled, but the measured
generation is incomplete in the trace. Its 1.580% decode overhead describes
the **capped collector**, not the cost of complete collection. Calling that
a successful low-overhead calibration would be wrong.

All five native captures contain **84,038 activities**, zero drops/invalid
timestamps and normal completion. Their complete-data decode overhead
exceeds 3%, with the entire paired interval above 4.6%. Stable repeated
outputs do not remove that instrumentation bias.

No cap, warmup count, workload or threshold was changed after observing
these failures. No quantitative GPU-stage engine delta is reported. The
activity-only mode remains useful for diagnostic observation, not an
unperturbed profiling guarantee. No second collector configuration is
rescued or tuned in this checkpoint.

## Fixed Execution And Behavior

M40 retains split-f32x4-128 FFN, packed-warp-256 QKV/output, dequant-SGEMM
prefix, prepared down/vocabulary and explicit dense FP16 KV/context 1024.
All **109 prepared identities / 2,917,580,800 bytes** remain unchanged within
each process. Actual IDs, backend choices, prefix mode, dispatch and tracked
allocation cleanup are checked by the ignored throughput test. Every process
returns tracked bytes zero to zero.

Each request records 576 split-FFN, 576 packed QKV, 576 packed output,
592 prepared down/vocabulary and 252 conversion/SGEMM operations. It also
records **8136 residual-add**, **5184 RoPE** and **2880 KV-append** operation
launches. These are counter counts, not timings or measured traffic. The
per-row prefix bias/RoPE/KV loops are visible in `src/infer/forward.rs`;
their count alone does not prove enough removable cost for an optimization.

The profile counts **21,253 launch actions plus 252 cuBLAS calls** per
request. The reset action is counted but has no concurrent-kernel activity;
excluding that action gives 21,504 physical kernel activities, times five
requests = 107,520. The audit does not equate wrapper counts with physical
launches without this reconciliation.

Native uses unchanged official 0.40.11 runner/backend/model hashes, full
**37/37 GPU offload**, 36 MiB dense FP16 KV, head128, context 1024, flash
attention off and greedy neutral-penalty sampling. The prompt is the frozen
65-ID array; `return_tokens=true` preserves actual sampled IDs. Explicit
`cache_prompt=false` yields zero cached and 65 processed prompt tokens in
every warmup and measured request, not an assumed cache replacement.

## Source And Process Evidence

- `c6ee954`: fixed calibration contract.
- `bcff929`: explicit callback-free activity mode; buffer-boundary flushing
  only in this mode, with original launch modes and caps preserved.
- `3b3d94e`: ignored resident probe and reference tests, reusing the existing
  workload, GPU-state and launch-count code. No inference/build change.
- `1e12a72`: measured M40 orchestration source. All ten M40 processes finish
  successfully. The following native preflight stops before any native HTTP
  generation request because direct backend discovery finds no GPU.
- `49beaeb`: native admission correction and explicit engine-only resume.
  It sets `GGML_BACKEND_PATH` to the actual CUDA 12 backend file, matching
  the pinned Ollama parent's source. `LD_LIBRARY_PATH` alone was insufficient.
  Only native is resumed; completed M40 pairs are not repeated or pooled.

The failed CPU-only preflight log/maps are retained and excluded from all
native timing. It cannot silently count as a GPU comparison. Native runtime
maps and offload logs verify the corrected backend for all ten processes.

Actual shared observer SHA-256 is
`cdebace8dfd0a502f9d8395a75f19e1ba404b84622a85cee525dce6c1018f2f2`;
release M40 probe is
`725b35fc534e7ae62bf17871765a9545c845b401aca1a6af3761aa36af59cb2a`.
Official runner/backend remain `234b05b2...` / `7a7b6696...`; original
Qwen2.5-3B Q8_0 model remains `4420ccb0...`. No model download or network
service is used; HTTP requests are loopback only.

Controller session **32682** exits 1 only after all M40 pairs, on the native
preflight. Its owned runner is stopped; GPU emptiness is verified before
native-only resume. Session **63681** completes all native pairs and exits
zero. Every new process starts with idle GPU checks, each generation checks
sole-process ownership, and native shutdown waits for the owned process.
All measured boundaries are P0 / 1113 MHz SM / 3004 MHz memory. Some M40
warmup boundaries have lower clocks and are not pooled into primary data.
No inference, build, test or MemPalace overlaps timed generation.

## Validation And Decision

The four observer smoke modes preserve actual output 96. Callback-free mode
has exactly two valid activities and zero callback records; malformed and
empty configurations fail. Exact timestamp-normalization checks pass.
The reference regression, release build, warning-as-error CUDA/server
Clippy, formatting and shell syntax pass. No unrelated CUDA parity suite
is claimed for these SDK/test-only changes. Initial duplicate-module/import
Clippy errors were corrected before measurement and source commit.
The warning-as-error non-CUDA suite passes **217** tests after measurement.
The CUDA-gated pure reference regression passes **one**, separately from
the actual hardware calibration requests.

The independent integrity audit accepts the documented **failed calibration**
and rejects twelve corrupted reports, including fabricated no-drop M40
captures, missing native work, changed IDs, wrong cache/dispatch and leaks.
It does not turn audit success into collector qualification.

- [All request reports, warmups and failed admission evidence](2026-10-26-maxwell-activity-calibration-data.tar.gz).
- Raw model captures are separate per-process gzip files for pairs 0-4:
  [M40 pair 0](2026-10-26-maxwell-activity-calibration-m40-0-trace.jsonl.gz)
  and [native pair 0](2026-10-26-maxwell-activity-calibration-native-0-trace.jsonl.gz)
  show the naming convention. The audit reads all ten; unrelated empty
  child-process traces are excluded from model coverage and the archive.
- [Observer smoke, reference, build, Clippy and non-CUDA logs](2026-10-26-maxwell-activity-calibration-validation.tar.gz).
- [Audit command](2026-10-26-maxwell-activity-calibration-audit.sh), [rules](2026-10-26-maxwell-activity-calibration-audit.jq), [derived pairs/statistics](2026-10-26-maxwell-activity-calibration-audit.json) and [checksums](2026-10-26-maxwell-activity-calibration.sha256).
- [Twelve rejected corruption controls](2026-10-26-maxwell-activity-calibration-controls.log.gz).

Retain plain diagnostic tools without changing inference or defaults. Further
exact work needs independently bounded cost in the qualified runtime, not
engine-stage subtraction from this failed collector or a larger record cap
called a speedup. Qualified status stays **49.146 vs 43.106 HTTP tok/s**;
both **53.06** and **56.037 HTTP tok/s** targets remain unmet and active.

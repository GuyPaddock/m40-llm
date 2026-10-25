# Diagnostic Layer-Mask Sensitivity: 2026-10-25

C20 is complete. Skipping one/two/three fixed central layers reduces matched
step cost, but worsens language-model loss and sometimes task behavior. Keep
all masks opt-in. This is model-changing research, not an unchanged-engine win.

## Configuration and Controls

All four release cases use runner `261faf9` with a clean worktree, the same
Qwen2.5-3B Q8_0 source SHA `4420ccb0...e0111d`, and the selected C11 native-Q8
engine. Exact dense KV/context 1024 is explicit; preparation, materialization
and graphs are off. The global compressed-KV and all-layers runtime defaults
are unchanged. Complete hashes, flags, token IDs and toolchains are in JSONL.

The [C19 V1 corpus](2026-10-25-model-quality-dense-reference.md) remains
unchanged, SHA `2a127095...c38`. Masks none, `[17]`, `[17,18]`, `[17,18,19]`
were fixed in `261faf9` before observing any mask result. IDs are zero-based;
first/last layers remain. Each case loads separate unchanged-reference and
candidate models; only the candidate gets a mask. No held-out-based choice,
rubric adjustment, ranking or compact export occurs.

The empty mask exactly matches all 12 C19 generation traces and all 203
teacher-forced token logits/losses. Every later case's reference also matches
C19. Manifests distinguish identical source hashes from different execution
models, with explicit transforms and original active/skipped IDs.

## Quality

Each split has six tasks. The reference passes 3/6 calibration and 4/6 held-out;
the five dense failures remain inconclusive for pruning conclusions.

| Skipped IDs | Active Layers | Candidate Passes Cal / Held | New Dense-Valid Failures Cal / Held | Calibration PPL | Held-Out PPL |
| --- | ---: | --- | --- | ---: | ---: |
| None | 36 | 3/6 / 4/6 | 0 / 0 | 31.556848 | 39.612159 |
| 17 | 35 | 3/6 / 4/6 | 0 / 0 | 34.162595 | 47.656541 |
| 17,18 | 34 | 2/6 / 2/6 | 1 / 2 | 38.410081 | 52.716052 |
| 17,18,19 | 33 | 3/6 / 3/6 | 0 / 1 | 41.494590 | 53.784796 |

Loss is token-weighted over the same 102 calibration and 101 held-out targets.
Held-out PPL increases 20.31%, 33.08%, 35.78% for one/two/three layers. One-layer
task passes therefore do **not** establish preserved quality. Held-out maximum
full-logit drift is 15.490280 / 15.904721 / 20.064199; passage/token mean drift,
target ranks/logits and top-ten overlap remain in the raw comparisons.

The regressions are concrete, and not all are factual errors:

- Two layers return commented `RETRY_BUDGET=19` instead of active `6`.
- Two layers retain the correct Cedar deck and workshop facts but add forbidden
  JSON keys. Fixed format grading fails despite fact score 1.0. Do not describe
  these as missing facts or widen the grader after seeing them.
- Three layers return obsolete Indigo hall 3 rather than 12 and split the answer
  into two JSON objects. This is factual and format drift.
- Three layers recover the config answer that two layers lose. Task behavior
  is non-monotonic; the number of removed layers alone is not a selection rule.

All outputs and dense-failed cases are retained. These authored short tasks and
small loss passages are a bounded diagnostic, not a general capability claim.

## Matched-Step Timing

Five serial reference/candidate pairs per split alternate execution order.
Each trace resets KV, creates reusable scratch, and warms eight observed
prefix tokens before timing. The first fixed passage supplies 42 timed steps
for calibration and 43 for held-out. Predicted IDs never enter the next prefix.
Every step calls the real greedy API; timing includes GPU argmax/readback and
final stream drains. Per-model predictions are stable across all five trials.

| Mask | Split | Reference Median ms | Candidate Median ms | Median Paired Step-Rate Gain |
| --- | --- | ---: | ---: | ---: |
| None | Calibration | 1163.787 | 1161.958 | +0.175% |
| None | Held-out | 1193.530 | 1191.969 | +0.137% |
| 17 | Calibration | 1163.825 | 1134.631 | +2.572% |
| 17 | Held-out | 1193.990 | 1164.174 | +2.564% |
| 17,18 | Calibration | 1166.734 | 1109.761 | +5.149% |
| 17,18 | Held-out | 1195.967 | 1137.054 | +5.181% |
| 17,18,19 | Calibration | 1166.441 | 1081.205 | +7.894% |
| 17,18,19 | Held-out | 1195.322 | 1107.889 | +7.870% |

Gain is median over paired `reference_ms / candidate_ms - 1`, not a latency
reduction or ratio of independently chosen trials. All per-model MAD/median
values are below 0.040%; raw pairs and individual variation are retained. The
empty-control difference is a small nonzero floor, not an optimization.

These are forced-prefix prediction rates, **not free-generated or HTTP TPS**.
Five pairs characterize this fixed workload; no broad confidence or exact
promotion claim follows. One-layer gain is also below the exact-kernel 3%
screen, which is not relaxed. More trials would be needed for any borderline
promotion claim; none is attempted because model semantics have changed.

## Free Generation Is Not Equal Work

Task records retain prompt/generated counts, stop reasons, prefill/decode and
admission-to-return wall time. One observation per model/case is not a paired
E2E performance gate. For example, held-out workshop QA uses 110 prompt tokens:

| Candidate Mask | Pass | Generated Tokens | Prefill s | Decode s | Wall s |
| --- | --- | ---: | ---: | ---: | ---: |
| None | Yes | 39 | 2.759 | 1.056 | 3.861 |
| 17 | Yes | 39 | 2.679 | 1.031 | 3.755 |
| 17,18 | No, extra key | 49 | 2.616 | 1.273 | 3.932 |
| 17,18,19 | No, old fact/format | 55 | 2.535 | 1.396 | 3.974 |

All four stop naturally. Removing more computation can still increase request
time when output gets longer or changes. Do not infer E2E speed percentages
from these unequal outputs or replace the unchanged-Qwen HTTP target with them.

## Memory and Execution Proof

All cases retain 36 physical layers. Each model still uploads 3,279,519,744
weight-payload bytes and reserves 37,748,736 dense-KV bytes, exactly 36 MiB.
Each reports 66,560 bytes of head128 workspace after use. Prepared/materialized
weights are zero. Combined Rust-tracked bytes are 6,559,588,368 after warmup;
this includes both resident models and is not standalone or peak VRAM. Native
allocations such as KV are reported separately, not fully covered by the tracker.
**No weight or KV-capacity saving is claimed from a mask.**

CUDA tests compare masked forward with explicit retained-layer calls, including
equal launch/cuBLAS counts, NaN-poisoned skipped weights, unchanged seeded
skipped KV and original layer/sequence slots. They also cover head64/head128
mixed packed prefill/batched decode and compressed state across q4-V aging.
Invalid/late masks and masked graph/reconstruction requests fail explicitly.

There is no GPU overlap: cases run serially, each pair drains before the next
model, and no test/build runs during capture. The script checks for compute
processes before each mask. Initial and final GPU use is 0 MiB / 0%; final compute
process list is empty. Every report completes with tracked bytes 0 -> 0.

## Artifacts and Reproduction

- [Empty control](2026-10-25-layer-mask-empty.jsonl), SHA
  `47b0365b6f245cad2115c32611fee6c5c04891dc5084588ef84f8417a5879df0`.
- [One layer](2026-10-25-layer-mask-one.jsonl), SHA
  `0954701f79c8e6d8c0f36376404d689f4cec4b8df0139f4670597b859a301672`.
- [Two layers](2026-10-25-layer-mask-two.jsonl), SHA
  `64ffe40ab0ad2c2bdb2088e0664f73468df0c46163afb84fa08fb5cd1dc2d581`.
- [Three layers](2026-10-25-layer-mask-three.jsonl), SHA
  `d145a8e747b6236812b0dccc021f5e9e5fd0d5ff3ed77563085e710a4cdd9837`.

Each file has 36 records. Audit verifies all source/reference/corpus identities,
actual active IDs, all 12 reference generations and four reference losses
against C19, matched timed prefix arrays, stable prediction IDs, completed
cleanup, and correct reference-dependent conclusions. See [runner commands](../model_quality.md)
and [mask constraints](../layer_masks.md). Add `--skip-layers 17,18`,
`--step-timing-trials 5 --step-timing-warm-prefix-tokens 8` to the C19 release
command; omit skip flags for the control. Use fresh report destinations.

Implementation commits are contract `9674452`, integration `04739c3`, runner
`261faf9`. Independent fixture fix `fe48028` explicitly selects dense semantics
for the legacy minimal-forward test. Validation passes 45 CUDA tests with
cuBLAS, 21 without, the full warning-as-error non-CUDA suite, both Clippy lanes,
six runner CUDA tests, 13 focused CPU tests, release build and formatting.

Next is C21's bounded fixed 16-of-32 sparse-Q8 format/kernel experiment against
matched-zero dense weights. Layer influence ranking remains C24. Keep these
masks diagnostic, with no runtime promotion. Exact HTTP targets 53.06 and
55.9416 tok/s remain unmet; changing model depth does not satisfy them.

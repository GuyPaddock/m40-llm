# Calibrated SwiGLU Input Sparsity: 2026-10-25

## Decision

C23 is complete as a bounded approximate-model experiment. The synthetic
selection-plus-down speed gate does not translate into comparable full-model
gains. Targets 50/60% improve matched-prefix step rate about 4.3/6.7%, but held-out
perplexity rises 8.35/17.38%. Target 25% is slower, and target 40% regresses a
dense-valid task. Keep application explicit and research-only; promote none
of these thresholds or kernels to normal generation.

The original model file and weights are unchanged, but discarding activations
changes model semantics. These results do not satisfy either unchanged-Qwen
HTTP speed objective. Global compressed-KV, sampling and all-layer defaults
are unchanged. C22/C27 remain postponed under the separate negative static
sparse-weight gate. Next is C24's calibration-based layer influence/ranking.

## Frozen Calibration and Execution

All five release variants use clean source
`5697e645ff8878d30752db044d53e3bd1d19cba5`, executable SHA-256
`f974287442bde8e3cbc9f9c1ccfd096f07da5fe7c0a0560af1713d421e149f84`.
Rust 1.95.0, NVCC 12.8.93, g++ 14.3.1, CUDA `-O3`, sm_52 SASS plus compute_52
PTX, cuBLAS enabled, no fast math. The selected C11 native-Q8 engine and C07
parallel head128 attention remain the reference. Dense KV/context 1024 is
explicit. No graphs, materialization, prepared execution-weight policy,
layer masks, packed prefill or server scheduling is used.

The unchanged Qwen2.5-3B Q8_0 GGUF SHA-256 is
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
The [frozen C19 corpus](2026-10-25-model-quality-dense-reference.md) SHA-256 is
`2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38`.
Capture and evaluation share base-execution SHA-256
`faf6939dfd4c3447858236bd4f76dfe9b12857f6bc0678f4708881ee2c31b890`.
Complete environment, model/tokenizer identities and compiler flags are in
each manifest. Both resident models execute serially; residency is not
concurrent inference.

The [calibration artifact](2026-10-25-activation-calibration.json) was frozen
before any variant or held-out evaluation, SHA-256
`a1721123485a8f1195f751538c56dc1db4dc450592df312b44cd58fc06f7fcc9`.
The post-matrix hash is identical. All targets 0/25/40/50/60% and width16/128
were declared beforehand. No threshold is fitted to held-out/browser output.

Capture observes completed SwiGLU vectors at generated-input positions only,
with a predeclared cap of 32 per task. Prompt inputs and the final sampled token
that is never forwarded are excluded. Every layer observes the same 110 rows:

| Calibration Task | Prompt Tokens | Generated Tokens | Captured Rows | Absolute Positions |
| --- | ---: | ---: | ---: | --- |
| chat-reminder | 86 | 21 | 20 | 86-105 |
| fact-depot | 73 | 11 | 10 | 73-82 |
| multi-delivery | 101 | 34 | 32 | 101-132 |
| config-retry | 80 | 2 | 1 | 80 |
| arithmetic-invoice | 61 | 16 | 15 | 61-75 |
| long-photo | 69 | 79 | 32 | 69-100 |

All six capture outputs and actual generated IDs exactly match C19's dense
calibration reference. Capture readbacks/barriers are diagnostic, not a speed
measurement. Observation payload is 174,366,720 bytes, reserved capacity
304,349,184 bytes; fitting takes 5701.533 ms. These are bounded early-generation,
token-weighted observations, not a representative model-wide activation corpus.

## Quality

The reference passes 3/6 calibration and 4/6 held-out tasks. Three calibration
and two held-out dense failures remain inconclusive at every target, including
cases a candidate happens to pass. Fixed rubrics and prompt templates are
unchanged. Loss covers the same 102 calibration and 101 held-out target tokens.

| Target | Candidate Passes Cal / Held | New Dense-Valid Failures Cal / Held | Calibration PPL | Held-Out PPL | Held-Out PPL Change |
| --- | --- | --- | ---: | ---: | ---: |
| 0%, native control | 3/6 / 4/6 | 0 / 0 | 31.556848 | 39.612159 | 0% |
| 25% | 3/6 / 4/6 | 0 / 0 | 32.530219 | 40.025773 | +1.04% |
| 40% | 3/6 / 3/6 | 0 / 1 | 34.130595 | 40.582514 | +2.45% |
| 50% | 4/6 / 4/6 | 0 / 0 | 32.764237 | 42.920954 | +8.35% |
| 60% | 4/6 / 4/6 | 0 / 0 | 35.955195 | 46.495763 | +17.38% |

Target 40% returns the correct workshop facts, then repeats an unfinished JSON
object and reaches 64 tokens instead of the reference's natural 39-token stop.
The fixed format rubric fails. It is not a missing-fact failure, and the grader
is not relaxed. Targets 50/60% preserve that task's exact generated trace, but
their higher loss still rules out a claim of unchanged quality. Task behavior
and calibration loss are non-monotonic across thresholds.

Full-vocabulary teacher-forced drift is measured on identical observed prefixes:

| Target | Cal Max / Mean Absolute Drift | Held Max / Mean Absolute Drift | Held Mean Top-Ten Overlap |
| --- | --- | --- | ---: |
| 0% | 0 / 0 | 0 / 0 | 10.000 |
| 25% | 3.098878 / 0.107549 | 1.854109 / 0.077633 | 9.634 |
| 40% | 6.251141 / 0.235104 | 5.351094 / 0.175163 | 9.188 |
| 50% | 15.506486 / 0.372343 | 12.521607 / 0.312733 | 8.832 |
| 60% | 20.660709 / 0.532106 | 16.773723 / 0.441693 | 8.554 |

Means weight each scored token equally; vocabulary size is fixed. Raw records
retain per-token target rank/logit/NLL, top IDs and generated divergence steps.
All five dense references match all 12 C19 task traces and all 203 reference
teacher-forced token statistics. The zero control is exact for every task,
full logit vector and timed prediction, not merely within a relaxed tolerance.

## Achieved Sparsity and Timing

Counters report actual omitted inputs. The aggregate below weights retained/
input counts across each split's six tasks and two loss passages. Every case
resets counters; these are disjoint populations. Prompt inputs are included,
so these percentages are not decode-only or the fitted 110-row population.
Summary-record counters after timing are not pooled again.

| Target | Achieved Cal / Held | Per-Layer Cal Range | Per-Layer Held Range |
| --- | --- | --- | --- |
| 0% | Native bypass, unmeasured | Not selected | Not selected |
| 25% | 24.00% / 23.80% | 18.31-26.56% | 17.92-26.12% |
| 40% | 37.86% / 37.76% | 26.88-41.55% | 26.24-40.95% |
| 50% | 47.31% / 47.06% | 38.00-51.25% | 35.77-50.51% |
| 60% | 56.82% / 56.61% | 48.99-60.36% | 47.40-59.66% |

The existing C20 timing method forces the same first passage prefixes after
eight warm tokens, 42 timed prediction steps for calibration and 43 for held-out.
Five pairs alternate order per split. Reset, scratch creation, warmup, counters
readback and model preparation are outside timing. Selection, sparse down,
all remaining forward/logits work, greedy argmax/readback and final GPU
completion are included. Predicted IDs never feed the next prefix; each model's
predictions are stable across trials. Each report contains 425 paired steps.

| Target | Cal Reference / Candidate Median ms | Cal Paired Rate Gain | Held Reference / Candidate Median ms | Held Paired Rate Gain |
| --- | --- | ---: | --- | ---: |
| 0% | 1162.926 / 1162.020 | +0.079% | 1193.165 / 1192.418 | +0.059% |
| 25% | 1164.567 / 1174.313 | -0.823% | 1193.229 / 1203.642 | -0.875% |
| 40% | 1165.167 / 1139.846 | +2.215% | 1193.962 / 1168.355 | +2.197% |
| 50% | 1164.579 / 1115.583 | +4.413% | 1193.945 / 1144.477 | +4.325% |
| 60% | 1164.165 / 1092.057 | +6.603% | 1193.348 / 1118.570 | +6.684% |

Gains are medians of paired `reference_ms/candidate_ms - 1`, not latency
reductions. Within-cell MAD/median is below 0.049% for both models; raw pairs
are preserved. The zero difference is noise, not a gain. Five pairs do not
characterize broader workload variation; more confirmation is needed for a
borderline promotion. None is attempted. Target 40% is also below the exact
3% screening threshold, which is not relaxed for this experiment.

These are matched-prefix prediction rates, not free-generated or HTTP TPS.
The isolated synthetic 50% screen's 30.4% down-latency reduction becomes only
about 4.3% complete matched-step throughput here. Actual activations, retained
input distributions and unchanged gate/up/QKV/attention/logits all matter.
Requested load savings are not measured DRAM traffic.

Every task also retains ordinary prefill/decode/total timing, caller wall time,
actual output IDs/counts and stop reason. One observation per free generation
is not an E2E promotion gate. The workshop example makes unequal work explicit:

| Target | Pass | Generated Tokens | Prefill ms | Decode ms | Generation Total ms | Stop |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0% | Yes | 39 | 2750.371 | 1056.165 | 3806.551 | stop |
| 25% | Yes | 39 | 2775.639 | 1062.933 | 3838.587 | stop |
| 40% | No | 64 | 2684.692 | 1705.545 | 4390.250 | length |
| 50% | Yes | 39 | 2623.061 | 1007.130 | 3630.205 | stop |
| 60% | Yes | 39 | 2558.472 | 984.249 | 3542.735 | stop |

The 40% decode total increases because it emits 25 more tokens; that total is
not an isolated projection slowdown. Other tasks can also change output length.
Caller wall times have a wider boundary and remain separately available.

## Memory, Setup and Safety

Each model retains 3,279,519,744 uploaded weight bytes and 37,748,736 dense KV
bytes. Positive thresholds prepare all 36 down tensors, adding 862,322,688
bytes, 822.375 MiB or 26.29% of original uploaded weights. Source weights are
not reclaimed. One shared selection workspace adds 88,068 bytes and 36 pairs
of u64 counters add 576 bytes, totaling 862,411,332 extra requested device bytes.
Capacity does not shrink with selected count. No model or KV compaction occurs.

The zero control uses native projection for all 36 layers and allocates no
prepared weights, selection scratch or counters. Its application still costs
8993.110 ms for identity/validation; positive preparation costs 10280.775-
10538.725 ms including uploaded-payload identity checking and packing/upload.
These are single setup observations, not timed decode or packing-only latency.
Candidate load/setup is 15.20-16.53 s versus 5.94-6.03 s for each native reference.
File hashing recorded later is outside that load boundary.

Both models report 66,560 bytes of head128 workspace after use. Combined tracked
allocation after warmup is 7,421,999,700 bytes for positive variants; it includes
both resident models and workspaces, and is neither standalone nor peak VRAM.
Native KV allocations are separately accounted. Every terminal record reports
tracked bytes 0 -> 0. After the matrix, the M40 has no compute process and reports
0 MiB/0% GPU use; clocks are not locked. The final 60% command was initially
blocked by an overlap precaution; it ran only after the earlier process exited
and GPU idleness was verified. There are no duplicate/overlapping timed rows.

Capture/application are model-local and immutable during sessions. Identity,
tensor shape, budget and unsupported-mode errors fail before publishing a
partial cache. Positive application adds deterministic compaction and down
launches without per-token allocation or host retained-count readback. The
zero full-model control deliberately keeps native reduction order: forcing
repacking at threshold 0 previously crossed an FP16 KV rounding boundary in a
two-layer fixture. That 0.00449 logit difference was not hidden by loosening the
tolerance. Independent same-zeroed sparse-kernel tests still cover threshold 0.

## Validation

Post-measurement M40 validation passes the CUDA library, activation capture/
application, selected-input/tiled-Q8 parity, 11-case attention grid, projection/
MLP wrappers, 24-case forward suite and five transfer-order cases. The focused
activation/attention/projection suite also passes with cuBLAS disabled. All
seven shared quality-runner tests pass with explicit hardware inclusion.
The optional real-Qwen forward fixture without a supplied model is not counted
as additional quality evidence; the five explicit-model reports are.

Both warning-as-error CI lanes pass: full non-CUDA locked tests and NVCC
locked `--all --no-fail-fast` with `CUDA_VISIBLE_DEVICES=''`. New hardware
checks are explicitly ignored in the no-device lane. An explicit ignored-test
request with the device hidden fails at context creation, as required; see
[the negative-check log](2026-10-25-activation-calibration-no-device-negative.txt).
GPU-less CI is not numerical parity evidence. CUDA/server all-targets Clippy
passes with cuBLAS on/off; non-CUDA all-targets Clippy, fmt, whitespace and
local report-link checks pass. NVCC's Maxwell deprecation warning remains.

All tests/builds run after the timed matrix, not during it. Source
`scripts/dev-env.sh` before these representative hardware commands:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --lib --test activation_calibration_cuda --test sparse_activation_cuda \
  --test q8_tiled_projection --test attention_parity_cuda_grid \
  --test proj_wrappers --test mlp_wrappers --test forward_with_layer_smoke \
  --test cuda_transfer_order -- --include-ignored --test-threads=1
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=0 cargo test --features cuda \
  --test activation_calibration_cuda --test sparse_activation_cuda \
  --test q8_tiled_projection --test attention_parity_cuda_grid \
  --test proj_wrappers --test mlp_wrappers \
  -- --include-ignored --test-threads=1
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --example model_quality -- --include-ignored --test-threads=1
```

## Artifacts and Reproduction

- [Frozen calibration](2026-10-25-activation-calibration.json).
- [Capture records](2026-10-25-activation-capture.jsonl) and
  [capture log](2026-10-25-activation-capture-log.txt).
- [Zero control](2026-10-25-activation-quality-0.jsonl).
- [Target 25%](2026-10-25-activation-quality-25.jsonl).
- [Target 40%](2026-10-25-activation-quality-40.jsonl).
- [Target 50%](2026-10-25-activation-quality-50.jsonl).
- [Target 60%](2026-10-25-activation-quality-60.jsonl).
- [Derived summary and identity audit](2026-10-25-activation-quality-summary.jsonl).

All five files contain 36 records and complete cleanup. Capture SHA-256 is
`3d7934b2304b89aaadc58074b03babe2442da827d7578c7a96d8d6ee4e11e9a9`.
Quality file SHA-256 values in target order 0/25/40/50/60% are:

```text
7521ff45585f5ff95ed031183ca94161c15d12b348f516e1309efb638622c0c7
5595c31ef31b209ea6e84bf7b1b379bd1e58d23aa9069c6a17514425ab532b3f
0b977003955fd63e7c2aa152fb2f820c7b2c92526e97930422c59f9e207c4e3d
ee2430880421fdc5c353fe5e53fcd369a14f72a813940730de576a977feb7223
a27f04ce1c52c8766328a6b852ec3c1f7b76b81deac9f3ec3c50ec855d1ebd29
```

See [API restrictions and complete commands](../activation_sparsity.md).
Implementation commits are artifact contract `c89b939`, model-local runtime
`1f2a65c`, and shared capture/evaluation runner `5697e64`. No experiment or
threshold is silently selected by the normal CLI/server.

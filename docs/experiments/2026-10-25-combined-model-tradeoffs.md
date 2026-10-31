# Frozen Layer/Activation Composition

## Decision

C26 now includes combined-model measurements, not just individual comparisons.
Keep both combinations research-only. Frozen mask `[8]` plus 50% activation
target improves held-out matched-step rate 7.012%, with PPL up 11.82%.
The 60% combination improves rate 9.247%, with PPL up 26.73% and a new
calibration config-lookup failure. Individual quality effects do not safely
compose, and neither result satisfies the unchanged-model objective.

## Predeclared Scope and Identity

The clean implementation/measurement revision is
`26a3a5dd7a85edf64c4b2f304a33bb78f5ecb841`, release executable SHA-256
`f8d2cc8cc90e752f2d49ebf498c2d524b82b090552958bfcc3f23377396cd2f5`.
It uses Rust 1.95.0, NVCC 12.8.93, g++ 14.3.1 and the unchanged C11 CUDA
`-O3`/sm_52/compute_52 flags without fast math. The complete environment,
model/tokenizer hashes, outputs and actual counts are in each manifest.

The original Qwen2.5-3B Q8_0 source remains SHA-256
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
No compact file or layer renumbering is used. The frozen ranking SHA-256 is
`a9f0f2cc8c952eacaf96d6fe2576e30a35ef225a5f85f4574630545af3ab4d3c`,
and frozen activation artifact SHA-256 is
`a1721123485a8f1195f751538c56dc1db4dc450592df312b44cd58fc06f7fcc9`.
Both bind the same C19 corpus and base execution. There is no new capture,
threshold fitting, ranking or held-out selection.

Before measurement, declare `[8]` with targets 0/50/60%, both frozen splits,
five alternating timing pairs, context 1024, explicit dense KV, sequential
prefill and greedy top1. The reference keeps all 36 layers; the candidate
keeps original IDs and skips only 8. Graphs, materialization, prepared lossless
execution policies, compressed attention and server scheduling remain off.

All GPU work is serial. Device queries show no competing compute process
before each run. The zero control starts at 33 C; positive-run boundary
samples are preserved below. Clocks are not locked. No tests, builds,
other inference or MemPalace work overlaps timing.

## Controls and Routing

The [zero-control audit](2026-10-25-combined-zero-control-audit.json) matches
all 12 current candidate task traces and all 203 teacher-forced token
statistics against C24's frozen masked candidate. Full references match the
frozen full model as well. Zero thresholds allocate no prepared storage.
This historical comparison checks recorded token statistics, not archived
full-vector logits. The separate live three-layer CUDA test proves exact
full-logit masked-native/zero-threshold parity.

The [three-run audit](2026-10-25-combined-model-audit.jsonl) verifies complete
36-record runs, clean source, unchanged base environment/corpus, matching full
reference traces/loss tokens, and original-layer routing. Masked layer 8 has
zero retained/input counts in every case; each positive active layer records
inputs under its original ID. All runs return tracked device bytes 0 -> 0.

## Quality and Achieved Sparsity

| Mask + Activation Target | Cal / Held Passes | New Dense-Valid Failures Cal / Held | Cal PPL | Held PPL | Held PPL Change |
| --- | --- | --- | ---: | ---: | ---: |
| Full reference, no transform | 3/6 / 4/6 | 0 / 0 | 31.556848 | 39.612159 | reference |
| `[8]` + 0%, control | 4/6 / 5/6 | 0 / 0 | 31.458426 | 40.984621 | +3.46% |
| `[8]` + 50% | 4/6 / 4/6 | 0 / 0 | 33.867852 | 44.293885 | +11.82% |
| `[8]` + 60% | 3/6 / 4/6 | 1 / 0 | 39.179986 | 50.199624 | +26.73% |

The 60% combination returns `19` instead of `6` on `cal-config-retry`,
diverging at generated step 0. The full reference and both individual
components pass this row. Reference/candidate generate 2/3 tokens and both
stop naturally. This is a measured composition regression, not a reason to
refit thresholds or relax the answer. The 50% combination keeps every
dense-valid task but still incurs larger loss than either component alone.

The three calibration and two held-out dense-failed rows remain inconclusive.
Do not interpret an extra candidate pass on those rows as an improvement.
There are only 102/101 loss tokens per split, not a broad quality benchmark.

Achieved sparsity pools each split's task and passage counter populations
once, excluding timing-summary snapshots. Prompt inputs are included:

| Target | Achieved Cal / Held | Full-Logit Max Absolute Difference Cal / Held |
| --- | --- | --- |
| 50% | 47.320% / 47.013% | 16.403569 / 11.994549 |
| 60% | 56.853% / 56.604% | 23.540970 / 19.048495 |

Zero uses native bypass and has no selected-input count. Skipped-layer
inputs are excluded naturally because their counters are zero. Calibration
thresholds are reused unchanged even though layer removal shifts activations.

## Matched-Step and Ordinary Generation Timing

| Target | Cal Reference / Candidate ms | Cal Rate Gain | Held Reference / Candidate ms | Held Rate Gain |
| --- | --- | ---: | --- | ---: |
| 0%, control | 1163.430 / 1134.892 | +2.507% | 1193.362 / 1163.630 | +2.561% |
| 50% | 1163.896 / 1087.513 | +7.017% | 1195.471 / 1117.707 | +7.012% |
| 60% | 1164.822 / 1063.021 | +9.580% | 1193.699 / 1092.645 | +9.247% |

These are five alternating pairs of identical observed prefixes, including
selection/down, remaining forward/logits, greedy readback and final stream
completion. Reset/setup/warmup/counter readback are outside the interval.
Predictions do not feed subsequent prefixes and are stable within each model.
The zero control measures 35 versus 36 active layers, not a new zero-threshold
kernel speedup. Raw MAD/median values are retained, all below 0.08%.

All ordinary generation rows retain prefill/decode/total, caller wall time,
token IDs/counts, output and stop reason. They are single observations and can
do unequal work. No controlled generated/HTTP E2E gain, confidence interval or
default promotion is claimed, and individual percentages are not added.

## Allocation and Preparation

Both models still upload 3,279,519,744 weight bytes and allocate 37,748,736
native dense KV payload bytes across 36 physical layer slots. Positive
combinations keep all 36 original prepared down tensors, including unused
layer 8: 862,322,688 bytes plus 88,068 selection scratch and 576 counter bytes.
Total extra is 862,411,332 bytes, about 822.46 MiB. Zero adds none.
The shared head128 workspace is 66,560 bytes/model after use. This is not
compact export or standalone peak VRAM; native KV is outside the Rust ledger.

Positive activation application records 10,547.507/11,192.620 ms for 50/60%,
including source identity, packing and upload. Reference/candidate load
intervals are 6080.966/16532.428 and 6227.701/17378.778 ms, respectively.
Zero load is 5982.308/14790.553 ms. Artifact preflight and later file hashes
have different boundaries; no complete cold-I/O or amortization claim follows.
Previously frozen calibration/ranking is reused, not measured again.

## Reproduction and Validation

Use the selected native-Q8 environment from the
[C24 command](../layer_ranking.md#frozen-ranking-runner), then:

```bash
target/release/examples/model_quality \
  --reference-model "$model" --model "$model" \
  --layer-ranking docs/experiments/2026-10-25-layer-ranking.json \
  --remove-layer-count 1 \
  --activation-calibration docs/experiments/2026-10-25-activation-calibration.json \
  --activation-target 0.5 --variant-id combined-mask8-target50 \
  --split all --context-tokens 1024 --step-timing-trials 5 \
  --output /tmp/new-combined-target50.jsonl
```

Run zero first, audit its historical control, then 50/60 serially with fresh
output paths. Do not retune artifacts on these results or apply original
threshold identities to a compact model.

Hardware validation passes three activation tests, 17 runner tests including
real compact Qwen parity, 46 cuBLAS-on baseline checks and 22 cuBLAS-off checks.
Both warning-as-error CI lanes, non-CUDA/CUDA-server all-targets Clippy, fmt
and whitespace checks pass. The explicit new hardware test fails without a
GPU rather than skipping. All code checks precede timed runs; this report
adds no code or new measurement.

Initial commands omitted the compact test's explicit paths and used a
nonexistent layer-mask target. Corrected commands pass; neither invocation
error is a runtime or timing result. The preserved logs distinguish them.
See [the runner suite](2026-10-25-composition-runner-validation.txt),
[cuBLAS-on parity](2026-10-25-composition-parity-on.txt),
[cuBLAS-off parity](2026-10-25-composition-parity-off.txt),
[explicit missing-device failure](2026-10-25-composition-no-device.txt),
[missing model paths](2026-10-25-composition-missing-model-paths.txt) and
[mistyped target](2026-10-25-composition-mistyped-target.txt).

## Artifacts

- [Zero raw JSONL](2026-10-25-combined-mask8-target0.jsonl), SHA-256
  `abb86f73d23ae0e641067b0d168e8585e4224f6e076aa16e140fd5dac4ecc8e7`.
- [50% raw JSONL](2026-10-25-combined-mask8-target50.jsonl), SHA-256
  `9b0f1c43a0ad0eef7bc2c5ac1bf89f8a21a15c28aa8f1b277b8dc8b4c5e0a909`.
- [60% raw JSONL](2026-10-25-combined-mask8-target60.jsonl), SHA-256
  `ff6b6eb8bcce22228ec57ad40c3a5d2b9a66b19223618c9fe75a5e5ae4ef8f99`.
- [Build](2026-10-25-combined-model-build.txt), execution logs for
  [zero](2026-10-25-combined-mask8-target0-run.txt),
  [50%](2026-10-25-combined-mask8-target50-run.txt),
  [60%](2026-10-25-combined-mask8-target60-run.txt).
- GPU boundary samples for [50%](2026-10-25-combined-target50-gpu-before.csv)
  and [60%](2026-10-25-combined-target60-gpu-before.csv).

No model, sampling, attention, server or graph default changes. The original
53.06 tok/s and 1.30x fresh-Ollama goals remain unmet; model-changing gains
cannot satisfy them.

# Compact Layer-8 Live Equivalence

C25 is complete. The separate 35-layer export exactly reproduces the source
with original layer 8 masked, while reclaiming physical weight and KV storage.
It does not improve the masked model's quality or establish another speedup.
The unchanged 36-layer model remains the exact-inference reference.

## Identities and Scope

The clean release runner is `0dfff9306191a85a0441abd96d2ec7bffb690e0f`,
executable SHA-256
`41ce877955658d0336fdfbd07945be4b6c02d0bf17f0207194f64d07ac78405e`.
The original Qwen2.5-3B-Instruct Q8_0 source SHA-256 is
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
The compact SHA-256 is
`7d04260214a3b57379310e963c4f6b4d39208be9b644d45c83aefa4430d03070`.
The [completed export record](2026-10-25-compact-layer8-export.json) binds
the frozen ranking, C24 quality report and unchanged C19 corpus.

Rust 1.95.0, NVCC 12.8.93 and g++ 14.3.1 build conventional sm_52 CUDA with
`-O3`, compute_52 PTX and no fast math. The complete environment and build
flags are in the [48 raw records](2026-10-25-compact-layer8-quality.jsonl),
SHA-256 `6517f8705e985d34c381c721cda4a72c3da1911819a7f9fec31f763340933f2c`.
The [build log](2026-10-25-compact-layer8-quality-build.txt) and
[execution log](2026-10-25-compact-layer8-quality-run.txt) are preserved.

The selected C11 native Q8 engine and C07 head128 parallel attention use
explicit dense KV/context 1024, greedy top1, sequential prefill, no graphs,
no materialization and no prepared execution-weight layouts. The source alone
has the frozen `[8]` mask; the compact model runs all 35 retained layers.
Both models are resident, but all inference is serial. The M40 was idle at
27 C with no compute processes before launch. No other test, build, inference
or MemPalace work overlaps the measured run.

## Equivalence and Quality Cost

All 12 tasks produce identical token IDs, text, token counts and stop reasons.
All stop naturally. Full-vocabulary differences are zero at 951 task prefixes,
including every final-prompt and first-decode boundary, and at all 203 scored
passage prefixes. These task diagnostics are outside generation/paired timing;
their dummy target is never forwarded and is not corpus loss.
All warm and timed forced-prefix greedy predictions match as well.

| Split | Masked Source / Compact Task Passes | Scored Tokens | Both PPL | Full-Logit Max / Mean Difference |
| --- | --- | ---: | ---: | --- |
| Calibration | 4/6 / 4/6 | 102 | 31.458426 | 0 / 0 |
| Held out | 5/6 / 5/6 | 101 | 40.984621 | 0 / 0 |

The [independent record audit](2026-10-25-compact-layer8-quality-audit.json)
also matches all current masked-source task traces and teacher-forced token
statistics against C24's frozen candidate. No mask or rubric is fitted again.
These counts are equivalence evidence, not unchanged-source quality approval.
C24's unmodified held-out PPL is 39.612159, so the compact model retains the
3.46% cost. Dense-failed tasks remain inconclusive for quality improvement.

## Actual Requested Storage

| Storage | Masked Source | Compact Export | Reclaimed |
| --- | ---: | ---: | ---: |
| Successfully owned uploaded weights | 3,279,519,744 B | 3,197,605,888 B | 81,913,856 B |
| Native dense KV payload capacity | 37,748,736 B | 36,700,160 B | 1,048,576 B |
| Physical KV layer slots | 36 | 35 | 1 |
| Head128 attention workspace after use | 66,560 B | 66,560 B | 0 B |
| Materialized / prepared weights | 0 B | 0 B | 0 B |

Owned upload requests fall 78.119 MiB or 2.498%. Dense KV falls 1 MiB or
2.778% at context 1024 without changing heads, dimensions or token capacity.
The native KV figure is successfully allocated payload capacity; native length
metadata and CUDA allocator overhead are excluded. It is not in the Rust
allocation ledger. Do not add both models' shared tracker snapshots as if
they were standalone peaks. That ledger returns from zero to zero after both
models are dropped; these records do not measure driver peak VRAM.

## Timing and Preparation

Five alternating pairs time identical observed prefixes with greedy readback
and final stream completion. Reset, scratch setup and warm prefixes are outside
these timings. This is not free-generated TPS or HTTP throughput.

| Split | Masked Median | Compact Median | Median Paired Rate Change |
| --- | ---: | ---: | ---: |
| Calibration | 1135.154 ms | 1134.544 ms | +0.0466% |
| Held out | 1164.965 ms | 1164.296 ms | +0.0603% |

This difference is far below the 3% promotion threshold. No extra trials or
speed claim are warranted. Both execute 35 layers. C24's separate +2.484%
held-out matched-step result compared 35 active layers against 36; it must
not be attributed again to file compaction or added to this noise.

The [export preparation checkpoint](2026-10-25-compact-layer8-export-checkpoint.md)
records 38.683 s through create-new publication, including hashes/copy/fsync.
This run separately spends 26.296 s on consistency/hash preflight and
6.313/13.061 s on source/compact load. Subsequent identity hashes are outside
those load timers. These one-shot values are not a cold-page-cache experiment
or evidence that compact files load more slowly in general.

## Reproduction and Validation

Use the selected environment from C24's manifest, then the
[live comparison command](../compact_gguf.md#live-masked-source-comparison).
Use new output paths; the runner never overwrites evidence. Check complete
records, not successful file construction or a model label alone.

All 16 runner tests pass on M40 with cuBLAS enabled and disabled, including
the real source/export artifact, reset, full-logit parity and owned allocation
checks. Baseline parity totals are 46 cuBLAS-on and 14 cuBLAS-off checks.
Both warning-as-error CI lanes, non-CUDA and CUDA/server all-targets Clippy,
formatting and whitespace checks pass. An explicit hardware request with
`CUDA_VISIBLE_DEVICES=''` fails at context creation as intended.

The initial suite invocation incorrectly imposed Qwen's head128-only kernel
on five head64 fixtures; the guard rejected them. The corrected test scopes
that override to the real Qwen case, and both complete reruns pass. That failed
debug invocation is not timing evidence. Raw validation logs cover
[runner cuBLAS on](2026-10-25-compact-validation-runner-on.txt),
[runner cuBLAS off](2026-10-25-compact-validation-runner-off.txt),
[baseline parity on](2026-10-25-compact-validation-parity-on.txt),
[baseline parity off](2026-10-25-compact-validation-parity-off.txt),
[explicit missing-device failure](2026-10-25-compact-validation-no-device.txt)
and the [rejected suite environment](2026-10-25-compact-validation-rejected-suite-env.txt).

Normal model files, runtime defaults and exact-engine objectives are unchanged.
C26 next compares separate model-changing quality, speed, memory and
preparation costs. Conditional negative-gated research remains postponed.

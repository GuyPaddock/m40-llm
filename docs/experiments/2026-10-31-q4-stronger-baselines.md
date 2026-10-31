# Sparse Q4 passes the stronger-reference timing gate

All twelve comparisons pass the frozen complete-FFN gate. Supplied-index
Q4 execution at 10% and 20% retention is faster than both full-retention
packed Q4 and the qualified Q8 compound/packet FFN. Screening, selection and
recovery still have to fit a small remaining budget. Output drift remains
unqualified; this result does not promote a runtime mode.

The clean release source was `c16559ee5412820609ba4a9ffc58230d5f3dd3f8`.
The [frozen protocol](2026-10-31-q4-stronger-baselines-protocol.md) used
the original captured batch-one input at layers 0, 17 and 35, with exact-Q4
hidden-magnitude importance selections. Every arm retained all three Q4
owners plus the existing Q8 split, joint and packet diagnostic owners.
Peak tracked residency was 4,745,945,272 bytes. No timed workload overlapped.

## Qualified Q8 comparisons

Times are contemporaneous complete-FFN CUDA-event medians in milliseconds.
The remaining budget uses `Q8 / 1.05 - selected Q4`, converted to microseconds.
It is a point estimate for retaining a 5% gain, not a confidence bound.

| Layer | Retained | Q8 packet ms | Selected Q4 ms | Speedup | Remaining budget, microseconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 10% | 0.324390 | 0.156682 | 2.070x | 152.26 |
| 17 | 10% | 0.324803 | 0.155686 | 2.086x | 153.65 |
| 35 | 10% | 0.324778 | 0.158432 | 2.050x | 150.88 |
| 0 | 20% | 0.324598 | 0.250483 | 1.296x | 58.66 |
| 17 | 20% | 0.324554 | 0.249312 | 1.302x | 59.79 |
| 35 | 20% | 0.324627 | 0.252275 | 1.287x | 56.89 |

All six lower gain bounds are positive. The smallest is 28.67%, for layer 35
at 20% retention. Use the tightest layer budgets, 150.88 microseconds at 10%
and 56.89 microseconds at 20%, when screening the next implementation.
These budgets cover rotation, score computation, top-k, recovery and any
additional launches or data movement together.

Against full-retention packed Q4, the 10% cases improve by 7.21x to 7.35x
and the 20% cases by 4.52x to 4.59x. This isolates a substantial incremental
benefit from skipping channels after the full-retention kernel/layout gain
observed in the first grid. Full-retention packed Q4 itself takes about
1.141 ms here and is slower than qualified Q8.

Each arm warmed for at least one second of GPU work. All twelve comparisons
completed five alternating pairs, sixty pairs total, with ten repetitions
per interval. No variation extension or borderline confirmation was needed.
GPU events, host submission/completion and dispatch counters are in the raw
artifact; setup, uploads, scoring and numerical readbacks are outside timing.

## Numerical and ownership evidence

All nine Q4 cases pass their independent masked FP64 reference. The largest
absolute implementation error is 0.000072494. Existing Q8 captured-input,
FP64 and intermediate checks also pass. Q8 output repeats bit for bit;
Q4 output remains within its original tolerance after timing. Immutable
owners, inputs, guards and per-layer allocation cleanup pass. After process
exit the M40 has no compute process and zero allocated MiB.

The approximate outputs differ from full Q8. For layer 35, mean absolute
FFN-output differences are 0.496625 at full Q4 retention, 0.788820 at 10%,
and 0.647166 at 20%. The other layers' values and maxima are retained too.
These are differences at one captured FFN input per layer. They establish
neither a model-accuracy failure nor acceptable generation quality.

## Decision and next work

Proceed to one bounded screening-cost experiment. The required budget is
now measured against the qualified Q8 path, rather than inferred from the
much slower native-Q4 consumer. Stop a retention setting if screening alone
exceeds its remaining budget; do not spend that setting's remaining work on
recovery or runtime integration. A cheaper screen still needs complete
screening-plus-selection-plus-recovery timing and quality validation.

The paper's screen rotates inputs and weight blocks, estimates gate/up
activations with ternary codes, and ranks channels using down-column norms.
Its low-read configuration examines 80 of each 128 rotated coordinates;
rank-32 weight-derived recovery approximates skipped output contributions.
These MoE mechanisms guide the next dense-Qwen experiment; their reported
accuracy does not transfer automatically.
[Primary paper, sections 5.1 to 5.3](https://arxiv.org/html/2610.37386v1#S5)

The source and benchmark remain opt-in. No teacher generation, training,
new corpus admission, model-quality run, whole-model timing or runtime
default change occurred.

## Artifacts and reproduction

- [Timing and numerical records](2026-10-31-q4-stronger-timing.jsonl.gz)
- [Summary and per-layer budgets](2026-10-31-q4-stronger-summary.json)
- [Run log](2026-10-31-q4-stronger-timing.log.gz) and [release-build log](2026-10-31-q4-stronger-release.log.gz)
- [CPU tests](2026-10-31-q4-stronger-tests.log.gz) and [Clippy](2026-10-31-q4-stronger-clippy.log.gz)

Use the [original reproduction command](2026-10-31-q4-channel-screen.md),
restoring its exact capture environment. Add `--ffn-q4-channels-baselines`,
use `--rows 1 --iterations 10 --trials 5`, omit correctness-only, and select
a new output path. The benchmark's 65 CPU tests and CUDA/server all-target
Clippy passed. Twenty unrelated hardware tests ignored in the CPU invocation
are not counted as passes.

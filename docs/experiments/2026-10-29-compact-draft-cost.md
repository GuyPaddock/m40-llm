# Compact Drafter Cost Checkpoint

## Decision

The one fixed rank256 proposal passes its **cost-only admission**. Across five
trials and 1000 complete decisions, median latency is **1.5380845ms**, p95
**1.5555975ms**, p99 **1.5633085ms**, below the frozen **3.884ms** median limit.
No alternate architecture, kernel, vocabulary restriction or geometry was tried.

Stop here for the required user checkpoint. There is no trained drafter,
measured acceptance, teacher dataset or live speculative HTTP result. The next
decision is whether to approve a frozen data/licensing/split manifest and a
small training pilot, not whether to promote a runtime mode. No training or
device switch is authorized by this cost result alone.

## What Was Measured

Source `c9c1853`, release Rust, the existing CUDA12.8/sm52 build flags and
cuBLAS, no new CUDA kernel. All 138 compiled CUDA instruction streams match
the qualified controls. The original Qwen2.5-3B Q8_0 model and dense context1024
KV are resident, with qualified compound and joint-only ownership. The exact
portfolio and the removed ILP4 probe are not selected.

Three independent affine maps transform a synthetic 2048-float hidden vector
to three width256 SiLU rows. They share one full151936-vocabulary FP32 matrix.
Parameters are deterministic, untrained coefficients, not downloaded weights.
The target is loaded but never called to produce teacher features or labels.

Primary wall-clock timing includes the 8192-byte D2D feature snapshot, two
cuBLAS calls, bias addition, SiLU, three original argmax launches, one decode
stream synchronization, 12-byte D2H and ID-range validation. All buffers are
persistent. Every 200-call trial reports exactly 400 cuBLAS calls, 1000 CUDA
launches, 200 synchronizations and 200 D2H copies, with no device allocation,
free or H2D upload. D2D is timed but absent from the existing counter schema.

One warmup interval of at least one second precedes five serial trials. GPU
UUID is `GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852`, P0, SM1113/memory3004.
Only this process appears before/after each timed trial. No builds, embeddings,
other inference or A4000 work overlapped timing.

| Trial | Median complete proposal, ms | p95, ms |
| --- | ---: | ---: |
| 0 | 1.5384420 | 1.5552035 |
| 1 | 1.5387880 | 1.55552595 |
| 2 | 1.5377515 | 1.5581045 |
| 3 | 1.5376455 | 1.5553675 |
| 4 | 1.5378035 | 1.5548742 |

Overall min/max are 1.525980/1.790907ms. This is an absolute cost-admission
measurement, not a paired runtime speedup claim. Five separate event-instrumented
diagnostics follow the primary trials: vocabulary is about0.708ms and the
three original argmax operations about0.754ms. Their extra events/syncs perturb
the intervals; do not sum them into an independently measured request saving.
No argmax replacement or other rescue was attempted.

## Correctness, Memory And Setup

All 768 hidden values and 455808 vocabulary logits pass the unchanged FP64
tolerance. Maximum absolute errors are 1.325033e-7 and 5.418706e-8 respectively.
Greedy IDs match the independent FP64 oracle on all three rows. These arbitrary
synthetic IDs are not evidence of target-model prediction quality. Inputs and
parameters remain bit-identical; tracked bytes return to zero after teardown.

| Resource or cold interval | Measured |
| --- | ---: |
| FP32 parameters | 161,876,992 bytes |
| Additional parameters, feature and scratch device bytes | 163,728,908 bytes |
| Additional scratch/feature bytes | 1,851,916 bytes |
| Retained host parameter payload | 161,876,992 bytes |
| Synthetic coefficient creation | 120.972585ms |
| Parameter upload/allocation | 23.336161ms |
| All proposal allocation/upload | 23.634416ms |
| Qualified target load in this process | 16,741.985264ms |

Parameter hashes and exact owner accounting are in the raw setup record.
Synthetic coefficient creation is not a measured trained-checkpoint load.
CUDA events/context/cuBLAS internal allocations are not included in logical
DeviceBuffer accounting. Existing target setup is not charged repeatedly as
proposal latency. No DRAM-traffic, achieved-occupancy or warp-stall claims.

The cost-probe session consumed **0.005923 GPU-hours**, conservatively including
resident setup and CPU validation. Training, teacher generation and acceptance
evaluation consumed zero hours. The 4/20/40/8-hour phase caps and 72-hour total
remain unchanged. Any M40/A4000 switch still requires explicit confirmation.

## Acceptance Still Required

The [updated cost model](2026-10-29-compact-draft-model.json) imports the
preserved verifier recurrence without modifying it. It retains measured
full-batch target costs, rejection correction/rollback, bookkeeping, finite
tails, ineligible decisions and historical HTTP overhead. Five perfect-draft
reproductions match the existing measured decode totals. All numbers below
are **projections**, not measured HTTP throughput or learned acceptance.

Using the measured median proposal cost and equal conditional acceptance at
each of the three guessed positions:

| Eligible coverage | Acceptance needed for 53.06 HTTP tok/s | For 56.153149 |
| --- | ---: | ---: |
| 100% | 88.4824% | 92.5948% |
| 75% | 89.3271% | 93.7749% |
| 50% | 90.9866% | 96.0774% |

These thresholds satisfy all five measured target-cost scenarios. Charging
p95 proposal latency on every decision changes the full-coverage stronger
threshold to92.6131%. At95% equal conditional acceptance and full coverage,
projected median HTTP throughput is58.250944 tok/s; at90%,54.264701. Perfect
acceptance projects62.823848. None is an achieved runtime result.

At95% acceptance, adding an unmeasured1ms per verifier attempt lowers the p95
cost scenario to57.372647 projected median HTTP tok/s; adding5ms lowers it to
54.162274. Live target cache interference, the exact learned feature tap,
serialization and late rejection costs remain limitations. Do not infer that
a 1.538ms proposal automatically meets the benchmark goal.

## Review Checkpoint

Cost no longer rules out this fixed architecture. Acceptance near93% remains
unproven and demanding for three independent low-rank heads. Before teacher
generation or training, freeze and review:

- Exact dataset revisions/IDs, attribution and target-output license terms.
- Document/template-isolated train, calibration and new held-out splits, with
  existing benchmark continuations and frozen answers excluded.
- Causal alignment: h(t) can predict t+2/t+3/t+4; t+1 is the known target seed.
  Later hidden features must never enter a proposal at t.
- A fixed optimizer, seed, training schedule and small-pilot stop criterion,
  with joint conditional acceptance, full-batch success and coverage evaluation.
- Bounded teacher feature throughput/storage measurements before committing
  to the planned dataset size, and a second review after the small pilot.

This report does not freeze an incomplete data manifest or start a training
job. The qualified reference remains50.910426 HTTP tok/s and the target remains
56.153149. Exact portfolio, approximate numerics and drafting results remain
separate; no runtime defaults change.

## Reproduction And Artifacts

Use `scripts/dev-env.sh` and the exact environment in
`2026-10-27-compound-reference.json`, then the preserved source:

```bash
cargo build --release --features cuda,server --example joint_only
target/release/examples/joint_only \
  --model /mnt/array-fastest/home/guyep/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d \
  --ownership joint-only --phase compact-draft-cost --output NEW_COST_REPORT.jsonl
```

- [Pre-measurement protocol](2026-10-29-compact-draft-cost-protocol.md)
- [Raw samples/setup/validation](2026-10-29-compact-draft-cost.jsonl)
- [Independent raw-result audit](2026-10-29-compact-draft-audit.json)
- [Audit program](2026-10-29-compact-draft-audit.jq)
- [Compiled-control identity](2026-10-29-compact-draft-compiled-audit.json)
- [Cost-model program](2026-10-29-compact-draft-model.pl)
- [Artifact checksums](2026-10-29-compact-draft-cost.sha256)

Executable, SASS and stderr are retained in
`local-artifacts/2026-10-29-compact-draft/`. Non-CUDA warning-as-error tests,
the cost-statistics test and CUDA/server all-target Clippy pass. The preceding
ILP4 removal also passes all seven retained compound/joint/four-row hardware
tests. The independent raw audit accepts the original data and rejects six
corruptions; the unchanged verifier recurrence passes its five behavior tests.
The attention parity grid passes all eleven cases. All34 immutable reference
artifact/model/executable/archive checksums still match.

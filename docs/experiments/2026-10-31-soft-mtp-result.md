# Soft MTP compatibility and cost assessment

The bounded Soft-MTP study is complete as a training-method and feasibility
assessment. It finds no drop-in change for the current frozen-target drafter.
Retain the existing three-guess cost candidate and its unresolved acceptance
requirement. No teacher generation, dataset acquisition, training, inference
integration or runtime-default change follows from this result.

## Training compatibility

[Multi-Token Prediction Done Right, v1](https://arxiv.org/html/2610.38331v1)
uses a common linear vocabulary projection for NTP and MTP states. It
subtracts the aligned NTP cross-entropy gradient from the MTP hidden-state
gradient through stop-gradient routing. Upstream modules receive the
self-distillation gradient; the output matrix retains cross-entropy training.
The reported training-overhead result compares with an existing MTP training
pipeline, not a separate inference-only drafter. This is a training change,
not a verification-kernel optimization.

The local [fixed candidate](2026-10-29-compact-draft-training.md) has three
2048-to-256 affine/SiLU maps and a learned 256-to-151936 vocabulary matrix.
Its [cost-probe implementation](../../examples/joint_only/compact_draft/reference.rs)
confirms these dimensions and independent synthetic coefficients. The target
uses its own 2048-to-151936 projection. Sharing the drafter matrix among three
heads does not mean sharing it with the target.

Inference from these dimensions: a target hidden-state gradient of width2048
cannot be subtracted directly from a drafter gradient of width256. Replacing
the independent output matrix with a shared target projection would change
the frozen architecture and invalidate its measured cost. Explicit teacher
distribution distillation is another possible objective, but the existing
greedy-label/full-vocabulary-CE proposal does not supply those distributions.
Neither change is admitted here. This incompatibility does not prove that
head-only distillation or Soft-MTP adaptations are impossible.

Offline alignment must still preserve the local causal contract: at prefix
ending at t, h(t) predicts t+2, t+3 and t+4. The target already knows t+1.
Later target states can supply training labels or supervision, never inference
features for the proposal at t. Any future alternative objective needs its
own frozen architecture, data approval, cost probe and held-out acceptance
study before comparison with the current candidate.

## Rechecked local evidence

This assessment reran the retained CPU cost audit and finite-request model.
The regenerated audit and model are byte-identical to their committed outputs.
All five recurrence behavioral cases pass, and all five perfect-draft request
reproductions match. The verifier's 18 checksum entries and compact-draft
probe's 12 entries pass in the original repository, including retained
executables and archives. Worktree-only checking initially found two absent
ignored verifier artifacts; the full original-repository check resolves those
paths. See [replay evidence](2026-10-31-soft-mtp-evidence.json).

The [complete verifier evidence](2026-10-28-direct-verifier-feasibility.md)
contains a 57.820ms median four-row target pass. Four rows mean one known seed
plus three guesses. They do not mean four independently accepted proposals.
The [synthetic proposal probe](2026-10-29-compact-draft-cost.md) costs
1.5380845ms median and 1.5555975ms p95 per complete decision. It includes a
feature snapshot, projections, activation, full vocabulary, argmax,
synchronization and ID readback. Its 1000 decisions establish execution cost,
not learned prediction quality.

The unchanged [finite cost model](2026-10-29-compact-draft-model.json) gives
the following minimum equal conditional acceptance for the historical
56.153149 HTTP tokens/s scenario, satisfying all five target-cost samples:

| Eligible coverage | Median proposal cost | P95 proposal cost |
| --- | ---: | ---: |
| 100% | 92.5948% | 92.6131% |
| 75% | 93.7749% | 93.7988% |
| 50% | 96.0774% | 96.1122% |

These are projected thresholds, not observed acceptance or HTTP performance.
At equal conditional acceptance p, rejection/acceptance probabilities are
1-p, p(1-p), p²(1-p), p³. Corresponding useful progress is 2,3,4,4 input rows
after ordinary correction. The model charges proposal work on ineligible
decisions, target verification, rollback, correction, bookkeeping and the
finite 511-input-row request tail. Rejected guesses are not credited as work
saved. It reserves 554.538ms historical HTTP overhead and 0.370ms extra session
setup; these remain assumptions, not a fresh HTTP measurement.

At 95% conditional acceptance and full coverage, the median-cost scenario
projects 58.250944 HTTP tokens/s. With p95 proposal cost and another unmeasured
1ms/5ms per verifier attempt, projected medians fall to 57.372647/54.162274.
The probe's 23.634416ms allocation/upload cost is separately charged in its
first-request sensitivity. Trained-weight loading and live learned feature
capture remain unmeasured. Target rejection costs are already charged and
must not be counted a second time.

## DSpark interface review

[PR 31372](https://github.com/ggml-org/llama.cpp/pull/31372) was open at the
October 31 inspection. The pinned head is
`cf6bf9d9a095d15ad81256a4ac87af5059f6bac9`. Its eight-file patch concerns Spark
conversion, model metadata and graph construction. The PR description also
claims probabilistic drafting and GPU rejection sampling, but those claims
are not established by this patch. In the inspected
[common/speculative.cpp](https://github.com/Septend-fun/llama.cpp/blob/cf6bf9d9a095d15ad81256a4ac87af5059f6bac9/common/speculative.cpp),
the `is_dspark` branch selects `cur_p->data[0].id` and appends the token;
it does not populate proposal distributions there. Other draft paths use
`result_q` and request temperature/seed. Do not treat the description as a
verified probabilistic DSpark implementation or import its acceptance table.

The useful local interface distinction is proposal tokens versus proposal
probabilities versus target decisions. The current exact-greedy verifier needs
tokens and causal state, then commits the accepted prefix, clears rejected KV
and executes correction. A later stochastic verifier would additionally need
normalized conditional proposal probabilities, sampler state, rejection and
residual-sampling validation. Greedy parity cannot qualify that algorithm.

## Coverage and stopping decision

No new draft-length/context sweep is needed to settle this method-compatibility
decision. Existing measurements resolve the four-row candidate's cost, but
do not establish a learned acceptance rate. The measured workload starts at
65 prompt tokens and covers its 512-token continuation; later wrong-prefix
rejection costs are modeled from early rejection samples. Neither long-context
verification nor other draft lengths are declared qualified by this report.
They become necessary measurements for a compatible admitted candidate, along
with its joint context, coverage, cost and rejection distribution.

The existing [live-controller correctness evidence](2026-10-28-draft-live-correctness.md)
preserves greedy IDs, stop behavior, valid KV and rollback across 21 prompt-copy
cases. That proves controller behavior for its source, not Soft-MTP acceptance.
Training remains outside this branch's executable scope under the
[branch protocol](2026-10-30-soft-mtp-protocol.md). No compatible trained
artifact was established. A separately admitted candidate with measured
conditional acceptance can reopen integration; this report does not close the
independent lossless-KV, quantized-KV or cooperative-serving studies.

## Reproduction

From this branch, the following CPU-only checks replay the retained evidence:

```bash
perl docs/experiments/2026-10-28-direct-verifier-model.pl --self-test
jq -s -f docs/experiments/2026-10-29-compact-draft-audit.jq \
  docs/experiments/2026-10-29-compact-draft-cost.jsonl
perl docs/experiments/2026-10-29-compact-draft-model.pl
```

Compare the last two outputs with their existing audit/model JSON files.
Run both original checksum manifests from the original repository, where
`local-artifacts/` retains the executable/archive inputs. No GPU command or
training program is part of this replay.

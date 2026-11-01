# N-Gram Feasibility: 2026-10-25

## Decision

C11A's formal browser screen **fails**. The best optimistic estimated HTTP E2E
throughput gain is **4.949%**, below the fixed **10%** threshold, even with a
1.0x verifier and zero drafting/commit/rollback/transfer/correction overhead.
Postpone C12-C17, record C18, then proceed to C19's separately quality-tested
model research. Do not build the speculative verifier for this drafter/workload
or tune another lookup policy to force a pass in this task.

No speculative GPU implementation or speedup is measured here. Ordinary
generation, native Q8/C07 benchmark settings, and runtime/KV defaults do not
change. The qualified measured result remains 31.850 HTTP tok/s versus
43.032 Ollama; both original objectives remain unmet.

This conclusion applies to the frozen lookup rule and recorded browser
continuation. It is not a universal upper bound on prompt lookup or evidence
that neural drafting, different workloads, or future verification cannot help.

## Inputs and Implementation

Use the selected actual m40 trace from the [C11 checkpoint](2026-10-25-exact-kernel-checkpoint.md),
not the mismatching prepared continuation. Both the prompt and output IDs are
actual captured IDs that reproduce the measured output. The 65 prompt tokens
and 512 length-limited generated tokens use Qwen2.5-3B-Instruct Q8_0, full dense
FP16 KV/context 1024, greedy sampling, and C07 parallel head128 split8/256.
Every accepted source trial verifies fresh KV and stable output.

Source revision is `f5cda0d`; study/capture revision is `739c195`, with clean
initial worktrees. Lookup is independently implemented in `fe392cd`, with no
change to model/kernel dispatch. The native model/runtime configuration is
unchanged from C11. Model SHA-256 is:

```text
4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d
```

The selected configuration fingerprint is
`cff7373a76ff1cf3d08b1e5cacb78c55344bb3f09e33c717aaf4e23b5a1f4561`.
The C11 input artifact SHA-256 is
`89a61055972c13c3d6f25264becc7ebf1f93675441349a2c5d73696a7a220f55`;
the regression capture SHA-256 is
`8db7487cb95df409e881f586a117c75ac1608a0b1fa01ac7b178f1eb93f445ba`.
Full provenance, including the analysis executable hash, is in the study manifest.

Lookup searches suffix lengths 3-8, chooses the longest, then the most recent
usable occurrence. Proposals borrow only already-known continuation tokens,
with no extrapolation beyond history. The first stop token and remaining
output budget cap drafts. Reference future tokens are available only to the
offline scorer. Rejection commits the accepted prefix plus correction; full
acceptance emits a bonus only when it exists. Post-EOS traces are rejected.

See [tool semantics and commands](../ngram_feasibility.md). The concept has a
[llama.cpp prompt-lookup reference](https://github.com/ggml-org/llama.cpp/blob/master/examples/lookup/README.md);
no upstream implementation code is copied.

## Browser Acceptance

At all four limits, **70/512 positions, 13.672%**, have a usable match. That is
not the simulated decoder's eligibility: committed rounds skip intermediate
positions. The visited-round results are:

| Candidate limit | Visited rounds | Eligible rounds | No-match rounds | Mean offered / eligible round | Mean accepted / eligible round | Total accepted draft tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 489 | 47 | 442 | 2.000 | 0.489 | 23 |
| 4 | 487 | 45 | 442 | 3.978 | 0.578 | 26 |
| 8 | 485 | 43 | 442 | 7.791 | 0.651 | 28 |
| 16 | 485 | 43 | 442 | 15.023 | 0.651 | 28 |

The stderr label `accepted/offer` means accepted tokens per eligible proposal,
not the accepted fraction of individual offered tokens. JSONL names both
quantities separately. Candidate16 offers 646 tokens versus 335 for candidate8,
but accepts the same 28 and commits the same rounds. Larger drafts do not
recover useful new continuations on this trace.

| Candidate limit | Zero rejection | Partial rejection | Full acceptance | Accepted-prefix histogram, length:count |
| --- | --- | --- | --- | --- |
| 2 | 33/47, 70.21% | 5/47, 10.64% | 9/47, 19.15% | 0:33, 1:5, 2:9 |
| 4 | 33/45, 73.33% | 9/45, 20.00% | 3/45, 6.67% | 0:33, 1:4, 2:4, 3:2, 4:2 |
| 8 | 33/43, 76.74% | 8/43, 18.60% | 2/43, 4.65% | 0:33, 1:4, 2:3, 3:1, 7:1, 8:1 |
| 16 | 33/43, 76.74% | 9/43, 20.93% | 1/43, 2.33% | 0:33, 1:4, 2:3, 3:1, 7:1, 8:1 |

Candidate8's 28 accepted draft tokens save 27 rounds, not 28: the final
budget-limited accepted prefix has no bonus token. All 24 case/limit studies
reconstruct the exact reference IDs. Raw records retain every round's position,
match, proposal, accepted prefix, committed tokens, and completion kind, plus
full position-level and visited-round histograms.

## Cost Sensitivity

The baseline is measured C11 median HTTP 16075.239 ms, internal prefill
1682.991 ms, and decode 14347.922 ms. The model fixes HTTP minus decode at
1727.317 ms and calibrates a target step at 14347.922 / 511 = 28.078 ms.
That fixed residual retains prefill and response costs but is not a measured
CPU stage; independent medians and different phase boundaries prevent that
interpretation.

For R ordinary/verification round cost units, estimated HTTP time is
fixed_residual + max(R - 1, 0) * target_step_ms. One prefilling prediction is
credited. Each no-match round costs one unit; each eligible draft round costs
the assumed verifier factor, irrespective of its acceptance. Actual drafting,
commit, rollback, copies, and extra correction processing are not charged.
These are deliberately optimistic engineering assumptions, not measured GPU
verification or a proof of the best possible implementation.

Estimated **throughput** changes against the measured ordinary baseline:

| Candidate limit | Verifier 1.0x | Verifier 1.5x | Verifier 2.0x | Verifier 4.0x | Estimated HTTP at 1.0x |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | +4.185% | -0.087% | -4.023% | -17.089% | 15.429 s |
| 4 | +4.566% | +0.439% | -3.375% | -16.117% | 15.373 s |
| 8 | +4.949% | +0.970% | -2.719% | -15.122% | 15.317 s |
| 16 | +4.949% | +0.970% | -2.719% | -15.122% | 15.317 s |

CPU drafting totals in the non-CUDA release run are 3.846/3.562/3.855/3.615 ms
over visited rounds, about 7.3-8.0 us per lookup. These are single-run CPU
observations with timer overhead, not repeated latency claims. They are
reported separately and ignored by the zero-overhead formal screen. Including
overheads would not rescue a result already below the required 10%.

## Frozen Regression Checkpoint

Before inspecting acceptance, freeze the existing canonical arithmetic/OK/config
smokes plus one binary-search explanation and one Java palindrome request.
Each is captured twice serially on the same M40 with identical actual IDs and
output. The first three stop naturally; the latter two reach their 128-token
limits. No output is extended to fill a budget. Short smoke outputs are `4`,
`OK`, and `37`; full explanation/code output and IDs remain in the capture.

| Case | Generated IDs | Eligible positions | Visited rounds at limits 2/4/8/16 | Accepted tokens at limits 2/4/8/16 |
| --- | ---: | ---: | --- | --- |
| Arithmetic | 2 | 0 | 2 / 2 / 2 / 2 | 0 / 0 / 0 / 0 |
| Exact OK | 2 | 0 | 2 / 2 / 2 / 2 | 0 / 0 / 0 / 0 |
| Config lookup | 3 | 0 | 3 / 3 / 3 / 3 | 0 / 0 / 0 / 0 |
| Explanation | 128 | 2 | 128 / 128 / 128 / 128 | 0 / 0 / 0 / 0 |
| Java code | 128 | 13 | 122 / 121 / 121 / 121 | 6 / 7 / 7 / 7 |

The code prompt has some repetition but cannot override the browser gate.
Regression captures are untimed in-process quality/ID observations, not HTTP
performance baselines; the study deliberately emits no E2E cost estimates for
them. These are not C19's calibration or held-out model-modification corpus.

## Reproduction and Validation

Use the [capture/study commands](../ngram_feasibility.md), selecting the committed
C11 JSONL and the captured regression artifact below. Analysis is release Rust
1.95.0. A non-CUDA release build reproduces all acceptance, replay, and cost
results from the CUDA-feature executable's CPU path exactly; only CPU timing
observations differ. Both source worktrees were clean at analysis start.
The non-CUDA executable also rejects `capture` explicitly with exit 1 before
opening any model/output, not a passing skip.

Lookup's five behavioral/exhaustive tests pass. The study's eight CPU and
CUDA-build tests cover future isolation, all rejection positions, subsequent
history, correction/bonus/EOS/budget handling, no-match baseline calibration,
cost sensitivity, and rejection of wrong/incomplete trace sources. Non-CUDA
warning-as-error tests, both Clippy lanes, and formatting pass. No CUDA kernel
or production inference logic changes in this unit.

Artifacts:

- Formal non-CUDA release [study JSONL](2026-10-25-ngram-study.jsonl) and
  [run/build log](2026-10-25-ngram-study.txt).
- Actual [regression IDs/output](2026-10-25-ngram-regressions.jsonl) and
  [capture log](2026-10-25-ngram-capture.txt).
- Initial CUDA-feature CPU-path [confirmation JSONL](2026-10-25-ngram-cuda-feature-confirmation.jsonl.gz),
  compressed without changing its records.

The gate record has passes_browser_screen=false, a complete regression
checkpoint, may_proceed_to_verifier_feasibility=false, and no measured speedup
or runtime default change. All GPU allocations are released after capture;
the final query reports 0 MiB / 0% utilization.

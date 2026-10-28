# Causal Draft-Acceptance And Cost Feasibility

## Decision

**General browser acceleration: no-go for every measured source.** The new
verifier is profitable with perfect drafts, but cheap CPU sources do not
provide enough correct guesses on novel prose/code. Do not reopen lookup
tuning, implement a neural drafter without evidence, or claim the original
HTTP target is reached.

**Verbatim copying: go for the bounded research implementation, completed.**
The unchanged prompt-copy source now drives the real direct verifier and
passes all correctness and paired performance gates on two predefined copy
workloads. It remains explicit research-only. No automatic class routing,
retry/fallback, kernel change, runtime default or server integration is added.

The immutable **50.910426 HTTP tok/s** checkpoint, preferred joint-only ownership
and **53.06 / 56.153149 HTTP tok/s** goals remain unchanged. Specialized copy
requests are not progress against the browser512 benchmark without separate
controlled qualification on that exact workload.

## Candidates, Data And Causality

Source/prompt revision `d2f8070` freezes three rules before new target capture:
historical longest-suffix lookup 3-8, prompt-only copy lookup 4-16, and online
most-frequent backoff orders 1-4 with stable token-ID ties. All request state
is fresh. Counts may learn only the current prompt and committed past, never
other held-out outputs. No rule or parameter was tuned after inspecting results.

Each proposal receives one already-computed target seed and produces three
guesses. Known seed acceptance is not counted as a guessed success. Guessing
stops is declined rather than padded. Eligibility uses the configured output
budget, never a future reference EOS. Offline futures are available only to
scoring/commit simulation. The live runner uses actual target predictions;
frozen IDs are checked only after evaluation, not supplied as drafts.

Disclosed development data are browser512 and twelve previously used frozen
tasks. The eight new held-out prompts cover prose, code, literal copying and
structured extraction at nominal 128/384 prompt lengths. Actual lengths are
130-137 and 393-397, with 52-128 actual output IDs. Each held-out continuation
was captured twice using qualified serial compound/joint-only/dense generation,
with identical actual IDs/counts/stops. All eight fit the measured target-cost
context range. The copy live follow-up reuses these fixed cases; it is not an
additional independent holdout or broad production qualification.

The [independent causal audit](2026-10-28-draft-audit.pl) reconstructs all 3,543
visited decisions and 621 proposals from prompt/past alone. It checks joint
conditional counts, rejection progress and all-in cost sums across five trials
for every one of the 63 source/case combinations. The [result](2026-10-28-draft-audit.json)
passes. Future/EOS isolation and rejected-guess state tests also pass.

## Ranked Approaches

| Approach | Cost/acceptance evidence | Decision |
| --- | --- | --- |
| Prompt-copy lookup | Measured CPU cost and live target performance; perfect conditional acceptance on two literal-copy tasks; poor coverage elsewhere | Viable only as explicit copy-specialized research drafter; implemented |
| Historical history lookup | Measured, cheap; browser first-guess acceptance 39.0%, full-batch acceptance 7.3% | No general implementation/promotion; original negative gate remains valid |
| Online backoff counts | Measured, cheaper but browser first-guess acceptance 30.8%; frequent wrong proposals | No-go; lower CPU cost cannot offset target rejection cost |
| Smaller vocabulary-compatible model | Could reduce proposal weight work, but artifact compatibility, real acceptance, prompt setup, three-token latency and draft-KV rollback are unmeasured | Pending feasibility only; not a viable measured candidate yet |
| Learned auxiliary heads / hidden-state drafter | Could avoid full draft-model passes, but trained artifact, disjoint training data, acceptance and all-in cost are absent | Unmeasured; no implementation, training or EAGLE in this task |

All three CPU sources succeed on the two simple copy tasks; prompt-copy is not
uniquely successful there. Its declared input-copy role and least-harmful
development browser behavior justify selecting it for the conditional live
prototype. No source qualifies as a general browser accelerator. Unknown neural
costs are not assigned zero or substituted with a perfect-drafter oracle.

## Joint Acceptance And CPU Costs

Five release CPU replays per source/case measure construction, every proposal
including misses, four-candidate assembly and committed-history maintenance.
Offline future scoring/JSON diagnostics are excluded. CPU is Xeon E5-2697A v4,
Rust 1.95.0; scheduling/frequency are not controlled. Raw trials and MAD/ranges
are reported, not presented as paired GPU confidence. Projections charge the
largest complete CPU cost across the five trials, including initialization.

Browser512, actually visited decisions:

| Source | Eligible / decisions | Conditional acceptance guesses 1 / 2 / 3 | Full-batch acceptance | All-in CPU median / max ms per request | Projected resident gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| History lookup | 41 / 447 | 39.0% / 43.8% / 42.9% | 7.3% | 3.680 / 3.707 | -10.32% |
| Prompt-copy | 5 / 498 | 100% / 60.0% / 66.7% | 40.0% | 1.443 / 1.468 | -0.12% |
| Online backoff | 156 / 275 | 30.8% / 66.7% / 28.1% | 5.8% | 0.358 / 0.372 | -31.21% |

The historical [2026-10-25 study](2026-10-25-ngram-feasibility.md) failed its
10% optimistic browser screen at only +4.949%, before real verifier/correction
costs. That rejection is preserved. The current baseline reuses its unchanged
lookup routine but adapts to the known seed and fixed three-guess interface;
it is a new negative measurement, not a reinterpretation of the old result.

Held-out resident projections, medians across the five measured target samples:

| Task | Actual prompt / output tokens | History lookup | Prompt-copy | Online backoff |
| --- | ---: | ---: | ---: | ---: |
| Prose, short | 130 / 128 | -2.23% | +0.94% | -29.77% |
| Prose, long | 395 / 128 | -0.16% | -0.13% | -17.01% |
| Code, short | 137 / 128 | -3.23% | +0.68% | -14.49% |
| Code, long | 397 / 128 | -1.90% | -1.17% | -17.44% |
| Copy paragraph | 132 / 52 | +12.48% | +12.48% | +12.47% |
| Copy config | 393 / 68 | +6.49% | +6.49% | +6.49% |
| Extraction, short | 134 / 79 | +1.09% | -1.81% | -20.40% |
| Extraction, long | 397 / 58 | -2.96% | -0.03% | -12.60% |

For prompt-copy on the two copy cases, visited eligibility is 80.0%/84.2%,
conditional acceptance is [1,1,1], and full-batch acceptance is 100%. Maximum
all-in CPU cost is 0.034/0.083 ms per whole request. On long extraction it is
ineligible everywhere; source initialization and misses still cost time.
Good conditional acceptance on a handful of positions is not broad coverage.

The [joint summary](2026-10-28-draft-summary.json) retains every source/case's
conditional denominators, full acceptance, rejection histogram, position-wise
versus visited coverage, absolute-context bins, CPU variation, construction
cost, logical host payload and projections together. The
[raw study](2026-10-28-draft-study.jsonl) contains all five timed visit paths,
proposal IDs/provenance and per-decision proposal/maintenance costs. Logical
payload excludes standard-library hash/tree overhead; it is not peak RSS.

## Cost Model And HTTP Scenarios

The model replays **actual joint visit/rejection paths**, not one IID acceptance
average. Each path uses all five measured complete-verifier/serial samples,
context-dependent target costs, observed rollback, ordinary correction,
finite output tails and all-in CPU overhead. A target-produced stop ends
without an unnecessary correction; it never suppresses a prior legal attempt.
Extra direct-session setup and loop work remain charged. Fresh serial capture
supplies prompt/setup/teardown anchors for the new contexts. Old development
tasks without suitable fixed-cost anchors receive decode-only projections.

All held-out contexts fit measured serial positions 65-575 and four-row starts
65-569. Clamped/extrapolated positions on older development tasks are marked.
Late wrong-draft target costs remain unmeasured; the model charges observed
early overhead and at least the measured ordinary correction at the absolute
position. Additional +1/+5 ms per attempt are explicit uncertainty scenarios.
They are not claimed upper bounds or alternate acceptance gates.

HTTP projections add a clearly borrowed historical overhead reservation. For
browser, use the largest original fixed HTTP residual, 554.538 ms. For fresh
prompts, retain measured resident prompt/setup time and add the positive gap
between that historical residual and the corresponding old resident fixed
cost. These unpaired intervals do not measure new HTTP/client buffering.

| Copy case | Projected HTTP tok/s range | Projected HTTP gain range | Resident +1 ms/attempt scenario | Resident +5 ms/attempt scenario |
| --- | ---: | ---: | ---: | ---: |
| Paragraph | 29.471-29.602 | +11.24%-11.33% | +11.59%-11.67% | +8.35%-8.42% |
| Config | 15.994-16.024 | +6.20%-6.28% | +6.04%-6.12% | +4.41%-4.49% |

These are projections for different short-answer workloads, not measurements
or claims against the browser512 HTTP target. The completed verifier model
still requires about 86.9% conditional acceptance for a 1 ms drafter to clear
the resident 3% floor, or 92.0% for the historical-overhead HTTP target scenario,
assuming high eligibility. At p=0.95 the all-in HTTP-target draft budget is
3.884 ms per three-guess decision; at p=0.90 even a free drafter misses that
target scenario. A neural candidate must fit its own prompt/state/rollback/
transfer costs in the budget as well as proposal computation. None has done so.

## Conditional Live Implementation And Stop

The [live measurement](2026-10-28-draft-live-measurement.md) completes the user's
conditional implementation request. Real causal prompt-copy gives **+12.7955%**
resident throughput on paragraph copy and **+6.6723%** on config copy, with
paired intervals [12.6511%,12.8623%] and [6.6499%,6.8529%]. Five alternating
release pairs per case pass the unchanged gates without extensions. Complete
timing includes setup, proposals/misses, history, target transactions, prompt
and teardown. These are real drafts, not frozen continuations fed to target.

Across all 21 preflight cases, actual IDs/counts/stops match, 1,544 full-vocabulary
vectors and 85,948,416 valid KV words are bit-identical, and 516,096 rejected
suffix words are checked zero across 13 natural rejections. The original
numeric tolerance and ownership/transaction requirements are unchanged.
The same copy source has no broad browser claim and is not preferred globally.
The runner is greedy, dense, batch-one and restricted to the qualified
Qwen2.5-3B setup. It adds no stochastic, compressed-KV or server-batch support.

Go/no-go rules remain: causal access, exact greedy behavior, complete numerical
and KV/cleanup checks, meaningful full-request gain above `max(3%,2*variation)`
with positive paired bounds, equal actual arm counts/stops and exclusive GPU
access. Offline projections cannot promote a runtime. Broader copy promotion
would need new independent prompt shapes/lengths and regressions, not tuning
against this final held-out set. New general drafting approaches need separate
artifacts and measured acceptance/cost mechanisms. No further source search,
neural training, EAGLE, fallback policy or default change follows automatically.

Reproduction commands and retained executables are in the live report. CPU
replay is `draft_feasibility study --heldout PATH --output NEW_PATH` in a release
build; capture is a separate explicit CUDA operation. The
[summary transformation](2026-10-28-draft-summary.jq) runs with `jq -s -f` on
the raw study. [Live audit](2026-10-28-draft-live-audit.pl) rejects eight altered
reports and independently reproduces paired intervals. All 138 compiled CUDA
entries and 113 prior checksums are unchanged. Tests and logs are linked from
the measurement report. This bounded task is complete; stop for review.

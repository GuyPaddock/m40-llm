# Hardware-adaptive drafting results

The frozen adaptive policy does not pass the broad-workload gate. Browser and
several non-copy cases remain slower than modeled serial execution. No non-copy
case reaches a 3% complete-request gain. The two known copy cases retain their
previous benefit, with the same attempted batches as always-attempt copying.
This is a negative controller study, not a new general speculative decoder.

The release study completed all 126 trace/policy/scale combinations and 630 CPU
trials. It uses all 21 retained traces without changing inputs or policy after
seeing results. The eight formerly held-out captures are retrospective inputs
here. No GPU speedup, HTTP throughput, fresh held-out generalization or optimal
draft length is established by this replay.

## All cases at the measured hardware scale

A/B denotes always-attempt/adaptive. Coverage is attempted batches divided by
visited decisions for the adaptive path. The histogram counts executed batches
accepting 0/1/2/3 guesses, excluding the known target seed. CPU time is the
maximum of five complete adaptive CPU trials in milliseconds. Gains are median
modeled complete-resident-request percentages over five historical target pairs,
with zero additional unknown per-attempt overhead. Negative means slower.
Unknown means no qualified fixed prefill cost exists for that trace; it is not
zero. The summary artifact retains ranges, conditional acceptance, all CPU
trials, decision reasons, decode-only costs and sensitivity scenarios.

| Trace | Attempts A/B | Coverage % | Histogram | CPU max ms | Resident gain % A/B |
| --- | ---: | ---: | --- | ---: | ---: |
| browser | 5/3 | 0.595 | 0/2/0/1 | 1.496 | -0.124 / -0.27 |
| cal-arithmetic-invoice | 0/0 | 0 | 0/0/0/0 | 0.022 | unknown / unknown |
| cal-chat-reminder | 1/1 | 5.556 | 0/1/0/0 | 0.102 | unknown / unknown |
| cal-config-retry | 0/0 | 0 | 0/0/0/0 | 0.004 | unknown / unknown |
| cal-fact-depot | 0/0 | 0 | 0/0/0/0 | 0.039 | unknown / unknown |
| cal-long-photo | 0/0 | 0 | 0/0/0/0 | 0.237 | unknown / unknown |
| cal-multi-delivery | 0/0 | 0 | 0/0/0/0 | 0.188 | unknown / unknown |
| held-arithmetic-crates | 0/0 | 0 | 0/0/0/0 | 0.005 | unknown / unknown |
| held-chat-tour | 0/0 | 0 | 0/0/0/0 | 0.094 | unknown / unknown |
| held-code-long | 3/1 | 0.794 | 1/0/0/0 | 2.504 | -1.176 / -0.765 |
| held-code-short | 7/7 | 6.422 | 1/1/0/5 | 0.935 | 0.687 / 0.681 |
| held-config-cache | 0/0 | 0 | 0/0/0/0 | 0.007 | unknown / unknown |
| held-copy-long | 16/16 | 84.211 | 0/0/0/16 | 0.11 | 6.486 / 6.486 |
| held-copy-short | 12/12 | 80 | 0/0/0/12 | 0.028 | 12.485 / 12.485 |
| held-extract-long | 0/0 | 0 | 0/0/0/0 | 1.185 | -0.036 / -0.037 |
| held-extract-short | 2/2 | 2.703 | 0/2/0/0 | 0.536 | -1.805 / -1.807 |
| held-fact-observatory | 0/0 | 0 | 0/0/0/0 | 0.027 | unknown / unknown |
| held-long-garden | 1/1 | 1.25 | 1/0/0/0 | 0.245 | unknown / unknown |
| held-multi-workshop | 0/0 | 0 | 0/0/0/0 | 0.213 | unknown / unknown |
| held-prose-long | 2/2 | 1.639 | 0/1/0/1 | 2.647 | -0.128 / -0.136 |
| held-prose-short | 3/3 | 2.542 | 0/0/1/2 | 0.675 | 0.949 / 0.949 |

The browser median worsens from -0.124% to -0.270%, despite attempting fewer
batches. Decisions change the visited path, so fewer attempts do not by
themselves prove a saving. Long code improves from -1.176% to -0.765% by
reducing three attempts to one, but remains a loss. Short prose and short code
stay below 1%. The copy paths retain all 12 and 16 fully accepted batches;
the tiny changes in their modeled gains reflect measured CPU variation.
Their adaptive complete-request gain ranges are 12.427–12.514% for short copy
and 6.457–6.538% for long copy. These were already specialized positive cases.

Across all primary-scale paths the adaptive policy executes 48 batches versus
52 for always-attempt. It makes ten initial probes, 36 expected-gain attempts
and two recovery probes; twelve proposals are declined for expected loss.
There are 1,338 source misses and fifteen budget-tail decisions, all charged.
All-in CPU time reaches 2.647 ms for one complete request. The twelve task
traces lack fixed prefill anchors and cannot qualify a complete-request gain.
Their decode-only projections are all negative; the detailed summary preserves
them without substituting decode-only gains for end-to-end evidence.

## Hardware and unknown-overhead sensitivities

Only the 1.0 scale uses the actual historical M40 costs. Scaling verifier-side
work by 1.25 or 2 is hypothetical; ordinary serial and correction work keep
their measured costs. It does not describe another GPU.

| Verifier cost scale | Always-attempt batches | Adaptive batches | Adaptive hardware-ceiling refusals | Cases meeting 3% resident gain in all five samples |
| --- | ---: | ---: | ---: | --- |
| 1.0 | 52 | 48 | 0 | Two known copy cases only |
| 1.25 | 52 | 20 | 109 | None |
| 2.0 | 52 | 0 | 174 | None |

The policy responds to cost changes, but source lookup and maintenance still
cost time when it chooses serial execution. At scale 2 the maximum complete
CPU trial is 2.683 ms. No thresholds, history windows or source parameters are
retuned to these results.

At scale 1, adding 1/5 ms of unknown overhead per attempt gives browser median
resident gains of -0.300%/-0.420%. Short copy falls to 11.643%/8.396%, and long
copy to 6.069%/4.437%. These are model sensitivities, not measured transfer or
scheduler costs. The positive copy cases do not satisfy the separate browser
and non-copy non-regression requirement.

## API and cost-model inventory

The preserved direct backend qualifies four input rows: one known target seed
and three proposed guesses. The existing selected-Q4 operator has 1/2/4/8-row
geometry, but that is not a two/three-row full target-verification measurement.
This controller therefore chooses between serial execution and the qualified
four-row verifier. It never divides a four-row cost to invent shorter-batch
performance or claims an optimal draft length.

The policy uses pointwise medians of five retained M40 control/direct pairs,
including rejection, rollback, correction and bookkeeping. First and recovery
probes pay their full cost. Its cumulative acceptance histogram contains only
executed attempts from the current request. Source lookup sees the prompt,
committed history and known seed; future IDs are used only for scoring.
The independent serialized audit reconstructs that boundary and the changing
visited path from retained source records.

The decision table is prepared once from historical hardware calibration before
request timing. CPU timing includes source/controller construction, lookup,
proposal assembly, decision, committed-history maintenance and teardown. Future
scoring and diagnostic serialization are excluded. Five warmed release trials
are run for each case; the cost projection charges their maximum CPU time.

No adaptive verifier attempt starts outside measured positions 65 through 569.
The two arithmetic traces each have two serial rows outside measured serial
positions; the existing boundary-clamped extrapolation remains explicitly
reported. Rejection overhead measured at the early calibration position is
reused later; late-position rejection cost is unmeasured. Historical HTTP
fields remain in raw projections under their original assumptions and are not
new HTTP measurements. These limits prevent treating a small modeled change
as a measured speedup.

## Validation and retained evidence

The implementation at `1892432ebf5944e8ac87a12b1fe2f826ba7652b4` passed fifteen
behavioral example tests, all 220 non-CUDA tests and both non-CUDA and
CUDA/server all-target Clippy before timing. The complete release run began
after the independent KV-quality process terminated and checked for an idle
GPU before and after CPU timing. No live decoder implementation or GPU timing
was added for this negative study.

The retained executable SHA-256 is
`6f97d4622bfe6c320a1660713f2048de10f7b0445e314b4612027b1641b4a831`.
Its source archive and original raw JSONL are retained in
`local-artifacts/qualified-references/adaptive-drafting/run-01/` in the original
repository. The [compressed raw records](2026-10-31-adaptive-drafting-results.jsonl.gz)
contain every trial and decision. The [summary](2026-10-31-adaptive-drafting-summary.json)
is reproducible with [its jq projection](2026-10-31-adaptive-drafting-summary.jq).

The independent [Perl audit](2026-10-31-adaptive-drafting-audit.pl) completes
successfully across all 126 cases, 630 trials and 8,660 causal rounds. It
reconstructs prompt-copy proposals, policy state, committed progress, target
costs and charged CPU time from retained inputs. It checks 224 attempted
batches and 300 skipped proposals across policies and scales. All seven
corrupted-report controls are rejected. The
[audit result](2026-10-31-adaptive-drafting-audit.json) and
[artifact hashes](2026-10-31-adaptive-drafting-artifacts.sha256) retain this
evidence. The audit is independent of the Rust implementation, but shares
the disclosed historical cost assumptions; it cannot validate unmeasured
late-position rejection or live GPU performance.

The [frozen protocol](2026-10-31-adaptive-drafting-plan.md) requires stopping
without policy tuning or live integration if the broad gate fails. That is the
result here. The specialized copy implementation remains an independent record;
no production dispatch, model acquisition, learned-drafter evaluation or training
follows from this experiment.

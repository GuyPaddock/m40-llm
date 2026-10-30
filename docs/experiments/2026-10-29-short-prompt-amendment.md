# Short-Prompt Eligibility And Verifier Cost Amendment

Date: 2026-10-29. Status: paper-only proposal for review. This amendment does
not authorize GPU measurements, teacher generation, training, device switching
or runtime integration. No architecture, training schedule, isolation rule or
budget increase is proposed.

## Scope And Preserved Evidence

The [CPU admission stop](2026-10-29-drafter-data-admission-stop.md) remains valid
under the frozen specification. All 964 MBPP prompts are shorter than its
65-token minimum, including the actual QwenChat wrapper. This is a dataset
eligibility mismatch, not evidence against learned drafting.

Preserve the original [data specification](2026-10-29-drafter-data-spec.json),
SHA256 `6ba7fc6606525bb142de110c7a7cd78cbe26d01a6f1edb8f03646871095d4bcd`,
and [pilot plan](2026-10-29-drafter-pilot-admission.md), SHA256
`42ea10028a648533111a56b9adc2140f6deec4336aa485ae14577bb4aeb3d516`.
Do not rewrite their hashes, failed decision or capacity report. A future
approved specification must have a new version, parent hashes and an explicit
per-stratum eligibility rule. The current capacity executable still enforces
the original specification; it is not changed by this document.

The immutable compound reference remains 50.910426 HTTP tok/s. Preserve
joint-only ownership, dense reference semantics, the original down/vocabulary
consumers, compiled controls, exact-output gates and runtime defaults. Do not
add the separate exact portfolio to this experiment. The compact rank256
cost probe passed a cost-only gate; learned acceptance remains unmeasured.

## Proposed Eligibility Amendment

Add a natural 1-64 formatted-token bucket for `mbpp_code` only. Retain its
existing 65-447 eligibility. Count tokens using the pinned model's runtime
Qwen tokenizer and QwenChat formatting, with no system message or extra BOS.
Require nonempty source `text`; the lower bound is not permission to admit
empty wrapper-only prompts. Do not pad, truncate, concatenate tasks, add tests
or reference solutions, or rewrite instructions to meet a length threshold.

Keep all three Dolly strata at 65-127, 128-255 and 256-447 tokens. The original
minimum of 32 held-out prompts in each of those three buckets still applies
across the corpus. Short MBPP rows cannot satisfy those longer-context quotas.
The MBPP held-out stratum supplies short-context evaluation separately.

Existing counts give the following length-only upper bounds, not admitted
split counts. Source data and tokenization do not need to be downloaded again.

| MBPP split | Observed formatted length | Newly length-eligible upper bound | Unchanged quota |
| --- | --- | ---: | ---: |
| Train | 16-58 | 374 | 256 |
| Calibration | 17-36 | 90 | 48 |
| Held-out | 16-55 | 500 | 96 |

The new rule does not prove enough independent groups survive. Preserve the
source revisions, attribution/license obligations, upstream document review,
group-before-split procedure, duplicate/template checks, original MBPP ID
partitions, benchmark exclusions across every split and quarantine rules.
Keep at least 16 independent held-out groups per stratum, the selection seed,
and totals of 1024 train, 192 calibration and 384 held-out prompts.

Any group, provenance, exclusion or quota failure still stops admission. Do
not move source partitions, replace missing rows with another source, reduce
quotas or reuse calibration/held-out templates. Short code prompts especially
need the existing substantive-template and paraphrase review: absence of a
shared 32-token span is not evidence of independence. Freeze the realized
manifest and expanded exclusion audit before any GPU admission.

## Alignment And Training Stay Fixed

For a prefix ending at absolute position `t`, use raw final residual `h_t`
after layer35 and before output RMSNorm. The committed KV length is `q=t+1`.
The known seed is `x[t+1]`; the three labels remain `x[t+2]`, `x[t+3]`,
`x[t+4]`. At the end of a prompt of length `P`, `q=P`. Do not skip early
features merely to make the cost model start at 65.

Keep the three independent 2048-to-256 affine-SiLU heads, shared
256-to-151936 vocabulary matrix, no seed embedding and frozen target. No
future hidden state may enter a feature. Include the first stop label and
mask subsequent labels; a length limit is censoring, not EOS. Retain the
original greedy teacher, 128-token generation cap, FP32 training, AdamW
settings, seed, batch128, four epochs, 4096-step ceiling and final-checkpoint
rule. The maximum feature payload remains 1,677,721,600 bytes for 204800 rows.

## Why The Cost Model Needs New Evidence

The [existing model code](../../examples/draft_feasibility/cost.rs) clamps
positions below 65 to the first measured serial or verifier sample. It counts
extrapolations, but still produces numerical estimates. That substitution is
not short-context measurement and is not a demonstrated conservative bound.

The [direct verifier evidence](2026-10-28-direct-verifier-measurement.md)
covers serial positions 65-575 and four-row starts 65,69,...,569. Intermediate
batch starts use interpolation. Rejection costs were measured at start65;
transferring those costs to other positions is a separate assumption. The
historical HTTP fixed interval is from the browser workload, not short MBPP.

Dataset eligibility, verifier eligibility and cost coverage are different:

- A naturally short prompt may satisfy every data rule.
- A verifier attempt additionally needs a nonterminal seed, sufficient output
  allowance, valid session ownership and the existing transaction rules.
- A numerical speed conclusion needs measured costs covering each visited
  current position and transaction, or an explicitly labeled assumption.

A prompt starting below 65 can cross that boundary during generation. Index
costs by current committed length `q`, not initial prompt length `P`. Do not
import the previous roughly 92.59% acceptance requirement as a short-prompt
threshold or call shorter prefill an automatic speculation benefit.

## Future Short-Context Measurement Protocol

This protocol requires review before execution. First finish the amended
CPU-only manifest, grouping, provenance and exclusion gates. Then obtain
explicit authorization for the bounded calibration below. Existing
conditional R1 approval does not bypass the new cost-coverage prerequisite.

### Fixtures And Correctness

Freeze all fixtures and measurement order before timing. Use each prefix of
length `q=1..65` of the existing frozen 65-token browser prompt as structural
fixtures. Prefixes may end inside ChatML; they test execution and must not be
described as natural prompts or inserted into training data. The benchmark
and its derivatives remain excluded from every dataset split.

For each truncated prefix, obtain a fresh qualified serial greedy continuation
in the future calibration run. The full prompt's continuation is not a valid
oracle for a truncated prompt. Freeze actual token IDs, full-logit references
and valid KV snapshots before timing candidate verification. These perfect
drafts test cost/correctness only, never learned acceptance.

At every start `q`, compare the unchanged fully direct four-row verifier with
the qualified serial control. The four input rows occupy absolute positions
`q..q+3`; do not renumber them. Test full acceptance, rejection at each of the
three guessed positions, partial commit, rejected-suffix cleanup and session
isolation. Retain the existing prefix-zero rollback diagnostic separately
from ordinary rejection after a valid known seed. Exercise terminal seeds,
natural stop handling and remaining output allowances of 1,2,3,4.

Require identical actual greedy IDs, counts and stop reasons, bit-identical
valid KV state, unchanged source/owner buffers, and full-vocabulary numerical
agreement under the existing `1e-3 + 2e-5 * abs(reference)` contract. Record
intermediate bitwise equality without adding or weakening a numerical gate.
Any failure stops measurement admission. No new CUDA kernel or SGEMM verifier.

Explicitly cover starts62-64 whose four-row work crosses the old65 boundary,
and continuation from below65 into the historical domain without resetting
the session. Keep start65 as a contemporaneous historical anchor. A material
anchor drift above `max(3%, 2 * larger timing variation)` blocks splicing old
and new tables; request review rather than silently rescaling history.

### Timing And Natural-Prompt Check

Use release builds, the same model/hash, compound/joint-only preparation,
dense KV, original prefill and down/vocabulary execution. Verify GPU isolation
before timing; do not run both GPUs. Include target computation, all vocabulary
work, dependencies, transaction bookkeeping, rollback, correction and any
per-attempt allocation/copy/synchronization. Measure session setup/cleanup
separately and charge them to complete requests. Do not hide setup transitions
in an operator timer. Record peak/resident bytes and cleanup.

Measure serial rows and complete four-row attempts at each `q=1..65`, including
each rejection outcome. Start with five alternating paired sweeps. Treat
paired sweeps/requests, not correlated layer or token samples, as independent
timing observations. Follow the unchanged
[controlled protocol](../controlled_benchmark.md): variation above5% extends
to ten pairs once; unresolved instability blocks conclusions. A claimed gain
must exceed `max(3%, 2 * larger variation)` with positive paired95% bounds.
Borderline results require the prescribed twenty fresh pairs, not repeated
extensions until significant. If the time cap prevents confirmation, report
inconclusive. Microtimings alone do not qualify request throughput.

After CPU manifest admission, preregister four distinct training-only MBPP
groups nearest the short-prompt length minimum, median, 90th percentile and
maximum, with the existing stable group order breaking ties. Use the next
nearest unused group when necessary. Do not select by output, acceptance or
speed. Record their identities before any GPU execution. Keep them in train;
do not consult calibration or held-out answers to select timing fixtures.
Diagnostic traces are not automatically teacher-feature artifacts.

Run serial and perfect-draft complete resident requests for these four natural
prompts, up to the unchanged128-token cap with natural stops. Match actual IDs,
counts and reasons per pair. Report prefill, setup, decode and complete time.
If a prompt stops before speculation is eligible, report that limitation, not
a fabricated four-token generation. These bounded measurements check whether
the structural cost table represents real short prompts; they do not measure
learned acceptance or buffered HTTP throughput. Prompt-content disagreement
invalidates a universal position-only cost claim and requires review.

### Recurrence And Required Reporting

A later versioned replay must reject uncovered costs for admission instead of
silently clamping them. Preserve original model code/results for reproducibility.
Label each cost as measured, interpolated or assumed, including the remaining
rejection-cost extrapolation above65. Short-context calibration alone does not
remove that longer-context uncertainty.

Replay actual eligible/ineligible positions and conditional acceptances
jointly. With `j=0,1,2` accepted guesses, retain `j+1` input rows including the
known seed and roll back to `q+j+1`. Charge correction when nonterminal and
permitted by the output limit. Full acceptance advances four inputs. Keep
finite-tail/stop behavior from the existing recurrence rather than assuming
every attempt yields four output tokens.

Use the measured proposal cost1.5380845ms and a separate p95 scenario1.5555975ms.
Retain additional unmeasured-attempt scenarios0/1/5ms. Do not double-count
copies, argmax or synchronization already inside the proposal probe. Charge
eligibility checks on ineligible positions and all real integration overhead
when eventually measured. The synthetic proposal timing is not a trained
inference measurement.

For a fixed observed trajectory, let `A` be attempt count and `T_other` include
all target, transaction, correction, serial-tail and fixed request costs.
The allowable proposal cost for speed factor `s` is
`(T_serial / s - T_other) / A`. Report negative budgets as infeasible; `A=0`
has no drafting budget or speculation gain. Recompute with observed rejection
positions, rather than multiplying three unconditional accuracies. Use the
unchanged performance threshold for `s`, not an easier short-prompt target.

Report conditional acceptance at guesses1/2/3 with reached denominators,
full-batch acceptance, eligible coverage, rejection position and proposal cost
jointly by task and current-length bands1-15,16-31,32-47,48-64,65-127,128+.
Keep whole document/template groups together for acceptance bootstrap and
retain the original sample-count gates. Missing coverage is inconclusive.

HTTP projections must use prompt-specific prefill/setup evidence or label
those terms unknown. The browser's historical fixed overhead may appear only
as a sensitivity scenario, not a measured short-prompt overhead. No projected
short-prompt number qualifies the immutable512-token benchmark or the53.06 /
56.153149 HTTP targets. No speculative HTTP runtime is authorized here.

## Budget And Review Gates

The72 aggregate GPU-hour ceiling and nontransferable parent caps remain
4 cost /20 teacher /40 training /8 evaluation. Usage remains0.005923 cost
hours, zero in the other phases. Preserve the conditional five-hour pilot
allocation of2 teacher /2 training /1 evaluation, including the existing
0.25-hour teacher scout inside teacher time.

Propose at most0.25 GPU-hour for this future short-context calibration,
charged inside the pilot's existing one-hour evaluation allowance. This
leaves at most0.75 hour for pilot acceptance evaluation. Count GPU setup,
warmup, correctness, failed runs and cleanup, not just timed kernels. Whether
the fixed protocol fits is unmeasured; stop incomplete if it cannot fit.
Do not borrow from teacher/training, spend unused parent capacity or omit
correctness/statistical checks to finish. Every device switch still needs
explicit user approval. This paper task consumes no GPU time.

Review sequence and dependency-ordered future commit units:

1. Review this amendment and protocol. Preserve the original failed admission.
   This is the only deliverable authorized now.
2. If approved, version the CPU specification and capacity tools, then complete
   isolation, provenance, exclusions and realized-manifest review. Suggested
   commit: `feat(research): admit natural short prompts`.
3. With separate GPU authorization, add the bounded host-side measurement
   driver and its behavioral checks, preserving existing kernels. Suggested
   commit: `test(research): cover short verifier contexts`.
4. Record calibration costs, correctness and budget evidence separately.
   Suggested commit: `docs(research): record short verifier costs`.
5. Version the coverage-aware replay and evaluate R1 against all unchanged
   gates. Suggested commit: `feat(research): enforce verifier cost coverage`.
   A successful cost table alone does not admit teacher generation or training.

Stop for review on any unmet prerequisite. No larger training allocation,
architecture change, held-out tuning or runtime integration follows
automatically from this amendment.

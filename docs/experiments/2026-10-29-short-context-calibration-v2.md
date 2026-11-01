# Frozen Short-Context Calibration Protocol V2

2026-10-29. CPU preparation only is authorized. This protocol is frozen for
review, not GPU execution. It supersedes the timing-grid proposal in the
[paper amendment](2026-10-29-short-prompt-amendment.md), without changing the
original failed admission or its immutable specification.

The [versioned specification](2026-10-29-drafter-data-spec-v2.json) admits
natural short MBPP prompts only. Dolly length requirements, upstream MBPP
partitions, quotas, document/template isolation, benchmark exclusions and all
training settings remain unchanged. A census or unresolved-candidate ledger
is not an admitted training manifest.

## Evidence And Position Definition

The existing runtime-tokenized MBPP census has lengths16-58. Train length
minimum/median/p90/maximum are16/24/29/58; calibration17/23/29/36 and
held-out16/23/29/55. These are source-census statistics, before grouping or
selection. They justify the fixed timing positions below, not data selection
based on target outputs.

`q` is committed KV length before the known seed is processed. Four input
rows occupy absolute positions `q..q+3`. A prompt of length `P` starts at
`q=P`, then traverses larger positions as generation proceeds. The original
cost tables start at65; their short-context clamp is not measured evidence.
There are no short-context GPU measurements in this protocol.

## Mandatory Correctness Coverage

Retain exhaustive structural coverage at every `q=1..65`, without collecting
statistical performance samples at every position. Browser-prefix fixtures
come from the frozen65-token browser prompt and stay excluded from every
training/calibration/held-out split. Prefixes inside ChatML are structural
inputs, not natural MBPP prompts. Generate each truncated prefix's own serial
reference; never borrow the full prompt's continuation.

For each position require full-vocabulary logits under the unchanged
`1e-3 + 2e-5 * abs(reference)` contract, exact actual greedy IDs/counts/stop
reasons, bit-identical valid dense KV and absolute causal positions. Cover
all three rejection positions, full and partial acceptance, rollback to
`q+j+1` for `j` accepted guesses, rejected-suffix cleanup, producer/consumer
dependencies, session isolation, source-owner immutability and teardown.
Keep the existing prefix-zero transaction diagnostic separate from normal
rejection after a valid seed. Record intermediate bit equality diagnostically.

Exercise terminal seeds and remaining output allowances1/2/3/4. A naturally
terminal fixture must take the no-proposal path, not be forced to continue
to create a timing sample. Use the existing synthetic transaction fixtures
for structural rejection/partial-commit cases unavailable in that natural
continuation; report those separately from actual greedy trajectories.
Missing mandatory coverage blocks calibration admission, not a claimed pass.

Explicitly cover starts62,63,64,65 and a continuing session crossing65 without
reset. No altered CUDA kernels, SGEMM verifier, model semantics or numerical
thresholds are allowed. Preserve compiled controls and original model hashes.

## Fixed Performance Samples

Run only after mandatory correctness passes and GPU execution is separately
authorized. Use the qualified serial/direct-four-row consumers, dense KV,
compound/joint-only preparation and original down/vocabulary/prefill paths.
Do not stack the exact portfolio or enable approximate numerics.

| Sampling role | Fixed committed lengths |
| --- | --- |
| Fit samples spanning MBPP and generated-prefix lengths | 16,20,24,29,32,36,48,55,58 |
| Boundary samples and historical anchor | 62,63,64,65 |
| Preregistered interpolation checks, not used in initial fit | 18,27,42,52,61 |

At each sampled position measure ordinary serial rows, complete four-row
verification and rejection outcomes. Charge vocabulary evaluation, correction,
rollback, dependencies, copies, allocations and transaction bookkeeping.
Do not infer the batch cost from isolated FFN timing. Session setup, prefill,
cleanup and high-water memory are separate recorded costs that remain in
complete-request accounting. Record natural-stop-ineligible cases explicitly.

Start with five alternating paired sweeps. Use independent sweeps/requests,
not layers or correlated tokens, for paired uncertainty. Follow the existing
[controlled protocol](../controlled_benchmark.md): variation above5% extends
once to ten pairs; unresolved instability is inconclusive. Request gains
must exceed `max(3%, 2 * larger timing variation)` with positive95% paired
bounds. Borderline claims require twenty fresh confirmation pairs within the
same cap. Never keep extending until significant or drop slow observations.

Compare the contemporaneous65 anchor with the historical same configuration.
Drift above `max(3%, 2 * larger variation)` prohibits joining the tables
without review. Do not rescale immutable evidence or automatically rerun a
long-context benchmark.

## Interpolation And Bounded Expansion

Every cost record must identify position, transaction outcome, fixture class,
sample count, raw timing source and whether it is measured or modeled. Retain
serial and rejection component costs, rather than treating all outcomes as
one mean accepted length.

Linear interpolation between fit samples is an estimate, not measurement.
Check it at the five preregistered validation positions, separately for serial,
full acceptance and rejection/correction costs. Report error and variation.
Do not claim that the largest neighboring latency is a proven upper bound.

Expansion is permitted only when an interpolation check differs by more than
5% beyond observed paired uncertainty, or the choice between plausible costs
changes the pilot's go/no-go conclusion. Measure at most four additional
integer positions, one at a time, chosen by the midpoint of the affected
interval. Prioritize greatest validation error; break ties by lower position.
For a decision-sensitive interval without validation error, use the interval
with the largest replay visit count, then lower position. Record the triggering
evidence before expanding. No architecture or kernel search is permitted.

After each expansion, report both the original and refined model. If four
positions or the time cap are insufficient, stop with incomplete cost coverage.
Do not silently interpolate across a failed interval. No extrapolation below16
is admitted; current natural prompt eligibility cannot produce such positions.
If a later admitted artifact does, return for protocol review.

For a validated interval, show the interpolation projection and sensitivity
using the larger endpoint cost plus5%, or the observed validation error if
larger. This is a stress scenario, not a physical bound. Preserve the existing
additional0/1/5ms attempt-cost scenarios. A success depending on optimistic
interpolation alone is inconclusive. Longer-context rejection costs remain an
explicit historical assumption; this short protocol does not measure them.

## Natural MBPP And Acceptance Reporting

After the CPU manifest is admitted, freeze four distinct training-only groups
nearest the admitted train length minimum, median, p90 and maximum. Use the
existing stable group order for ties and the nearest unused group when needed.
Do not choose by generated output, acceptance or timing. Record IDs and hashes
before GPU execution. Do not use held-out/calibration answers to construct
timing fixtures or insert browser structural fixtures into the dataset.

For these four natural prompts, compare complete serial versus perfect-draft
resident requests, up to128 generated tokens with natural stops. Record actual
IDs, counts and stop reasons; report prefill/setup/decode/total time separately
from structural samples. Perfect drafts supply cost evidence only. No learned
acceptance is measured by an oracle continuation. A disagreement between the
natural-request results and the position-only model blocks a universal cost
claim and requires review.

The later pilot still reports conditional acceptance at guesses1/2/3 with
reached denominators, full-batch acceptance, eligible coverage, rejection
positions and proposal cost jointly by source/task/current-length band. Keep
whole document/template groups as the acceptance bootstrap unit and preserve
all original sample-count requirements. Held-out traces cannot tune gates,
schedule, checkpoint selection or sampling positions.

Use proposal median1.5380845ms and p951.5555975ms as distinct measured synthetic
cost scenarios, not learned acceptance evidence. Include ineligible positions,
serial tails, correction and real integration overhead. A replay needing an
uncovered cost must report that cost unknown and cannot pass admission.
Do not call historical browser fixed overhead a measured short-prompt HTTP
cost. Resident measurements and projected HTTP results remain distinct.
Nothing here qualifies the exact512-token HTTP speed target.

## Budget And Stop Decision

No GPU work occurs under the present approval. Future calibration remains
bounded to0.25 GPU-hour INSIDE the original one-hour pilot evaluation cap,
leaving at most0.75 hour for final evaluation. Include setup, warmup,
correctness, failures and cleanup. Mandatory coverage takes priority over
timing; inability to finish within the cap is inconclusive, not permission to
omit gates or transfer time. Teacher2 hours including0.25 scout, training2
hours, and parent4/20/40/8 totaling72 hours remain unchanged. Used GPU time
remains0.005923 hours from the earlier cost probe.

Return the realized CPU manifest, source/license/provenance/group/exclusion
audits, this protocol and remaining blockers for review before GPU execution.
If the fixed data requirements cannot be met, return a rejection ledger and
failed admission decision, not a fabricated training manifest. Training and
teacher generation remain blocked until both manifest and cost requirements
pass. Every GPU switch needs approval. No speculative runtime promotion or
default changes are authorized.

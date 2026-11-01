# Hardware-adaptive drafting follow-up

## Question and scope

Can the already implemented prompt-copy drafter avoid unprofitable M40
verification attempts by using measured target costs and earlier acceptance
outcomes from the same request? This follows the October 31 runtime objective
on `research/1031-adaptive-drafting`, based on feature commit `b9333e7`.
The independent KV-quality run may continue while this protocol is prepared;
CPU timing and any GPU experiments must wait for it to finish.

Keep the existing prompt-copy algorithm, target weights, four-row direct
verifier, exact dense attention, joint-only ownership and runtime defaults.
No model acquisition, learned-drafter evaluation, training, CUDA geometry
search or production dispatch is included. The V3 data work remains stopped.
The previous negative general-drafting result and positive specialized copy
result retain their original scope.

The current qualified direct backend verifies four input rows: one known
target seed and three guesses. It does not establish measured two/three-row
verification costs. This study adapts between serial execution and that
four-row operation. It must not claim optimal draft length or fabricate
shorter-batch costs by dividing the four-row measurement. Inventory the API
and preserved kernel evidence explicitly in the final report.

## Fixed data and cost model

Reuse all 21 existing traces: browser512, twelve task continuations, and eight
October 28 held-out captures. Those eight are now retrospective regression
inputs, not fresh held-out evidence for this controller. Preserve all actual
IDs, configured budgets, natural stops, prompts and source hashes. Do not add
or remove cases after seeing the result. Source lookup receives only prompt,
committed history and the known seed; future IDs belong only to the scorer.

Use the five existing complete M40 target pairs, including ordinary serial
rows, four-row passes, early rejection/rollback/correction and bookkeeping.
The policy uses their pointwise median costs. Score every resulting visit
path against each pair independently using the retained finite-request model.
Keep its disclosed late-rejection and historical HTTP assumptions. Refuse
speculation outside measured four-row starts 65..569; charge serial execution
and disclose any serial-cost extrapolation. No HTTP performance claim follows.

## Frozen controller

Compare ordinary serial execution, unchanged always-attempt prompt-copy, and
one adaptive policy. Reset all policy state per request. Do not choose rules
from task labels, previously captured outputs or known future stop positions.

1. Respect the configured remaining budget and stop-ID proposal checks used
   by the original source. Pay source lookup and maintenance costs even when
   lookup fails or the controller chooses serial execution.
2. Before any attempted batch, require the full-acceptance cost ceiling to
   beat four serial rows by at least 3%, including a fixed 0.1 ms proposal and
   controller allowance. This allowance is a decision input, not a substitute
   for measuring and charging actual CPU costs in the final projection.
3. With no observed attempts, allow one qualifying probe. Thereafter retain
   the cumulative histogram of accepted guesses 0,1,2,3 from executed attempts.
   Do not score or learn acceptance from skipped proposals. Estimate expected
   batch cost and useful serial work from that histogram at the current
   position, including rollback and ordinary correction on rejection. Permit
   another ordinary attempt only when the estimated gain is at least 3%.
4. If that estimate rejects speculation, permit a new qualifying probe only
   after at least 32 committed input rows since the last attempted batch.
   Every probe pays its full cost. A source miss does not count as a probe.
5. Update source history only with committed target tokens, including ordinary
   correction. Preserve finite tails and verified terminal tokens. Never use
   future EOS to decide whether to propose, or count the known seed as a guess.

Expected costs use the same complete rejection/correction convention as the
existing model. The no-stop estimate is conservative at terminal positions;
actual replay still charges only the work that the existing controller would
execute. Record every decision, histogram before decision, cost estimate,
reason, proposed IDs, observed outcome and next committed position.

## Validation, measurements and gate

First test decision behavior with known cost tables: expensive verification
stays serial, a profitable first probe executes, rejection disables ordinary
attempts, recovery probes wait 32 rows, misses/tails/out-of-range positions
stay serial, and future suffix changes cannot affect earlier decisions.
Audit actual causal replay independently from serialized records, including
proposal provenance, outcomes, retained progress, stops and finite accounting.
Reject corrupted evidence rather than merely checking row presence.

After the M40 quality workload finishes, warm each CPU replay once, then run
five complete release trials per trace and policy with source construction,
proposal, policy, correction-history maintenance and teardown charged. Keep
the existing cost-model +1/+5 ms unknown-overhead scenarios. Also run two
explicitly hypothetical hardware sensitivities with verifier-side work scaled
by 1.25 and 2 while ordinary serial costs stay fixed; these are policy-response
checks, not measurements of another GPU. No parameter retuning follows.

Report all 21 cases, decision coverage, acceptance, rejected probes, CPU cost,
modeled complete-request gain and differences from always-attempt copying.
A broad lead requires browser and non-copy non-regression plus at least 3%
modeled gain over serial on a declared workload in all five target samples.
If no lead passes, preserve the negative study and stop. A passing lead needs
a separate frozen live correctness/measurement protocol before claiming speed:
exact greedy IDs, logits, KV/rollback, then controlled paired resident requests.
Retrospective projection alone cannot qualify production or generalization.

## Dependency-ordered commits

1. `docs(research): plan adaptive drafting study`: freeze this scope and policy.
2. `feat(research): replay adaptive draft decisions`: isolated CPU controller,
   cost access, causal replay, records and behavioral tests; commit before study.
3. `docs(research): record adaptive drafting results`: raw release trials,
   independent audit, sensitivities, limitations and conditional go/no-go.
4. Only for a passing lead, preregister live validation and then implement and
   measure the unchanged policy as separate verified research commits.

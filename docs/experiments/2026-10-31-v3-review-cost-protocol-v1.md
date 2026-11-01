# V3 local review-cost pilot v1

Preregistered October 31 before sample construction or timing. The user confirms
that no independent human reviewers are available. Measure automated effort
only. Human source inspection, semantic judgment, rights decisions, adjudication,
independent second review and agreement stay **unmeasured**, not zero or passed.
Codex's own inspection is not independent human review.

## Fixed sample

Use the existing four-source sample-input and original Qwen token ledgers.
No new source requests, tokenizer substitution, model output or training labels.
Select sixteen review locators in existing ledger order among natural inputs
of 65-447 tokens. This is a purposive workload sample, not a statistical census.

| Family | Fixed four locators | Coverage intent |
| --- | --- | --- |
| UltraChat | First two eligible train SFT and first two eligible test SFT records | Ordinary prompt packets plus missing message/origin evidence |
| CommitPackFT Python | First two eligible permissive-label records and first two eligible AGPL/EPL-label records | Repository aliases, code dependencies and different rights-screen burdens |
| SQuAD | First two eligible train questions from the first eligible article, and first two eligible validation questions from its first eligible article | Same-document dependencies, native boundaries and ambiguous historical page identity |
| DocRED | First two eligible annotated-train, first eligible dev and first eligible test record | Shared task wording, document/entity provenance and missing test labels |

Permissive labels for this selection only are MIT, Apache-2.0, BSD-2-Clause,
BSD-3-Clause and ISC; restrictive-screen labels begin AGPL or EPL. They are
not legal clearance. If a stratum cannot supply its specified records, report
that failure without replacement. Freeze exact IDs and source/input/token
hashes in a review-only manifest commit before timing. Do not choose or replace
packets based on dependency, benchmark or later model outcomes.

## Automated workflow and measured scope

1. Bind all 846 retained four-source prompts to their original local records,
   token counts, native split locators and source hashes. Inspect retained raw
   field availability, never infer missing answers/messages/page revisions.
2. Build the full retained-sample dependency graph, including unselected and
   ineligible rows. Reuse all original raw overlap pairs; add exact provisional
   document/conversation IDs, shared repository aliases/commits and supported
   code-template fingerprints. Preserve unresolved AST/provenance cases.
   Repeated task instructions are not automatically exempted. This is not
   complete semantic grouping or full-population dependency closure.
3. Expand each packet's transitive local dependencies. Record component size,
   unselected members, cross-family/native-split flags and unresolved external
   links. Packet members do not become training examples.
4. Run existing literal benchmark checks on packet prompts against the frozen
   final exclusion inventory and original comparison-token ledger. Preserve
   coverage gaps and missing audit-field tokenization; zero literal hits is
   not semantic clearance. Do not use benchmark responses as training labels.
5. Attach local source cards/rights evidence by hash and provenance blockers.
   Export blocked dispositions, not machine-generated approval. Record each
   unavailable human stage explicitly and do not calculate reviewer agreement.

Measure CPU and wall time by stage, packet/dependency counts, input bytes and
process RSS. One run, 4 GiB address-space cap, 600 CPU/900 wall seconds with
660 CPU-second hard cap; no retries or new matching architecture. Existing
raw-overlap SQLite evidence is a reusable input, not a newly timed population
matching result. Its original construction cost is reported separately.

## Review cost and economic interpretation

Report automated preparation costs as automation, never reviewer time. Separate
shared ledger/index setup from packet-level work; do not multiply shared setup
sixteen times or infer human minutes from Python CPU seconds. Report bounds on
local dependency counts, with unseen dependencies explicitly unknown.

For 1,600 examples, retain an explicit cost equation with unmeasured human
rates for content review, dependency inspection, rights chains, adjudication
and second review. No agreement statistic or numeric human-resource estimate
is measured without reviewers. Prior assumed-rate scenarios remain scenarios.
At eight hours, even zero overhead permits only eighteen seconds per example.
The automated pilot cannot make that human allowance credible.

Keep raw protected source content local. Commit locators, hashes, counts and
blocked decisions rather than republishing source text. Report what a future
independent-review pilot needs, but do not recruit reviewers, expand resources,
start a census or claim R1 readiness automatically. Stop for user review.

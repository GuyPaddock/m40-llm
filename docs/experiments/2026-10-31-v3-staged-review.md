# V3 staged review and resource decision

Review proposal v1, October 31. This does not authorize acquisition, source
selection, a larger review allocation, teacher work or training. The approved
[census protocol](2026-10-30-v3-census-protocol-v1.md), source pins, quotas and
isolation rules remain unchanged. No target outputs enter this process.

## Separate population coverage from manual adjudication

The proposed population contains 390,220 records. An automated edge index is
not a semantic clearance certificate. It cannot establish that two paraphrases
describe different facts merely because their words differ. Prioritize manual
work without removing unselected or length-ineligible records from the graph.

| Stage | Complete scope and evidence | Stop condition |
| --- | --- | --- |
| Population ingestion, after separate approval | Every pinned native row and audit field; IDs, file/record hashes, split provenance, raw text, adapter and tokenizer identities, length, rights status | Missing fields, count mismatch or resource limit leaves coverage incomplete |
| Deterministic grouping | All source documents, repository/fork/commit aliases, code/AST dependencies, full-content literal overlaps and benchmark matches; include ineligible bridge records and cross-family edges | Unresolved identities or unsupported syntax cannot receive clearance |
| Semantic candidate ledger | Document/fact/answer-instance aliases, substantive templates, paraphrase and benchmark-topic candidates; preserve method, scope and recall limitations | Literal-negative alone never closes semantic review |
| Prospective review queue | One deterministic count-only allocation on the frozen graph, if canonical evaluation completes; expand the entire dependency closure of prospective components | A feasible but noncanonical witness is not a selection or review-priority shortcut |
| Independent manual adjudication | Every prospective selected example, its component, related unselected/ineligible records, rights chain and benchmark dependencies | Any unresolved relevant edge blocks the entire affected component |
| Release assessment | Coverage, reviewers, disagreements, quarantines, independent-group capacity, valid witness, rights and benchmark-exclusion evidence | Graph changes invalidate the witness; return for a versioned decision |

The queue is for review, not a selected training manifest. Its ordering uses
the frozen seed and canonical component identities, never acceptance, target
logits, teacher answers or a favorable model response. No repeated allocation
or replacement of difficult rows to obtain a passing review result.

## Dependencies cannot be hidden outside the selected subset

For each prospective component, reviewers receive all members and typed edges,
including links to otherwise unused records and other sources. A source-wide
unresolved identity problem that could connect it to another partition blocks
clearance; an empty literal adjacency list does not resolve that problem.

Record every link as confirmed, refuted with evidence, or unresolved. Preserve
raw overlaps even when a reviewed generic-only disposition later applies.
Cross-boundary matches and source payloads stay actionable. Normalized-offset
mapping is a separate implementation obligation; exact-byte registry checks do
not authorize normalized-text masking. The four entries still need independent
second review. No exemption is active in the new frequency index.

Manual review must check content, not just IDs: source-code transformations,
shared passages, entity/relation instances with qualifiers, answers tied to the
same evidence, substantive worked examples, and benchmark-specific templates.
Shared generic task wording may be classified only through the approved exact
registry procedure. Cross-family and cross-native-split components retain the
existing quarantine policy. Entire task-family transfer is outside this task.

If review discovers an edge or rights issue that changes the frozen graph,
invalidate its allocation certificate and stop. A new graph version and
authorization are required before refilling or reallocating. Do not silently
drop an unselected neighbor to preserve a quota.

## Source-specific review packets

These obligations follow retained evidence, not new source inspection or legal
clearance. Keep copyrighted source material local; publish locators, digests
and decisions rather than assume permission to redistribute full records.

| Source | Required evidence and remaining uncertainty |
| --- | --- |
| UltraChat SFT | First-user/message consistency, conversation and quoted-document provenance, original/H4 transformations, generation-time terms and quoted-content rights. Retained samples lack messages, so that consistency check remains unavailable. MIT collection labeling is not blanket content clearance. |
| CommitPackFT Python | All repository aliases and forks, commit/file license at the source revision, copied dependencies, old/new code and tests as isolation evidence, AST/paraphrase/benchmark review. Collection MIT and permissive-looking row labels do not override file rights. |
| SQuAD | Article/redirect/passage identity and historical revision evidence, context/question plus audit-only answers, source attribution and CC-BY-SA conditions. Titles alone are not verified document identities. |
| DocRED | Article/source aliases, text/entities and available relation annotations, separate Wikipedia/Wikidata/publisher rights. Missing test labels prevent claiming complete answer-instance coverage; preserve the later post-teacher collision gate. |

Use independent reviewers with recorded identities, evidence hashes, decisions
and disagreements. Automated witness verification is independent of the
optimizer, but it is not independent human registry or semantic review. No such
human approval is asserted by this follow-up.

## Cost model, not a measured human throughput result

The unchanged quotas require 1,600 examples. Eight hours allow 18 seconds per
example even with zero setup, dependencies, licensing or second-review work.
A complete manual pass over 390,220 records would allow only 0.074 seconds per
record. Neither figure supports the proposed audit standard.

Let `N=1600`, `D` be additional dependent records needing inspection, `E` be
unresolved semantic links, and `L` be distinct source/license chains. Then:

```
review_hours = setup_hours
             + (N * minutes_per_example + D * minutes_per_dependency
                + E * minutes_per_link + L * minutes_per_rights_chain) / 60
             + independent_review_hours + adjudication_hours
```

No human timing study has measured these rates. As a planning sensitivity
range, 2/5/10 minutes per example alone costs 53.3/133.3/266.7 reviewer-hours.
These are arithmetic scenarios, not staffing estimates or an approved budget.
Dependencies and rights disputes add time; shared component review may amortize
some provenance work but does not waive individual content inspection.

The current eight hours can fund only a bounded workflow-cost pilot, if
separately approved. A proposed pilot would freeze a queue from existing local
evidence, time complete packets, record unresolved dependencies and benchmark
checks, and include independent second review. It would not clear an entire
source, estimate corpus-wide semantic recall from a few rows, select training
examples, or enlarge any GPU budget. Review rates, dependency sizes and rights
chain counts would then support a priced continuation request.

## Decision and requirements before a finite census

The eight-reviewer-hour allowance is not credible for complete admission.
The actual required effort is unknown until packets and dependency counts are
measured; the selected-example-only scenarios already exceed the allowance by
6.7 to 33.3 times. Stop for review rather than reduce the standard.

Before requesting the finite four-source census, resolve the engineering
limits in the accompanying CPU results, approve a workable review allocation
or explicitly separate coverage-only census authorization from later admission,
and retain all unresolved rights/identity and benchmark coverage blockers.
Independent registry review, normalized offsets, full audit-field coverage and
semantic dependency closure remain mandatory. Acquisition permission would not
itself authorize a selected manifest, R1, GPU calibration or teacher generation.

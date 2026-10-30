# V3 isolation policy amendment proposal v1

## Status and decision requested

Policy identifier: `v3-isolation-proposal-1`, October 30, 2026.
**Paper-only, inactive, pending review.** This proposes a change in evaluation
scope, not a reinterpretation of earlier failed admissions. No data audit code,
source manifest, runtime setting or active specification changes with it.

Approve or reject a primary evaluation of **unseen examples of known tasks**,
with document and substantive example isolation, while treating transfer to
entirely unseen task families as a separate evaluation objective. Permit only
explicitly reviewed generic instruction spans to recur across partitions.
Do not exempt repeated text merely because it occurs often or is called an
instruction.

If approved, the smallest candidate source design becomes CommitPackFT Python,
UltraChat first-user SFT, SQuAD grounded QA and DocRED extraction. All four
remain provisional. This is not permission to acquire complete corpora,
implement the policy, generate teachers, train or enter R1. Those steps require
their own reviewed manifests and authorization.

The [original source comparison](2026-10-30-original-source-comparison.md) at
`c037ca3` and [earlier V3 rejection](2026-10-30-hf-dataset-feasibility.md) at
`c1597dd` remain correct under their original contracts. V1/V2 specifications,
short-context amendments, failed admissions and raw evidence remain immutable.
Further revisions must receive new policy versions rather than replacing this
review artifact or rewriting old conclusions.

The [V2 specification](2026-10-29-drafter-data-spec-v2.json) remains the
reference for unchanged quotas, teacher alignment, exclusion seeds, numerical
contracts and budgets. The proposed delta is the instruction/example boundary
and evaluation objective, plus the explicit allocation and conservative
cross-family quarantine rules below. No exception is active yet.

## Three distinct isolation questions

| Category | Proposed primary evaluation rule | Examples and limits |
| --- | --- | --- |
| Document and passage leakage | Prohibited across train, calibration and held-out. | All questions/annotations from one paper or article stay together; overlapping excerpts, translations, summaries and paraphrases of the same source join the group. |
| Entity-instance and answer leakage | Prohibited for the same answer-bearing instance, fact or target solution. | The same subject/relation/object assertion, with matching time and other qualifiers, cannot cross partitions just because a new article or question describes it. A common entity name or label alone is not an instance identity. |
| Source-code leakage | Prohibited across partitions. | Repository/fork lineage, commits, copied files/functions, renamed or reformatted solutions, tests and substantive problem variants must be linked. |
| Substantive example-template leakage | Prohibited across partitions. | The same worked problem, scenario, table, code/config fixture or answer-producing scaffold remains one example family after names, numbers, labels or surface wording change. |
| Generic task instructions | May recur only through a frozen, reviewed registry. | A request to extract entities/relations or a generic output schema may be shared if it contains no example-specific support, worked answer or benchmark scaffold. |
| Entire task family | May recur in the primary evaluation; report separately for task transfer. | Training and testing relation extraction on independent documents measures known-task generalization, not transfer to an unseen task. |

An entity-instance is not every occurrence of a globally common entity. It is
a provenance-linked mention/event or answer-bearing assertion. Reusing a fact
about a person in different documents is a dependency; mentioning that person
in genuinely unrelated facts is not automatically the same example. Alias and
fact-resolution uncertainty requires review or quarantine, not an invented ID.

Likewise, equality of short answers such as `yes`, a relation label or a common
name does not alone prove leakage. Reusing the corresponding fact, evidence,
solution or source answer does. A literal ban on common labels would silently
turn known-task evaluation into label-transfer evaluation. Long duplicated
answer passages, copied code and answer-equivalent paraphrases retain the
content-overlap checks below. No dataset answer becomes a teacher label.

## Narrow generic instruction registry

Registry approval precedes row selection and any target-output inspection.
Each entry must contain the source/adapter revision, exact UTF-8 bytes and
SHA256, structural location and character boundaries, task/schema identifier,
purpose, reviewer decision and rationale. The registry itself is versioned and
hashed. Unlisted wording, unreviewed paraphrases and changed adapters receive
no automatic exemption.

An entry qualifies only if all these conditions hold:

- It specifies an operation, response format or fixed task vocabulary without
  carrying a particular example's entities, facts, numbers, answer or solution.
- It contains no few-shot demonstration, quoted document, test case, worked
  reasoning, source-code fragment or benchmark-specific narrative scaffold.
- It can be applied unchanged to independent documents without supplying
  evidence for any particular answer. Frequency of repetition is not proof.
- Its boundary follows an explicit adapter field. No regular expression may
  erase arbitrary repeated spans from document content or free-form prompts.
- Benchmark-specific patterns remain ineligible even if superficially reusable.
  In particular, the frozen browser-loading topic/templates and reserved
  benchmark markers remain excluded.

The retained DocRED adapter's exact fixed request is a candidate registry
entry. This paper does not activate it. Its request should not be confused
with the source sentences that follow it. SQuAD's `Context:` and `Question:`
headings are candidate formatting entries, not permission to mask the question.
An instruction embedded inside an UltraChat user message is not automatically
an adapter field. Its location and role require explicit review.

Example decisions:

| Repeated material | Proposed disposition |
| --- | --- |
| Fixed generic relation-extraction request over unrelated documents | Eligible for registry review; not a document-group edge by itself. |
| Same request plus one shared worked relation example | Keep the example as payload; link/exclude its dependencies. |
| A copied article followed by a different question | Same document group regardless of instruction wording. |
| Renamed entities in the same synthetic table or benchmark scenario | Same substantive example-template group; no exemption. |
| Same generic task over two genuinely different facts | Potentially independent, subject to document, fact and semantic audits. |

## Reproducible content and grouping policy

The following is a proposed future procedure, not an audit performed here.

1. Freeze source revisions, native split mappings, adapters, tokenizer identity,
   registry, audit-tool versions and the complete benchmark inventory. Retain
   each source record locator, native IDs, rights evidence and source hashes.
   Preserve emitted natural prompts exactly; use NFKC, lowercase and collapsed
   whitespace only for comparisons.
2. Build the audit inventory before length filtering or quota selection.
   Include ineligible records and source fields needed to discover connecting
   documents/examples. Partition each record into adapter instruction,
   context, natural question, demonstrations, code and audit-only source
   answers/annotations. Unknown field roles stay unexempted. Full population
   coverage would need separate acquisition approval; bounded samples cannot
   certify an unseen corpus.
3. Retain full-text match results. Also compare unexempted content fields with
   the existing thresholds: exact normalized matches, shared contiguous
   32-token spans using the original Qwen comparison tokenizer, and five-word
   shingle Jaccard at least 0.5. Preserve per-field boundaries; do not create
   artificial matches by joining nonadjacent fragments after masking.
4. A raw match may be dismissed as generic only when all of its matched
   content is wholly inside approved registry spans and no payload or
   substantive dependency remains. Record raw scores, content-only scores,
   exact span locations, registry entry and decision. Matches crossing into
   payload remain actionable. Unreviewed boilerplate is not exempt.
5. Add hard graph edges for shared source documents/passages, source-derived
   variants, repository/code lineage, duplicated solutions, answer-bearing
   entity/fact instances, content matches and adjudicated substantive example
   templates. Multi-document examples connect every source document they use.
   Apply the graph jointly across all sources and task families, including
   SQuAD/DocRED articles and any quoted UltraChat or code documentation.
   A match consisting solely of a task-vocabulary label or bare entity name
   is logged for instance review rather than automatically becoming an edge.
   Dismissal requires evidence of distinct underlying facts/examples, not
   frequency alone; unresolved cases stay quarantined. This does not waive
   duplicated answer passages or solutions.
6. Audit semantic paraphrases and aliases separately from literal matches.
   Require documented review of candidate question/problem skeletons, code
   variants and fact/evidence relationships. A generic task-level pattern such
   as asking for an author is not alone a duplicated example; the same work,
   source fact or instantiated scenario is. Ambiguous cases are quarantined.
   For the bounded pilot, review every selected record's provenance and
   semantic family; hashes and exact spans alone cannot clear it.
7. Compute transitive connected components, retaining typed edge evidence.
   Canonicalize member locators as sorted `(source revision, native record ID)`
   pairs; hash their canonical UTF-8 JSON array for the component ID. A hash
   binds the audited group, not proof of document identity. Unresolved source
   aliases or a component touching a prohibited benchmark quarantine the
   entire affected component.

For this pilot, also propose quarantining components that span more than one
task-family stratum. This conservative additional rule keeps the existing
family-stratified group bootstrap well-defined and avoids counting related
examples as independent in different families. Report its capacity cost
separately; a more permissive policy would need a joint statistical protocol.

Manual adjudications must be reproducible artifacts, with record pairs,
reason codes, evidence, reviewer and decision. Use a second review for any
proposed exemption or contested example-template classification; unresolved
disagreement stays quarantined. Freeze decisions before target acceptance
results. This proposal does not claim that semantic paraphrase detection is
complete or that adding an embedding-similarity threshold would prove it.

Source answers and annotations may be inspected only in the isolation audit,
not exported to the drafter trainer or used to tune held-out prompts. They
identify shared facts/solutions and support benchmark exclusion. Missing
answer fields must be recorded. DocRED test lacks relation labels; do not
invent them or import answers from another split. Pre-teacher admission needs
the strongest available document/fact audit and an explicit unresolved-answer
ledger. Final training admission also needs the existing post-teacher overlap
check after separately authorized frozen-target generation. Any new collision
quarantines whole components and triggers review if quotas fail; no automatic
refilling, repartitioning or held-out tuning follows.

## Preserve native splits and assign whole groups

| Source | Permitted research destinations |
| --- | --- |
| SQuAD original train | Train or calibration, split by whole audited components. |
| SQuAD original validation | Held-out only. No validation-to-training promotion. |
| DocRED annotated train / dev / test | Train / calibration / held-out respectively. Exclude distant-train. |
| CommitPackFT Python original train | New research train/calibration/held-out partitions by repository/fork/example components; no native held-out partition is claimed. |
| UltraChat train_sft / test_sft | Train or calibration / held-out only; first-user prompts, no assistant-turn padding. |

A component spanning different original splits of the same source is
quarantined, not repaired by moving rows. Cross-source components must have a
common permitted destination; conflicting fixed destinations quarantine the
whole component. Retain all native split labels even when a research split
is assigned. Resolve article redirects and shared facts across SQuAD and
DocRED before making this decision.

Proposed deterministic allocation uses the unchanged seed `20261029` and
`SHA256(UTF-8("20261029:" + component_id))`, interpreted as a big-endian
unsigned integer. A singleton permitted destination is fixed. For flexible
train/calibration groups, remainder modulo 304 below 48 selects calibration;
the other remainders select train. For groups permitting all three destinations,
remainder modulo 400 selects train for 0-255, calibration for 256-303, and
held-out for 304-399. Any other permitted-destination set stops for review.
These are assignment rules, not a guarantee that row quotas will be met.

Within each destination and family, order eligible components by the same
hash, breaking ties by component ID. Order member record locators by their
seeded hashes and select round-robin across components until the fixed quota
is reached. Do not change the seed, thresholds, registry or group decisions
to obtain a passing count. Retain excluded and unused records in the ledger.
One source example cannot fill two family quotas; ambiguous family assignment
must be resolved before selection, not based on where a deficit occurs.

## Benchmark exclusion and evaluation integrity

The existing exclusion specification remains binding. Apply it to all three
partitions, before quota selection, across prompts, contexts, code, available
source answers and later frozen-target continuations. Exclude the browser-
loading topic/templates, `ZXQ-NEEDLE-41729`, `M40_BATCH_LIMIT`, frozen quality
tasks, draft/copy held-out prompts and outputs, KV fixtures, experiment traces
and referenced external artifacts. Renaming a benchmark's entities, constants
or codes does not make its substantive example template eligible.

Use the same raw/content dual audit for benchmark matches. Generic text shared
with a benchmark is not by itself benchmark-specific content, but dismissal
requires the frozen registry and evidence that no benchmark scenario, answer,
code or topic rule matches. A benchmark template can never be placed on that
registry simply because it looks like an instruction. Preserve every such
decision for review.

The retained 265,771-text exclusion inventory and zero literal hits are useful
prior evidence, not complete clearance. Its documented script/external-artifact
and semantic gaps must be closed or explicitly block admission. Newly reviewed
policy labels do not erase unresolved exclusions. Audit reviewers may inspect
held-out metadata solely for isolation; model development must not use
held-out outputs, teacher features or acceptance results for selection or tuning.

## Independent groups and scientific tradeoffs

Keep four families, each with **256 train, 48 calibration and 96 held-out
prompts**. Require at least **16 independent held-out components per family**
after all document, fact, example-template and benchmark edges. Multiple
questions, relation pairs, token positions, paraphrases or instruction variants
from a component count once. Generic instruction identity alone does not add
or merge components under this proposed primary objective.

Keep natural 65-447-token inputs for the proposed replacement sources, 128
generated-token limit and at least 32 held-out prompts in each global
65-127/128-255/256-447 bucket. The accepted MBPP-only short-context amendment
does not automatically extend to these sources. No truncation, padding, quota
transfer, new teacher labels or sampling based on target performance.

Report component counts in every partition/family, largest-group share,
prompts per component, topic/source/task distribution and all quarantine
losses. Use the existing group-level bootstrap, 10,000 resamples and seed
20261029, stratified by family; token positions are not independent samples.
The proposed cross-family quarantine prevents the same component from being
independently resampled twice. The minimum of sixteen groups is an admission
floor, not a claim of high statistical power.

This amendment permits learning task format and label conventions. It gives a
cleaner test of new-document/instance prediction for a known workload, but a
weaker claim about transfer to novel task instructions. One dataset per family
also confounds family with source style; a good result cannot establish broad
cross-source or unseen-task generalization. The pilot still measures agreement
with the frozen target, not a new model's factual accuracy or benchmark score.

Task-transfer evaluation needs a separately frozen protocol that withholds
entire task families from both training and calibration, and accounts for any
retraining, data and compute. Do not infer it from the primary held-out score,
pool its metrics into the known-task result, or spend the existing budget on
it automatically. Instruction-paraphrase transfer is another distinct axis,
not an excuse to split duplicate examples.

## Effect on existing bounded evidence

No new sampling, tokenization or inference was performed for this amendment.
The following are conditional interpretations of the retained comparison.

| Source | Observed evidence | Effect if this policy is approved | Still missing |
| --- | --- | --- | --- |
| SQuAD at `7b6d24c440a36b6815f21b70d25016731768db1f` | 122/126 eligible questions in each sampled split, covering nine train and eleven validation article titles. Four shared-span pairs are within articles. | Generic QA headings need not connect unrelated examples. Article and substantive question/fact dependencies still do; there is no new admitted count. | At least sixteen held-out components, fixed quotas after cross-source/semantic exclusions, article identity/alias evidence and rights review. |
| DocRED at `7985b4e0371e6c61a756feb41b7b27becf71c666` | Eligible sampled documents: 69 annotated train, 67 dev, 71 test; 207/214 total. No sampled 32-token span pairs. | A reviewed generic extraction request would no longer collapse all records into one template group. Native document and answer-bearing entity/relation dependencies remain. | Enough independent components and 96 held-out prompts after full grouping, relation/fact duplication audit, Wikipedia provenance/rights and missing-test-label handling. |

DocRED therefore becomes a **plausible provisional extraction source**, not an
admitted replacement. Its 71 sampled eligible test documents are not 96
prompts or 71 cleared independent groups. SQuAD's eleven sampled validation
articles still fall below the independent-group minimum. Neither nonrandom
sample establishes population yield. Do not multiply sample pass percentages
by metadata row counts to claim quota capacity.

The minimal conditional four-source design is CommitPackFT, UltraChat, SQuAD
and DocRED. Code/prose provisional status is unchanged. The policy does not
clear repository/file rights, synthetic-content provenance or upstream text
licenses. The retained SQuAD CC-BY-SA and DocRED publisher/upstream license
distinctions still need review for actual uses and redistributed artifacts.
No new legal conclusion is made here.

NCBI's fixed-adapter upper bound of 86 eligible test documents remains below
96 even if its generic instruction dependency were removed. QASA's inspected
SciRIFF chain still has test data only. Neither old failure is repaired by
this policy. No additional source search is recommended before this review.

## Remaining admission evidence and stop gates

1. Explicit policy approval, including the generic-task versus substantive-
   example distinction, deterministic allocation and separate transfer claim.
   A reviewed policy is not yet acquisition or GPU approval.
2. A frozen generic-instruction registry and source-specific field adapters,
   plus an approved bounded census/acquisition protocol. Native identities,
   split roles, copyright/provenance and answer-field limitations must be
   documented before treating a row as usable.
3. Joint document/passage/entity/code/answer/example grouping across all four
   sources, complete benchmark exclusions, semantic adjudications and an
   auditable disposition for every raw overlap. Unresolved provenance,
   contested exceptions and benchmark dependencies block affected components.
4. A realized hash-bound manifest meeting unchanged quotas, group minima,
   natural lengths and context buckets without split repair or quota rescue.
   Report bounds and missing evidence rather than calling partial audits a pass.
5. Separate R1/cost-coverage and teacher authorization, then the post-teacher
   answer/benchmark audit before training. Preserve causal teacher alignment,
   all three conditional guess-position metrics, full-batch acceptance,
   eligible coverage and the measured-verifier cost recurrence. No projection
   becomes a measured runtime-speed claim.

If any required count or provenance gate fails, stop for review. Do not
reinterpret another repeated example as generic, move held-out rows, change
prompts or activate a smaller-family experiment to force admission.

## Preservation and delivery

Original Qwen2.5-3B Q8_0, the rank-256 architecture, fixed training schedule,
[1.538085 ms median/1.555598 ms p95 cost-only probe](2026-10-29-compact-draft-cost.md), exact-output requirements,
runtime defaults and qualified performance artifacts remain unchanged. Keep
the 72 GPU-hour parent ceiling, phase caps 4/20/40/8, non-transferable pilot
caps 2/2/1 and no automatic unlock; recorded usage remains 0.005923 hours.

One atomic documentation unit records this proposal and the research
checkpoint. No runtime implementation or dataset audit execution accompanies
it. Local formatting/link/whitespace and immutable-evidence hash checks are
sufficient; no GPU tests, source downloads, teacher generation, training or
GPU switching are authorized. The [policy evidence checksum list](2026-10-30-v3-isolation-policy-v1.sha256)
binds the proposal to the retained source evidence without modifying it.

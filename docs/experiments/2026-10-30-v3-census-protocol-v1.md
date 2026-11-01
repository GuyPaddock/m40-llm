# V3 census and registry protocol v1

Protocol `v3-census-review-1`, October 30, 2026.
Frozen proposal for review, not an executable or admitted data specification.

## Authorization and decision

The user approved the known-task versus example-leakage distinction in
[isolation proposal v1](2026-10-30-v3-isolation-policy-v1.md) at `0eee578`.
Generic instructions may recur only through reviewed exact-byte entries.
Document, passage, code, fact, answer-instance and substantive example-template
isolation remain mandatory. Entirely unseen task-family transfer remains a
separate objective requiring separate authorization.

This document proposes the census scope, resource bounds, reporting contract
and deterministic allocation rule. The accompanying
[registry v1](2026-10-30-v3-generic-instruction-registry-v1.json) contains four
candidate spans and no UltraChat exemption. Entry approval and implementation
are pending. No new source sampling, acquisition, tokenization, selection,
GPU work or target-output inspection was performed for this proposal.

Recommended review decision: approve the four narrow registry entries and
quota-aware allocation rule below before implementation. Keep the original
hash allocation as a one-shot diagnostic, not a reason to reject a source
with enough independently assignable groups. Approving the design does not
authorize bulk acquisition. A separate acquisition decision must name the
pinned files and accept the finite bounds below.

Preserve all prior failed admissions and specifications unchanged. In
particular, the original policy document remains a record of what was proposed
before this approval, not a file to edit retrospectively.

## Frozen scope

| Family and repository | Revision | Native rows reported by retained metadata | Research destinations |
| --- | --- | --- | --- |
| Prose: `HuggingFaceH4/ultrachat_200k` | `8049631c405ae6576f93f445c6b8166f76f5505a` | `train_sft` 207,865; `test_sft` 23,110 | Train SFT to train/calibration; test SFT to held-out only |
| Code: `bigcode/commitpackft`, Python only | `fc56fe33c030c6daa414c2b112c932b8eed085e6` | 56,025 Python records | Native Python train to new research train/calibration/held-out by whole groups |
| Grounded QA: `rajpurkar/squad` | `7b6d24c440a36b6815f21b70d25016731768db1f` | Train 87,599; validation 10,570 | Train to train/calibration; validation to held-out only |
| Extraction: `thunlp/docred` | `7985b4e0371e6c61a756feb41b7b27becf71c666` | Annotated train 3,053; dev 998; test 1,000 | Train/dev/test to train/calibration/held-out respectively |

These are metadata counts, not audited group counts. The proposed full scope
is 390,220 source records. Exclude UltraChat generation splits, non-Python
CommitPackFT, DocRED distant supervision and all mixtures or replacement
datasets. Do not expand the scope if a quota fails. A file count differing
from the frozen metadata stops for reconciliation rather than truncating it
or silently accepting a different population.

The retained repository inventories identify these exact data paths at the
pins above. This is an acquisition allowlist proposal, not permission to fetch
them. README/license metadata and separately logged provenance evidence do
not add records to the population.

| Repository | Allowed data paths |
| --- | --- |
| UltraChat | `data/train_sft-00000-of-00003-a3ecf92756993583.parquet`; `data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet`; `data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet`; `data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet` |
| CommitPackFT | `data/python/data.jsonl` |
| SQuAD | `plain_text/train-00000-of-00001.parquet`; `plain_text/validation-00000-of-00001.parquet` |
| DocRED | `data/train_annotated.json.gz`; `data/dev.json.gz`; `data/test.json.gz`; `data/rel_info.json.gz` as audit-only relation metadata, not extra examples |

Retain four families with **256 train, 48 calibration, 96 held-out per family**:
1,024/192/384 total. Require at least 16 independent held-out components per
family and at least 32 held-out prompts in each global length bucket:
65-127, 128-255 and 256-447 tokens. One example cannot fill two families.
No truncation, padding, answer appending or quota transfer. The MBPP-only
short-prompt amendment does not extend to these sources.

Use the original Qwen2.5-3B Q8_0 embedded tokenizer, GGUF SHA256
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
Preserve the existing QwenChat rendering, no added system prompt or BOS,
including the assistant generation prefix in natural input lengths. The
registry changes audit classification, never prompt bytes or token counts.
Keep the 128-generated-token limit and frozen-target-only teacher labels.
Dataset answers and assistant responses are isolation evidence only.

## What the existing evidence establishes

Counts below come from the retained
[HF sample summary](2026-10-30-hf-sample-summary.json) and
[original-source summary](2026-10-30-original-source-summary.json), not a new
census. Samples are bounded prefixes or spaced row groups, not random
population estimates. Do not multiply their yields by metadata row counts.

Bucket counts are eligible prompts in 65-127 / 128-255 / 256-447 order.

| Source/split | Sample rows; eligible | Bucket counts | Median / p90 tokens | Observed identity counts, before full grouping |
| --- | --- | --- | --- | --- |
| UltraChat train SFT | 126; 72 | 45 / 13 / 14 | 97 / 591 | 72 eligible prompt IDs |
| UltraChat test SFT | 126; 70 | 47 / 14 / 9 | 88.5 / 432 | 70 eligible prompt IDs |
| CommitPackFT Python | 128; 127 | 11 / 56 / 60 | 245.5 / 332 | 124 provisional eligible commit identities; 111 license-label-screened rows span 109 repository-set strings |
| SQuAD train | 126; 122 | 5 / 96 / 21 | 190.5 / 305 | Nine eligible article titles |
| SQuAD validation | 126; 122 | 11 / 91 / 20 | 185 / 289 | Eleven eligible article titles |
| DocRED annotated train | 71; 69 | 0 / 43 / 26 | 238 / 344 | 69 eligible titles |
| DocRED dev | 69; 67 | 0 / 38 / 29 | 233 / 355 | 67 eligible titles |
| DocRED test | 74; 71 | 0 / 40 / 31 | 249.5 / 356 | 71 eligible titles |

No row above establishes a cleared connected-component count. Repo-set strings
are not fork-independent repositories; titles are not resolved page revisions;
prompt IDs are not source-document identities. Full native document counts,
joint component counts and policy-qualified capacity remain **unknown**.
SQuAD's eleven sampled held-out articles do not meet the sixteen-group floor,
but do not prove population failure. DocRED's missing short bucket is not
itself failure because bucket requirements are global, not per family.

## Source-specific audit contract

### UltraChat

Use the `prompt` field unchanged and verify that it represents the first user
message in `messages`. Quarantine inconsistencies; do not choose the shorter
version. Retain `prompt_id`, source file, row locator and raw record hash.
Retain native `train_sft`/`test_sft` labels even though historical sample files
abbreviate them to train/test.

Group duplicate conversation IDs and copied/paraphrased requests, quoted
documents, facts and substantive examples. Natural user instructions have no
automatic registry exemption. Audit assistant turns for copied solutions,
documents and benchmark content without exporting them as prompts or labels.
Report unresolved source-document origins rather than inventing identities.

The [retained rights review](2026-10-30-hf-dataset-feasibility.md) records an
MIT collection label, ChatGPT-generated prompts/responses and H4 processing.
Generation-time terms, underlying quoted content, original/H4 attribution and
redistribution scope remain unresolved. The collection label is not training
clearance. General coding/browser questions also create substantive benchmark
overlap risk even without reserved marker strings.

### CommitPackFT Python

Keep the existing natural input: subject, the registered file delimiter,
`old_file`, a newline and `old_contents`. `new_contents`, commit message and
other source fields are audit-only. No gold patch becomes a target label.
Record commit SHA, all `repos` aliases, old/new paths, file hashes, native row
locator and source revision. A locator identifies a record, not a repository.

Build repository/fork and shared-commit components before filtering. Link
copied files/functions, normalized ASTs, renamed solutions, test fixtures,
problem paraphrases and documentation facts across repositories and sources.
Unsupported syntax or unresolved provenance cannot receive semantic clearance.

The prior sample contains 16 AGPL and one EPL label despite an MIT collection
card. The 111-row permissive-label screen is not file-level approval. Inspect
the license at the source commit, file notices and copied dependencies; record
training/use and redistribution decisions separately. Quarantine unresolved
rights. Code/config fixtures, common benchmark solutions and browser examples
need the full benchmark audit, not just marker matching.

### SQuAD

Retain `id`, `title`, `context`, `question`, native split and audit-only
`answers`. The original bounded extraction did not acquire answer columns;
that missing evidence must be explicit in a future census. Render the two
registered headings with untouched context/question bytes.

Group all questions from an article, overlapping passages, redirects and
source-derived paraphrases. Native titles and paragraph hashes are evidence,
not verified historical Wikipedia page/revision IDs. Preserve resolution
confidence and unresolved alias records. Link shared answer-bearing facts to
DocRED and quoted UltraChat material. A repeated short answer alone is not a
fact identity, but duplicated evidence or the same fact is a dependency.

The retained source review records CC-BY-SA 4.0 and Wikipedia-derived text.
Attribution, changes, source revisions and applicable redistribution conditions
need explicit review. No new legal clearance is inferred here. SQuAD-style
question wording alone is not a substantive template; instantiated benchmark
questions/facts remain excluded.

### DocRED

Use `title`, `sents`, `vertexSet` and available `labels` for provenance and
isolation. Prompt only the registered extraction request plus newline-joined
source sentences with tokens joined by a space. Never inject gold entity or
relation annotations. Bind native record locators to split/file hashes;
resolve titles and source identities separately.

Group document aliases, shared evidence passages and entity/relation instances
including qualifiers. A shared relation class or bare entity name is not
alone a shared answer instance. Generic extraction wording can be registered;
worked examples and source sentences cannot. Test lacks relation labels.
Report that limitation and audit available text/entities; do not borrow labels
from another split or assert complete answer clearance. Preserve the separate
post-teacher collision gate before training.

The retained review distinguishes Tsinghua's publisher MIT notice, Wikipedia
text and Wikidata structured content. None substitutes for the others' rights
or provenance. SQuAD/DocRED cross-source article/fact aliases are a specific
capacity risk, not independent data merely because publishers differ.

## Registry and overlap handling

Registry entries bind source pins, adapter SHA256, exact UTF-8 text and digest,
half-open byte/Unicode-scalar boundaries, rationale and pending reviewer
decision. DocRED contributes one 62-byte instruction; SQuAD two headings of
9 and 12 bytes; CommitPackFT one 8-byte delimiter. UltraChat contributes none.
These byte counts include literal newlines. They are not token counts.

Require a second review before activating an exemption. Match only at the
adapter boundary; never locate exempt text by searching the payload. Verify
the adapter hash and exact text before any normalization. Changed source,
adapter or wording stops for a new version. No catch-all instruction regex.

Keep original full-content matches alongside content-only results. Preserve
the existing normalized exact match, contiguous 32-token and five-word
Jaccard >=0.5 checks. A generic-only disposition needs every matched span to
lie wholly inside approved entries. Cross-boundary or payload matches remain
actionable. Do not concatenate separated content fragments into artificial
spans after masking. Retain raw scores, normalized/token offsets, registry
IDs, reviewers, decisions and reasons, including rejected exemptions.

Build one dependency graph across all four sources, before length filtering
or quota selection. Include ineligible records that bridge otherwise eligible
examples. Add document, passage, code, fact, answer-instance and substantive
example edges with evidence. Keep semantic/paraphrase review separate from
literal detection; unresolved candidates quarantine affected components.

Keep the conservative proposal to quarantine cross-family components and
components crossing native splits within a source. Report those losses
separately. Never move native held-out documents into training. Preserve the
original component-ID serialization from policy v1; namespace native record
IDs by source and split, and retain raw IDs separately to avoid collisions.
Freeze canonical JSON serialization and its test fixtures before census code
runs. Components receive one research destination at most.

Benchmark exclusions apply to every partition and all available input and
audit-only answer fields. Preserve the 265,771-text inventory and its known
coverage gaps. Resolve or report missing script/external artifacts; zero
literal matches is not clearance. Browser-loading topics/templates, frozen
twelve-task outputs, copy/draft held-out material, KV fixtures, traces,
`ZXQ-NEEDLE-41729` and `M40_BATCH_LIMIT` remain excluded. Generic status cannot
waive a benchmark-specific scaffold, even with renamed constants.

## Bounded execution proposed for later authorization

Nothing in this section runs now. Approve implementation separately, then
review its local fixtures before authorizing acquisition. Existing bounded
sample tools must not be repurposed into full downloads by removing limits.

1. Reproduce retained sample hashes and token counts locally. Validate exact
   registry bytes, Unicode boundaries, payload lookalikes, raw-match retention,
   group transitivity and native-split quarantine on CPU fixtures. No network.
2. Freeze a pinned file inventory from retained metadata. Only SFT parquet,
   Python records, SQuAD plain-text train/validation and DocRED annotated
   train/dev/test files are in scope. Bind upstream locators, known sizes and
   repository revisions; calculate content hashes if acquisition is approved.
   No automatic trust-remote-code loaders, new repositories or model downloads.
3. Request explicit permission for that finite census. Proposed hard ceilings
   are 3 GiB transferred total, at most 1 GiB each for UltraChat/CommitPackFT,
   128 MiB SQuAD and 32 MiB DocRED. Within the total, allow at most 64 MiB and
   256 requests for rights/identity evidence and 16 MiB for metadata. Count
   retries/partial responses against limits. The caps are cost limits, not
   measured file-size estimates or a promise that every source fits.
4. Stream all in-scope records if authorized. Bound peak RAM at 4 GiB, retained
   workspace at 12 GiB, compute at 8 aggregate CPU-hours, two worker threads
   and 8 hours wall time. Include tokenization, indexes and allocation checks
   in those limits. Reuse the original local tokenizer without loading GPU
   weights. Stop on a cap or source/metadata mismatch; do not truncate and
   call it a completed census. Network evidence uses pinned or archived
   source identities and records every unresolved lookup.
5. Produce raw identity/length counts and the joint dependency/rights ledger.
   Reserve at most 8 reviewer-hours in this bounded attempt, with unresolved
   cases reported rather than cleared by sampling. A complete semantic audit
   may need a separately approved continuation. Resource exhaustion is an
   incomplete audit, not evidence of true source-capacity failure.
6. Compute the diagnostic hash result and capacity certificates below without
   target outputs. Do not emit a selected training manifest until the registry,
   grouping and allocator are approved. Return coverage, bounds and blockers
   for review even if every count appears feasible. No R1 or teacher approval
   follows automatically.

No entire-source clearance follows from a random review of a few records.
All prospective selected records require provenance and semantic review, and
their unselected/ineligible neighbors remain part of the dependency graph.
An unresolved link to a prospective component blocks it. Raw protected content
stays in controlled local artifacts; publish hashes, locators and decisions
rather than redistributing full corpora under assumed collection rights.

## Capacity is not hash allocation

Use seed `20261029` once. Retain policy v1's exact hash allocator: fixed native
roles stay fixed; train/calibration groups use modulo 304 with calibration
below 48; unrestricted groups use modulo 400 with train below 256, calibration
below 304 and held-out otherwise. No retries with new seeds. Report per-family
role and bucket deficits after its specified round-robin selection.

Assess feasibility independently of that assignment on the same frozen
graph. For every component retain permitted roles and eligible unique row
counts by bucket. Enforce one role per component, all family quotas, held-out
group minima and global bucket coverage simultaneously. Independent per-role
totals alone are insufficient when flexible groups compete for roles.

Report these outcomes separately:

| Outcome | Required evidence | Decision |
| --- | --- | --- |
| Incomplete/uncertain | Missing source rows, unresolved edges/rights, resource cap, or solver timeout | Capacity unknown; stop with coverage and optimistic bounds |
| Source-capacity failure | Even an optimistic complete-scope upper bound cannot satisfy native-role quotas/groups/buckets | Proven failure for this pinned scope; no source expansion |
| Isolation/rights-qualified failure | Complete audit and certified constraints are infeasible after required exclusions | Explain losses and stop; distinguish this from raw source scarcity |
| Hash-allocation shortfall | Historical hash assignment fails, but a checked legal assignment witness meets every constraint | Allocation artifact, not source failure; alternative needs approval |
| Feasible counts | Checked witness on cleared components, with required scope/audit coverage | Capacity evidence only, not license, teacher or training admission |

For unresolved cases, publish both a cleared-pool bound and an optimistic
bound allowing unresolved rows/edges their most favorable legal treatment.
Feasibility of the optimistic problem proves nothing about admission.
Infeasibility requires a verifiable bound/proof, not a greedy algorithm getting
stuck. A feasible witness must be independently checked against the ledger.

Illustration only: two flexible groups of 256 and 48 eligible rows could both
hash to train. Calibration would be empty despite a legal 256/48 assignment.
Conversely, fewer than 96 total eligible native DocRED test rows cannot be
repaired by a hash rule. Neither example asserts a measured population count.

## Proposed quota-aware alternative

`quota-aware-v1` is a single frozen assignment method, proposed before any row
selection or target-output inspection. It is not permission to keep revising
partitions until a model performs well. Recommend it for the eventual census;
run the historical allocator once for comparison regardless of its outcome.

Specify a finite integer feasibility problem. For component `g`, destination
`s` and length bucket `b`, use binary assignment `y[g,s]`, integer row count
`x[g,s,b]` and audited available count `n[g,b]`. Only permitted roles exist.
Require sum of `y[g,s]` over roles to equal one for every usable component,
and `0 <= x[g,s,b] <= n[g,b] * y[g,s]`. Unused rows may have zero counts but
retain the component's assigned destination.

For each family/destination, sum of row counts equals 256/48/96. A held-out
component counts toward the sixteen-group minimum only if at least one of
its rows is actually selected. Introduce binary `z[g]` with
`z[g] <= sum_b x[g,heldout,b] <= 96*z[g]` and require at least sixteen per
family. Require at least 32 selected held-out rows per global bucket. Do not
split a component, duplicate rows or count tokens as independent groups.

Choose one reproducible solution without target quality signals:

1. Minimize the number of flexible components assigned differently from their
   original seed-20261029 hash destination. Fixed native roles never move.
2. Break ties lexicographically over component destinations in ascending
   original hash/component-ID order, with destination order train, calibration,
   held-out. Then choose the lexicographically smallest flattened row-count
   vector in component/destination/bucket order. This freezes a unique
   assignment/count result instead of relying on solver traversal order.
3. After separate selection approval, satisfy each component/bucket count
   using member records ordered by the same seeded hash and canonical locator.
   Retain unused rows and all component memberships. The row-count tie-break
   replaces v1 round-robin for this alternative; report largest-group share
   and concentration rather than implying population-representative sampling.

Freeze solver version, integer formulation, serialization, objective order
and digest before execution. Use a checked witness for feasibility and a
checkable infeasibility certificate or sufficient mathematical upper bound
for failure. An unsupported proof format or exhausted solve is unknown.
Lexicographic optimization is one declared algorithm, not repeated experiments
with seeds, constraints or target outputs. It stays inside the census CPU cap.

This method avoids accidental quota loss while conditioning sampling on
component size and length coverage. It is not an unbiased random sample of
the source population. Preserve group-level, family-stratified bootstrap with
10,000 resamples and seed 20261029; report group concentration and the limits
of the sixteen-group floor. One source per family still confounds source
style with task family. No claim of task-transfer generalization follows.

Freeze the audited graph before allocation. If later provenance review or
post-teacher collision checks change it, invalidate the witness and stop for
review. Do not refill, rerun allocation or select another held-out set without
a new explicit version and authorization. No output-based choice of allocator.

## Required census delivery

| Artifact | Required contents |
| --- | --- |
| Source coverage ledger | Pins/files/hashes, expected and seen rows, native split counts, processed bytes, incomplete fields, acquisition/CPU limits |
| Registry decision ledger | Exact spans, digest/adapter checks, two-review decisions, raw and content-only overlap evidence; no erased original matches |
| Provenance/rights ledger | Native IDs and source locators, document/repository aliases, evidence hashes, confidence, attribution/terms, use versus redistribution, unresolved/quarantined records |
| Group graph and counts | Typed edges, source documents, repositories and components per native split/family, length filtering, exclusions, cross-split/cross-family losses, cleared versus optimistic capacity |
| Length report | Full natural QwenChat min/median/p90/max and bucket counts before/after exclusions, component-level bucket support, no sample-to-population extrapolation |
| Allocation comparison | Original hash result, deficits, certified capacity/witness, proposed quota-aware roles, group concentration; no seed search |
| Benchmark report | Frozen inventory digest and coverage gaps, literal and semantic dispositions, source answers audit-only, unresolved test-answer limitations |
| Readiness decision | Pass/blocked/fail with causes; no R1 claim until all evidence and later calibration requirements are satisfied |

Do not reuse source labels as teacher targets. The frozen runtime alone later
produces continuations and causal hidden-state features. Post-teacher answer
collisions remain a separate pre-training gate, never a reason to tune prompts
or inspect held-out acceptance now.

## Implementation sequence after review

This turn changes documentation and an inert registry only. Proposed later
atomic units, not authorized execution orders:

| Unit | Dependency | Purpose and draft Conventional Commit |
| --- | --- | --- |
| Registry audit support | Reviewed entries/protocol | Validate exact spans and retain both overlap views with CPU behavioral fixtures: `feat(data): audit registered instruction spans` |
| Census reporting | Registry support, acquisition decision | Stream approved pinned scope with hard caps and source/group/rights ledgers: `feat(data): report bounded V3 census coverage` |
| Capacity and allocation | Reviewed allocator, census schema | Distinguish proven capacity from assignment artifacts with witnesses and CPU fixtures: `feat(data): certify quota-aware allocations` |
| Measured census checkpoint | Completed authorized audit | Preserve realized counts, costs, unresolved cases and gate decision: `docs(data): record V3 census readiness` |

Freeze implementation/tool hashes before executing the census. A reviewed
protocol does not itself approve all four units or any corpus download.

## Preserved controls and stop

No runtime, source adapter or active training specification changes here.
Keep rank-256 and all three guess positions, the original 36-layer Qwen target,
fixed training schedule, numerical gates and exact-output benchmarks. Preserve
[proposal-cost evidence](2026-10-29-compact-draft-cost.md): 1.538085 ms median,
1.555598 ms p95; learned acceptance remains unmeasured. Keep the immutable
50.910426 HTTP tok/s reference and all later qualified artifacts unchanged.

Keep 72 aggregate GPU-hours, phase caps 4/20/40/8 and non-transferable pilot
teacher/training/evaluation caps 2/2/1. Usage remains 0.005923 GPU-hours. Zero
GPU time, teacher generation, training, evaluation or device switching is
authorized by this document. The accepted V2 short-context protocol remains
planning evidence, not calibration permission.

Return this protocol and registry for review. Do not begin acquisition,
implementation, selection or calibration automatically.

The accompanying `2026-10-30-v3-census-protocol-v1.sha256` binds this proposal,
registry, source pins, existing adapter/tokenizer code and prior evidence
bundles. Any approved revision gets a new version rather than overwriting
these review artifacts.

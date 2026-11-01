# V3 CPU preflight decision

October 30, 2026. This is implementation and resource evidence for the
[frozen census protocol](2026-10-30-v3-census-protocol-v1.md), not a census,
selected training manifest or R1 admission.

## Decision

**No-go for complete-source acquisition under the current implementation.**
The registry boundary checks and small-instance allocation formulation pass.
Population-scale canonical allocation is not implemented, repeated-header
candidate generation exhausts its probe budget, and independent second review
and source provenance remain incomplete. These are preflight limitations, not
evidence that the four sources cannot supply the unchanged quotas.

The user approved the four registry candidates and quota-aware methodology,
subject to second review and validation. Neither this agent's implementation
nor its tests count as an independent second reviewer. No registry exemption
is active. No source, seed, quota, native split or matching threshold changed.

## Implementation and validation

| Commit | Scope | Validation |
| --- | --- | --- |
| `d64e76b` | Hash-bound registry, unchanged rendering, byte/Unicode spans and raw overlap dispositions | Five CPU behavioral tests |
| `f427d8d` | SQLite literal candidates, complete-content verification and transitive group quarantine | Three CPU tests, including exhaustive candidate comparison |
| `919a5f2` | Whole-component integer reference, independent witness check and deterministic objective | Eight CPU tests covering feasibility, infeasibility and unknown |
| `58cbfc3` | Local-evidence replay and bounded synthetic scale driver | Completed local/synthetic measurement run below |

The complete audit suite passes **55 tests**, including the prior 39. See
[test output](2026-10-30-v3-preflight-tests.txt) and
[machine-readable measurements](2026-10-30-v3-preflight-results.json).
`cargo fmt --all -- --check` and whitespace checks pass. No Rust/CUDA runtime,
old source adapter, prior specification or prior evidence artifact was edited.

The first test invocation used system Python, which lacks PyArrow. It failed
to import the historical audit modules; the existing project CPU virtualenv
ran the tests successfully. No dependency was installed. The first scale
attempt stopped because UltraChat's retained projection omits `messages`.
The completed run records that missing check explicitly instead of acquiring
more data or treating it as passed.

## Registry review evidence

The immutable registry remains at SHA256
`b8c250e9a15ec5e53c8970094adb5c14b7190f6de10694a636dd3966deda100b`.
Source revisions and both historical adapter digests were verified. The
preflight reconstructs inputs from retained source records and compares their
complete text multisets against the original tokenization inputs.

| Source | Replayed records | Verified candidate spans | Boundary decision |
| --- | --- | --- | --- |
| UltraChat SFT | 252 | 0 | Entire natural prompt remains payload |
| CommitPackFT Python | 128 | 128 | Only the eight-byte adapter delimiter; not path, subject or code |
| SQuAD | 252 | 504 | Nine-byte context heading and twelve-byte question delimiter at structural offsets |
| DocRED | 214 | 214 | Only the 62-byte generic extraction prefix; no document sentences or annotations |
| Total | 846 | 846 | Original prompt bytes identical |

Tests cover UTF-8/scalar offset differences, literal heading copies inside
payload, filename/code boundaries, an unlisted UltraChat instruction,
invalid coordinates, source revisions and tampered adapter identity. Raw
overlap text and coordinates survive classification. A match extending one
character into payload is not generic-only. No global search/replace, regex
masking or automatic instruction exemption is used.

Second-review checklist, all still pending independent sign-off:

- Confirm each fixed string supplies no instance-specific evidence or answer.
- Confirm the structural byte/scalar boundaries match the pinned adapters.
- Confirm payload lookalikes, cross-boundary matches and unlisted instructions
  stay subject to full-content checks.
- Confirm the registry changes audit classification only, not actual inputs,
  tokenization, source grouping or benchmark exclusions.

This run reuses the original Qwen comparison IDs. It does not claim a fresh
tokenizer benchmark: unchanged prompt bytes and unchanged tokenizer identity
preserve the existing length evidence. UltraChat first-user consistency cannot
be checked without its missing `messages` column. SQuAD answer columns and
DocRED test relations also retain their documented evidence limitations.

The raw matching index deliberately does not apply registry waivers. Its
full-content result is conservative while second review remains outstanding.
The scalar-span classifier is not a validated normalized-token-to-original-
offset mapper. Automatic exemptions in normalized matches require that extra
integration validation, including normalization-induced boundary changes.

## Candidate generation and grouping

The new index uses complete exact-text and 32-token posting keys, plus a
globally ordered prefix for five-word Jaccard >=0.5. For a set of size `n`,
the indexed prefix has `n - ceil(n/2) + 1` elements. Qualifying sets must share
a prefix key. Full original comparison sets verify every candidate; SHA256
posting collisions can add candidates but cannot certify a false match.
Empty shingle sets do not create a Jaccard match.

No frequent key is dropped. All verified raw pairs, reasons, original text
and comparison IDs remain in local SQLite evidence. A limit raises an
incomplete-audit result. The literal index does not claim to discover all
document aliases, AST variants, shared facts or paraphrased examples. Those
edges still need the existing source/semantic audits before admission.

Across the **846 retained records**, indexed results match an exhaustive
357,435-pair oracle exactly: **22 matching pairs**, after 23,515 candidate
checks. Matching took 11.820 CPU seconds and 11.826 wall seconds, using a
13,053,952-byte logical database. This validates the retained raw-input view,
not full source-answer fields, benchmark coverage or semantic clearance.

| Synthetic probe | Completed work | CPU seconds | Logical database bytes | Meaning |
| --- | --- | --- | --- | --- |
| 1,000 unique records, 256 words/IDs each | 353,000 postings; no candidate pairs | 5.255 | 22,118,400 | Best-case content indexing |
| 10,000 unique records, 256 words/IDs each | 3,530,000 postings; no candidate pairs | 103.246 | 225,882,112 | Larger best-case index; cost is not linear from the smaller probe |
| 2,000 intended records, twenty shared header words | Stops after 200 records and 19,900 pair checks | 8.809 | 4,304,896 | Unknown at the 20,000-candidate cap; zero verified matches so far |
| 390,220 unique one-word/one-ID records | All 390,220 exact keys | 10.774 | 30,826,496 | Record-count plumbing only, not realistic token workload |
| 390,220 graph vertices, 387,171 edges | 3,049 components, 382 quarantined | 0.952 | Not a database probe | Union/quarantine cost only, not identity discovery |

Synthetic IDs are not Qwen tokenizer measurements or natural-length eligibility
evidence. Grouping includes every vertex before filtering and propagates
native-split/family conflicts transitively. Its speed does not price the work
needed to discover or adjudicate those edges.

Peak process RSS across the entire run was 1,745,788 KiB, **1.665 GiB**, under
the 4 GiB address-space cap. Per-stage RSS fields are cumulative high-water
marks, not separate stage allocations. The completed run consumed 149.091 CPU
seconds, about 0.0414 CPU-hours. It had a 900-second hard CPU cap and used no
parallel worker pool. Each matching probe had a separate 180 CPU-second cap.

## Allocation formulation and proof limits

The reference enumerates the protocol's binary whole-component assignments
and integer bucket counts. It checks exact family/destination quotas, native
permitted roles, unique capacity use, global held-out buckets and contributing
held-out group minima. Groups assigned to held-out but supplying zero rows do
not count toward the independence floor. Synthetic reduced quotas exercise
the formulation; population probes keep 256/48/96, sixteen groups per family
and global bucket minima 32/32/32.

The objective is unchanged: minimize departures from seed `20261029` hash
allocation, then the destination vector in seeded group order, then bucket
counts. Unassigned-role counts are implicit zero; this preserves the flattened
vector order when destination vectors tie. The search compares the full key,
not the first feasible result. Input permutations produce identical canonical
witnesses in completed tests.

An independent checker verifies each candidate witness. Sufficient upper
bounds prove native-role, joint-role, bucket and group-minimum shortages.
Small exhaustive enumeration can prove indivisible-group infeasibility even
when those bounds pass. A state/CPU stop returns `unknown`, retaining any
checked feasible witness but making no canonical-optimum claim. The component
limit is a reference-solver guard, not an eligibility or grouping rule.

| Probe | Result | CPU seconds | Interpretation |
| --- | --- | --- | --- |
| Four synthetic components | Feasible canonical optimum, 4,145 states | 0.0140 | Complete reference proof |
| Eight synthetic components | Unknown at 50,000 states; feasible witness found | 0.1374 | Feasibility witness is not completed canonical allocation |
| Twelve synthetic components | Unknown at 50,000 states; feasible witness found | 0.1556 | Same distinction |
| 1,000 synthetic components, full quotas | Unknown at reference size guard | 0.0040 | Not a failed dataset quota |
| 390,220 synthetic components, full quotas | Unknown at reference size guard | 1.8520 | Validation/bounds only; no large integer solve measured |
| Two flexible groups both hashing to train | Canonical feasible witness, 45 states, one departure | Included in total | Genuine hash-allocation shortfall, not capacity failure |

The last fixture has capacities two and one and train/calibration quotas two
and one. Original hash roles are both train; quota-aware roles put the one-row
component in calibration. No real source record was selected or reassigned.
Separate tests establish actual native-role/bucket shortages and an indivisible
group unable to satisfy two destinations. Seeds and assignments were not
searched against target outputs.

This standard-library reference was chosen to validate constraints without
installing a solver or claiming a large-model proof certificate. It is
**not a scalable canonical allocator**. Production solver choice, proof/witness
formats, exact objective preservation and scalable tie-breaking remain open.

## Resource assessment against frozen limits

| Requirement | Evidence or estimate | Admission consequence |
| --- | --- | --- |
| Literal index within eight CPU-hours | Extrapolating only the 10,000-row unique synthetic probe gives about 1.119 CPU-hours for 390,220 equally shaped records | Conditional projection, not a census measurement; common-key and large-field costs are excluded |
| Workspace within 12 GiB | The same projection gives 8.209 GiB for one logical index | Excludes raw corpora, answer fields, separate content views, benchmark index, temporary files and solver storage; aggregate fit unproven |
| Common instructions without weakened matching | Twenty shared header words exhaust candidate cap at 200 records despite no true matches | Current hash-ordered prefix index has poor common-key selectivity; completion at source scale not demonstrated |
| Canonical allocation within eight CPU-hours | Exact reference stops at eight/twelve groups under its bounded state allowance | No credible population-scale estimate from this algorithm; do not extrapolate cheap capacity checks to full solves |
| Semantic/provenance review within eight reviewer-hours | At least 1,600 eventual selected prompts require review | Only eighteen seconds per selected prompt if all eight hours were spent on them; leaves no time for aliases, rights, benchmark/template adjudication or second review |
| Source-field coverage | UltraChat messages and some source-answer evidence absent locally | No first-user/answer-provenance clearance from this run |
| Benchmark and semantic isolation | Literal logic validated; previously documented inventory and semantic gaps remain | A fast literal matcher cannot certify these requirements |

No reviewer-throughput measurement supports eighteen seconds per example.
Do not treat that arithmetic as proof of a feasible review schedule, nor
declare the task mathematically impossible. A complete provenance and semantic
review is not credible within the current eight-hour allowance without much
stronger evidence and automation. The limit has not been increased or spent
on a claimed independent review.

Worst-case pair work remains material: 390,220 records contain 76,135,629,090
pairs. This is a combinatorial bound, not measured corpus overlap. The current
stress result motivates better complete candidate generation, not deletion of
common instructions from raw evidence or relaxation of thresholds.

## Review requested before a census

Do not authorize bulk acquisition yet. A follow-up CPU-only engineering proposal
should address these items without changing the isolation contract:

1. Obtain independent sign-off on the four exact spans. Validate normalized
   match-to-original offsets before applying any exemption automatically.
2. Replace the weak common-key candidate ordering with a complete, more
   selective strategy, such as a measured global-frequency prefix order.
   Retain an exhaustive oracle, raw evidence and incomplete-work semantics.
3. Choose and validate a scalable integer formulation/solver, including the
   frozen minimum-departure and exact lexicographic objective. Do not silently
   replace it with a greedy allocator or repeated seed trials.
4. Price the missing field, benchmark, rights and semantic work. Demonstrate
   a realistic reviewer workload or return a separate budget proposal; do not
   claim the current eight reviewer-hours suffice.

Those are recommendations for review, not work automatically started here.
The full-census capacity of the four pinned sources remains unknown. No quota
was lowered and no source substituted to repair preflight limitations.

## Reproduction and preservation

```bash
local-artifacts/2026-10-29-drafter-data/venv/bin/python \
  -m unittest discover -s scripts/drafter_data -v
local-artifacts/2026-10-29-drafter-data/venv/bin/python \
  scripts/drafter_data/run_v3_preflight.py \
  --output local-artifacts/2026-10-30-v3-preflight-run2
```

The recorded output path already exists; reproduction requires a new output
directory. The driver refuses replacement. Request escalation for tests under
project policy. These commands read retained evidence and create synthetic
artifacts only. They do not acquire sources or run a model.

Python 3.11.14 and SQLite 3.50.4 were used. Local databases retain raw source
evidence under the ignored run directory; no third-party prompt corpus is
redistributed in Git. The committed JSON is byte-identical to the local report.
The accompanying checksum bundle binds implementation, tests, reports, local
database artifacts and the prior immutable evidence chain.

Rank-256, the original Qwen target, cost-only 1.538085 ms result, numerical
gates and all performance references are unchanged. GPU usage remains
0.005923/72 hours; phase and non-transferable pilot caps remain unchanged.
No GPU work, device switch, teacher generation, training, selected training
manifest, speculative integration or runtime-default change occurred.

# V2 CPU Manifest Preparation Checkpoint

## Decision

The approved MBPP-only eligibility change is implemented in a new
[v2 specification](2026-10-29-drafter-data-spec-v2.json), commit `28195c2`.
The original specification, plan, failed admission, source artifacts and
tokenizer ledger remain unchanged; all27 entries in the original checksum
manifest pass. No GPU work, teacher generation, training, device switch or
runtime change occurred.

CPU preparation produced a realized candidate/quarantine ledger for all964
MBPP and15011 Dolly source rows. It did **not** produce an admitted selected
training manifest. Dolly upstream-document provenance remains unresolved,
and the remaining template/exclusion audits have not passed. No row was
assigned to training to conceal those gaps.

The [calibration protocol](2026-10-29-short-context-calibration-v2.md) is frozen
for review. It distinguishes mandatory correctness at positions1-65 from
13 representative timing positions and five preregistered interpolation
checks. At most four additional timing positions require a recorded cost-model
reason. All GPU calibration still requires separate authorization; its proposed
0.25-hour allocation stays inside the existing one-hour pilot evaluation cap.

## MBPP Result

Implementation `5d69cb6` binds every task to the original runtime-tokenized
ledger and its unchanged Parquet source. No alternate tokenizer, character
estimate or rewritten prompt was used. All964 rows satisfy the new natural
short-prompt rule. Official source partitions are preserved.

The conservative automated grouping connects exact normalized problem text,
identical parsed reference programs, and normalized five-word Jaccard >=0.5.
Programs are parsed, never executed or supplied as prompt content. Identical
parsed programs preserve identifiers, operators and constants; this is only
a subset of the required normalized-AST/manual template audit.

There are127 duplicate edges and879 connected components.41 components span
official partitions, quarantining105 rows. Grouping occurs before eligibility
filtering, so an ineligible member cannot hide a cross-partition connection.

| MBPP split | Source / length-eligible | Cross-partition quarantine | Remaining upper bound | Required quota |
| --- | ---: | ---: | ---: | ---: |
| Train | 374 | 40 | 334 | 256 |
| Calibration | 90 | 14 | 76 | 48 |
| Held-out | 500 | 51 | 449 | 96 |

These bounds do not yet prove independent group capacity, benchmark isolation
or manual template clearance. No solution parse failures occurred. Shared
normalized32-token spans, richer AST/template equivalence, paraphrases and
the expanded benchmark exclusion audit remain pending. Additional filters can
only reduce the available rows. Do not call this an MBPP isolation pass.

## Dolly Provenance Blocker

The source card identifies Wikipedia-derived context passages, but the retained
payload has only `instruction`, `context`, `response` and `category` fields.
No reviewed canonical source-document map is available. Neither a paragraph
hash nor its opening subject establishes document-disjoint train/held-out sets.
The candidate ledger therefore records null upstream document IDs and null
manual reviewers, rather than inferring provenance.

| Required stratum | Source rows before length/group filters | Unresolved context rows |
| --- | ---: | ---: |
| Prose/open QA | 4451 | 0 |
| Grounded QA | 1773 | 1773 |
| Extraction/classification | 3642 | 1506 |

All grounded-QA candidates are currently quarantined. Its unchanged quotas
are256/48/96 across train/calibration/held-out. With zero document-resolved
candidates, no complete pilot manifest can presently satisfy those quotas.
This is unresolved provenance, not proof that provenance can never be
reconstructed or that Dolly is unsuitable in principle.

The other rows are not implicitly cleared. Dolly runtime tokenization,
duplicate/template grouping, split allocation and longer-context held-out
bucket quotas remain unverified. Source/license notices are retained, but
combined artifact-use and document-provenance clearance is incomplete.

## Artifacts And Reproduction

- [CPU preparation report](2026-10-29-drafter-v2-preparation.json).
- [Content-free MBPP duplicate evidence](2026-10-29-drafter-v2-duplicates.json).
- [Independent consistency audit](2026-10-29-drafter-v2-audit.json).
- [Consistency audit program](2026-10-29-drafter-v2-audit.jq).
- [Checksums](2026-10-29-drafter-v2.sha256).
- [CPU tool instructions](../../scripts/drafter_data/README.md#v2-cpu-preparation).

The complete local ledger is
`local-artifacts/2026-10-29-drafter-data-v2/candidates.json`, SHA256
`8aec641405b976e696a1c295d6139daa82cbd3dfe8fbd5f6cb99a3ff641e6020`.
It records all source IDs/line numbers, source hashes, unselected status and
quarantine reasons. MBPP records additionally bind original split, token
length, solution fingerprint and duplicate component. The original local
tokenizer ledger retains prompt bytes and complete IDs. Neither is a selected
training split. Preserve them with the original source/license artifacts.

The independent jq audit recomputes residual counts, checks the exact official
ID/partition sets, duplicate-edge/component consistency, cross-partition
quarantine, unresolved grounded-QA rows and absence of any selected records:

```bash
jq -e --slurpfile report docs/experiments/2026-10-29-drafter-v2-preparation.json \
  -f docs/experiments/2026-10-29-drafter-v2-audit.jq \
  local-artifacts/2026-10-29-drafter-data-v2/candidates.json
```

Twelve CPU unit tests pass: nine grouping/provenance/evidence-ownership cases
and three existing Parquet conversion cases. The first test run used a pretty
JSON object instead of a one-line JSONL fixture; correcting the fixture made
the real JSONL reader test pass. No production parsing rule was relaxed.
Formatting hooks and whitespace checks pass. No CUDA tests were run or needed.
The preparation command's nonzero exit is the explicit incomplete-admission
decision, not a test failure or GPU issue.

## Next Review

Review the frozen protocol and the partial CPU evidence. A complete realized
training manifest remains blocked on a reviewed upstream-document mapping for
Dolly, then the unfinished grouping/exclusion/length checks. MBPP's remaining
upper bounds must also survive those checks. Do not remove the grounded-QA
stratum, substitute source partitions or weaken isolation to fill quotas.

This checkpoint does not authorize GPU calibration. Teacher generation and
training remain blocked until manifest and cost-coverage requirements pass.
The original rank256 architecture and training schedule are unchanged. The
72-hour parent ceiling, nontransferable4/20/40/8 phase caps and pilot2/2/1 caps
remain intact. GPU usage remains0.005923 hours from the accepted cost-only
probe. No learned acceptance or new speed claim is available.

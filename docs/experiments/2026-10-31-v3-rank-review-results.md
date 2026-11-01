# V3 count amendment and local review-cost results

October 31, 2026. Two CPU-only follow-ups to `a7479ba`, with no corpus
acquisition, teacher work, training, GPU access or inference changes.
Preregistration `2c1c735` freezes the
[rank-count amendment](2026-10-31-v3-rank-amendment-v2.md) and
[review-cost protocol](2026-10-31-v3-review-cost-protocol-v1.md).
Allocator implementation is `cc5be21`; review tooling and the
[sixteen fixed locators](2026-10-31-v3-review-cost-manifest-v1.json) are
committed at `5eb63db` before the timed review run.

## Allocator comparison

V2 retains the exact native constraints, minimum departures and destination
tie-break. Only the final count objective changes to a one-based rank-weighted
sum. Each returned witness is serialized, reloaded and checked independently
against the original constraints. Integer rank cost is recomputed outside
CP-SAT. Tiny exhaustive cases check objective order, nonzero hash repairs,
tied rank costs, native shortages and independence/bucket requirements.

Reproducibility means full witness equality in two fresh processes after
canonical input sorting, including reversed caller order. It does not mean
V1's unique mathematical count vector or reproducibility across untested
solver/platform changes. A feasible incomplete result cannot pass this gate.
All probes retain the same 90 CPU/120 wall seconds, 128 calls and 4 GiB address
space. The V1 small control is contemporaneous; other V1 costs are explicitly
historical. These are bounded single-run timings, not paired speedup estimates.

| Components | V1 allocator CPU s / status | V2 allocator CPU s / status | V2 repeat |
| ---: | --- | --- | --- |
| 72 | 0.040 historical, canonical | 0.022, all objectives proven | Identical witness, 0.021 s |
| 3,200 | 13.718 fresh, canonical | 1.603, all objectives proven | Identical witness, 1.596 s |
| 11,776 | 90.102 historical, count objective incomplete | 10.003, all objectives proven | Identical witness, 10.042 s |
| 79,803 | 90.458 historical, count objective incomplete | 90.722, count objective incomplete | Not run; incomplete result |
| 390,220 | Historical memory-limit unknown, no witness | 72.359, memory-limit unknown, no witness | Not run; unknown result |

The large result retains a checked feasible witness with zero departures and
canonical destinations. It does not prove the final rank optimum. There was no
retry, cap increase or objective adjustment. A simplified objective does not
eliminate the original integer model's time and memory costs.

### Selection concentration is not unchanged

The 3,200-component fixture has zero departures and identical destination
priorities in both versions, but V2 contributes 1,030 groups versus V1's 1,037.
Only 303 groups overlap. This is expected: V1 minimizes early counts toward
zero, while V2 favors low-rank contributions. Lower rank cost is an objective
result, not evidence of better data quality or representative sampling.

| Small fixture partition | V1 groups / HHI | V2 groups / HHI | Maximum group share |
| --- | --- | --- | --- |
| Code train | 127 / 0.009247 | 121 / 0.009460 | 1.1719% in both |
| Code calibration | 23 / 0.050347 | 27 / 0.041667 | 6.25% in both |
| Code held-out | 52 / 0.023003 | 47 / 0.024306 | 3.125% in both |
| QA held-out | 16 / 0.143880 | 16 / 0.143880 | 16.6667% in both |

The sixteen-group floor does not imply sixteen equally weighted groups. QA
calibration remains concentrated in three groups in this fixture, as in V1.
All twelve family/destination concentration rows and selected-set overlaps are
retained in raw reports, including explicitly noncanonical historical controls.

## Review-cost findings

The user confirmed that independent human reviewers are unavailable. Human
source inspection, semantic adjudication, rights clearance, second review,
reviewer agreement and total human effort remain null/unmeasured. Codex's own
inspection and Python timing are not substitutes for independent human review.

The fixed sample has four records per family and natural Qwen lengths 88-332.
It includes two BSD-3-Clause-labeled and two AGPL-3.0-labeled code records,
paired questions from the same SQuAD article in each native split, UltraChat
with missing message evidence, and native DocRED train/dev/test cases. These
are purposive review strata, not a population-representative random sample or
a selected training manifest. Collection/row labels do not establish rights.

Automation binds all 846 retained inputs to raw source fields, replays preserved
overlap evidence, expands conservative local dependencies, inventories rights
evidence and runs literal benchmark checks on the sixteen inputs. Unselected
and length-ineligible members stay in the graph. Missing fields, outside links,
semantic overlaps and rights decisions stay blocking. No registry exemption is
activated, and no template or benchmark rule is relaxed.

### Measured automated effort

| Stage | CPU seconds | Wall seconds |
| --- | ---: | ---: |
| Source identity and available-field inventory | 0.250 | 0.259 |
| Existing raw overlap identity replay | 0.088 | 0.094 |
| Local dependency graph | 0.067 | 0.067 |
| Rights evidence hash inventory, not legal review | 0.002 | 0.003 |
| Literal benchmark input screen | 7.015 | 7.019 |
| Local closure and blocked review packets | 0.002 | 0.002 |
| Complete automated run | 7.425 | 7.446 |

The graph contains 597 components and 261 typed edge records. The sixteen
packets require 106 record visits if reviewed independently, or 59 distinct
local records after deduplication, including 43 unselected dependencies. Twelve
packets are local singletons; the paired SQuAD packets belong to components of
12 and 35 records. External dependencies remain unknown even for singletons.
These purposive counts cannot estimate population-wide review expansion.

No literal prompt matches were found against 265,771 frozen benchmark texts.
This does not clear semantic or audit-only-field overlaps. All sixteen packets
remain blocked: external closure, rights chains, semantic benchmark review,
complete audit-field coverage and independent review are unresolved. The global
registry's second review is also pending; no exemption is active. UltraChat
message consistency is unverified; SQuAD source answers are absent from the
retained projection; one DocRED test record lacks labels. Historical article
identity is unresolved for all eight SQuAD/DocRED packets. Across the retained
code population, 239 old/new code fields require unsupported-template review.
None of these fields is silently treated as clean or used as teacher labels.

This run reuses the preserved overlap index. Its construction previously cost
4.584 CPU seconds, or 10.030 seconds including exhaustive equivalence checks;
that work is not included again in the 7.425-second packet preparation timing.
Process RSS for this run is reported as 1,745,788 KiB, a conservative inherited
process high-water observation, not a packet-specific working-set estimate.

### Human resource projection

Human review time, independent second-review time, adjudication and agreement
are **unmeasured**, with zero available reviewers. A defensible 1,600-example
budget needs, at minimum:

```text
human_seconds = 1600 * primary_content_seconds_per_example
              + distinct_dependencies * dependency_seconds_per_record
              + rights_chains * rights_seconds_per_chain
              + disputed_cases * adjudication_seconds_per_case
              + second_review_units * second_review_seconds_per_unit
              + shared_provenance_and_benchmark_review_seconds
```

Use deduplicated review units and do not charge shared work twice. All rates
and population dependency/rights counts are unknown. Eight hours permits only
18 seconds per selected example before any of those additional tasks. The
automation measurement neither validates that allowance nor measures a larger
human-hour requirement. An actual cost estimate requires an independently
staffed review pilot; no such pilot occurred here.

## Decision and remaining admission requirements

**No-go for full census acquisition.** The count amendment is useful at moderate
synthetic scale and preserves independently checked constraints, but does not
establish population-scale optimization or review readiness.

1. The amended objective completes at 11,776 components, where V1 did not.
   Larger fixtures remain incomplete or memory-limited, not quota-infeasible.
   Three fully optimal cases reproduce the entire witness in fresh processes.
   Accepting rank-tied pinned-solver reproducibility instead of V1 count-vector
   uniqueness is an explicit scientific tradeoff, not a hidden relaxation.
2. The prior lossless matching population probe still stops during staging.
   This follow-up does not establish matching completion inside eight CPU hours
   or complete-source independent-group capacity. No indexing redesign ran.
3. Automated packet preparation is inexpensive, but all human, semantic and
   rights gates remain open. No independent reviewer is available. Full census
   acquisition is not economically justified by these measurements alone.
4. Before acquisition authorization, review the rank objective and concentration
   changes, resolve population resource readiness, and fund or otherwise supply
   independent review with measured costs. Preserve unselected dependency
   closure, source-field coverage, exact registry review and benchmark/rights
   gates. No automatic corpus acquisition or further infrastructure work follows.

## Evidence and reproduction

[Allocator reports](2026-10-31-v3-rank-results.json) contain nine serialized
probe summaries, independent witness checks, objectives, concentration, resource
limits and repeat identities. [Review results](2026-10-31-v3-review-cost-results.json)
contain every blocked packet, source hashes, dependency IDs and unavailable human
stage. Raw problem/witness files remain local under
`local-artifacts/2026-10-31-v3-rank-review/`; protected source text is not republished.

All [72 CPU tests](2026-10-31-v3-rank-review-tests.txt) pass, including four new
allocator tests and four review tests. Nine serial solver probes use 210.311
CPU seconds; review automation uses 7.425, for 217.736 seconds excluding tests.
The largest process RSS is 3,792,216 KiB (3.617 GiB), under a 4 GiB address-space
limit that nevertheless prevents the upper allocation from finishing. These
are bounded local/synthetic costs, not a full-census budget projection.

The existing CPU environment pins OR-Tools 9.14.6206. Reproduction commands
below are documentation, not authorization to extend the experiment:

```bash
PY=local-artifacts/2026-10-31-v3-engineering/venv/bin/python
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" -m unittest discover -s scripts/drafter_data -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" scripts/drafter_data/run_rank_evaluation.py medium --output /tmp/rank-medium
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" scripts/drafter_data/run_rank_evaluation.py medium --repeat-of /tmp/rank-medium --output /tmp/rank-medium-repeat
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" scripts/drafter_data/run_review_cost.py --manifest docs/experiments/2026-10-31-v3-review-cost-manifest-v1.json --output /tmp/review-cost
```

The [checksum bundle](2026-10-31-v3-rank-review.sha256) binds implementations,
protocols, reports, raw witnesses and previous checksum bundles. All 726 entries
in the nine prior data/source/policy/preflight/engineering bundles still verify.
No source pin, native split, quota, length rule, isolation requirement, target,
rank-256 architecture, runtime default, numerical contract or GPU budget changed.
Additional GPU hours: zero. Stop for user review.

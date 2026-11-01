# V3 quota-count amendment v2

Version `quota-aware-rank-v2`, preregistered October 31 before evaluation.
This CPU-only amendment changes the final count objective, not source admission.
The original [V1 protocol](2026-10-30-v3-census-protocol-v1.md), implementation
and [failed scale admission](2026-10-31-v3-engineering-results.md) stay immutable.

## Unchanged constraints and objective order

Keep native roles, whole-component assignment, exact 256/48/96 family quotas,
sixteen contributing held-out components per family, and global held-out bucket
minima 32/32/32 for 65-127/128-255/256-447 tokens. Preserve seed 20261029, source
pins, graph/benchmark exclusions, and all isolation and rights requirements.
No target-model behavior enters the allocator or the review sample.

First minimize departures from the original hash assignment. Then preserve
the exact destination-vector lexicographic tie-break in ascending original
hash/component-ID order. Zero departures force the original destinations;
otherwise use the same exact integer chunked optimization as V1.

Replace only the last objective. For zero-based ordered component `i` and
bucket `b`, minimize the integer sum:

```
sum((3*i + b + 1) * selected_rows[i,b])
```

Here each component has its already fixed destination; all other destination
counts are zero. No greedy assignment, epsilon objective, seed search or
quota relaxation is used. Reuse CP-SAT 9.14.6206, one worker and random seed
20261029. Canonicalize model insertion order before solving. Require completed
optimal objectives and identical full witnesses in two fresh runs, the second
given reversed input order. Equal rank-cost optima use the pinned solver's
deterministic traversal, not V1's unique count vector. This reproducibility
contract does not promise identical witnesses across solver/platform upgrades.
Incomplete solves cannot qualify as reproducible admitted assignments.

Rank cost favors earlier hash-ranked contributions, whereas V1's smallest
count vector favors zeros in earlier positions. These rules need not select
the same groups. Report contributing-group counts, maximum group share and
Herfindahl concentration by family/destination, plus selected-component overlap.
Neither rule gives an unbiased population sample. No actual source rows are
selected by this evaluation.

## Frozen CPU evaluation

Use existing V1 synthetic problems `fixed72`, `small`, `medium`, `large`,
`upper`: 72/3,200/11,776/79,803/390,220 components, unchanged capacities,
native roles, quotas and buckets. Bind their existing JSON hashes. Existing
V1 reports/witnesses supply labeled historical controls, not paired speed
claims. Also rerun the original `small` control once for a contemporaneous
check. Run V2 once per fixture; repeat only fully optimal checked results
in a fresh process with reversed input. Do not retry an incomplete solve.

Each invocation retains 90 CPU/120 wall seconds, 128 solver calls and a
4 GiB address-space cap. Whole-process hard cap is 180 CPU seconds, allowing
bounded serialization/cleanup; no core dumps. No memory/model reformulation
or second solver is authorized. Check tiny fixtures against independent
enumeration of the amended objective, including nonzero departures, native
shortages, global buckets, held-out groups, tied rank costs and interruptions.
Validate serialized witnesses separately from CP-SAT and recompute objectives
with integer arithmetic. Separate feasibility, proven minimum departures,
canonical destinations, proven rank optimum and reproducibility.

The prior upper-count memory failure may persist: replacing count tie-breaking
does not shrink the integer model. Keep failures as unknown where no proof of
infeasibility exists. Do not raise limits or weaken objectives after results.

## Atomic delivery and stop

1. Freeze this amendment and the separate review-cost protocol:
   `docs(data): freeze rank and review pilots`.
2. Add opt-in research allocator, independent objective audits and tests:
   `feat(data): evaluate rank-weighted quotas`.
3. Prepare fixed local review packets and automated-effort measurements:
   `feat(data): measure local review preparation`.
4. Preserve measurements, reproducibility, review limitations and economic
   assessment: `docs(data): record V3 follow-up decisions`.

No acquisition, teacher work, GPU calibration, training, GPU switching or
inference changes. Preserve rank-256, frozen Qwen target, causal alignment,
exact-output contracts, 1.538085 ms proposal-cost evidence, qualified runtime
artifacts and all GPU-hour caps. Stop for user review after these two follow-ups.

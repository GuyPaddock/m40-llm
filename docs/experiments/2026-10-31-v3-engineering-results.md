# V3 bounded CPU engineering follow-up

October 31, 2026. CPU measurements and final decision are recorded below.
The [frozen probe plan](2026-10-31-v3-engineering-plan.md) and
[staged review proposal](2026-10-31-v3-staged-review.md) define this attempt.
No complete corpus, model weights, target output, GPU work or training was used.
All prior failed admissions and immutable artifacts remain unchanged.

## Match contract and completeness

Implementation `df9a4dc` uses one two-pass SQLite index. Pass one retains raw
text and original comparison-token IDs and counts document frequency for each
distinct five-word shingle. Pass two orders every set by the same global
frequency, digest and content order. It indexes the first
`n - ceil(n/2) + 1` shingles, all contiguous 32-token spans, and normalized
exact-text keys. Full-content verification decides every candidate's original
match reasons. It applies no registry exemption or normalized-offset masking.

For Jaccard at least one half, intersection size is at least `ceil(n/2)` for
each set. If their prefixes were disjoint, the first common element would lie
beyond at least one prefix, leaving too few elements for that intersection.
Thus every qualifying pair shares a prefix key under the global order. The
size test `n/2 <= other_n <= 2*n` is also necessary. Digest collisions can
produce extra candidates, but not remove true matches; final verification uses
the original content. Exact-text and token-span candidates bypass the size
filter. No frequent key or required pair is discarded.

Preserving all true raw pairs necessarily costs quadratic output for a cluster
where every record shares a qualifying passage. Frequency ordering fixes false
common-header candidates, not that lower bound. A CPU/disk deadline leaves an
explicit unknown status and retained partial evidence, never a complete empty
match report. No second indexing architecture was attempted.

## Allocation contract and proof scope

Implementation `324d09a` evaluates one solver, OR-Tools CP-SAT 9.14.6206, with
one worker and seed 20261029. Integer y/x/z constraints retain whole-component
roles, exact family quotas, positive-contribution held-out group counts and
global context buckets. The original checker validates every feasible witness.
Serialized evidence is bound to its synthetic input hash and rechecked without
importing CP-SAT by implementation `80e9a16`.

The first objective minimizes hash departures. A proven zero fixes all original
hash roles exactly. Otherwise destination tie-breaking uses exact mixed-radix
chunks in seed order. Bucket-count tie-breaking follows after destinations are
fixed. Each chunk has integer range below 2^50 and is fixed at its proven optimum
before the next solve. There are no epsilon penalties, changed seeds or greedy
repairs. Dropping implicit zero counts for unchosen roles preserves the original
full-vector lexicographic objective.

Feasible, minimum-departure and canonical are separate result fields. A timeout
can preserve a checked feasible witness without completing either objective.
Infeasibility requires an independently recomputable capacity bound or completed
small exhaustive proof; unsupported large solver proof claims remain unknown.
Small fixtures compare canonical results against the preserved exhaustive
allocator. The fixed-role 72-component fixture additionally has a unique
all-capacity solution, independently provable without solver traversal.

CP-SAT status meanings follow its [official documentation](https://developers.google.com/optimization/cp/cp_solver).
The pinned [v9.14 Python API](https://github.com/google/or-tools/blob/v9.14/ortools/sat/python/cp_model.py)
supports the integer objectives and solver statuses used here. No upstream
implementation code was copied. Installed dependency versions are recorded in
`scripts/drafter_data/solver-requirements.txt`; this CPU-only environment is
separate from the runtime toolchain.

## Measurement boundaries

Each probe uses a fresh process and output directory, serial execution and one
solver/BLAS worker on the Intel Xeon E5-2697A v4 host. No CPU affinity was pinned.
The runner sets 4 GiB address space and 720 CPU-second hard
limits after module imports. Matching allows 600 CPU/900 wall seconds including
frequency preparation; allocation allows 90 CPU/120 wall seconds including model
construction, with at most 128 exact optimizer calls. Limits were not increased
after observing results. Reports include total measured CPU/wall time and
process high-water RSS; those clocks exclude interpreter/module startup.

RSS from `getrusage` is a conservative process-lifetime high-water value, not
the index or solver's incremental memory. The launcher can contribute to it.
The 10,000-header process was also observed at 124,744 KiB current RSS and
419,528 KiB virtual size after 202 CPU seconds, below that reported high-water.
This observation is not a separate full-run peak measurement.

Retained probes reuse original Qwen token IDs. Synthetic lengths are word/ID
counts, not measured Qwen eligibility. Synthetic native-role and bucket shapes
exercise constraints but do not establish source-group capacity or population
distributions. Population stress uses 64 words, below much of the real audit
content; success there would not bound arbitrary long source/answer fields.
All source acquisition, tokenization, provenance, AST, benchmark and semantic
work would still have to fit the eight-hour census budget separately.

## Results and decision

### Frequency index

| Probe | Records | Candidate checks / true pairs | CPU s | Wall s | DB MiB | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Retained raw inputs | 846 | 22 / 22 | 10.030 | 10.048 | 17.301 | Complete; all 357,435 exhaustive pairs agree |
| Twenty-word header, 256-word records | 2,000 | 0 / 0 | 23.849 | 23.876 | 61.750 | Complete |
| Same header, larger population | 10,000 | 0 / 0 | 248.096 | 248.375 | 311.027 | Complete |
| Shared 64-token passage | 200 | 19,900 / 19,900 | 2.385 | 2.399 | 2.371 | Complete; no frequent true matches discarded |
| Population-size attempt, 64-word records | 390,220 intended | Not reached | 602.961 | 611.339 | 742.160 | Unknown; staged 240,451 records, matching not started |

The population attempt stored 10,579,860 frequency keys before its 600-second
index deadline. Cleanup, counts and file hashing account for the small excess
in total measured CPU. Its zero candidates/pairs are **not** a negative match
finding. The frequency pass itself was incomplete. All reports retain status,
partial counts and database hashes; there was no larger retry.

Common-header selectivity is fixed in the tested cases, and the raw-match gates
pass. End-to-end population readiness remains unestablished. This is not proof
that the index cannot finish in eight hours: the bounded probe stopped much
earlier. Nor is linear extrapolation a census measurement. Longer audit fields,
true-match clusters and the benchmark corpus remain unmeasured at full scale.

### Allocation

| Synthetic components | Checked feasible | Minimum departures | Full canonical | Calls | Allocator CPU s |
| ---: | --- | --- | --- | ---: | ---: |
| 72, fixed native roles | Yes | 0, proven | Yes; independent uniqueness proof | 9 | 0.040 |
| 3,200, mixed native roles | Yes | 0, proven | Yes; CP-SAT objective completion | 80 | 13.426 |
| 11,776, mixed native roles | Yes | 0, proven | No, deadline | 91 | 90.102 |
| 79,803, mixed native roles | Yes | 0, proven | No, deadline | 2 | 90.458 |
| 390,220, one-row upper-count case | No witness, unknown | Unknown | No, memory limit | 1 attempted | 70.405 |

The 11,776/79,803-component results are useful feasibility evidence for their
synthetic problems. They are not canonical results, source quota failures, or
authorization to select examples. The serialized checker validates the witness
but does not independently replay large optimality proofs.

The 390,220-component case exhausted address space during the first optimizer
invocation, before a solver status or witness was returned. Its status remains
unknown, not infeasible. Peak reported RSS was 3,807,720 KiB, or 3.631 GiB, under
the 4 GiB virtual-address ceiling. No memory-limit increase or alternate solver
followed. The proposed count tie-break alternative below would not by itself
resolve this model-memory problem.

### Verification and resources

All **64 CPU tests pass**, including the preserved registry, native-split,
exhaustive allocation and overlap fixtures. New tests cover common headers,
copied passages, 32-token boundaries, Jaccard exactly 0.5, raw retention,
incomplete runs, canonical agreement under reversed input order, infeasibility
versus timeout, and serialized witness corruption. The
[test log](2026-10-31-v3-engineering-tests.txt) includes 1.06 CPU seconds for the
entire test process. No CUDA tests were run or needed for these Python changes.

The [ten raw reports](2026-10-31-v3-engineering-results.json) total
1,169.041 measured CPU seconds and 1,177.784 wall seconds, about 19.5 minutes.
Peak process RSS was 3.631 GiB. These are probe costs, not a completed census
estimate. The solver table isolates allocation time; total reports also include
synthetic construction, serialization and independent witness verification.
The retained directory uses about 890 MiB of allocated filesystem space after
the probes, including the separate environment; this is not a workspace
high-water measurement. Original-source acquisition and GPU use were both zero.
All eight original data/source/policy/preflight SHA256 bundles verify unchanged,
covering 680 checksum entries. The new engineering bundle binds new code,
tests, reports, local synthetic evidence and those original bundles.

### Canonical objective limitation and review-only alternative

This implementation repeatedly solves the full model to fix exact count-vector
prefixes. With 390,220 one-row components and proven zero departures, even the
best case needs 7,805 count chunks plus the primary solve. That exceeds the
frozen 128-call probe cap independently of runtime. The cap never changes the
objective; it prevents a completion claim. Smaller actual document-component
counts may help, but have not been measured. Do not interpret these probes as
proof that every possible exact implementation exceeds eight CPU-hours.

If this tie-break remains unaffordable, a narrowly defined **proposal for
review**, not an implementation, is to retain the exact minimum-departure and
destination objectives, but replace only flattened-count lexicographic
minimization with one integer rank-weighted count objective. Use the existing
seed/component order and bucket order as ranks. Freeze the pinned single-worker
solver traversal to resolve equal optima, require proven objective completion,
independent constraint verification and reproducible witnesses on repeated
identical inputs. A disagreement or timeout remains unknown, not a new seed.

This would change the scientific sampling rule: minimum rank sum is not the
original lexicographic vector, and ties would no longer have a solver-independent
unique mathematical answer. Group concentration and length support could change.
It needs an explicit new policy version and user approval before coding or row
selection. It is not automatically preferred, does not solve unresolved rights
or semantic review, and may still leave expensive nonzero-departure destination
tie-breaking. The current implementation and all measurements keep the original
objective unchanged. No alternative optimizer or greedy allocator was run.

## Go/no-go for the finite census

**No-go under the current implementation and review allowance. Stop here for
review.** This is not a source-capacity failure or a rejection of the drafter.

1. **Does matching scale without losing required matches?** The tested required
   matches are preserved, and the 10,000-record repeated-header probe avoids
   the previous false-pair explosion. Full-population completion is still
   unknown. The population probe did not finish frequency preparation, much
   less matching, and no complete-census cost claim follows.
2. **Does allocation produce checked feasible and canonical assignments?** Yes
   on the small fixtures and 3,200-component case. Larger probes demonstrate
   checked feasibility without canonical completion. They do not establish a
   population-scale canonical result within the shared census budget.
3. **Is the review process realistic within eight hours?** The staged procedure
   preserves leakage rules, but the allowance is not credible for admission.
   At an assumed 2-10 minutes per selected example, those examples alone cost
   53.3-266.7 hours, before dependencies, rights and independent review. Actual
   staffing needs remain unmeasured; do not treat that scenario range as an
   observed human rate or spend it without approval.
4. **What remains before acquisition authorization?** A credible bounded full
   indexing/allocation completion case, a reviewed decision on canonical cost,
   an approved review-resource plan, independent registry review and normalized
   span integration, and a complete coverage plan for audit-only fields,
   document/code/fact dependencies, licensing and benchmark exclusions. Preserve
   the frozen file inventory and all limits. Ask separately for finite acquisition;
   it would not itself admit a manifest or authorize GPU work.

No source was cleared by these synthetic witnesses. Source pins, raw overlap
dispositions and all previous negative decisions stand. In particular, unknown
document identities, unavailable UltraChat messages, SQuAD answer columns,
DocRED test answer coverage and benchmark-inventory gaps cannot be fixed by
calling the new index or optimizer successful. The manual review proposal
describes how these blockers remain attached to prospective components and
their unselected dependencies.

## Reproduction and retained evidence

The frozen runner is `ee552df`; the staged review-process proposal is
`e32009c`. Run one probe at a time from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  local-artifacts/2026-10-31-v3-engineering/venv/bin/python \
  scripts/drafter_data/run_v3_engineering.py matching header10k \
  --output /tmp/v3-header10k-fresh
```

For allocation, replace `matching header10k` with `solver fixed72`, `solver
small`, `solver medium`, `solver large` or `solver upper`. Output directories
must not exist. Synthetic problem/witness JSON and SQLite raw-match evidence
remain under the ignored `local-artifacts/2026-10-31-v3-engineering/` root.
They are research evidence, not a selected training manifest. Compact reports
and file digests are committed separately from implementation.

The frozen four sources, native splits, 65-447-token rule, 256/48/96 family
quotas, sixteen held-out groups, global bucket minima and benchmark inventory
remain unchanged. Exact registry bytes and their adapter identities are not
edited. Original causal feature/label alignment, rank-256 architecture,
1.538085 ms proposal-cost evidence, numerical gates and runtime defaults are
unchanged. GPU usage remains 0.005923 hours against the 72-hour parent ceiling;
phase and non-transferable pilot limits remain unchanged. This follow-up used
no GPU time and authorizes no acquisition or further infrastructure search.

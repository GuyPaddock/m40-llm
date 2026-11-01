# Runtime research follow-up disposition

The nine original research branches remain independent records. The feature
branch carries the Q8 cache synchronization correction and explicit selected-Q4
execution components. No experimental algorithm has been added to production
dispatch, and no runtime default changes.

The cheaper-selector study closes negative for its fixed signals. Graph
attention-dispatch parity is repaired and verified on a separate branch.
The real-model KV study preserves all seven dense-passing cases in its bounded
grid. Adaptive drafting fails its broad-workload performance gate; its full
independent audit passes, including seven corrupted-report controls.

## Carry-forward and API review

Integration starts from clean feature commit `b9333e7`. Its runtime/dependency
paths are identical to the previously validated carry base `a7479ba`:

```bash
git diff a7479ba b9333e7 -- src cuda build.rs Cargo.toml Cargo.lock tests
```

The supported FP16-K/Q4-V direct consumer already exists on that feature base.
It consumes compressed values inside attention; no duplicate implementation is
needed. Supported block-select-exact, direct mixed backing, plain top-k/top8
and explicit dense-reference configuration remain unchanged.

| Component | Feature commit | Scope and qualification |
| --- | --- | --- |
| Host Q4 layout | `3df1dd2` | Exact `1e3a6d3` implementation; three host behavioral tests |
| Q8 scale synchronization | `e930d21` | Exact `6aa52d8` correction; 26 cuBLAS-on checks and the cuBLAS-off regression |
| Selected-Q4 device execution | `9a68003` | Exact `22ee0d3` component; 20 cuBLAS-on checks, two selected-FFN cuBLAS-off checks, 223 non-CUDA tests and CUDA/server Clippy |

The Q8 fix separates shared K-scale reads from later V-scale scratch writes.
Its original unfixed negative control passed once; the race was intermittent,
so this is not evidence of a reliably reproduced failure. The dormant bulk
builder is compile-covered, not separately hardware-qualified.

The Q4 host layout preserves original Q4_0 scales and codes. The CUDA owner
supports fixed 1/2/4/8-row selected FFNs with FP32 accumulation, owns immutable
indices and reusable scratch, and requires live disjoint device spans,
context identity, producer ordering and serialized workspace use. Numerical
checks cover masked references, native/repacked addressing, irregular/full
selection, Qwen-sized synthetic dimensions, guards, immutable readback,
cross-stream ordering, invalid spans and tracked allocation cleanup.

It is an explicit research API. Both down layouts remain resident for
comparison, and selection is immutable. Production sparse execution would
need shared immutable weights, per-session selection/scratch, charged index
updates and one chosen down layout. The negative selector study does not
justify that integration. No model conversion, screening kernel or dynamic
sparse FFN dispatch was imported.

All runtime/dependency/test paths at feature `9a68003` match `22ee0d3` exactly:

```bash
git diff 22ee0d3 9a68003 -- src cuda build.rs Cargo.toml Cargo.lock tests
```

The [integration plan](2026-10-31-runtime-carry-plan.md) records dependencies,
atomic commits and fresh checks. The [feature source/log manifest](2026-10-31-feature-q4-artifacts.sha256)
verifies current source and retained validation logs. Final GPU snapshots are
empty. These are correctness checks; importing an operator does not establish
new throughput or sparse-model quality.

## Four independent investigations

Each report lives on its named branch. Inspect it without switching worktrees
using `git show BRANCH:docs/experiments/FILE`.

| Investigation | Branch and result commit | Report |
| --- | --- | --- |
| Cheap selector | `research/1031-selector-followup`, `928a0b3` | `2026-10-31-selector-result.md` |
| Graph dispatch parity | `research/1031-graph-dispatch`, `314581e` | `2026-10-31-graph-dispatch-result.md` |
| Real-model KV quality | `research/1031-kv-quality`, `9da2390` | `2026-10-31-kv-quality-result.md` |
| Adaptive drafting | `research/1031-adaptive-drafting`, `115f765` | `2026-10-31-adaptive-drafting-result.md` |

**Cheap selector.** Six complete Qwen2.5-3B Q8_0 calibration generations provide
128 emitted tokens, 122 consumed-token positions and five observed layers.
Observer-off/on generations match. Previous-token labels, refresh-history
frequency and fixed 16-dimensional current/previous-layer sketches are tested
at 10%/20% retention with refresh intervals eight/sixteen. None of 5,382 causal
policy-row comparisons passes the frozen recall-plus-weighted-mass gate.
Independent replay checks all 8,296 records and 476 summaries; weighted-mass
arithmetic has behavioral tests rather than independent recomputation.
No heavyweight predictor, training, masked-generation accuracy claim or selector
CUDA implementation follows. This bounds the tested signals on six prompts,
not all possible forms of dynamic sparsity.

**Graph parity.** The baseline mismatch is reproduced. Shared ordinary and
device-length dispatch fixes specialized head64/head128 and generic paths,
including the length-8192/8193 boundary. All four original full-forward cases
match logits and FP16 cache exactly over 128 token comparisons. The release
attention grid passes 228 exact comparisons; affected checks pass with cuBLAS
on/off. One optional real-model diagnostic skipped and is excluded from the
54 executed regression checks. Parallel graph capture is explicitly rejected;
ordinary parallel attention remains available. The fix is retained separately,
with no graph performance claim or default promotion.

**KV quality.** The frozen three-arm Qwen grid completes 36 generations and
1,149 emitted tokens over six tasks at short/long contexts. Top64 and top8
mixed direct attention preserve the seven dense-passing cases. Five dense
failures remain inconclusive. Top8 omits a concept in the dense-failed long
garden case, and long-prefix logit differences increase. The independent audit
reconstructs scores, repetition, IDs, counts/stops, dispatch/cache coverage and
top10 overlap, rejects twelve corruptions, and passes eight known function
cases. Full logits are not retained for an independent KL recomputation.
Mixed allocation is 116,499,600 bytes versus 113,246,208 dense at capacity 3,072;
this configuration provides no cache memory saving. No throughput or broad
quality equivalence is claimed.

**Adaptive drafting.** A fixed controller uses historical M40 target costs and
only acceptance outcomes from executed attempts in the current request.
All 21 existing traces, two policies, three cost scales and five CPU trials
produce 126 cases and 630 trials. It adapts between serial and measured
four-row verification; two/three-row target costs remain unqualified.
At the measured scale, browser projected complete-request gain worsens from
-0.124% to -0.270%, while long-code loss decreases from -1.176% to -0.765%.
Only the two known copy cases exceed 3%; no non-copy case does. The broad gate
fails, so no live implementation or retuning follows. Independent source,
policy, cost and CPU-accounting audit passes all 126 cases, 630 trials and
8,660 causal rounds, and rejects seven corruptions. These projections are retrospective, not new GPU/HTTP measurements.

## Original branch preservation

All original heads remain unchanged and their worktrees are clean:

| Topic branch under `research/1030-` | Preserved head | Disposition |
| --- | --- | --- |
| `activation-preparation` | `f8e7a18` | Experimental rotated preparation; no production dispatch |
| `channel-selection` | `cf42931` | Original screening result retained; Q4 components carried separately |
| `dualpace` | `69b8a3e` | Experimental CPU/GPU scheduling; no production dispatch |
| `ffn-fusion` | `b44d71e` | Experimental multi-row fusion; no production dispatch |
| `legacy-graphs` | `2470940` | Original graph result retained; parity fix on follow-up branch |
| `lossless-kv` | `73a9417` | Experimental lossless cache; no production dispatch |
| `quantized-kv` | `e23d162` | Original operator result retained; direct consumption already present |
| `shiftadd` | `753f3ca` | Experimental quantization; no production dispatch |
| `soft-mtp` | `ee7a226` | Original feasibility record; no training or learned-drafter work |

The final source diff is limited to the Q8 barriers, Q4 modules/build/FFI and
associated tests. It contains no production dispatch change from the preserved
five experimental implementations. The separate V3 acquisition/training work
and its stop instructions remain untouched. Follow-up branches and feature
commits remain local; this objective does not include another push.

## Completion audit

The final review checks each requested disposition against source, branch heads,
raw results and independent audits. The
[requirement record](2026-10-31-runtime-completion-audit.tsv) maps every objective
item to its evidence. All nine original branch heads are preserved; no whole
research branch is merged. The feature runtime source equals the qualified
carry-forward source, with defaults unchanged. Each of the four investigations
has a committed result and an explicit scope or stop decision.

The selector, graph, KV-quality, adaptive and current feature manifests verify.
All test and measurement processes have terminated. No remaining experiment,
audit or implementation is required by this objective. Follow-up ideas outside
the fixed studies remain separate work, not implicit continuation orders.

The [final branch record](2026-10-31-runtime-final-branches.txt) pins every
original and follow-up head. The [final evidence manifest](2026-10-31-runtime-final-evidence.sha256)
covers this disposition, requirement record and branch-local final reports and
audits. Absolute worktree paths in that manifest identify the local copies;
the branch/commit table above identifies their durable Git objects.

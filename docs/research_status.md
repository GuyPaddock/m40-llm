# Research Status

This is the task checkpoint, not agent instructions. Update it after meaningful
progress; keep working rules in [AGENTS.md](../AGENTS.md). Verify this snapshot
against the current code, worktree, Git history, and linked raw results.

## Active Plan

The authorized research sequence is the workspace's
[M40-LLM research plan](../../M40_LLM_RESEARCH_PLAN.md), outside this Git
repository. It orders controlled measurement, bounded exact-kernel/layout
experiments, the offline n-gram feasibility gate, conditional speculation, and
separately evaluated model-changing research. It is not replaced by the
[legacy task roadmap](archive/legacy-task-roadmap.md).

## Current Phase: Separate Model Research

C18's [unchanged-model audit](experiments/2026-10-25-unchanged-model-checkpoint.md)
is complete. C01-C11, conditional C10A, and C11A are verified checkpoints.
C12-C17 are **postponed**, not implemented, under the negative n-gram gate.
The full research plan remains active; C20-C27 remain subject to their own
quality/performance gates. Neither exact-engine speed objective is achieved.

**C19 is complete.** Disjoint inputs/rubrics were frozen before the dense run;
teacher-forced loss and the explicit-model release runner now have a recorded
Qwen2.5 Q8_0 reference over both splits. See the [C19 evidence and limitations](experiments/2026-10-25-model-quality-dense-reference.md).
Modified-model speed must not be presented as an unchanged-engine win.

Next is **C20**, diagnostic interior-layer masks with original layer/KV IDs,
empty-mask parity, invalid-mask rejection, and explicit memory/quality/timing
records. Then C21 tests exactly 50% static sparsity with matched-zero dense
projections before any calibration-driven pruning. Do not tune on held-out or
browser inputs, and do not reopen the failed n-gram/layout screens.

C20 is split into dependency-ordered commits:

1. `feat(infer): define diagnostic layer masks`: checked interior IDs, original
   chronological execution order, empty-mask behavior and CPU tests.
2. `feat(infer): apply masks to full-model forward`: model-local setup before
   KV allocation, all full-model forward paths, active-index scratch routing,
   and CUDA parity/skip tests. Depends on the mask contract.
3. `test(quality): evaluate masked layer variants`: candidate-only CLI masks,
   transform/memory identity, and matched forced-prefix step timing. Depends
   on forward wiring and C19's frozen evaluation runner.
4. `docs(perf): record layer-mask tradeoffs`: fixed one/two/three interior
   layer diagnostics against the unchanged reference, with quality, loss,
   logit drift and scoped timing/memory. Depends on verified implementation.
   No ranking or compact export in C20; those remain C24/C25.

The mask contract is committed in `9674452`. Forward integration covers
sequential, batched, packed-prefill and device-parameter paths with original
KV IDs; session graphs and all-slot packed-then-compress reconstruction are
explicitly rejected for nonempty masks. See [the mask contract](layer_masks.md).
The broader validation exposed an independent minimal-forward fixture using
dense head_dim=4 KV with the compressed default; `fe48028` makes that reference
explicitly dense. Runner integration and measured mask results remain pending.

C19 was split into dependency-ordered commits:

1. `d1294fc`, `test(quality): freeze pruning evaluation splits`: immutable authored V1
   data, split/leakage validation, deterministic task rubrics and tests.
2. `c32ec4e`, `feat(eval): score teacher-forced tokens`: stable token-weighted NLL/PPL,
   full-logit drift, strict-prefix prediction, and CUDA session integration
   tests. Depends on the frozen scoring/data contract.
3. `f974162`, `test(quality): evaluate explicit model variants`: release CLI with model/
   corpus/config hashes, serial dense-reference comparisons, output/task/loss/
   memory/timing records. Depends on the loss evaluator and existing sessions.
4. `docs(perf): record dense quality reference`: evaluate both frozen splits,
   retain dense failures as capability evidence, and publish the exact current
   C19 completion status before C20/C21. No model tuning in this unit.

Corpus/scoring is frozen in `d1294fc`, before dense results. Its SHA-256 is
`2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38`.
Teacher-forced CPU loss and CUDA session comparison are committed in `c32ec4e`,
including strict prefixes, token weighting, full-vocabulary drift, cache reset,
and shared-KV rejection. See [the evaluation contract](model_quality.md).
The explicit-model release runner now adds model/corpus/config identity,
reference-dependent scoring, full-logit loss comparisons, actual token IDs,
timing boundaries, and scoped memory accounting. It preserves both frozen
splits and rejects tokenizer mismatch, truncation, or implicit compressed KV.
The release self-reference run matches all 12 generated traces and all 203
teacher-forced tokens exactly, including zero full-logit drift. Fixed-rubric
passes are 3/6 calibration and 4/6 held-out; five dense-failed tasks remain
inconclusive for pruning quality. Token-weighted PPL is 31.556848 calibration
and 39.612159 held-out. These small synthetic passages are not a standard
benchmark claim. Single in-process timing observations are not HTTP speed
promotion evidence. Both models are released, with zero tracked/device use.

## N-Gram Feasibility Checkpoint: 2026-10-25

C11A is complete. The selected actual C11 browser trace gives at most 4.949%
optimistic estimated HTTP throughput improvement, below the fixed 10% gate.
Limits8/16 both replay in 485 visited rounds, with 28 accepted draft tokens;
33/43 eligible proposals reject immediately. No GPU verifier or measured
speedup is claimed. See the [study and raw artifacts](experiments/2026-10-25-ngram-feasibility.md).

Actual fixed regression captures are stable across two runs; the canonical
smokes return `4`, `OK`, `37`. CPU-only release reproduces acceptance/replay/cost
results from the CUDA-feature CPU path. Lookup/study tests, non-CUDA
warning-as-error tests, both Clippy lanes, and formatting pass. GPU allocation
returns to zero after capture. Runtime/KV defaults and the measured engine
configuration are unchanged.

C18 records this negative gate and the two unmet speed objectives in the
phase audit above. Do not build a verifier or reopen layout/lookup tuning to
force the screen to pass; C19 is next.

C11A was split into dependency-ordered commits:

1. `fe392cd`, `feat(decode): add bounded ngram lookup`: history-only borrowed drafts,
   longest/recency matching, stop/budget bounds, and CPU behavioral tests.
2. `739c195`, `test(bench): study ngram acceptance offline`: position/round replay,
   actual-trace validation and regression capture, hypothetical cost sensitivity,
   and the formal screening gate. Depends on the lookup and C11 artifacts.
3. `docs(perf): record ngram feasibility decision`: release CPU analysis and
   raw results, with proceed/postpone status before conditional verifier work.
   Depends on the tested study tool and selected actual traces.

## Exact Kernel Checkpoint: 2026-10-25

C01-C11 are complete, including conditional C10A evaluation and the preserved
failure removals. The selected opt-in benchmark remains native Q8 plus C07
parallel head128 attention, partitions=8/threads=256, with preparation off.
Global runtime/KV defaults are unchanged.

Fresh five-pair release HTTP confirmation at `f5cda0d` measures 31.850 tok/s
m40-llm versus 43.032 Ollama. Every row has 65 prompt IDs, 512 generated tokens,
length stopping, and verified fresh KV. The actual m40 trace matches the prior
qualified trace exactly. Both objectives remain unmet: 53.06 tok/s and 1.30x
fresh Ollama, now 55.9416 tok/s. See the [C11 report and raw records](experiments/2026-10-25-exact-kernel-checkpoint.md).

The slower prepared fused MLP is removed in `9044eea`, after preservation in
`e2dbc61` and measurements in `2e2bb5c`. Native fusion and shape/pointer checks
remain. The slower tiled coalesced-lane mapping is removed in `f5cda0d`, after
preservation in `bb4d30e`; vector-only research remains. Remaining preparation
paths stay opt-in, with no qualified HTTP win. CUDA/cuBLAS-on/off, non-CUDA
warning-as-error, Clippy, and formatting checks pass. Benchmark processes are
stopped; final GPU allocation is zero.

This selected trace is the source for the completed C11A study above. Its
negative >=10% gate postpones C12-C17; C18/C19 follow without another layout
tuning round.

## Prepared Q8 HTTP History: 2026-10-25

C01-C10 and C10A implementation/HTTP validation are complete. No prepared
candidate passes the combined timing and actual-ID parity gate. Keep the
qualified C07 native-Q8/split8/256 attention benchmark configuration and
preparation off. Runtime/KV defaults are unchanged; both overall speed goals
remain unmet.

Five serial release pairs per subset, each with 65 prompt IDs and 512 generated
IDs, give:

| Prepared families | Native / prepared HTTP tok/s | Actual IDs equal | Decision |
| --- | --- | --- | --- |
| All | 31.946 / 11.433 | Yes | Timing regression |
| QKV/output/down/vocabulary; native MLP | 31.907 / 29.765 | Yes | Timing regression |
| Down/vocabulary only | 31.847 / 35.837 | No | Timing lead, parity rejection |

The last subset first diverges at zero-based generated step 310. It is not a
new qualified best result. Canonical arithmetic/OK/config smokes pass native
and both reduced subsets; the first run's user-only OK/config reference
failures remain inconclusive. The three runs never overlap; the isolated
oracle has been stopped and GPU allocation returned to zero.

See the [full release report and raw artifacts](experiments/2026-10-25-q8-prepared-http.md)
for confidence intervals, timing boundaries, memory/setup costs, and limitations.
Implementation/test contracts and earlier correctness canaries remain in
[Q8 execution weights](q8_execution_weights.md).

C10A commits preserve cache/loading (`11f2fb6`), fused QKV (`4db3c1e`), fused
MLP (`e2dbc61`), greedy vocabulary (`b76177d`), immutable families (`3d176fe`),
and HTTP comparison (`26bc810`). `e74ed87` fixes only quality-smoke formatting,
without changing the measured browser prompt. CUDA family/ownership/forward/
attention tests, cuBLAS on/off checks, non-CUDA warning-as-error tests,
Clippy, and protocol/CLI tests pass.

This checkpoint led to C11's keep/remove decision and exact-kernel confirmation
above. It did not authorize another layout/geometry search or reinterpret
isolated gains as qualified generation speedups.

### C11 Commit Sequence

1. Completed `9044eea`, `refactor(cuda): remove failed prepared MLP`: remove
   the preserved
   `e2dbc61` fused consumer after the negative `2e2bb5c` release gate, retaining
   native fusion, shape/pointer checks, and cache coexistence tests.
2. Completed `f5cda0d`, `refactor(cuda): drop coalesced Q8 experiment`: remove
   the losing lane
   mapping preserved in `bb4d30e`, keeping useful vector-tiled projection
   research. This is independent of MLP removal.
3. Completed in this checkpoint,
   `docs(perf): record exact Qwen kernel results`: rerun the selected native
   configuration through the controlled release HTTP comparison after cleanup
   and reconcile all C07-C10A keep/remove decisions. Depends on both removals.
4. Completed `739c195`, `test(bench): study ngram acceptance offline`: deterministic history-only
   lookup and simulation over the selected configuration's actual traces, with
   cost sensitivity and the formal proceed/postpone gate. Depends on C11.

The prepared MLP consumer is now removed. CUDA native/cache fusion, family,
prepared QKV/greedy, and MLP wrapper tests pass, as do non-CUDA warning-as-error
tests and CUDA/server all-targets Clippy. The
remaining cache and split/vector consumers stay opt-in research, not a promoted
configuration. No new layout tuning or default change is authorized here.
The coalesced-lane mapping preserved in `bb4d30e` is now removed from CUDA,
Rust launch descriptors, and the microbenchmark CLI. Its original measurements
remain unchanged. Retained vector/library/cache tests, the 11-case attention
grid, 24-case forward suite, projection/MLP wrappers, cuBLAS-off checks,
non-CUDA warning-as-error tests, and CUDA/server Clippy pass. The controlled
native/Ollama release confirmation is complete in the C11 report above.

## Committed Checkpoint: 2026-10-24

The following checkpoint was moved from `AGENTS.md` at `81948df`. Paths in
backticks are repository-root relative. Earlier implementation history is in
[archived progress](archive/agent-progress-through-2026-10-24.md); authoritative
measurements and raw records remain in [performance history](perf_baselines.md)
and `docs/experiments/`.

- C01/C02 are complete: MemPalace practice and opt-in buffered HTTP metrics.
- C03 now has a release-only `compare_ollama` example with identity checks,
  verified cache resets, paired trials/bootstrap, and untimed actual token traces.
- The controlled five-pair baseline is 23.735 HTTP E2E tok/s m40-llm versus
  43.035 for prebuilt Ollama 0.40.11 on M40. Both use the same Qwen Q8_0 blob,
  65 shared ChatML IDs, dense FP16 KV, context 1024, and 512 generated IDs.
  All rows are length-limited/cache-valid. See `docs/controlled_benchmark.md`
  and the dated raw records in `docs/perf_baselines.md`.
- The initial 66-versus-65-ID attempt was rejected. `2f4d7ca` fixes extra BOS
  insertion for preformatted Qwen Raw/Auto prompts; standard automatic Qwen
  chat formatting is unchanged.
- C04 now records buffered per-layer CUDA-event spans with session-owned reused
  events and nonblocking resolution at existing logits completion. CUDA tests
  preserve results and launch/sync/copy/allocation counts, including two streams.
  Five same-workload HTTP pairs show 23.780 tok/s off versus 23.667 on, a 0.47%
  profiling reduction with identical generated IDs. Mid-token FFN spans total
  18.17 ms and attention 11.42 ms; late attention rises to 20.39 ms. First-capture
  uncovered wall time is not ordinary CPU-cost evidence. See the dated profile
  record in `docs/perf_baselines.md`.
- C05's optional standalone CUPTI preflight enumerates 154 M40 metrics, but
  actual legacy event-group creation and domain enumeration both fail with
  `CUPTI_ERROR_NOT_SUPPORTED` on driver 580.159.04/CUPTI 12.8.90, even as root.
  Traffic/occupancy measurements remain unavailable, not estimated counters.
  See `docs/maxwell_counters.md`; ordinary builds do not depend on CUPTI.
- C06 now consolidates the raw baseline, sampled stage medians, counter
  limitations, actual tensor inventory, and explicitly estimated Amdahl limits
  in `docs/experiments/2026-10-24-qwen-baseline-checkpoint.md`. FFN parameters
  are 78.90%; gate/up is the larger fixed cost, while attention grows with
  context. The report's jq program reconciles source records and calculations.
- C07 compares eight exact head128 geometries with reusable RAII split scratch
  and stable softmax merge. Split8/256 passes the paired HTTP gate: 31.852
  tok/s versus 23.774, +33.98% with 95% interval [33.868%,34.203%]. Actual
  512-token traces match. Keep it as the selected opt-in benchmark config:
  `M40LLM_ATTN_HEAD128_KERNEL=parallel`, partitions=8, threads=256; full env
  names/commands and short-context limitations are in `docs/diagnostics.md`.
  Runtime/graph/scheduler/compressed-KV defaults are unchanged. Scratch is
  65 KiB per KV cache, not per layer/token. The separate middle/late stage
  diagnostic measures attention at 0.791/0.973 ms and FFN at about 64% of the
  middle GPU interval union. See `docs/experiments/2026-10-24-head128-attention.md`.
  Canonical-system Qwen arithmetic/OK/config smokes pass both kernels; the
  earlier user-only reference EOS is retained, not called an attention bug.
- C08's allocation-error prerequisite now releases the CUDA context mutex
  before querying error properties. An impossible-allocation hardware test
  verifies contextual errors, unchanged accounting, live data, and subsequent
  allocation instead of deadlocking before partial-buffer cleanup.
- C08 now provides benchmark-only Q8 source identities/bounded views, explicit
  column/split/tiled allocation plans, and typed RAII host/device ownership in
  `infer::execution_weights`. Identity includes model SHA, tensor, offset,
  dtype/shape, and execution layout rather than device addresses. CPU failure
  fakes plus a real second-plane CUDA allocation failure verify cleanup; device
  roundtrip and tail-shaped projection tests pass. Production GGUF views,
  materialized weights, loader/cache, dispatch, and defaults are unchanged.
  See `docs/q8_execution_weights.md` for ownership contracts.
- C09 now copies original Q8 scale/value bytes into separate planes and tests
  four bounded Maxwell configurations with a benchmark-only async consumer.
  Real Qwen release projection medians improve gate 0.229->0.139 ms, down
  0.191->0.143 ms, vocabulary 2.584->1.624 ms; this is not an HTTP speedup.
  K/V needed a longer ten-pair check and confirms only about 14% improvement.
  Tail/CPU/GPU parity, invalid-call rejection, CUDA grid/wrappers, non-CUDA
  warning-as-error tests, and Clippy pass. Repacking is lossless; retaining
  original plus execution buffers doubles tensor device payload. Defaults,
  fused generation kernels, and model loader/cache remain unchanged. See
  `docs/experiments/2026-10-24-q8-separate-projections.md` for raw trials,
  memory/setup costs, clock/measurement limitations, and negative configs.
- C10 completes independent input-tiled Q8 packing and twelve CUDA cases:
  widths4/8/16, threads128/256, vector/coalesced-lane mappings. All source
  bytes/scales and tails are preserved; padded output guards and CPU/GPU parity
  pass. Best vector M1 gate/down/vocabulary are 0.141/0.161/1.661 ms; refreshed
  split planes are 0.140/0.142/1.623 ms. Vector rows=64 Q/down are promising
  at 1.244/6.480 ms, but no model/HTTP speedup is claimed. Coalesced lanes
  were slower and did not reduce registers; code/results are preserved before
  any later removal. Width4 adds 5.88% storage, and tiled preparation is slower.
  All 2,460 measured intervals are serial; raw records, compiler output,
  configuration bounds, and clock limits are in
  `docs/experiments/2026-10-24-q8-input-tiled.md`. The C10 tuning budget is closed.
  CUDA parity with cuBLAS on/off, forward/ownership, non-CUDA warning-as-error
  tests, and Clippy pass. Loader/cache, generation dispatch, and defaults do not change.
- Next: conditional C10A useful per-family integration, not another layout
  tuning round. Preserve useful fused operations and explicitly budget duplicate
  source/prepared weights, then require model parity and HTTP E2E gates. Use the selected
  C07 benchmark config for subsequent controls, require parity and repeated
  HTTP E2E gates, and bound experiments before C11/C11A. Missing counters or
  the original speed target do not block the later n-gram feasibility decision
  and separate model-changing research. No further attention tuning round yet.
- C10A preparation tests exposed pre-existing uploaded-weight accounting drift
  from `bd9b68fd`: final context drop freed native GPU weights but never removed
  their allocation records or subtracted tracked bytes. Final-owner cleanup now
  accounts only successful frees, with a clone/readback/repeated-model regression.
  This is an independent correction, not a kernel performance result.
- The unchanged-model target requires both 53.06 E2E tok/s and 1.30x the fresh
  Ollama median, currently 55.945 tok/s. Neither is reached. The short benchmark
  explicitly uses dense `off`; the project's compressed-KV defaults are unchanged.

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

## Work in Progress: 2026-10-24

C01-C10 and the uploaded-weight accounting prerequisite `81948df` are complete.
C10A.1/2 add opt-in model-owned Q8 preparation, standard projections, fused
consumers, and greedy vocabulary integration. HTTP validation is pending. Runtime/KV
defaults and the selected C07 benchmark configuration are unchanged.
The real split-cache Qwen Q8 arithmetic canary returns `4`; cache/stream/failure
tests, existing CUDA regressions with cuBLAS on/off, non-CUDA warning-as-error
tests, and both Clippy lanes pass. See [ownership and validation details](q8_execution_weights.md).
This is correctness evidence, not a generation speedup.

The dependency-ordered C10A units are:

1. `feat(infer): prepare model-owned Q8 layouts`: loader/cache configuration,
   semantic identity, budgets, stream-aware standard dispatch, and ownership/
   numerical tests. Depends on C08-C10 and the allocation prerequisites.
2. `perf(cuda): reuse Q8 layouts in fused decode`: prepared fused QKV and
   gate/up/SwiGLU consumers plus greedy vocabulary integration, preserving
   useful fusion and producer streams. Depends on C10A.1 and parity tests.
3. `test(bench): validate prepared Q8 HTTP gains`: immutable native/prepared
   model controls, serial paired HTTP measurements, actual token traces,
   setup/resident-memory reporting, and the promotion decision. Depends on
   C10A.2 and model parity; no microbenchmark-only promotion.

C10A.1 is committed as `11f2fb6`. Split C10A.2 into focused consumer commits:
`perf(cuda): use prepared Q8 for fused QKV`, then
`perf(cuda): reuse prepared Q8 in fused MLP`, then
`perf(infer): use prepared Q8 greedy logits`. Each depends on the cache and
its own numerical/stream tests; the HTTP gate waits for all three.
Prepared fused QKV now preserves one-launch dispatch, optional biases, and both
producer streams; CPU/native-GPU tail and shape tests pass. The real Qwen Q8
canary returns `4` and records 756 prepared fused-QKV launches.
Prepared fused gate/up/SwiGLU now retains a one-column/128-thread launch with
shared activation reads and aligned split values, not the failed four-column
fusion. Independent-weight, tail, stream, CPU/native tests pass. Greedy
vocabulary integration now uses prepared projection plus device argmax, with
exactly one u32 readback and no device allocation in selection. CPU/native/
full-logits/tie/stream checks pass. There is no new HTTP speed claim.
The real fully prepared QKV/MLP canary returns `4` with 756 launches of each
consumer. Its unpaired debug generation total is higher than the previous
native-MLP canary; preserve that evidence and compare native/prepared MLP
explicitly in the release gate instead of assuming all prepared fusion wins.
Next is C10A.3: immutable model controls, per-family native/prepared choices,
actual-token parity, and the serial paired release HTTP promotion gate.
The real Qwen vocabulary/output-norm probe matches full logits and copies only
one u32 D2H; its subsequent text canary still answers `4`. C10A.3 must provide
per-model native/prepared family controls before timing, particularly for MLP,
without mutating shared dispatch configuration between active requests.

Preserve useful fusion and account for duplicate source/prepared weights.
Do not reopen the completed layout tuning rounds or claim a runtime speedup
from their projection microbenchmarks.

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

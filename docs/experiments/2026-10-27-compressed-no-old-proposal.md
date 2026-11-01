# Compressed KV With No Old Tokens

## Decision Requested

Conditional feasibility only. Review this proposal before implementing a
runtime adapter or changing CUDA wrappers. The
[immutable compound reference](2026-10-27-compound-reference.md) remains
50.910426 HTTP tok/s with explicit dense KV. Project defaults stay unchanged.
This experiment would improve the preferred compressed cache's short-prefix
execution, not qualify the exact-dense 30%-over-Ollama target.

## Evidence And Avoided Work

The [historical stack comparison](2026-10-26-maxwell-packed-kv-stack.md)
measured 49.104 dense versus 19.694 compressed resident tok/s. It used the
earlier split FFN, not compound, and excluded model load/KV allocation.
It is neither an HTTP comparison nor a contemporary compound baseline.
All 512 generated IDs matched in the default-window compressed row.

Its 65 prompt tokens and final 576-token KV prefix fit within the 1024-token
recent window. No old token exists, no old block is scored, and no old q4 V
is dequantized. Do not attribute this row's penalty to old-block scoring or
q4 arithmetic. Reserved old capacity still increases final allocation from
37,748,736 dense bytes to 61,747,488 compressed bytes.

Source inspection identifies work that dense attention avoids:

- The preferred direct kernel launches one 128-thread block per query head
  and builds a generic candidate list, even with no old blocks.
- Each candidate's QK dot uses a shared-memory reduction. At 128 threads,
  its seven reduction strides and surrounding barriers cause nine block
  barriers per candidate, before softmax/V work.
- The qualified dense head128 kernel distributes positions across warps
  and eight context partitions, then merges stable softmax statistics. It
  does not perform a block-wide reduction for every context position.

These facts explain a credible opportunity, not the fraction of the measured
15.57-second resident-time gap attributable to attention. The historical
run collected dispatch counts, not attention timings. Obtain contemporary
compound measurements and separately calibrated attention timings before
making that attribution. No DRAM/occupancy/warp-stall claim follows.

Inspect the direct attention/append/reconstruction functions in
[native CUDA](../../cuda/kernels.cu),
[dense attention ownership](../../src/cuda/dense_attention.rs), and
[inference routing](../../src/infer/kv.rs). Use measured source `63240c5`
when reviewing this proposal's source claims.

## Bounded Adapter

Proposed opt-in `M40LLM_KV_NO_OLD_ATTENTION=off|dense-recent`, default `off`.
Capture it in immutable model/session configuration. Do not add process-global
mutable selection or per-token environment parsing. Reject malformed values.

Only preferred `block-select-exact` / `fp16-k-q4-v` /
`fp16-k-q4-v-direct`, plain top-k, head128 enters the fast path. Existing
diagnostic policies and head64 retain their original execution, explicitly
reported as ineligible. No new head64 or wrapped-ring kernel is proposed.

For validated sequence length N and physical recent capacity W, require
`0 < N <= W`, N within the existing dense kernel's 8192-token bound, matching
physical/configured W, valid GQA dimensions and the original physical
layer/sequence slot. Use the authoritative session length; do not introduce
device-to-host length reads. Never infer eligibility from closed-block count.
At N=W+1 there is already one old token, even before a full block closes.

Before rollover, the physical recent address is
`((slot * W + absolute_position) * kv_heads + kv_head) * head_dim`.
That matches the existing dense kernel's token-major layout when its storage
stride parameter is W, not the cache's maximum sequence capacity. Neither
positions nor RoPE need renumbering. Confirm this for both append and
packed-prefix reconstruction; no-old eligibility alone is not permission
to assume a different builder's layout.

Add a separate validated enqueue adapter using borrowed recent K/V pointers,
their physical stride and the existing fixed eight-partition/256-thread
head128 kernel plus merge. Keep the existing dense-only wrapper's rejection
of compressed caches. Do not manufacture a dense cache, assign dense d_k/d_v,
weaken pointer validation or change the existing kernel bodies.

Retain partial-result scratch under the owning cache/context's RAII and
decode-stream rules. Prepare capacity before measured steady decode. No
per-token allocation, gather, copy, upload or synchronization is allowed.
Report scratch separately from resident KV. Existing producer events and
decode/output dependencies remain intact.

The adapter must only read KV. Leave compressed append, eviction, summary
accumulation, quantization, absolute positions, metadata and reset unchanged.
At the first eviction, use original direct compressed attention immediately.
Continue that original path after rollover; this proposal does not optimize
selected old blocks or the already-wrapped recent ring.

## Correctness Gates

Require exact greedy token equality, unchanged KV state and transitions,
and attention/logit parity under existing tolerances. Record max/mean error,
top-10 overlap, greedy IDs and bitwise attention/logit equality separately.
Do not widen tolerances or permit an output-token divergence.

For synthetic identical Q/K/V inputs, snapshot the cache before/after both
attention calls and require bit-identical payload and metadata. Exercise
N=1 and partial-warp lengths as well as W-1/W/W+1. Include multiple query/KV
head ratios, physical slots, reset and reuse. Require original compressed
dispatch at W+1, including a partial old block.

For full-model paired runs, require bit-identical compressed cache snapshots
between original and candidate at each checkpoint, including recent K/V,
old FP16 K, old q4 V/scales, block counts, accumulators/summaries and lengths.
Compare initialized/valid regions; use deterministic canaries for unused
capacity instead of treating uninitialized bytes as logical cache state.
Compare attention/logits to dense while no old KV exists, and to the original
compressed path at and after rollover. Define N as valid-token count; the
first eviction is append position W, giving N=W+1.

Cover sequential and packed-prefix construction at W-1/W/W+1,
W+block_size-1/W+block_size/W+block_size+1, and 2W-1/2W/2W+1.
Use small windows for repeated-rollover fixtures and W=1024 for a bounded
Qwen2.5-3B boundary continuation. Keep prompt IDs, sampling and prefill
construction identical between each original/candidate pair. Dense is not
the reference for selected-context semantics after rollover.

Parallel and generic reductions differ. Passing attention tolerances does
not prove bit-identical downstream K/V or exact generated IDs. This is an
unresolved feasibility risk. If either cache or output gate fails, preserve
the diagnostic and stop for review rather than changing the contract or
searching for another kernel geometry.

Run affected attention, forward, compressed reconstruction/append and
ownership tests, the CUDA parity grid, CUDA/cuBLAS on/off checks and
non-CUDA warning/Clippy checks. Request hardware escalation and run serially.
Skipped hardware checks are not passes.

## Measurement And Stop Decision

Only after review and all correctness checks, freeze a new release candidate
and compare three contemporaneous arms: qualified compound+dense,
compound+original compressed, and compound+candidate compressed. Keep all
source weights/prepared owners, top8, window1024, context1024, the canonical
65-ID prompt, greedy 512-token output and other runtime flags fixed.

Report five initial alternating original/candidate pairs separately for
resident generation and buffered HTTP. Use the unchanged variation and
fresh-confirmation rules from the [controlled protocol](../controlled_benchmark.md),
including its 3% minimum and positive paired 95% lower bound. Dense is a
contemporary comparator, not a pooled historical control. Keep the fresh
official Ollama reference unchanged; do not launch another Ollama benchmark
to qualify this compressed-only change.

Report prefill/decode/total, actual token IDs/counts/stop reason, eligibility
and dispatch counts, attention scratch, full compressed allocation, setup
peak, copies, allocations and synchronizations. Separate instrumented
attention measurements from primary timing; preserve the existing 3%
profiling-perturbation limit. No build, tests, MemPalace or other inference
may overlap timed GPU work.

Use a short bounded W-crossing continuation to confirm transition dispatch,
unchanged cache/output and no added post-rollover hot-path work. Report its
timing separately, not as the fixed 512-token HTTP result.

Stop on a layout/ownership/stream failure, any token or cache mismatch,
out-of-tolerance attention/logits, hot-path allocation/copy, or failed paired
performance gate. Preserve failed code/evidence before separate removal.
Do not expand the experiment to old-block scoring, selection policies,
kernel geometry search or default promotion.

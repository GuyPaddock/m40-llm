# Runtime research carry-forward

This implements the user's October 31 disposition of the nine research
branches. Preserve them as independent experimental records. Carry only the
verified direct-KV correctness fix and selected-Q4 execution components onto
`feature/phase4-multiple-experiments`; do not merge the nine whole branches.
The active feature started this integration at `b9333e7` with a clean worktree.
This work does not reopen V3 acquisition, training or learned evaluation.

## Dependencies and API decisions

The existing feature already contains FP16-K/Q4-V direct cache consumption,
including ordinary supported runtime dispatch. No duplicate consumer or staged
attention diagnostic is needed. Keep block-select-exact, mixed direct backing,
plain top-k/top8 defaults unchanged.

The correctness correction is the two Q8 K-scale/shared-scratch barriers
retained at `6aa52d8` on `research/1031-selector-followup`. Import its regression
test and original evidence. Its unfixed negative control happened to pass;
do not claim the intermittent race is reliably reproduced. The dormant bulk
builder remains compile-covered rather than hardware-qualified.

Host Q4 packing at `1e3a6d3` is independent of CUDA and preserves original
Q4_0 scale/code bytes. The device owner at `22ee0d3` depends on that packing,
the existing buffer/context RAII API, Maxwell build/FFI/stub plumbing and the
three selected-FFN kernels. Keep it an explicit research API with immutable
per-row selections, owned scratch and both comparison layouts. The unsafe
enqueue contract requires disjoint live spans, same-context storage, producer
ordering and serialized workspace use through completion.

This is not production sparse dispatch. Dynamic integration would require
separate immutable weights and per-session selection/scratch, charged device
index updates, and a choice of one down layout to avoid duplicate storage.
The cheaper-selector investigation closed negative, so that integration is
not admitted by this result. No Q8-to-Q4 conversion or predictor is imported.

At planning time, `git diff a7479ba b9333e7 -- src cuda build.rs Cargo.toml
Cargo.lock tests` is empty. Those are the runtime/dependency paths used by
the previous carry-forward validation. Apply only the intended commit diffs;
preserve intervening feature documentation and the complete V3 history.

## Atomic integration and checks

1. `feat(infer): retain Q4 channel packing`: the exact host component/tests from
   `1e3a6d3`; run its three CPU behavioral tests with warnings denied. This
   independent component can precede the CUDA fix while the GPU is occupied.
2. `fix(cuda): retain Q8 scale synchronization`: the barriers, regression and
   original evidence from `6aa52d8`. Run the Q8 regression and affected cache/
   forward checks with cuBLAS enabled and disabled, serially on the idle M40.
3. `feat(cuda): retain selected Q4 execution`: kernels, build dependency, C ABI,
   stub, owner and hardware tests from `22ee0d3`. Run selected-FFN numerical,
   guard, ownership and stream checks with cuBLAS on/off, affected projection/
   MLP/forward checks, full warning-free non-CUDA tests and CUDA/server Clippy.
4. Record final feature commits, source equivalence, fresh validation and the
   disposition of each original topic in the feature's research status and
   performance history. No pushes or default promotions are part of this step.

Wait for the independent KV-quality run and its following adaptive CPU timing
to finish before starting GPU checks. Never overlap hardware workloads.
The original experiments retain their measured scope and failures; importing
source does not create a new performance or model-quality result.

## Host component checkpoint

The host implementation and tests have been applied without modification from
`1e3a6d3`. Git blob IDs match `0d32fd9d00f9ebbb3bf487e24ad0aee245a8d937`
for `src/infer/q4_channels.rs` and `7c3bf976483b2446fd964362a9fe7043d859ea6d`
for `tests/q4_channels.rs`. Its module export is the same one-line addition.
All three warning-free CPU tests pass on the actual feature worktree: every
FP16 scale bit pattern/payload is preserved, selected values and odd-output
padding are correct, and invalid sources are rejected. Formatting and source
whitespace checks pass. See [the raw test log](2026-10-31-feature-q4-host.log).
The selected-Q4 CUDA component is still pending.

## Q8 synchronization checkpoint

The barriers and regression are unchanged from `6aa52d8`; their Git blobs are
`feb9df0442243dd4ea66900a97909cc8596cc6f8` for `cuda/kernels.cu` and
`6204ef232000812793a2689a3d7004bf8eb8c822` for the test at this checkpoint.
All 26 affected CUDA checks pass with cuBLAS enabled, followed serially by
the Q8 scale regression with cuBLAS disabled. No test skips; the final GPU
snapshot is empty. Fresh logs use the `2026-10-31-feature-q8-` prefix.
The initial compile-only command named a nonexistent test target; it ran no
tests and is retained as setup-failure evidence before the corrected build.

The imported [original report](2026-10-31-q8-scale-carry-forward.md) and its
checksums describe the research source and retain its negative-control and
dormant-builder limits. They are not new throughput measurements. The final
feature source manifest will include the later Q4 device addition separately.

## Independent follow-ups

- Cheaper channel selection: `research/1031-selector-followup`, bounded negative
  study. Its carry-forward source is distinct from the failed selector search.
- CUDA Graph dispatch parity: `research/1031-graph-dispatch`, verified fix retained
  separately. Graphs remain opt-in; no graph performance promotion follows.
- Real-model KV quality: `research/1031-kv-quality`, frozen grid still running at
  this checkpoint. Complete output and independent audit are required.
- Adaptive drafting: `research/1031-adaptive-drafting`, tested CPU replay at
  `1892432`; timed study waits for the KV run to finish.

ShiftAdd, multi-row FFN fusion, rotated activation preparation, lossless KV and
CPU/GPU scheduling remain on their original experimental branches. Their
negative or conditional results do not authorize production dispatch.

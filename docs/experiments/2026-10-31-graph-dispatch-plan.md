# CUDA Graph attention dispatch follow-up

## Scope and evidence

Base: `b9333e7` on `feature/phase4-multiple-experiments`. Work takes place on
`research/1031-graph-dispatch`. The original nine research branches remain
independent records. This runtime investigation follows the user's October 31
objective; it does not reopen the separately frozen V3 data/training work.

The original graph study at `2470940` failed exact FP16 cache parity in four
native-Q8 cases. Ordinary dense GQA uses specialized head64/head128 reductions
through length 8192, while device-length attention always uses a scalar kernel.
The old failures do not establish that this is their sole cause. No performance
claim or default promotion follows from source inspection.

## Ordered commits and gates

1. `docs(research): plan graph dispatch parity`: retain this protocol and current
   checkpoint before candidate implementation.
2. `test(cuda): retain graph state qualification`: import only the test-scoped
   failure injection and four existing state checks from `a45cfcf`. Reproduce
   the native replay failure on the current feature lineage. Add direct
   ordinary-versus-device-length attention checks before changing kernels.
3. `fix(cuda): preserve graph attention dispatch`: share the ordinary dispatch
   choices with device-length launches. Preserve each existing reduction order,
   head64 normal/ldg and head128 native/shared/warp32, and the scalar fallback
   above 8192 or for other head sizes. Device-length branches must use fixed
   launch resources sized for the cache and respect device-updated lengths.
4. `docs(research): record graph parity results`: retain raw commands/results,
   source identity, limitations and integration decision after validation.

The direct hardware checks compare exact FP32 outputs and guard values using
fixed K/V and distinct Q values, multiple physical cache slots and changing
device lengths through one captured graph. Include short/warp boundaries and
8191/8192/8193, all five specialized choices, and a generic odd head dimension.
Invalid device lengths must leave output unchanged. Verify producer ordering,
input immutability and tracked allocation cleanup. Ordinary output is the
reference; exact equality is a dispatch-parity gate, not a claim of mathematical
accuracy independently of existing CPU-reference tests.

After a candidate, rerun all four original full-forward state cases unchanged.
Require exact initialized FP16 cache values and lengths, numerical logits,
greedy IDs, single callbacks, independent sequences, repeated session lifetimes,
capture cancellation and pre-submission fallback. No tolerance relaxation.
These injected faults do not model asynchronous CUDA execution failures.

Run affected attention CPU-reference/parallel, projection/MLP, forward and
ownership checks serially on the idle M40, cuBLAS enabled and disabled where
relevant. Require warning-free non-CUDA tests, CUDA/server all-target Clippy,
formatting and whitespace checks. Tests and GPU access require escalation.
Store builds on disk, not tmpfs. Do not claim a skipped check passed.

This phase investigates dispatch correctness. Graph capture/setup, resource
occupancy, full-model decode speed and buffered HTTP serving performance need
separate measurements even if parity passes. Parallel/partitioned attention has
a separate API; inspect runtime reachability and document its graph contract.
Keep graphs and cache experiments opt-in. No changes to supported compressed
KV defaults or to experimental Q8 graph restrictions are authorized here.

## Validation execution amendment

The first full direct grid was deliberately cancelled during the boundary case
with SIGTERM, after verifying its process was still live. Build inspection
showed `DEBUG=true`; the installed `cc` crate adds NVCC `-G` in that profile.
This makes the scalar long-context fallback costly and does not validate the
optimized deployment binary. Preserve the partial log as cancelled, not passed.
Run the identical full grid and full-forward state cases in release mode,
followed by release CUDA regressions and cuBLAS-disabled parity. No input,
boundary, guard, expected output or acceptance threshold changes. Earlier debug
negative controls and the completed five-case debug state result remain valid
within their recorded build scope. CPU tests and Clippy retain their normal
profiles. This is a build-profile change, not a shortened boundary experiment.

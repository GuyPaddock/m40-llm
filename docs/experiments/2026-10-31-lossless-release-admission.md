# Lossless attention passes full release correctness

The clean `fe213e1` release completes all twelve real-input cases: layers
0/17/35 at 1K/2K/4K/8K context, each with batches 1/2/4/8. All 240 arm/batch
checks and 48 append checks pass. The four matched prototype consumers
produce identical output bits and reconstruct the original FP16 words
exactly. The native control passes its separate FP64 numerical gate.
Maximum absolute error across all arms is `4.269011991020477e-6`, below the
unchanged `1e-3` bound.

Every case restores tracked allocation bytes. The external post-process
query reports zero MiB and no M40 compute process. These are correctness
results with zero generated model tokens; no timing pair ran.

The [admission summary](2026-10-31-lossless-release-admission.json) records
the revision, executable and original input artifact hashes. The
[raw records](2026-10-31-lossless-release-correctness.jsonl.gz) retain per-case
input identities, errors, guards/ownership outcomes and setup measurements.
The [audit program](2026-10-31-lossless-release-admission.jq) checks all 86
record types/counts, the exact case and arm/batch grids, append admission,
completion and tracked cleanup.

The [executed script](2026-10-31-lossless-release-correctness.sh),
[build log](2026-10-31-lossless-release-correctness-build.log.gz) and
[post-process log](2026-10-31-lossless-release-correctness-queue.log.gz) are
retained with [checksums](2026-10-31-lossless-release-admission.sha256).
The prior [failed run](2026-10-31-lossless-release-failure.md) remains valid
for its original implementation. The [rounding correction](2026-10-31-lossless-rounding-correction.md)
resolves it without changing the codec, partition geometry or admission gate.

This admits the frozen paired timing grid. Attention, append/re-encoding,
setup costs and model-level performance must remain separate. No runtime
integration or default change is established.

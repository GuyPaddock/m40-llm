# Quantized KV passes full standalone release admission

Clean `1e445a5` completes all 36 real-input cases after the Q8 reduction
barrier correction. The grid covers layers 0/17/35 at 1K/2K/4K/8K, with
Q8/head64, mixed/head64 and mixed/head128 backing. Every case uses eight
distinct sequences and checks active batches 1/2/4/8.

Selected token positions match the independent reference exactly. Staged KV
words match reconstructed encoded inputs exactly. Query/output guards,
inactive rows, repeated serial output bits and tracked cleanup all pass.
After process exit M40 reports zero MiB and no compute process.

Maximum absolute FP64 errors are `1.0475234504703224e-5` for staged/direct
attention and `2.0414530030876676e-5` for the full-context dense reference,
below the unchanged `1e-3` limit. No numerical difference is observed between
staged/direct or batched/serial direct outputs in this grid.

Approximation is a separate question. The largest selection-only difference
from full-context original-FP16 attention is `1.6747894128815721`; the largest
quantization difference at fixed positions is `0.019369109227968778`.
Those are operator-output errors on these queries, not model logits or quality
metrics. A cost comparison with full-context dense attention cannot justify
changing token coverage. Model weights are not resident, queries are synthetic,
head64 is a diagnostic slice, and zero model tokens are generated.

The [summary](2026-10-31-quantized-kv-release-admission.json) pins revision,
executable and original capture identities. The
[audit](2026-10-31-quantized-kv-release-admission.jq) checks all 110 records,
case coverage, exact reconstruction, numerical admission and cleanup. The
[raw records](2026-10-31-quantized-kv-release-correctness.jsonl.gz),
[executed script](2026-10-31-quantized-kv-release-correctness.sh),
[build log](2026-10-31-quantized-kv-release-correctness-build.log.gz) and
[post-process log](2026-10-31-quantized-kv-release-correctness-queue.log.gz)
are retained with [checksums](2026-10-31-quantized-kv-release-admission.sha256).

This admits the frozen 384-group paired operator comparison. No timing pair
ran in this qualification. Both earlier standalone failures remain preserved
with the [barrier correction](2026-10-31-quantized-kv-barrier.md); passing
synthetic or isolated tests were not substituted for this complete grid.

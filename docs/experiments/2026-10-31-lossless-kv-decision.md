# Lossless KV study rejects runtime integration

The fixed exponent-offset codec saves storage but loses every matched
attention speed comparison on M40. Keep it as an isolated research fixture;
do not integrate it into decoding or change runtime defaults. This completes
the bounded lossless-KV topic. A different codec or kernel would be a new
experiment, not a continuation of this measured result.

## Complete comparison

Clean revision `941d485` uses executable SHA-256
`907c56d455e12af82d99828e142cb4fe3cae1b73c575ad401d86f212b8e26b88`.
The run covers layers 0/17/35, contexts 1K/2K/4K/8K and batches 1/2/4/8,
using eight distinct captured sequences and deterministic synthetic queries.
Head dimension is 128. The matched kernels use eight context partitions;
the native reference uses its existing eight-partition, 256-thread path.

All 288 comparison groups complete across twelve cases: 240 attention groups
and 48 append-plus-attention groups. There are 1,455 alternating pairs.
Three groups extend from five to ten pairs under the frozen variation rule;
none requires fresh confirmation and none ends with a timing-policy error.
Each attention interval contains ten complete B-row operations. Each append
interval contains one append plus attention. Warmup, setup and readbacks are
outside those intervals as specified in the [runner protocol](2026-10-31-lossless-attention-runner.md).

Ratios below divide candidate latency by its paired control latency. They
use reciprocals of median complete-operation rates. Values above one are
slower; each range spans all 48 layer/context/batch cases.

| Control and candidate | Candidate/control latency | Speed-gate passes |
| --- | ---: | ---: |
| Matched dense, then fused reconstruction | 7.074–8.511× | 0/48 |
| Matched dense, then separate reconstruction | 1.575–3.069× | 0/48 |
| Separate reconstruction, then fused reconstruction | 2.508–5.326× | 0/48 |
| Native serial attention, then fused batched attention | 6.633–36.314× | 0/48 |
| Fused serial rows, then fused batched rows | 0.270–1.0004× | 36/48 |
| Native append/attention, then CPU cache replacement/attention | 198.49–561.62× | 0/48 |

The 36 batching wins are all batches 2/4/8; batch one has no dispatch saving.
They compare the compressed kernel with itself and do not establish a codec
benefit. Both matched dense and native controls remain faster.

## Batch-one attention

These are completed host times in milliseconds per attention operation.
Ranges span the three sampled layers; each control comes from its own paired
comparison with fused execution. CUDA-event observations are retained in the
raw results and summary.

| Context | Matched dense | Fused in dense comparison | Native | Fused in native comparison |
| --- | ---: | ---: | ---: | ---: |
| 1K | 0.0905–0.1014 | 0.7117–0.7849 | 0.0331–0.0337 | 0.7118–0.7140 |
| 2K | 0.1724–0.1736 | 1.4072–1.4408 | 0.0532–0.0538 | 1.4073–1.4406 |
| 4K | 0.3985–0.3988 | 3.2825–3.3493 | 0.1009–0.1010 | 3.2822–3.3488 |
| 8K | 0.7858–0.7863 | 6.5864–6.6887 | 0.1837–0.1844 | 6.5892–6.6886 |

Fused attention is 7.74–8.51 times slower than matched dense at batch one.
The fixed packed reader reconstructs fields repeatedly while traversing KV;
removing a materialized cache does not make that reconstruction free. The
experiment does not isolate instruction cost from cache effects. There are
no measured DRAM transaction or dynamic-instruction counters.

## Storage and append costs

The [complete capture](2026-10-31-lossless-capture-results.md) establishes
4.574% device-representation savings at 8K/batch eight across the three
sampled layers: 201,326,592 original bytes versus 192,117,952 encoded bytes
including pointer directories. This is a KV representation saving, not a
whole-model VRAM reduction. Retained original copies, staging, query/output,
partial state and native comparison storage are reported separately.

The available CPU append path retains an original-FP16 host history, reads
the new GPU token back, re-encodes and validates complete streams, allocates
and uploads replacement device storage, retires the old storage, then runs
attention. All those costs are charged. No incremental GPU encoder or assumed
amortization is substituted. CUDA-event append intervals include CPU gaps;
they are not GPU-active execution time.

At batch one, append plus attention takes 10.79–12.83 ms at 1K and
85.55–107.46 ms at 8K. Native append plus attention takes 0.0535–0.0544 ms
and 0.2101–0.2115 ms respectively. These costs independently reject this
append implementation for interactive decoding.

## Correctness and limits

The timing run repeats all 240 arm/batch and 48 append admissions. Source
FP16 words reconstruct exactly; matched prototype outputs agree bit-for-bit.
Every timed output matches its own untimed reference. Maximum absolute FP64
error is `4.269011991020477e-6`, below `1e-3`. Each case restores tracked
allocation bytes. After exit, M40 reports zero MiB and no compute process.

The earlier [failed release](2026-10-31-lossless-release-failure.md) and
[rounding correction](2026-10-31-lossless-rounding-correction.md) remain
preserved. Initial and corrected tests cover codec bit patterns, guards,
ownership, stream dependencies and cuBLAS-enabled/disabled execution. The
[full release admission](2026-10-31-lossless-release-admission.md) separately
establishes correctness before timing.

This is an operator experiment using real captured KV and synthetic queries.
It generates zero model tokens, loads no resident model weights and measures
no whole-model speedup, long-generation quality or maximum model capacity.
The technical-document capture is not a claim of representative user traffic.
The negative decision applies to this fixed Maxwell implementation; it does
not refute results from a different codec or GPU.

## Reproduction and evidence

The [executed script](2026-10-31-lossless-timings.sh) freezes environment,
checks the prior admission, requires clean source and runs the serial grid.
The [raw records](2026-10-31-lossless-timings.jsonl.gz) retain all observations,
identities, variation, confidence intervals, setup costs and logical-byte
estimates. Sector footprints are estimates, not measured transactions.

The [summary program](2026-10-31-lossless-timing-summary.jq) selects the final
series for each comparison and checks full-grid completion, numerical
admission and cleanup. Its [summary](2026-10-31-lossless-timing-summary.json)
retains all 288 final rows. The [build log](2026-10-31-lossless-timings-build.log.gz),
[run log](2026-10-31-lossless-timings.log.gz),
[post-process log](2026-10-31-lossless-timings-queue.log.gz) and
[checksums](2026-10-31-lossless-timings.sha256) preserve the executed result.

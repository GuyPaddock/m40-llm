# Direct quantized KV consumption on Maxwell

The matched staged/direct comparison measures the benefit of the existing
direct consumers. The bounded study adds a head128 staged reference and
retains the separate Q8 reduction synchronization fix. Runtime attention
semantics, cache defaults and block selection stay unchanged.

## Scope and timing

The clean timing source is `a9b6131`. The grid covers layers 0/17/35,
contexts 1K/2K/4K/8K, and batches 1/2/4/8. It compares Q8/head64,
FP16-K/Q4-V/head64 and FP16-K/Q4-V/head128. Head64 inputs are diagnostic
slices of the original head128 capture. Queries are deterministic synthetic
vectors; the eight sequences contain distinct captured KV.

The recent window is 1024, blocks contain 32 tokens, and selection retains
plain top8 blocks in score order with no anchors or masks. The 1K cases
contain no old KV. Each staged/direct pair consumes the same encoded cache,
selected positions and queries. Timings include selection, reconstruction
where applicable, and attention. Ten complete B-row operations occur per
interval. Allocation, fixture uploads, append quantization and readbacks
remain outside attention timing and have separate setup records.

Three warm calls per arm precede five alternating pairs. The frozen variation
rule can extend a series to ten pairs. Borderline matched comparisons receive
one fresh twenty-pair confirmation; the initial and confirmation samples are
not pooled. Ratios below use reciprocals of median complete-operation rates.
CUDA-event timing remains available alongside completed host wall timing.

## Results and decision

All 144 matched staged/direct comparisons pass the operator speed gate.
Direct attention reduces complete latency by 3.53% to 20.40% across the grid.
This completes the bounded quantized-KV topic with a measured benefit for
an already implemented design. It provides no new runtime promotion and
requires no model-quality claim to characterize that matched execution cost.

All 36 cases and 384 comparison groups complete in 2,340 alternating pairs.
Twenty-one groups receive fresh twenty-pair confirmation. No final series
ends with a variation error, and none needs a ten-pair extension. The binary
SHA-256 is `f91040717c3ab9ab2cc7c971dee39eacaf189ea1d769b6aeb42160f7be40bb33`.

| Encoded KV and head dimension | Direct/staged latency | Speed-gate passes |
| --- | ---: | ---: |
| Q8, 64 | 0.8043 to 0.9355 | 48/48 |
| FP16 K and Q4 V, 64 | 0.8093 to 0.9647 | 48/48 |
| FP16 K and Q4 V, 128 | 0.7960 to 0.9569 | 48/48 |

All 36 batch-one staged/direct groups pass too. For the original head128
width, completed attention times in milliseconds span the three layers:

| Context | Staged | Direct |
| --- | ---: | ---: |
| 1K | 3.1031 to 3.1087 | 2.4699 to 2.4766 |
| 2K | 4.4440 to 4.4498 | 3.7184 to 3.7319 |
| 4K | 5.6074 to 5.6142 | 5.0984 to 5.1104 |
| 8K | 7.8988 to 7.9138 | 7.5493 to 7.5662 |

The separate mixed-cache batching comparison passes 78/96 groups. All 72
batch-2/4/8 groups pass. Their batched/serial latency ranges are 0.4920 to
0.5573, 0.2515 to 0.2622, and 0.1312 to 0.1370 respectively. Only six of
24 batch-one groups pass; their whole range is 0.9674 to 0.9984. These
comparisons use the same direct consumers and are separate from staging
removal. The remaining 144 dense-context comparisons have no adoption gate.

Maximum absolute FP64 error is `1.0475234504703224e-5` for both matched
consumers and `2.0414530030876676e-5` for the dense reference. Staged/direct
and batched/serial have zero observed numerical difference. The largest
selection-only difference from full FP16 is `1.6747894128815721`; the largest
fixed-selection quantization difference is `0.019369109227968778`. Those
approximation diagnostics prevent treating sparse-versus-dense timing as
an equivalent-output speedup. External residency is zero MiB after exit,
with no compute process remaining.

## Correctness and approximation

Every case checks selected positions against independent FP64 scoring,
reconstructed staged KV words, query/output guards, inactive rows and repeated
serial outputs. Each timed output must equal its own untimed output bits.
Matched staged/direct and batched/serial comparisons use numerical parity;
zero observed numerical difference is not a separate cross-arm bit assertion.
All tracked allocations return to their pre-case totals. The post-process
log records external device residency after teardown.

The separate full-context dense comparison changes token coverage. Its costs
are descriptive and have no adoption gate. Selection-only differences from
full FP16 and quantization differences at fixed positions are retained in
the numerical summary; neither is a generated-sequence quality evaluation.
The study generates zero model tokens and establishes no model-throughput,
logit-divergence or long-generation quality improvement.

## Storage and traffic

Both arms retain the same encoded cache, dense reference, staging workspace,
queries and outputs. Setup rows separate encoded payloads, scales, recent
FP16 storage, indices and summaries, staging, and temporary upload storage.
These benchmark allocations must not be presented as a production cache's
minimum residency. The direct consumer avoids materializing the staged KV
representation during its measured operation.

Logical traffic includes the old-key selection scan, selected old KV, recent
KV, and the staged representation's writes and reads. Per-head references
can hit cache, especially with grouped-query attention. Aligned 32-byte
sector footprint sums omit cache reuse and duplicate warp requests. No
observed DRAM-transaction or dynamic-instruction counters are available.
The raw logical records remain estimates, not measured bandwidth savings.

## Preserved correctness fix

The first two standalone releases failed exact selected-position admission.
The retained [failure records](2026-10-31-quantized-kv-release-second-failure.md)
show a material omitted-block score difference, not a near tie. The
[separate correction](2026-10-31-quantized-kv-barrier.md) adds a barrier after
all threads load the K maximum and before shared reduction storage is reused
for V. Both affected Q8 quantizers contain the missing synchronization in the
branch base. No tolerance, quantization arithmetic or selector was changed.

The eviction path passes the distinct-scale stress regression and real-input
replay. Both staged tests and nineteen runner tests pass with cuBLAS on and
off; affected runtime checks, all-target CUDA/server Clippy and formatting
also pass. One optional model diagnostic did not execute and is not a pass.
The bulk builder's matching correction is compiled and inspected, with no
separate end-to-end hardware invocation. Dynamic race checking was unavailable.
The [full clean release admission](2026-10-31-quantized-kv-release-admission.md)
passed all 36 cases before performance measurements began.

## Reproduction

The [executed script](2026-10-31-quantized-kv-timings.sh) checks clean source,
prior correctness admission and exclusive M40 use. The
[raw records](2026-10-31-quantized-kv-timings.jsonl.gz) contain source and
executable hashes, every trial, setup cost and logical-traffic record.
The [audit program](2026-10-31-quantized-kv-timing-summary.jq) requires the
complete 36-case and 384-group grids and retains the final series for each
comparison. Its [summary](2026-10-31-quantized-kv-timing-summary.json) preserves
all final rows, including errors or failed gates. The
[checksum manifest](2026-10-31-quantized-kv-timings.sha256) covers the raw data,
script, build/run/post-process logs and summary.

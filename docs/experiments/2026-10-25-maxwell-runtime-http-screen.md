# Remaining Maxwell Runtime HTTP Screen

M08's five remaining independent comparisons are complete. None qualifies
as an additional exact-output benchmark candidate. The previously qualified
[packed FP32/128](2026-10-25-maxwell-runtime-packed128-http.md) remains the
best measured exact-to-control typed candidate; runtime defaults are unchanged.
The subsequent [prefill attribution](2026-10-25-maxwell-prefill-drift.md)
finds a sequential buffer-reuse ordering defect. These measurements preserve
the historical control, not a corrected correctness baseline. Requalification
is required before treating any of their timing gates as a runtime recommendation.

## Protocol and Provenance

Release source `9d7fabb`, comparison executable SHA-256
`e07292f27c7da0d97c3e8aff1ac95fbfc5c376a421ddbb0aeeea1fbf7031d283`.
The frozen environment, original Qwen2.5-3B Q8_0 blob, prepared down/vocabulary,
dense context 1024, 65 ChatML prompt IDs, greedy sampling and actual 512-token
natural length stop match the [quality checkpoint](2026-10-25-maxwell-runtime-quality.md).
Each candidate's completed quality report is checked before timing. Explicit
request-owned prefill prevents ambient settings from changing either path.

All five cases run serially, in the table's order. Each starts with one
untimed official Ollama 0.40.11 tokenizer/offload oracle, then alternating M40
control/candidate requests. The isolated Ollama parent PID 935796 and runner
936917 remain idle during M40 timing; both were stopped after the matrix.
The preserved Ollama log has five `/api/generate` requests, not concurrent
timing work. No build, test, or MemPalace embedding overlaps these trials.
Later source edits only add an uncompiled diagnostic to `model_quality`;
the comparison executable, library and CUDA code remain fixed throughout.

Five pairs are the initial screen. The borderline FP32/256 result triggers
twenty **fresh** confirmation pairs; the table uses those twenty, not the
initial five or a pooled sample. Other cases have clear, low-variation
results and need no extension. Per-request workload/cache checks and sorted
client intervals pass. Every control's companion 512 IDs matches the earlier
qualified prepared trace. All GPU boundary samples observe 1113/3004 MHz.

## Results

HTTP E2E throughput, independently paired against native FFN/sequential
prefill with the same prepared down/vocabulary:

| Candidate | Pairs | Control tok/s | Candidate tok/s | Gain | Paired 95% gain interval | Exact IDs | Decision |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Packed FP32/256, sequential | 20 fresh | 35.482491 | 36.469939 | +2.782917% | [2.742349%, 2.825317%] | All 512 equal | Below 3% timing gate |
| Q8_1/128, sequential | 5 | 35.559443 | 41.545187 | +16.833063% | [16.723698%, 16.996039%] | Diverges at 42 | Numerical research only |
| Q8_1/256, sequential | 5 | 35.564578 | 40.262891 | +13.210654% | [13.149243%, 13.301173%] | Diverges at 42 | Numerical research only |
| Native FFN, native packed prefix | 5 | 35.547112 | 31.912737 | -10.224109% | [-10.261678%, -10.187292%] | Diverges at 0 | Timing and exact-output fail |
| Native FFN, dequant SGEMM prefix | 5 | 35.526490 | 38.393959 | +8.071356% | [8.022041%, 8.270331%] | Diverges at 0 | Exact-output fail |

FP32/256's initial five-pair gain is 2.589353%; the fresh confirmation still
falls below the unchanged 3% floor despite a positive interval. Do not repeat
it until it wins or promote it from the earlier FFN microbenchmark.

The 128-thread Q8_1 geometry is faster than 256 here. Both include activation
quantization, required dependencies and all request overhead. Their actual
512-ID continuations differ at 465 / 462 positions from the reference.
Neither regresses a frozen dense-valid task, but their already measured
held-out PPL increases and changed continuation prevent any original
exact-output claim. They do not establish the Ollama target.

Both packed-prefill candidates differ at 507 generated positions, beginning
with the first token. Their candidate continuations match each other in the
earlier quality checkpoint. This implicates a shared packed-path difference,
not specifically the SGEMM converter; attribution still requires diagnostics.
Neither candidate advances to an exact combined decode/prefill benchmark.

### Timing Boundaries

Independent medians in milliseconds; phase medians need not sum to total.
HTTP includes the full client-observed buffered request.

| Candidate | Control prompt | Candidate prompt | Control decode | Candidate decode | Control HTTP | Candidate HTTP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Packed FP32/256 | 1588.62 | 1541.97 | 12798.28 | 12451.89 | 14429.65 | 14038.96 |
| Q8_1/128 | 1583.55 | 1353.94 | 12768.21 | 10927.19 | 14398.43 | 12323.93 |
| Q8_1/256 | 1585.29 | 1396.02 | 12772.29 | 11276.21 | 14396.35 | 12716.42 |
| Native packed | 1590.57 | 3232.83 | 12768.96 | 12765.39 | 14403.42 | 16043.75 |
| Dequant SGEMM | 1584.99 | 510.02 | 12788.57 | 12779.36 | 14411.78 | 13335.43 |

Sequential FFN candidates also change prompt-time one-row projections.
The packed experiments hold native decode fixed; their decode intervals
remain similar even though generated continuations differ. Do not interpret
unmatched continuations as an exact decode speed comparison.

### Dispatch and Memory

Every model retains 1,192,935,424 prepared weight bytes and 37,748,736 dense
KV bytes. Native controls and FP32/256 issue 250,369 launches and five
request allocations/frees. Q8_1 adds 20,736 quantization launches and retains
2304 activation bytes; warm request allocations/frees remain five each.

Native packed reports 237,013 launches, 41 allocations/frees and 324 stream
waits. SGEMM reports the same launch count, 42 allocations/frees, 324 waits
and 252 cuBLAS calls. Its companion trace records a 90,177,536-byte temporary
weight capacity, released before post-request resident accounting. A zero
post-request capacity is not evidence that no temporary was used. All 252
per-prompt conversions and SGEMMs are inside the prompt timing. Native packed
has no temporary dequantized weight. Both process a 64-token prefix and the
last prompt token through the normal logits path.

## Raw Evidence

Each stem below has `.jsonl` raw requests/traces, `-audit.json` derived checks
and `.log.gz` process output under this directory:

- `2026-10-25-maxwell-runtime-packed256-http`
- `2026-10-25-maxwell-runtime-q81128-http`
- `2026-10-25-maxwell-runtime-q81256-http`
- `2026-10-25-maxwell-runtime-native-packed-http`
- `2026-10-25-maxwell-runtime-dequant-sgemm-http`

The [audit program](2026-10-25-maxwell-runtime-http-audit.jq) takes a slurped
report and `--slurpfile reference 2026-10-25-q8-native-order-http.jsonl`.
It checks serial intervals, actual workload, reference IDs, memory controls
and primary pair count before deriving medians. The
[isolated Ollama log](2026-10-25-maxwell-runtime-remaining-ollama.log.gz) and
[checksums](2026-10-25-maxwell-runtime-http-screen.sha256) preserve provenance.
Reports keep the reused tool's historical `model_variant="prepared"` label
for the candidate; both models have prepared down/vocabulary.

## Next Checkpoint

Resolve the shared packed-prefill drift before an exact combination can
qualify. Keep packed FP32/128 as the only qualified typed candidate so far,
then measure that selected configuration against fresh controlled Ollama.
Neither 53.06 HTTP tok/s nor 1.30x the fresh Ollama median is achieved.

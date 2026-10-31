# Combined Exact Runtime HTTP Qualification

Packed FP32/128 FFN plus dequant-SGEMM prefill qualifies against the repaired
native FFN/sequential control. The two changes first passed independent
[quality](2026-10-25-maxwell-completed-prefill-quality.md),
[HTTP](2026-10-25-maxwell-completed-http.md) and
[combined quality](2026-10-25-maxwell-completed-combined-quality.md) gates.

## Fixed Configuration and Result

Report revision `b7d2e24`; release comparison executable SHA-256
`40d818d6488f6818fd1d08030154b6f440c0bd7614c1767306969ea9c682b898`.
Original Qwen2.5-3B Q8_0 weights, prepared down/vocabulary, dense context
1024, canonical 65 ChatML prompt IDs, natural 512 generated tokens and
the frozen M02 environment remain unchanged. The candidate uses FP32
activations, not Q8_1 quantization.

Five alternating pairs yield **41.360036 vs 35.467001 HTTP tok/s**,
**+16.615544%**, with paired 95% gain interval
**[16.438789%, 16.707720%]**. MAD/median is 0.042134% control and 0.124404%
candidate. The fixed 3% floor dominates twice observed variation, and no
extension or borderline confirmation is required.

All actual 512 IDs and measured output match. Three HTTP arithmetic/OK/config
smokes pass, and completed combined quality has no dense-valid regressions.
This qualifies the typed experimental candidate, not a runtime-default change.

| Independent medians | Native control | Combined candidate |
| --- | ---: | ---: |
| Prompt interval | 1610.93 ms | 509.01 ms |
| Generated-token decode interval | 12778.87 ms | 11827.69 ms |
| Complete buffered HTTP | 14435.95 ms | 12379.10 ms |

Phase medians are independent and need not sum to HTTP. Unlike multiplying
the independent percentage gains, this is a measured combined result.

## Resources and Execution

Prepared down/vocabulary remain 1,192,935,424 bytes/model; dense KV remains
37,748,736. Candidate prompt processing includes all 252 conversions/SGEMMs,
using one 90,177,536-byte temporary capacity that is released before resident
post-request accounting. No full FP32 cache or Q8_1 activation storage exists.
The candidate reports 237,013 launches, 252 cuBLAS calls, 42 request
allocations/frees, 1097 syncs and 324 waits. The reference reports 250,369
launches, five allocations/frees and 1155 syncs. Decode scratch is reusable;
the extra allocations belong to prompt workspace changes, not each generated
token. All twenty primary boundaries observe 1113/3004 MHz clocks.

The isolated official Ollama server, parent PID 1058047, supplies an untimed
oracle for this comparison and then participates in the separately reported
fresh Ollama comparison. GPU workloads are serial. No build, test or MemPalace
embedding overlaps timing. The server is stopped after both comparisons and
an empty GPU process list is verified. Both M40 models are resident during
this candidate-pair run; residency is not concurrent inference.

## Evidence

- [Raw requests, counters, actual IDs and promotion decision](2026-10-25-maxwell-completed-combined-http.jsonl)
- [Derived serial-interval and resource audit](2026-10-25-maxwell-completed-combined-http-audit.json)
- [Comparison log](2026-10-25-maxwell-completed-combined-http.log.gz)
- [Shared isolated Ollama log](2026-10-25-maxwell-combined-ollama.log.gz)
- [Checksums](2026-10-25-maxwell-completed-combined-http.sha256)

The audit uses the existing HTTP audit program and derived corrected-control
trace from the completed quality report, not the old unsafe C11 continuation.
The reused `prepared` label means candidate; both models have prepared
down/vocabulary. This comparison alone establishes neither original target.
The next checkpoint is the fresh selected-runtime/Ollama measurement.

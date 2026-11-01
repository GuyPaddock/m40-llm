# Corrected-Control Independent Runtime Timings

Fresh M08 comparisons qualify packed FP32/128 FFN and dequant-SGEMM prefill
independently. Native packed prefill passes quality but loses on time.
These are corrected-control measurements, not reuse of historical unsafe
sequential timings, a combined result, or a fresh Ollama target comparison.

## Controlled Setup

Release source `3727014`, executable SHA-256
`40d818d6488f6818fd1d08030154b6f440c0bd7614c1767306969ea9c682b898`.
All three cases hold original Qwen2.5-3B Q8_0 weights, prepared down/vocabulary,
dense KV/context 1024, canonical 65 ChatML IDs, greedy sampling and natural
512-token length stopping fixed. The reference uses repaired sequential
prefill and native FFN. Matching completed quality reports are loaded before
timing. See [FP32 quality](2026-10-25-maxwell-completed-packed128-quality.md)
and [independent prefill quality](2026-10-25-maxwell-completed-prefill-quality.md).

The isolated official Ollama 0.40.11 server, parent PID 1030397, supplies one
untimed tokenizer/offload oracle before each case. Its runner is idle during
M40 pairs. Exactly three `/api/generate` requests appear in that server log.
No build, test, MemPalace embedding or other GPU workload overlaps timing.
Cases run serially in table order. The server was stopped after the group,
and an empty GPU process list was verified. Runtime defaults are unchanged.

## Paired HTTP Results

Five alternating pairs per case. The fixed 3% floor dominates twice observed
variation; no case needs variation extension or borderline confirmation.
Each primary request generates 512 actual tokens, resets prompt KV and has
the same output within its configuration. Untimed actual-ID replays match
the corrected reference and measured output in all three cases.

| Candidate | Native control tok/s | Candidate tok/s | Gain | Paired 95% gain interval | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| Packed FP32/128, sequential | 35.537469 | 38.381635 | +8.003288% | [7.815462%, 8.130796%] | Qualifies independently |
| Native FFN, native packed | 35.444783 | 31.870199 | -10.084936% | [-10.252013%, -9.939259%] | Reject on timing |
| Native FFN, dequant SGEMM | 35.420713 | 38.415733 | +8.455561% | [8.354389%, 8.485314%] | Qualifies independently |

All actual 512 IDs match. Both frozen quality splits have no dense-valid
regressions, and the three HTTP arithmetic/OK/config smokes pass for each
control/candidate pair. Source/candidate parameters and the corrected prefix
contract are recorded. Timing gates alone do not imply default promotion.

### Phase Medians

Milliseconds, with prompt/decode/HTTP medians calculated independently.
Phase medians need not add to HTTP; HTTP includes the complete buffered
client request. Diagnostic quality intervals are not substituted here.

| Candidate | Control prompt | Candidate prompt | Control decode | Candidate decode | Control HTTP | Candidate HTTP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Packed FP32/128 | 1597.23 | 1476.76 | 12755.93 | 11805.46 | 14407.33 | 13339.71 |
| Native packed | 1603.10 | 3239.71 | 12798.07 | 12774.00 | 14445.00 | 16065.16 |
| Dequant SGEMM | 1602.00 | 499.78 | 12800.59 | 12781.25 | 14454.82 | 13327.87 |

The FFN candidate also improves sequential prompt's one-row projections;
it is not a packed-prefill experiment. Both prefill candidates keep decode
native. SGEMM's benefit is in prompt processing, including every temporary
conversion. Native packed remains slower despite doing valid batched work.

### Dispatch and Allocation

Prepared weights remain 1,192,935,424 bytes/model; dense KV remains
37,748,736. All 60 primary GPU boundaries observe 1113/3004 MHz clocks.
These samples are not measured occupancy or DRAM traffic.

Control and packed FP32/128 issue 250,369 launches and five request
allocations/frees. The corrected sequential loop reports 1155 stream
synchronizations, including 64 prefix completion boundaries. Packed FP32
needs no quantized activation scratch or prefill temporary.

Native packed reports 237,013 launches, 41 allocations/frees, 1097 syncs and
324 waits. Dequant SGEMM reports the same launches/syncs/waits, plus 252
cuBLAS calls and 42 allocations/frees. Its companion trace records a
90,177,536-byte temporary weight capacity, released before post-request
resident accounting. No full FP32 model cache is retained. A zero resident
temporary after the request does not mean conversion used no memory.

## Evidence

Each stem has `.jsonl`, `-audit.json` and `.log.gz` artifacts:

- `2026-10-25-maxwell-completed-packed128-http`
- `2026-10-25-maxwell-completed-native-packed-http`
- `2026-10-25-maxwell-completed-dequant-sgemm-http`

Audits use the unchanged
[HTTP audit program](2026-10-25-maxwell-runtime-http-audit.jq). Its
`--slurpfile reference` input is the
[derived corrected-control trace](2026-10-25-maxwell-completed-control-trace.jsonl),
normalized from the completed FP32 quality report's actual control IDs, not
the old C11 unsafe continuation or a new GPU execution. Source provenance
is embedded in that derived record. Pair counts, serial client intervals,
fixed workloads, reference IDs, memory controls and clock samples pass.
The reused `prepared` variant label means candidate; both models retain
prepared down/vocabulary.

Also retained are the [isolated Ollama log](2026-10-25-maxwell-completed-ollama.log.gz)
and [artifact checksums](2026-10-25-maxwell-completed-http.sha256).

## Next Gate

Both useful independent changes now pass quality and E2E timing. Validate
packed FP32/128 plus SGEMM together against the corrected control before
timing that combination. Then use the explicit selected-runtime Ollama mode
for a fresh relative-target comparison. Neither 53.06 HTTP tok/s nor 1.30x
the controlled Ollama median is established by this candidate-only screen.

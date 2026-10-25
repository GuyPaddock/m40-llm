# Split-Plane FFN Incremental HTTP Qualification

The fixed split-plane FP32/four-value/128 FFN passes the full HTTP gate
against the current qualified packed128/SGEMM runtime. It first passed
[actual 512-token and frozen-task quality](2026-10-25-maxwell-split-runtime-quality.md).
This is an incremental comparison, not a new comparison with the older
native/sequential control.

## Fixed Configuration

Report revision `69baad2719f2590775f9212654fadbf2d1d64815`, release comparison
executable SHA-256
`63d616833d1f732b55a19d19c5a2bb8da495aab9bf2d9be4e7980f86d7378560`.
Tracked and untracked worktree status is clean at admission. Original
Qwen2.5-3B Q8_0 weights, canonical 65 ChatML prompt IDs, greedy natural
512-token generation, dense FP16 KV/context 1024 and frozen M02 flags remain.
Both arms use dequant-SGEMM for the 64-token prefix and normal final-token
decode. Prepared down/vocabulary are identical. Only fused gate/up storage
and its one-row FP32 consumer change; no activation quantization or model
modification participates.

The existing comparison tool receives:

```text
--compare-q8-runtime split-f32x4-128
--q8-runtime-prefill dequant-sgemm
--q8-runtime-control qualified-packed128-sgemm
--q8-runtime-quality-report docs/experiments/2026-10-25-maxwell-split-runtime-quality.jsonl
```

Use the isolated official Ollama setup and frozen environment from
[the controlled protocol](../controlled_benchmark.md). In this comparison,
Ollama supplies an untimed tokenizer/offload oracle, not throughput samples.

## Result

Five alternating pairs measure **44.971790 vs 41.204821 HTTP E2E tok/s**,
**+9.142060%**, with a paired 95% gain interval **[9.080723%, 9.295892%]**.
MAD/median is 0.142299% control and 0.148104% candidate. The fixed 3% floor
dominates twice observed variation. No extension or borderline confirmation
is required.

| Independent medians | Packed128/SGEMM | Split-plane/SGEMM |
| --- | ---: | ---: |
| Complete buffered HTTP | 12,425.73 ms | 11,384.91 ms |
| Prompt interval | 512.42 ms | 509.42 ms |
| Generated-token decode interval | 11,867.08 ms | 10,823.46 ms |

Phase medians need not sum to HTTP. All measured outputs match completed
quality; actual replayed 512-ID continuations match one another and that
same quality report. Arithmetic, exact-OK and config-lookup HTTP smokes pass.
This retains an experimental runtime candidate, not a global default.

## Resources And Execution

Candidate preparation adds **1,724,645,376 bytes** of gate/up planes to the
unchanged 1,192,935,424-byte down/vocabulary cache. Source plus prepared
candidate weight payload is **6,197,100,544 bytes** before KV/workspace;
the control is 4,472,455,168 bytes. Each model has 37,748,736 bytes of dense
KV. Both M40 models are resident during paired requests, alongside the idle
Ollama runner; inference is serial, not concurrent.

Every primary request has the same enqueue totals: 237,013 launches,
252 SGEMMs, 42 allocations/frees, 1097 synchronizations and 324 waits.
Each selected FFN consumer launches 18,432 times. All 252 temporary prefix
conversions remain; capacity is 90,177,536 bytes and is released before
warm decode. The per-request allocation count belongs to prompt workspace
transitions, not per-token FFN preparation. There is no Q8_1 scratch or full
FP32 materialization cache. Candidate cold load is 17,823.29 ms, including
10,979.38 ms hash/cache preparation, outside primary timing.

All twenty primary GPU boundaries report P0, 1113 MHz SM and 3004 MHz memory.
Absolute M40 client intervals do not overlap. No build, test or MemPalace
embedding overlaps timing. Official Ollama server PID 1246178 is owned by
the experiment and stopped after the subsequent fresh comparison; final
GPU process enumeration is empty.

## Evidence And Next Gate

- [Raw HTTP responses, counters, memory and actual IDs](2026-10-25-maxwell-split-incremental-http.jsonl)
- [Derived audit](2026-10-25-maxwell-split-incremental-http-audit.json)
- [Reproducible audit program](2026-10-25-maxwell-split-http-audit.jq)
- [Run log](2026-10-25-maxwell-split-incremental-http.log.gz)
- [Checksums for both comparisons](2026-10-25-maxwell-split-http.sha256)

The audit binds full quality-file SHA-256, configurations, actual IDs,
dispatch, memory and clocks; negative wrong-memory and changed-ID controls
are rejected. Rust JSON reserialization changes three preparation-time
values by about 1e-12 ms; only those four known timing fields allow <1e-9 ms
roundtrip error. All other copied manifest fields compare exactly.
The [fresh official Ollama result](2026-10-25-maxwell-split-ollama-http.md)
is separate. Neither original final throughput target follows from this
incremental comparison alone.

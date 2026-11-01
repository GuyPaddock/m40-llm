# QKV/Output Warp Incremental HTTP Qualification

## Decision

The model-local QKV/output warp combination passes the controlled HTTP
gate against the **current qualified split-FFN/SGEMM control**:
**47.322695 versus 44.949943 HTTP E2E tok/s**, **+5.278655%**, with a paired
95% gain interval of **[4.945304%, 5.864675%]**.

Five alternating pairs pass the unchanged 3% floor, positive-interval and
variation rules. MAD/median is 0.158922% control and 0.116935% candidate;
twice either value remains below 3%. No extension or fresh borderline
confirmation is required. Actual 512-ID equality and all HTTP smokes pass.
This retains the opt-in combination, not a new default.

This comparison does **not** time Ollama. It supplies an untimed tokenizer/
offload oracle only. Next perform fresh official Ollama timing for the
selected combination before making a new cross-engine claim. Neither
original goal follows from this result: 47.323 is below 53.06, and the
fresh 1.30x-Ollama requirement remains separate.

## Fixed Setup

Clean report revision `f8db9547092db3488178d76fd5520780cc91aae8`, release
comparison executable SHA-256
`7ffa5b7898affb3c67a93fd7040a1b590801c89fa66c6daf6ba3ffdef341a360`.
This is the same executable recorded by the completed
[actual-model quality gate](2026-10-25-maxwell-projection-runtime-quality.md).
The intervening commit records documentation, not a different kernel build.

Both arms use the unchanged Qwen2.5-3B Q8_0 GGUF, canonical 65 ChatML prompt
IDs, greedy sampling, natural 512-token length stops, dense FP16 KV/context
1024, and the frozen M02 environment. Split-plane gate/up/SwiGLU, prepared
down/vocabulary and the 64-token dequant-SGEMM prefix are identical. The
candidate changes only the single-row QKV/output reduction backends.

Start the [isolated official Ollama server](../controlled_benchmark.md#ollama-setup)
and export the frozen environment from the
[quality reproduction](2026-10-25-maxwell-projection-runtime-quality.md#configuration-and-reproduction),
then run the existing release tool with:

```text
--compare-q8-runtime split-f32x4-128
--q8-runtime-prefill dequant-sgemm
--q8-runtime-control qualified-split-sgemm
--q8-runtime-attention warp-256
--q8-runtime-quality-report docs/experiments/2026-10-25-maxwell-projection-runtime-quality.jsonl
```

Supply the usual `--model`, owned `--ollama-pid`, `--ollama-log` and new
`--output` path. Owned official Ollama PID 1492825 uses the recorded 0.40.11
binary and CUDA 12 backend; RAM prompt caching is disabled. Its runner is
idle during the paired M40 requests. No engines infer concurrently.

## Measurements And Ownership

| Independent medians | Current split-FFN/SGEMM | Same, QKV/output warp |
| --- | ---: | ---: |
| Complete buffered HTTP | 11,390.448 ms | 10,819.333 ms |
| Prompt interval | 499.903 ms | 499.311 ms |
| Generated-token decode interval | 10,824.594 ms | 10,280.475 ms |

Phase medians need not sum to HTTP. The sampled-stage estimate of roughly
4.7% was a feasibility model, not a strict measured bound or substitute for
this full workload. The actual unchanged-count run is the promotion evidence.
All measured outputs match completed quality, and actual untimed replayed
512-ID continuations match both models and that same quality report.
Arithmetic, exact-OK and config lookup HTTP smokes pass without regression.

Both arms retain the same **109 prepared tensor identities and bytes**:
3,279,519,744 source bytes plus 2,917,580,800 prepared bytes, totaling
**6,197,100,544 bytes per model** before KV/workspace. Unlike the earlier
split-FFN comparison, this experiment adds **no gate/up cache delta**.
Both models have 37,748,736 bytes of dense KV. Setup tracked payload is
6,197,100,544 before loading the second model and 12,394,201,088 afterward.
Boundary GPU memory also includes driver/library/workspace and idle Ollama;
it is not peak VRAM or DRAM-traffic measurement.

All primary requests retain 237,013 kernel launches, 252 SGEMMs, 42
allocations/frees, 1097 stream synchronizations and 324 waits. Each selected
QKV/output consumer launches 18,432 times. All other per-operation counters
match the quality report and control, including prepared FFN/down/head.
The 90,177,536-byte temporary SGEMM buffer is released after prefix work;
there is no retained FP32 full-model cache or quantized activation storage.
Per-request allocation counts cover admission/session work, not per-token
preparation. Candidate cold loading is 17,423.847 ms, outside primary timing.

All twenty primary GPU boundaries show P0, 1113 MHz SM / 3004 MHz memory
and only benchmark PID 1493249 plus idle official runner PID 1494146.
Recorded M40 client intervals do not overlap. No build, test, MemPalace
embedding or other GPU workload overlaps timing. The owned Ollama server
is retained temporarily for the next fresh comparison; it is not a new
background project service.

## Evidence

- [Raw HTTP and actual-token records](2026-10-25-maxwell-projection-incremental-http.jsonl)
- [Audit program](2026-10-25-maxwell-projection-http-audit.jq) and [derived result](2026-10-25-maxwell-projection-incremental-http-audit.json)
- [Run log](2026-10-25-maxwell-projection-incremental-http.log.gz) and [checksums](2026-10-25-maxwell-projection-incremental-http.sha256)

Reproduce the audit with `--arg mode incremental`,
`--arg revision f8db9547092db3488178d76fd5520780cc91aae8` and
`--slurpfile quality` pointing to the completed quality JSONL. It binds
the exact executable/model/quality hashes, all existing cache identities,
actual prompt/token output, complete per-operation dispatch, memory,
paired medians/variation, clocks and process isolation. Four copied
preparation/hash durations permit only <1e-9 ms JSON roundtrip differences;
all other copied quality fields compare exactly. Wrong dispatch and changed
actual-token negative controls are rejected.

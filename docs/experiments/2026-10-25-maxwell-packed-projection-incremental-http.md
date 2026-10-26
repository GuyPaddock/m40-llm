# Packed QKV/Output Incremental HTTP Qualification

## Decision

Packed loads pass the unchanged runtime performance gate against the
qualified scalar-load warp control: **49.082435 versus 47.186286 HTTP
tok/s**, **+4.018433%**, paired 95% gain interval **[3.802092%, 4.143568%]**.
Actual 512-token continuations and HTTP quality smokes match. Retain the
explicit opt-in combination; defaults do not change.

The initial five-pair screen gained 3.917978%, triggering the prespecified
borderline rule. The tool therefore ran **twenty fresh alternating pairs**.
The headline result uses only those forty confirmation requests, not the
initial ten. Confirmation MAD/median is 0.118681% control and 0.161675%
candidate, leaving the variation-adjusted floor at 3%. No further extension
is required. This is measured HTTP evidence, not the stage estimate.

Ollama supplies an untimed tokenizer/offload oracle only in this run. It
does not establish a new cross-engine gain. Next time the selected candidate
against fresh official Ollama. Neither original requirement is satisfied:
49.082 is below 53.06, and the fresh 1.30x requirement remains separate.

## Reproduction And Fixed Work

Clean measured revision: `e7aa655317b70970a0467471f359d6a2abafa1e3`.
Release executable SHA-256:
`4b7f15aa9361a29e42472e7c0194c92863b050ed1c3ab436f2f68514f4ff9cf7`.
This is the same build recorded by the
[full-model quality gate](2026-10-25-maxwell-packed-projection-runtime-quality.md).
The intervening commit records evidence only, not new kernel code.

Export that report's frozen environment, use the same model path, start
the [isolated official Ollama server](../controlled_benchmark.md#ollama-setup),
and invoke `target/release/examples/compare_ollama` with:

```text
--compare-q8-runtime split-f32x4-128
--q8-runtime-prefill dequant-sgemm
--q8-runtime-control qualified-warp-sgemm
--q8-runtime-attention packed-warp-256
--q8-runtime-quality-report docs/experiments/2026-10-25-maxwell-packed-projection-runtime-quality.jsonl
```

Supply `--model`, owned `--ollama-pid`, `--ollama-log` and a new `--output`.
Owned official 0.40.11 server PID 1723368 and CUDA 12 runner PID 1724899
use disabled RAM prompt caching. Benchmark PID 1724026 runs both M40
configurations **serially**; the oracle runner is idle during measured
requests. No engines infer concurrently.

Both arms retain unchanged Qwen2.5-3B Q8_0 bytes, canonical 65 prompt IDs,
greedy sampling, natural length stopping at 512 actual generated tokens,
all 36 layers, full dense FP16 KV/context 1024, split-plane FFN and
prepared down/vocabulary. The 64-token prefix remains on the same bounded
dequant-SGEMM path. Only QKV/output load instructions change.

## Measurements And Ownership

| Confirmation medians | Qualified warp control | Packed loads |
| --- | ---: | ---: |
| Complete buffered HTTP | 10,850.611 ms | 10,431.430 ms |
| Prompt interval | 504.179 ms | 501.079 ms |
| Generated-token decode interval | 10,302.623 ms | 9,883.993 ms |

Phase medians need not sum to HTTP. Even-sized medians use the mean of
the middle two observations. The initial screen is preserved separately;
it is not pooled into confirmation or used to stop confirmation early.
All timed outputs match completed quality; final untimed actual token
replays match both configurations and the frozen 512-ID continuation.

All **109 prepared tensor identities** remain fixed: 3,279,519,744 source
plus 2,917,580,800 prepared bytes, totaling **6,197,100,544 bytes per
model** before KV/workspace. Dense KV is **37,748,736 bytes** per model.
Temporary SGEMM capacity remains **90,177,536 bytes**, released after the
prefix. Packed loads add no weight cache or scratch. Both models may be
resident; this does not imply concurrent inference or report peak VRAM.

Per request, each selected projection dispatches 18,432 times. All other
operation counts and enqueue counters match completed quality, including
237,013 kernels, 252 SGEMMs, 1097 synchronizations, 324 waits and 42
session/admission allocations/frees. There is no added per-token allocation.

All **100 GPU boundaries** across the screen and confirmation show P0,
1113 MHz SM / 3004 MHz memory and only the benchmark plus idle owned
runner. Recorded client intervals do not overlap. No build, test or
MemPalace work overlaps timing. Session 71116 exits zero after actual-ID
qualification. Initial quality timing and isolated projection speedups are
not substituted for this result.

## Evidence

- [Raw trials and actual token traces](2026-10-25-maxwell-packed-projection-incremental-http.jsonl)
- [Audit program](2026-10-25-maxwell-packed-projection-http-audit.jq) and [derived result](2026-10-25-maxwell-packed-projection-incremental-http-audit.json)
- [Run log](2026-10-25-maxwell-packed-projection-incremental-http.log.gz) and [checksums](2026-10-25-maxwell-packed-projection-incremental-http.sha256)

The audit checks frozen quality/configuration binding, dispatch, prepared
identities, screen-to-confirmation progression, all measured workload/cache
records, actual IDs, phase medians, variation and GPU isolation. Missing
confirmation, altered dispatch/IDs, reused KV or changed clocks must fail.
The earlier five-pair audit remains unchanged; this experiment's audit
also handles the required even-sized independent confirmation.

# Packed Projection Runtime Versus Fresh Official Ollama

## Decision

Five alternating release pairs measure **49.146449 M40-LLM versus
43.105570 official Ollama HTTP tok/s**, **+14.014148%**, paired 95% gain
interval **[13.974351%, 14.370487%]**. Both engines generate 512 actual
tokens on the canonical workload. M40 actual IDs match completed quality;
the intended packed QKV/output kernels execute in every one-row layer.

This is the latest qualified cross-engine result, not completion of the
objective. Both original requirements remain unmet: **53.06 HTTP tok/s**
and **56.037241 HTTP tok/s** for 1.30x this fresh Ollama median. Defaults
remain unchanged. The selected exact configuration stays opt-in.

MAD/median is 0.114345% Ollama and 0.239715% M40. The unchanged promotion
floor remains 3%; neither variation extension nor borderline confirmation
is required for this cross-engine result. The earlier
[incremental M40 comparison](2026-10-25-maxwell-packed-projection-incremental-http.md)
did require twenty fresh confirmation pairs and independently passed at
+4.018%. Do not pool the two experiments or use their medians as paired
observations.

## Fixed Setup And Reproduction

Clean report source is `f276957c78d85e41bad99e2275fc584333b0d22b`.
Release executable SHA-256:
`4b7f15aa9361a29e42472e7c0194c92863b050ed1c3ab436f2f68514f4ff9cf7`.
It is the same build as the
[full-model quality checkpoint](2026-10-25-maxwell-packed-projection-runtime-quality.md)
and incremental comparison. The intervening commits record evidence only.
CUDA build flags and the complete frozen environment are unchanged.

Export the quality report's environment and use the
[isolated Ollama setup](../controlled_benchmark.md#ollama-setup). Add
`--time-ollama` to the incremental arguments:

```text
--compare-q8-runtime split-f32x4-128
--q8-runtime-prefill dequant-sgemm
--q8-runtime-control qualified-warp-sgemm
--q8-runtime-attention packed-warp-256
--q8-runtime-quality-report docs/experiments/2026-10-25-maxwell-packed-projection-runtime-quality.jsonl
--time-ollama
```

Supply the same `--model`, owned `--ollama-pid`, `--ollama-log` and a new
`--output`. Primary model loading captures and verifies `packed-warp-256`;
the control label in quality evidence refers to the prior M40 warp runtime,
not the official engine. Actual timed dispatch is checked against that
quality record, so a native model with a packed label cannot qualify.

Both engines use identical Qwen2.5-3B Q8_0 model bytes, canonical 65 ChatML
prompt IDs, greedy sampling, context 1024, natural length stops at 512
generated tokens, and explicit full dense FP16 attention. M40 retains
split-f32x4-128 FFN, prepared down/vocabulary and bounded dequant-SGEMM
prefix. No activation quantization, pruning, approximate attention or
compressed KV is added to this controlled workload.

Official Ollama is the pinned prebuilt 0.40.11 binary, SHA-256
`b0a4becfaf872e3ad5d2e9051244e0dfe55409f03d6e2cd9bd9c34400e201b4c`,
with the verified CUDA 12 backend on sm_52. RAM prompt caching is disabled.
Runner diagnostics verify zero reused prompt tokens and 65 evaluated tokens
for every accepted pair; replacing a prompt alone is not the cache proof.

## Measurements And Memory

| Independent medians | M40 packed runtime | Official Ollama |
| --- | ---: | ---: |
| Buffered HTTP | 10,417.843 ms | 11,877.815 ms |
| Reported prompt interval | 498.880 ms | 229.589 ms |
| Reported decode interval | 9,870.478 ms | 11,643.652 ms |

Phase timing boundaries differ between engines and medians need not sum to
HTTP. Only buffered client HTTP time establishes the cross-engine speed
claim. Both engines preserve their own warmup output; identical text or
arithmetic across engines is not assumed. Actual M40 replay IDs match its
completed 512-token quality continuation.

M40 retains all **109 prepared identities**, 3,279,519,744 source bytes
plus 2,917,580,800 prepared bytes: **6,197,100,544 source-plus-prepared
bytes** before KV/workspace. Its dense KV is **37,748,736 bytes**. Temporary
SGEMM capacity remains **90,177,536 bytes**, released after the prefix.
Packed loads add no weight preparation or scratch. These payload counts
are not peak VRAM or measured DRAM traffic.

Each M40 request records 18,432 launches of each packed projection,
18,432 split-FFN launches and 18,944 prepared down/vocabulary launches.
All other work matches quality, including 252 SGEMMs, 237,013 kernels,
1097 synchronizations, 324 waits and 42 admission/session allocations/frees.

All **twenty GPU boundaries** show P0, 1113 MHz SM / 3004 MHz memory and
only benchmark PID 1742689 and owned official runner PID 1724899. Engine
requests are serial; residency is not concurrent inference. No build,
test or MemPalace work overlaps timing. Session 98157 exits zero after
the final actual-ID qualification. The isolated server was stopped after
completion; no experiment-owned GPU process is intentionally left running.

## Evidence And Next Work

- [Raw paired requests, cache proof and token trace](2026-10-25-maxwell-packed-projection-ollama-http.jsonl)
- [Audit program](2026-10-25-maxwell-packed-projection-http-audit.jq) and [derived result](2026-10-25-maxwell-packed-projection-ollama-http-audit.json)
- [Run log](2026-10-25-maxwell-packed-projection-ollama-http.log.gz) and [checksums](2026-10-25-maxwell-packed-projection-ollama-http.sha256)

The audit binds clean source/build, immutable quality, actual dispatch,
prepared identities, token IDs, prompt processing, clocks, process ownership,
paired medians and both original targets. Corrupted configuration, cache
proof, replay IDs or target accounting must fail.

Next re-profile this newly selected runtime with the existing opt-in stage
diagnostic, preserving actual IDs and measuring profiling perturbation.
Only then choose another bounded exact hypothesis from the remaining stage
cost. Do not revive below-gate FFN candidates, change model semantics or
claim either original throughput goal has been reached.

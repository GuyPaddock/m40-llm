# Selected Maxwell Runtime vs Fresh Ollama

The combined FP32/128 FFN plus dequant-SGEMM runtime reaches **41.330620
HTTP tok/s** versus official Ollama's **42.945666**. M40-LLM is **3.760674%
slower**, paired 95% gain interval **[-4.391029%, -3.338984%]**.
The improvement over M40's own native control is real, but neither original
throughput target is achieved.

## Controlled Comparison

Report revision `b7d2e24`; release executable SHA-256
`40d818d6488f6818fd1d08030154b6f440c0bd7614c1767306969ea9c682b898`.
Ollama is the pinned official 0.40.11 binary with the already documented
llama.cpp `b11845` source/runtime provenance. Both use the identical
Qwen2.5-3B Q8_0 GGUF SHA-256 `4420ccb0...e0111d`, canonical 65 ChatML IDs,
dense FP16 KV/context 1024, greedy sampling and natural 512-token length stop.

The M40 side explicitly selects typed `packed-f32-128` plus `dequant-sgemm`,
with prepared down/vocabulary and no full FP32 model cache. This is not the
old native baseline under a new label. `--time-ollama` loads only this M40
model, verifies every measured output against completed quality evidence,
and verifies actual replay IDs before writing final qualification. The raw
`model_variant="control"` label is reused by the ordinary runner; manifest
`primary_runtime_config` and per-request dispatch identify the selected path.

The isolated Ollama server PID 1058047 is shared with the preceding combined
M40 comparison, but inference is serial. No build, test, MemPalace embedding
or other GPU job overlaps timing. Both engines are warmed. Ollama's RAM
prompt cache is disabled, an unrelated zero-common-prefix replacement is
sent before each measured request, and actual native runner records confirm
zero reused / 65 evaluated prompt tokens. M40 resets KV for every request.
The server exits after measurement; an empty GPU process list is verified.

## Results and Target Audit

Five alternating pairs, 512 actual tokens per primary request:

| Independent medians | Selected M40 runtime | Ollama |
| --- | ---: | ---: |
| Complete buffered HTTP | 12387.91 ms | 11922.04 ms |
| Reported prompt interval | 515.63 ms | 231.40 ms |
| Reported decode interval | 11823.71 ms | 11685.76 ms |
| HTTP tok/s | 41.330620 | 42.945666 |
| MAD / median | 0.128943% | 0.054927% |

The existing protocol needs no extension or fresh borderline confirmation
for this clear negative result. All twenty primary boundaries observe
1113/3004 MHz clocks. Both engines keep their own warmup output across all
pairs. Their text/IDs need not equal each other: Ollama's quantized-activation
arithmetic is different. M40's actual IDs match its completed native-reference
quality evidence, with no regressions across seven dense-valid tasks.

The prompt/decode boundaries differ somewhat between engines. Their medians
are indicative, not additive causal accounting. M40's remaining prompt
interval is about 4.2% of HTTP time, so even eliminating it cannot provide
the roughly 35.1% additional throughput required by the stronger target.
Do not infer DRAM traffic, occupancy or an exact stage share from this table.

| Requirement | Required HTTP tok/s | Measured M40 | Met? |
| --- | ---: | ---: | --- |
| Original fixed floor | 53.060000 | 41.330620 | No |
| 1.30x this fresh Ollama median | 55.829366 | 41.330620 | No |

The final `selected_runtime_ollama_qualification` record confirms completed
exact quality and 512 matching actual IDs, but explicitly reports
`reaches_original_target=false`. Neither a kernel microbenchmark nor the
separate +16.616% M40/native gain substitutes for these requirements.

## What the Investigation Establishes

| Aspect | Original M40 control | Retained typed M40 candidate | Pinned Ollama execution |
| --- | --- | --- | --- |
| Decode activations | FP32 | FP32 | Q8_1 quantized input |
| Gate/up reduction | Scalar Q8 dequantization and shared block tree | Alignment-safe packed loads, warp completion, 128 threads | Packed integer dot fallback, specialized warp reduction |
| Prompt projection | Sequential single-row | 64-token prefix, temporary FP32 weights and cuBLAS SGEMM | Traced dequantization/cuBLAS prefill |
| Weight model | Original Q8_0, prepared down/vocabulary | Same source and down/vocabulary | Same source GGUF |
| Approximate attention, pruning, graphs | None | None | No Maxwell graph/MMVQ fusion advantage in the pinned paths |

The [M01 source/runtime audit](2026-10-25-maxwell-q8-dispatch.md) establishes
which upstream paths actually execute, not just available source code.
The [independent corrected measurements](2026-10-25-maxwell-completed-http.md)
then demonstrate +8.003% from the packed FP32/128 FFN path and +8.456% from
SGEMM prefill. Their [measured combination](2026-10-25-maxwell-completed-combined-http.md)
gains +16.616%; this is not a multiplication of independent percentages.

This supports two explanations for the earlier gap: inefficient native
projection organization and sequential prompt work. It does not isolate
every component of packed loads, reduction geometry and instruction issue,
or prove why the remaining 3.761% differs. The Q8_1 real-input micro screen
shows additional arithmetic-organization potential, but its historical
full-model runtime quality/timing used the old unsafe sequential control.
Keep that numerical evidence labeled historical; it is not a new exact
runtime recommendation or an original-goal success.

## Retained Configuration and Next Strict Work

Retain the explicit typed FP32/128 plus SGEMM candidate for this workload.
Global FFN/prefill defaults remain native and the project compressed-KV
default remains unchanged. The scientific comparison explicitly uses dense
reference KV. The environment spelling `packed-f32` still selects its
original 256-thread geometry; do not mistake it for this typed selection.

Continue the plan's native-FFN follow-up before model-changing research:

1. Measure stage costs on the selected runtime, using a separate untimed
   diagnostic and proving profiling preserves actual IDs. The older native
   stage proportions are not automatically current selected-runtime shares.
2. Test one bounded FP32-only arithmetic hypothesis if profiling confirms
   gate/up remains dominant. The current loop applies each block scale per
   weight; an eight-value partial dot followed by scale application could
   remove repeated inner-loop scaling without quantizing activations.
   Start with one 128-thread configuration and existing real FFN inputs.
   This changes FP32 association and is a hypothesis, not a promised gain.
3. Require CPU-reference tolerance, complete-FFN paired improvement, actual
   512-ID equality, frozen dense-valid quality, and contemporaneous E2E
   comparison against the retained candidate before recommendation. Keep
   prepared down/vocabulary fixed and preserve a loser without expanding
   the geometry search to rescue it.

Sparse weights, layer pruning, approximate attention, speculation or scheduler
changes do not satisfy the remaining exact-engine targets. The goal remains
active; this is a measured gap checkpoint, not project completion.

## Evidence and Validation

- [Raw HTTP requests, native cache evidence and actual token traces](2026-10-25-maxwell-selected-ollama-http.jsonl)
- [Derived workload/configuration/quality audit](2026-10-25-maxwell-selected-ollama-http-audit.json)
- [Audit program](2026-10-25-maxwell-selected-ollama-audit.jq)
- [Comparison output](2026-10-25-maxwell-selected-ollama-http.log.gz)
- [Shared isolated Ollama log](2026-10-25-maxwell-combined-ollama.log.gz)
- [Comparison-tool tests](2026-10-25-maxwell-selected-tool-tests.log.gz)
- [All-target Clippy](2026-10-25-maxwell-selected-tool-clippy.log.gz)
- [Non-CUDA example checks](2026-10-25-maxwell-selected-tool-noncuda.log.gz)
- [Checksums](2026-10-25-maxwell-selected-ollama-http.sha256)

The tool has fourteen passing CPU tests, all-target CUDA/server Clippy and
warning-as-error non-CUDA example checks. Runtime CUDA/parity/ownership
validation is recorded in the preceding prefix-repair checkpoint; no runtime
kernel changes occur afterward. The actual controlled runs validate the new
selection route. The audit rejects deliberately changed IDs, reused cache
tokens and a false target-success flag. M40 absolute client intervals are
recorded; Ollama uses per-request runner-log byte ranges rather than absolute
client timestamps. Both are scheduled through serial awaited requests.

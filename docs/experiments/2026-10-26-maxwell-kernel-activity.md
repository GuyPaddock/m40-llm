# Actual Official Ollama GPU Stage Activity

## Finding

CUDA 12.8 concurrent-kernel activities **work on this M40/driver**, despite
unavailable hardware counters and PC sampling. A validated external observer
records GPU timestamps from the unchanged official Ollama 0.40.11 runner.
One canonical request produces sixteen tokens with fresh KV, then exits
cleanly. All **18,486 GPU activities** correlate one-to-one with successful
driver/runtime launch pairs; there are no drops or invalid timestamps.

The trace separates two startup output passes, one 65-token prompt pass,
and fifteen one-token decode forwards. A prompt forward supplies the first
sampled token, so sixteen sampled tokens do not imply sixteen decode forwards.
Startup kernels are not divided by generated-token count.

This supplies actual stage costs instead of an instruction-count inference.
It does **not** establish a new optimization, unprofiled engine comparison,
quality qualification or progress against either throughput threshold.

## Instrumented Decode

Medians across the fifteen sequential decode forwards of this **single cold
request**, not independent warmed trials. Times are sums of completed GPU
spans across all 36 layers, with no overlapping kernels in the captured
context/stream. Separate stage medians need not sum to the median total.

| Stage | Launches per decode | Median GPU ms |
| --- | ---: | ---: |
| Gate/up quantized projections | 72 | 9.126416 |
| Gate/up activation quantization | 72 | 0.172384 |
| SwiGLU | 36 | 0.121120 |
| Down projections | 36 | 4.413223 |
| Down activation quantization | 36 | 0.092318 |
| Complete FFN including quantization | 252 | 13.925996 |
| Q/K/V projections | 108 | 1.420792 |
| Attention matvecs plus softmax | 108 | 1.583123 |
| Output projections | 36 | 0.970266 |
| Vocabulary projection | 1 | 1.692855 |
| All activation quantization, including FFN | 253 | 0.584475 |
| All completed kernel spans | 1013 | 21.074601 |

Projection roles are assigned from the actual per-layer shape sequence
`2048,256,256,2048,11008,11008,2048`, followed by vocabulary `151936`.
Every one of the 253 projections is preceded by its own Q8_1 quantizer.
The audit checks that sequence, quantizer dependency and all launch geometry;
it does not classify every 2048-wide projection as down. Gate/up is unfused
and is followed by a distinct SwiGLU kernel, as source inspection predicted.

The full prompt pass contains 1193 kernels, **232.545177 ms** summed GPU
execution and **270.684377 ms** from its first GPU start to its last end.
The two startup passes contribute 2098 other kernels and are excluded.
No prefill dispatch or inference algorithm is changed.

## Perturbation And Limits

The median first-to-last GPU interval per decode is **44.538674 ms**, with
individual intervals **34.570501-70.890821 ms**, versus approximately
21.075 ms of summed kernel execution. This is a heavily instrumented
callback/activity process. Its large gaps cannot be attributed to the normal
engine or presented as an optimization opportunity. There is no paired
observer-off calibration in this bounded diagnostic.

The API reports 274.331 ms prompt processing, 696.281 ms decode and
3678.106 ms total, including 2705.524 ms loading. These are instrumented
cold-request fields, not the resident HTTP benchmark. Output is:

```text
Sure, I'll walk you through the entire process of loading a web page from
```

No actual generated-ID replay or frozen quality suite is claimed. The API
reports sixteen generated tokens and a natural length stop. Actual prompt
IDs are separately checked with the runner tokenizer against the frozen
65-ID canonical prompt; API and runner logs both report zero reused tokens.

Do not subtract these costs from the earlier M40 event profile: context
lengths, instrumentation, warmup and clock observations are not matched.
In particular, the post-request idle boundary shows 949 MHz SM / 3004 MHz
memory, not proof of a fixed clock throughout execution. No achieved
occupancy, cache-hit, stall or DRAM-traffic measurement follows.

The observation confirms that total activation-quantizer GPU execution is
small relative to this runner's projection work. It does not isolate the
benefit of quantized dot arithmetic or justify promoting Q8_1, whose existing
numerical/quality limitations remain unchanged. No rejected layout, cache,
reduction or geometry candidate is reopened.

## Implementation And Validation

Contract `23b910b`; observer `be82fa4c3cbad4cf97b59dfff1b0570be8aa0adb`.
The observer library hash, actual pinned binary hashes, model and owned
process IDs are in the [identity record](2026-10-26-maxwell-kernel-activity-identities.json).
This run uses the same official CUDA 12 backend and Qwen2.5-3B Q8_0 model
as the qualified comparison, with sm_52, context 1024, full GPU offload,
flash attention off, plain greedy sampling and neutral penalties.

The new option only affects the SDK-built `LD_PRELOAD` observer. Inference,
CUDA kernels, Cargo/build flags and global defaults are untouched. It enables
concurrent-kernel activity, never serialized profiling or hardware counters.
Buffers are capped at eight outstanding 1 MiB host allocations; each is
released by its completion callback. Records are capped at 100,000.

The real two-launch CUDA fixture returns **96** with activity enabled,
disabled and unset. The enabled case has exactly two GPU spans, successful
launch correlation and no activity for its deliberately rejected third API
call. Invalid option values fail before CUDA initialization. These SDK-only
changes compile with `-Wall -Wextra -Werror`; shell syntax, Perl syntax,
formatting and whitespace checks pass. Unrelated model/CUDA parity suites
are not rerun or claimed for observer-only work.

An initial sandboxed launch-only subscription returned status 15,
`CUPTI_ERROR_NOT_INITIALIZED`, before executing its command. The subsequent
elevated hardware checks pass; the initial result is preserved as a
[failed preflight](2026-10-26-maxwell-kernel-activity-sandbox-preflight.jsonl),
not a negative activity-capability finding or a measured request.

Absolute timestamps exceed double-precision JSON's exact integer range.
`7e6ca53` adds a tested 64-bit integer normalization step: subtract before
jq analysis, preserve absolute clocks as strings and retain the original raw
trace. A synthetic 17 ns difference at the observed epoch remains 17 ns;
out-of-order delivery and invalid/overflowing ranges are checked. An initial
normalizer incorrectly rejected valid 19-digit timestamps; this was corrected
before analysis and did not require another GPU request. A partial read while
normalization was live was discarded; analysis uses the completed file.

The independent audit checks all callback/activity joins, exact normalized
durations, serial GPU intervals, source/binary pins, prompt IDs, cache state,
phase boundaries and projection roles. Twelve corrupted reports are rejected.

## Ownership And Reproduction

Owned server session **16900**, parent PID **3636581**, runner PID
**3636932**, native runner port **44745**. The GPU is empty before launch;
only the verified runner is resident after the request. No inference, build,
test or MemPalace overlaps collection. The tokenizer request does not generate.
The runner is gracefully stopped first to flush activities, then the verified
parent. Session exit is zero; both final process and GPU lists are empty.
All discovery-process traces are excluded from model coverage.

Use the [isolated official server setup](../controlled_benchmark.md#ollama-setup)
with `M40LLM_CUDA_TRACE_ACTIVITY=1`, modules off, no forced-JIT/cache override,
and the [external observer wrapper](../cuda_launch_trace.md). Submit the saved
request once. Verify actual identities and prompt IDs before graceful shutdown.
Never rebuild the observer while observed processes are live.

- [Request](2026-10-26-maxwell-kernel-activity-request.json), [response](2026-10-26-maxwell-kernel-activity-response.json) and [actual prompt IDs](2026-10-26-maxwell-kernel-activity-tokenizer.json).
- [Original runner trace](2026-10-26-maxwell-kernel-activity-runner-trace.jsonl.gz), [server log](2026-10-26-maxwell-kernel-activity-server.log.gz) and [behavioral smoke records](2026-10-26-maxwell-kernel-activity-smokes.tar.gz).
- [Audit command](2026-10-26-maxwell-kernel-activity-audit.sh), [rules](2026-10-26-maxwell-kernel-activity-audit.jq), [derived stages](2026-10-26-maxwell-kernel-activity-audit.json) and [checksums](2026-10-26-maxwell-kernel-activity.sha256).
- [Owned processes](2026-10-26-maxwell-kernel-activity-owned-processes.txt), [loaded backend](2026-10-26-maxwell-kernel-activity-backend-maps.txt), [GPU state](2026-10-26-maxwell-kernel-activity-gpu.csv), [resident runner](2026-10-26-maxwell-kernel-activity-processes.csv), [final GPU](2026-10-26-maxwell-kernel-activity-final-gpu.csv) and [final processes](2026-10-26-maxwell-kernel-activity-final-processes.txt).

Next, any quantitative engine-stage comparison must use matched workload
phases and quantify collector perturbation on both engines. This diagnostic
does not authorize a new runtime candidate or relax qualification. Qualified
exact status remains **49.146 HTTP tok/s vs 43.106 official Ollama**; both
**53.06** and **56.037 HTTP tok/s** requirements remain unmet and active.

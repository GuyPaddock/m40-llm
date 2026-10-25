# Maxwell Q8 Execution Investigation

The active workspace plan is
[Close the Maxwell Q8 Execution Gap](../../M40_LLM_MAXWELL_Q8_EXECUTION_PLAN.md).
This guide links its evidence and current implementation boundaries. The
earlier C01-C27 experiments remain historical; their negative gates are not
permission to substitute pruning or approximate attention for this work.

## Fixed Control

Keep original Qwen2.5-3B Q8_0 weights, all 36 layers, full dense FP16 KV at
context 1024, the canonical 65-token ChatML browser prompt and greedy decoding.
The [qualified control](experiments/2026-10-25-q8-native-order-http.md) prepares
only down/vocabulary with split storage and a 6144 MiB extra-payload budget.
Single-row down retains native-order 4-column/256-thread reduction. Native
fused QKV/MLP, decode-stream scheduling, greedy head flags and parallel
head128 split8/256 remain fixed. The qualified rate is 35.724705 HTTP tok/s,
not the older 31.85 native-only result.

FP32-activation candidates must preserve actual 512-token output and pass the
existing E2E gate. Q8_1 activation quantization is a separate opt-in numerical
track with reconstruction, logit, task and perplexity reporting. It cannot
satisfy the original exact-output goal by changing the continuation. Nothing
in this investigation changes the project compressed-KV default or globally
enables weight preparation.

## Checkpoints

| Unit | Current State |
| --- | --- |
| M01 | Source pin and official-backend launch dispatch verified; [report and raw evidence](experiments/2026-10-25-maxwell-q8-dispatch.md) |
| M02 | In progress: [paired input artifact and read-only model observer](ffn_benchmark_inputs.md) have CPU/M40 coverage; real layer 0/17/35 fixtures and complete rows 1/4 FFN baseline remain |
| M03 | Pending: fixed FP32 packed-load/reduction ablation, then one additional 128-thread combined variant |
| M04 | Pending independently: packed Q8_1 activation/dot experiment, quantization-inclusive cost and numerical checks |
| M05 | Pending: complete-FFN five-pair screen and preserved negative results |
| M06 | Conditional: immutable runtime selection and reusable scratch only for qualifying decode candidates |
| M07 | Independent pending branch: bounded reusable Q8-to-FP32 prefill SGEMM, with no full-model FP32 cache |
| M08 | Pending: isolated decode/prefill and qualifying combination HTTP/quality comparisons |
| M09 | Pending: measured attribution, retained configuration, fresh Ollama comparison and remaining-goal audit |

The standalone [untimed launch observer](cuda_launch_trace.md), preserved in
`723a456`, is an M01 prerequisite with no Cargo/inference dependency. Its
callback support is distinct from the unavailable legacy traffic counters.
The official binary was not rebuilt for the diagnostic.

M01 confirms relevant execution differences, not how many milliseconds each
causes. Keep the scalar-Q8_1 and prior packed-prefill failures as evidence.
Do not repeat those implementations or use unmatched shapes to rule out every
Maxwell packed/warp strategy. Finish actual-input, complete-FFN measurements
before integrating a decode candidate; hold decode fixed for prefill work.

Use the shared [controlled benchmark protocol](controlled_benchmark.md), source
the development environment, request hardware escalation, serialize GPU work,
and keep trace/profile runs separate from authoritative timing. Both original
exact-output targets remain unmet: 53.06 HTTP tok/s and 1.30x a fresh controlled
Ollama median. Failed experiments need preservation before separate removal.

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
| M02 | Complete: [warm real-input FFN baseline](experiments/2026-10-25-maxwell-ffn-baseline-warm.md), genuine rows 1/4, all numerical/ownership checks pass; one-row complete about 0.457 ms/layer |
| M03 | Complete: [real-input five-pair screen](experiments/2026-10-25-maxwell-fp32-ffn-screen.md); packed/warp256 gains 7.75-8.27% complete FFN with native bits on the sampled one-row inputs; 128 gains 12.18-12.63% with small arithmetic drift |
| M04 | [Pinned Q8_1 encoder and 128/256-thread packed consumer](q8_activation.md) pass release-M40 byte/projection/SwiGLU parity; quantization-inclusive real-input FFN screen pending |
| M05 | Pending: complete-FFN five-pair screen and preserved negative results |
| M06 | Conditional: immutable runtime selection and reusable scratch only for qualifying decode candidates |
| M07 | Independent pending branch: bounded reusable Q8-to-FP32 prefill SGEMM, with no full-model FP32 cache |
| M08 | Pending: isolated decode/prefill and qualifying combination HTTP/quality comparisons |
| M09 | Pending: measured attribution, retained configuration, fresh Ollama comparison and remaining-goal audit |

The standalone [untimed launch observer](cuda_launch_trace.md), preserved in
`723a456`, is an M01 prerequisite with no Cargo/inference dependency. Its
callback support is distinct from the unavailable legacy traffic counters.
The official binary was not rebuilt for the diagnostic.

### M03 Primitive Checkpoint

`CudaContext::ffn_q8_f32_ablation_async` is an explicit experimental API,
not a generation policy. Its five fixed variants use scalar/packed loads and
shared/warp reduction at 256 threads, plus packed/warp at 128 threads. All
enqueue one kernel on the decode stream with no allocation, transfer or
synchronization. Packed payload reads require only two-byte alignment for
native 34-byte Q8_0 blocks. No upstream code was copied for these variants.

The release M40 tests in `tests/q8_ffn_fp32.rs` pass for distinct rows 1/4,
signed extremes, zero scales, hidden tails and two-byte-offset weight bases.
All four 256-thread variants match native split4 output bits across that
synthetic grid; the sole 128-thread variant passes CPU-reference tolerance.
Input/output guards, immutable weights, rejected arguments, enqueue counters
and tracked allocation cleanup are checked. This is not yet real-model
512-token parity or a measured speedup.

NVCC 12.8.93 release sm_52 inspection reports 32 registers, zero stack and
zero local bytes for all five variants and native split4. Dynamic shared
storage is 2 KiB at 256 threads and 1 KiB at 128 threads. The scalar variants
contain 16 byte-payload load sites plus two scale-load sites; packed variants
contain eight halfword-payload load sites plus two scale-load sites. Warp
variants contain one barrier and ten shuffle sites. Static shared-reduction
barrier-site counts are not dynamic counts: native uses a loop, whereas the
new fixed-size scalar/shared reduction is unrolled. Debug resource reports
are not performance evidence. Occupancy and DRAM traffic were not measured.

The existing complete-FFN benchmark now accepts `--ffn-f32-ablation` for
[paired native controls](ffn_benchmark_inputs.md#bounded-fp32-ffn-screen).
Release M40 tests verify all five variants through reusable pipeline scratch,
cross-stream four-row work, full CPU-reference outputs and enqueue counters.
The [real-input five-pair matrix](experiments/2026-10-25-maxwell-fp32-ffn-screen.md)
now passes the one-row micro gate for packed/warp256 and packed/warp128 at
all three layers. The latter is not bit-identical. Losing variants remain
preserved. No whole-model or HTTP qualification follows from this screen.
Keep prepared down/vocabulary fixed. The four-row experimental fused kernel
is a row-batched algorithm extension, not the production four-row two-GEMM
plus SwiGLU path; report that distinction rather than attributing all M4
differences to packed loads or reductions.

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

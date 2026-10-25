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
| M04 | Complete: [quantization-inclusive real-input screen](experiments/2026-10-25-maxwell-q81-ffn-screen.md), 21.50-27.41% one-row FFN gain; separate numerical track, later quality and historical timing evidence retained |
| M05 | Complete: [bounded screening decision](experiments/2026-10-25-maxwell-q81-ffn-screen.md#m05-screening-decision); packed/warp FP32 and Q8_1 advance conditionally, failed variants preserved |
| M06 | Complete: [model-local FFN selection, reusable scratch and runtime regressions](experiments/2026-10-25-maxwell-ffn-runtime.md); full-model qualification remains M08 |
| M07 | Complete: [bounded reusable Q8 prefill SGEMM, runtime/fixture parity and real dispatch smoke](q8_prefill.md); full-model qualification remains M08 |
| M08 | Corrected exact branch qualified: [independent gains](experiments/2026-10-25-maxwell-completed-http.md), [combined quality](experiments/2026-10-25-maxwell-completed-combined-quality.md) and [combined HTTP +16.616%](experiments/2026-10-25-maxwell-completed-combined-http.md); numerical historical evidence remains separate |
| M09 | [Fresh selected-runtime checkpoint](experiments/2026-10-25-maxwell-selected-ollama-http.md): 41.331 vs 42.946 HTTP tok/s, 3.761% slower; original targets unmet, continue the native-FFN next-work sequence |

The standalone [untimed launch observer](cuda_launch_trace.md), preserved in
`723a456`, is an M01 prerequisite with no Cargo/inference dependency. Its
callback support is distinct from the unavailable legacy traffic counters.
The official binary was not rebuilt for the diagnostic.

## Experimental Runtime Selection

### Explicit Runtime Qualification

The existing `model_quality` runner accepts independent
`--q8-runtime-candidate native|packed-f32-256|packed-f32-128|q8-1-128|q8-1-256`
and `--q8-runtime-prefill sequential|native-packed|dequant-sgemm` selections.
Both models use the qualified prepared down/vocabulary configuration, dense
KV/context 1024 and the same source GGUF. The reference always uses native FFN
and sequential prefill. Runtime choices are explicit, not inherited from
ambient experiment settings; model-changing/capture modes are incompatible.
Quality manifests now record `prefill_execution_contract` as
`prefix-decode-completed-v1`. HTTP qualification rejects older reports with
a clear instruction to rerun quality. This prevents historical unsafe-prefix
evidence from qualifying the repaired runtime merely because model, prompt
and candidate names still match.

After restoring the frozen M02 environment, run a single candidate:

```bash
cargo run --release --features cuda,server --example model_quality -- \
  --reference-model "$MODEL" --model "$MODEL" --variant-id packed256 \
  --q8-runtime-candidate packed-f32-256 --q8-runtime-prefill sequential \
  --split all --output /tmp/new-packed256-quality.jsonl
```

Source `scripts/dev-env.sh` first and request hardware escalation. The report
records actual 512-token browser continuations with natural stopping, prompt
IDs, first divergence, dispatch counters, memory/scratch and frozen task scores.
Full-vocabulary logit comparisons replay identical reference prefixes at the
final prompt, first decode, fixed later samples and the divergence step. The
generic full-logit head is labeled separately from actual GPU-greedy output.
Sequential candidates also report token-weighted calibration/held-out loss
and perplexity. Packed-prefill candidates do not claim sequential passage
loss as prefill validation; their actual packed task/browser generation and
forced-prefix logits exercise that path instead. Reports refuse overwrites.

These runs are quality diagnostics, not HTTP throughput qualification. A
different continuation, natural stop length or dense-valid task regression
blocks the FP32 exact-output gate. Q8_1 stays a separate numerical track.

### Packed-Prefill Attribution

The same runner accepts `--q8-prefill-drift` instead of a runtime candidate.
Use identical source/reference model paths, context 1024 and the frozen M02
environment. It compares the canonical prompt through session sequential,
manual sequential, explicitly drained sequential, a greedy head at every
prefix, native packed-prefix and SGEMM packed-prefix execution. All choices
retain native FFN, prepared down/vocabulary and explicit dense KV.

The untimed JSONL includes actual greedy token IDs, final hidden/full-logit
differences, physical KV lengths and K/V comparisons at sampled absolute
positions in every layer. A final-layer attention replay compares the real
query against its stored FP16 cache, widened to FP32 for packed attention,
and an independent FP64 CPU softmax. This distinguishes cache contents from
kernel arithmetic; it does not recover the original unrounded packed K/V.
The diagnostic adds no runtime synchronization policy or default change.

```bash
cargo run --release --features cuda,server --example model_quality -- \
  --reference-model "$MODEL" --model "$MODEL" --variant-id prefill-drift \
  --q8-prefill-drift --context-tokens 1024 \
  --output /tmp/new-prefill-drift.jsonl
```

Source the development environment and request hardware escalation first.
Run attribution separately from all timed comparisons.

### Load-Time Configuration

`M40LLM_Q8_FFN_KERNEL=native|packed-f32|q8-1` is captured once by
`LoadedModel::from_gguf`. Unset means `native`, preserving the existing
dispatch and all runtime defaults. `packed-f32` selects the screened
256-thread packed/warp FP32-activation kernel; `q8-1` selects the 128-thread
quantization-inclusive candidate. These are experimental choices, not
whole-model quality or throughput recommendations. Q8_1 changes arithmetic
and remains a separate numerical track.

The typed `Q8FfnConfig` constructors retain both 128/256-thread geometries
for full-model qualification. `from_gguf_with_q8_backends` accepts independent
weight-preparation and FFN configurations. The existing explicit
`from_gguf_with_q8_execution` constructor always uses native FFN, so ambient
experimental flags cannot alter frozen references. Down/vocabulary preparation
is unchanged and must be selected separately.

Only one-row forward uses the candidate. Sequential prompt tokens and the
final prompt token use that one-row path; multi-row packed-prefix prefill
keeps its native projections. Q8_1 uses one RAII activation allocation in the
leased forward workspace, shared across layers and reused across tokens.
Qwen's D=2048 needs 2304 bytes; packed FP32 needs no quantization allocation.
The selected kernels enqueue on decode. Other projection streams retain
explicit producer/consumer waits. No per-token allocation, transfer or
synchronization is added by FFN enqueue.

Explicit experimental requests require compiled CUDA kernels, sm_52, uploaded
Q8_0 gate/up tensors in every standard layer and complete 32-value input
blocks. Unsupported requests fail with a native-mode remedy rather than
silently benchmarking another implementation. Experimental graph capture,
batched decode, frozen input capture and model-changing research combinations
are rejected. This investigation adds no server scheduling behavior.

API migration: in-memory `LoadedModel` construction now supplies
`q8_ffn: Q8FfnConfig::native()`. Direct callers of the unsafe fused-FFN
wrapper supply a final scratch argument: `None` for native/packed FP32, or
an owned `Q8FfnScratch` view for Q8_1. Workspace allocation/estimation also
takes the FFN configuration. Configure before creating workspace/session
state; do not replace model components while sessions reference them.

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

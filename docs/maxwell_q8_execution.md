# Maxwell Q8 Execution Investigation

The active workspace plan is
[Close the Maxwell Q8 Execution Gap](../../M40_LLM_MAXWELL_Q8_EXECUTION_PLAN.md).
This guide links its evidence and current implementation boundaries. The
earlier C01-C27 experiments remain historical; their negative gates are not
permission to substitute pruning or approximate attention for this work.

## Fixed Control

Keep original Qwen2.5-3B Q8_0 weights, all 36 layers, full dense FP16 KV at
context 1024, the canonical 65-token ChatML browser prompt and greedy decoding.
The [current qualified configuration](experiments/2026-10-25-maxwell-packed-projection-ollama-http.md)
uses split-f32x4-128 fused gate/up/SwiGLU, packed-warp-256 raw-Q8 QKV/output,
prepared down/vocabulary with native-order 4-column/256-thread reduction,
and a bounded dequant-SGEMM 64-token prompt prefix. Decode-stream scheduling,
greedy head flags and parallel head128 split8/256 attention remain fixed.
All 109 prepared identities and their 2,917,580,800-byte payload are pinned.
Source plus prepared payload is 6,197,100,544 bytes before KV/workspace.

Five fresh paired HTTP trials measure **49.146449 versus 43.105570 tok/s**
for the official Ollama baseline. Both original targets remain unmet:
53.06 and 1.30x the fresh baseline, currently 56.037241 HTTP tok/s.
The [35.724705 down/vocabulary-only control](experiments/2026-10-25-q8-native-order-http.md)
is the historical M01-M08 comparison baseline, not the current best runtime.
Keep it explicitly named when reproducing those experiments.

FP32-activation candidates must preserve actual 512-token output and pass the
existing E2E gate. Q8_1 activation quantization is a separate opt-in numerical
track with reconstruction, logit, task and perplexity reporting. It cannot
satisfy the original exact-output goal by changing the continuation. Nothing
in this investigation changes the project compressed-KV default or globally
enables weight preparation.

## Admission For Further Optimization

Preserve the qualified 49.146 HTTP tok/s configuration. Negative bounded
experiments are not evidence that Maxwell's hardware ceiling has been
reached. DRAM transactions, achieved occupancy and warp stalls remain
unmeasured; static register counts and read-checksum timings cannot replace
those measurements.

Before implementing another candidate, write down:

- The expensive work in the actual qualified reader or dataflow that the
  candidate avoids. Name the loads, repeated data, conversion, addressing,
  synchronization or materialization involved. An arithmetic substitution
  alone is not a credible significant-E2E hypothesis without cost evidence.
- The expected saving at actual Qwen dimensions and its complete-FFN/E2E
  scope. Distinguish logical bytes/instructions from measured physical
  transactions; do not claim bandwidth saturation from payload estimates.
- New resident and peak setup bytes, scratch, padding, metadata, load-time
  repacking cost and per-token conversion/read/write overhead. Explain reuse
  and ownership, including whether source storage can actually be released.
- How the proposal differs from recorded losing layout/reuse experiments.
  Do not rescue them with another geometry/cache/register parameter sweep.
- One bounded test, its control, exact-output checks and a stop condition.
  A weight organization/reuse proposal must prove real complete-FFN gains,
  not just a faster checksum or isolated dot product.

Keep the existing gates: at least 5% complete-FFN improvement with positive
paired confidence before runtime integration, then HTTP improvement above
`max(3%, twice observed variation)` with positive paired confidence, actual
512-token equality and frozen dense-valid quality. Independent projection
work must meet its separately declared stage and E2E gates. Include all
conversion, launches and dependencies. Preserve failed code and measurements
before separate removal; no defaults or exact-goal relaxation.

The [remaining prefix profile](experiments/2026-10-26-maxwell-prefill-nongemm.md)
accounts for 106.794 ms in newly measured groups, only 1.029% optimistic
HTTP scope even if all disappeared. It does not justify a prefix rewrite.
Further work should first establish avoided dominant decode-reader work,
not produce another implementation on the strength of unmeasured headroom.

The [subsequent reader-work audit](experiments/2026-10-26-maxwell-reader-work-audit.md)
finds each logical payload word is already read once per output column,
with activation loads shared between gate and up. It separates repeated
source operands from physical requests and records the costs/results of
existing reuse/packing probes. No new kernel is admitted. Changing weight
organization must explain savings beyond those closed experiments and
account for any increased resident storage and per-token traffic.

## Checkpoints

| Unit | Current State |
| --- | --- |
| M01 | Source pin and official-backend launch dispatch verified; [original report](experiments/2026-10-25-maxwell-q8-dispatch.md) and [actual selected driver-JIT code](experiments/2026-10-26-maxwell-ollama-jit-inspection.md) |
| M02 | Complete: [warm real-input FFN baseline](experiments/2026-10-25-maxwell-ffn-baseline-warm.md), genuine rows 1/4, all numerical/ownership checks pass; one-row complete about 0.457 ms/layer |
| M03 | Complete: [real-input five-pair screen](experiments/2026-10-25-maxwell-fp32-ffn-screen.md); packed/warp256 gains 7.75-8.27% complete FFN with native bits on the sampled one-row inputs; 128 gains 12.18-12.63% with small arithmetic drift |
| M04 | Complete: [quantization-inclusive real-input screen](experiments/2026-10-25-maxwell-q81-ffn-screen.md), 21.50-27.41% one-row FFN gain; separate numerical track, later quality and historical timing evidence retained |
| M05 | Complete: [bounded screening decision](experiments/2026-10-25-maxwell-q81-ffn-screen.md#m05-screening-decision); packed/warp FP32 and Q8_1 advance conditionally, failed variants preserved |
| M06 | Complete: [model-local FFN selection, reusable scratch and runtime regressions](experiments/2026-10-25-maxwell-ffn-runtime.md); full-model qualification remains M08 |
| M07 | Complete: [bounded reusable Q8 prefill SGEMM, runtime/fixture parity and real dispatch smoke](q8_prefill.md); full-model qualification remains M08 |
| M08 | Corrected exact branch qualified: [independent gains](experiments/2026-10-25-maxwell-completed-http.md), [combined quality](experiments/2026-10-25-maxwell-completed-combined-quality.md) and [combined HTTP +16.616%](experiments/2026-10-25-maxwell-completed-combined-http.md); numerical historical evidence remains separate |
| M09 | [Current fresh official checkpoint](experiments/2026-10-25-maxwell-packed-projection-ollama-http.md): 49.146 vs 43.106 HTTP tok/s, +14.014%; both original targets still unmet. The earlier 41.331 checkpoint remains historical |

The [plan-evidence audit](experiments/2026-10-26-maxwell-plan-evidence-audit.md)
distinguishes completed exact-path qualification from historical numerical
evidence. The [refreshed Q8_1 reports](experiments/2026-10-26-maxwell-q81-corrected-quality.md)
now use completed sequential prefill for both geometries: zero dense-valid
task regressions, changed continuations at step 60 for 128 threads and step
10 for 256 threads, and worse held-out PPL. They remain numerical-only and
cannot qualify the exact-output goal. Earlier Q8_1 reports are historical.

The standalone [untimed launch observer](cuda_launch_trace.md), preserved in
`723a456`, is an M01 prerequisite with no Cargo/inference dependency. Its
callback support is distinct from the unavailable legacy traffic counters.
The official binary was not rebuilt for the diagnostic.

Its later opt-in module capture now verifies the actual official runner's
selected sm_52 MMVQ image against M01's offline assembly. The matching code
and whole-process counts close that evidence gap without a new timing or
runtime claim. No global forced-JIT setting is recommended.

## Experimental Runtime Selection

### Fixed-Memory Clock Sensitivity

The [two-state diagnostic](experiments/2026-10-26-maxwell-sm-clock.md)
keeps unchanged kernels and memory 3004 MHz while observing SM 1113/822 MHz.
One-row latency rises 6.60-6.79% gate/up, 13.13-13.31% prepared down and
9.00-9.11% complete FFN. Five outer pairs and exhaustive paired intervals
are distinct from the existing inner benchmark rounds and promotion gates.
Original hardware policy is restored. This does not measure pure arithmetic,
DRAM traffic, occupancy or stalls, nor justify clock changes or another
kernel automatically. Qualified runtime/defaults and original targets stay
unchanged; closed probes cannot be rescued from this sensitivity alone.

### Actual Runner Activity Costs

The [subsequent off/on calibration](experiments/2026-10-26-maxwell-activity-calibration.md)
fails joint attribution: complete native captures add 5.607% decode overhead;
M40's 1.580% point estimate comes from capped, incomplete captures. Keep
activity-only observation diagnostic. Do not subtract these engine stages,
raise the cap to claim success or change runtime/defaults from that evidence.

The [bounded official activity checkpoint](experiments/2026-10-26-maxwell-kernel-activity.md)
now measures GPU spans, complementing callback and loaded-code inspection.
Concurrent-kernel activities work on M40, unlike the unavailable counters
and PC sampling. A cold sixteen-token request yields fifteen one-token
decode forwards after separately identified startup and prefill work.
Complete correlated spans give 13.926 ms median FFN and 0.584 ms total
activation quantization within 21.075 ms summed kernel execution.
These are instrumented stage costs, not unprofiled throughput. Large host
gaps and unmatched warmup/clocks preclude quantitative engine deltas; matched
phases and collector perturbation checks must precede such comparisons.
No defaults, exact-output gates or qualified runtime change.

### Layer-Reuse Diagnostic

The shared benchmark accepts `--ffn-layer-reuse --rows 1` with the frozen
M02 model/fixture/capture and release settings. It compares the same two
original layer pipelines grouped versus alternating for pairs (0,17),
(17,35), (35,0), with separate scratch and unchanged kernels. It rejects
four-row or combined experimental modes instead of changing streams.
Use ten iterations and five initial trials; 10/20 fresh trials remain
available for uncertain results. No model generation or defaults change.

The [first measured checkpoint](experiments/2026-10-26-maxwell-layer-reuse-screen.md)
finds all stage confidence intervals within +/-0.45%, closing a material
tested temporal-order bias under the fixed +/-3% diagnostic threshold.
It is not a DRAM measurement, a true 36-layer forward or permission to
retune closed candidates.

### Diagnostic Library Read Reference

The [fixed public-CUB reference](experiments/2026-10-26-maxwell-cub-read-screen.md)
is closed: identical modular column checksums regress one-row throughput
22.87-23.01%, missing its 10% read-cost gate. Implementation `4c1680b` and
isolated-header fix `dfb6451` remain in history; results were committed in
`f058d4a` before removing the adapter, scratch API, test selectors and CUB
compilation from the active build. There is no current `--ffn-weight-read-cub`
flag or inference consumer. Reproduce at the documented historical commit,
not by passing the removed flag to a current build.

The [CUB license notice](third_party/cub-LICENSE.txt) remains with the archived
assembly. All 135 original kernel entries were verified unchanged before
measurement; global header flags that altered six RoPE entries were never
timed. No policy/geometry rescue or measured DRAM/occupancy claim follows.

### Explicit Runtime Qualification

The existing `model_quality` runner accepts independent
`--q8-runtime-candidate native|packed-f32-256|packed-f32-128|split-f32x4-128|q8-1-128|q8-1-256`
and `--q8-runtime-prefill sequential|native-packed|dequant-sgemm` selections.
Both models use the qualified prepared down/vocabulary configuration, dense
KV/context 1024 and the same source GGUF. The historical reference uses native FFN
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

For incremental split-plane qualification, both tools also accept
`--q8-runtime-control qualified-packed128-sgemm`. This names the current
qualified FP32/128 decode plus SGEMM prefill, rather than the historical
`native-sequential` control. The default control remains historical for
reproducibility. Quality and HTTP must request the same control; mismatched
evidence is rejected. Reference-prefix replay exercises the control's packed
prefill too, not a sequential substitute.

```bash
target/release/examples/model_quality \
  --reference-model "$MODEL" --model "$MODEL" --variant-id split-f32x4 \
  --q8-runtime-candidate split-f32x4-128 --q8-runtime-prefill dequant-sgemm \
  --q8-runtime-control qualified-packed128-sgemm \
  --split all --output /tmp/new-split-f32x4-quality.jsonl
```

The split candidate may add only gate/up cache entries. Shared validation
pins every prepared down/vocabulary key, byte count and launch configuration,
checks model/layout identity and total accounting, and rejects duplicates or
extra projection families. Reports preserve both full cache inventories and
setup costs. Each inventory must remain unchanged during evaluation. Actual
browser counters must cover every selected FFN layer forward plus all 252
dequantizations/SGEMMs for a Qwen 64-token prefix. This guards against caching
gate/up accidentally changing the already-qualified prompt strategy.

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

`M40LLM_Q8_FFN_KERNEL=native|packed-f32|split-f32x4-128|q8-1` is captured once by
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

The qualifying [split-plane screen](experiments/2026-10-25-maxwell-split-ffn-screen.md)
also has an opt-in runtime consumer, `Q8FfnConfig::split_f32x4()` or
`M40LLM_Q8_FFN_KERNEL=split-f32x4-128`. Its only geometry is 128 threads,
four FP32 input values per task, original per-weight scaling and FP32
accumulation. This is not Q8_1 activation quantization. It requires
`M40LLM_Q8_EXECUTION_WEIGHTS=split` and a family list containing `gate-up`.
Use `gate-up,down,vocabulary` to retain the qualified prepared down/head.
Missing preparation or insufficient budget is an error, not a native fallback.

The model's existing execution cache owns separate original FP16 scale and
signed-byte planes. Packing/upload occurs once at load, releases each temporary
host copy, and is charged to existing preparation/budget reporting. The Qwen
36-layer gate/up planes add 1,724,645,376 bytes to the down/vocabulary control;
that is a layout calculation, not a measured runtime memory result. Decode
borrows these owners without new allocation, transfer or synchronization.
Explicit `dequant-sgemm` prefill still converts the original Q8 tensor using
one bounded temporary buffer, even when gate/up is cached. Only the final
prompt token uses the new fused consumer. No defaults change; microbenchmark
qualification alone does not establish full-model token equality or HTTP speed.

Explicit experimental requests require compiled CUDA kernels, sm_52, uploaded
Q8_0 gate/up tensors in every standard layer and complete 32-value input
blocks. Unsupported requests fail with a native-mode remedy rather than
silently benchmarking another implementation. Experimental graph capture,
batched decode, frozen input capture and model-changing research combinations
are rejected. This investigation adds no server scheduling behavior.

API migration: in-memory `LoadedModel` construction now supplies
`q8_ffn: Q8FfnConfig::native()`. Direct callers of the unsafe fused-FFN
wrapper supply a final scratch argument: `None` for native/packed/split FP32, or
an owned `Q8FfnScratch` view for Q8_1. Workspace allocation/estimation also
takes the FFN configuration. Configure before creating workspace/session
state; do not replace model components while sessions reference them.

### Bounded Split-Plane Column-Reuse Follow-Up

After [qualified split-runtime profiling](experiments/2026-10-25-maxwell-split-stage-profile.md),
the shared FFN benchmark accepts `--ffn-f32-split-two-columns`. It compares
one fixed two-column/four-value/128-thread kernel with the current qualified
one-column split kernel, not packed/warp128. Both arms retain the same
existing gate/up planes and prepared down/vocabulary. There is no new layout,
activation quantization, preparation memory or generation configuration.

`CudaContext::ffn_q8_split_f32_two_columns_async` uses the same pointer,
alignment and lifetime contract as the one-column API. It enqueues on decode
without allocation, transfer or synchronization, and optionally captures
gate/up through the same kernel. Odd hidden tails skip only the missing
column, not the block-wide barrier. The original one-column kernel body
and C ABI remain intact. Tests require gate/up/SwiGLU bit equality with the
current control, CPU tolerance, guards, immutable planes and scratch reuse.
No runtime selection is added unless the complete-FFN screen qualifies.
The [completed screen](experiments/2026-10-25-maxwell-split-cols2-screen.md)
regresses 5.85-6.01% at one row despite bitwise parity. It remains
benchmark-only; do not widen the column search or enable it in generation.

### Bounded Joint Gate/Up Packing Follow-Up

The [instruction audit](experiments/2026-10-25-maxwell-joint-ffn-audit.md)
motivates `--ffn-f32-joint` in the same shared FFN benchmark. It compares one
fixed one-column/four-value/128-thread consumer of paired gate/up payloads
and scales against the qualified separate-plane one-column control. Both
retain original FP32 activation and per-weight arithmetic. No generation
configuration or runtime default selects this benchmark candidate.

`HostQ8FfnPair::pack` preserves both source identities and every stored bit;
`DeviceQ8FfnPair` owns two RAII buffers. Payload records are eight bytes,
gate's four signed bytes followed by up's four; scale records are four
bytes containing the original two FP16 encodings. The unsafe enqueue API
requires 16-byte-aligned FP32 input, eight-byte-aligned paired payloads,
four-byte-aligned scales/outputs, ordered producers and live spans through
decode completion. It allocates, copies and synchronizes nothing per call.

Benchmark setup retains both the control and joint layouts for both arms.
Report the joint pair and combined preparation bytes separately; each Qwen
pair is 47,906,816 bytes, so combined preparation is 95,813,632 bytes. Reject
combined preparation above 128 MiB before packing the joint pair. CPU/bitwise,
guard, cleanup and complete-FFN timing gates precede any runtime work.
The [completed screen](experiments/2026-10-25-maxwell-joint-ffn-screen.md)
and fresh twenty-pair confirmation show 4.66-4.74% complete one-row gains,
below the 5% gate despite bitwise parity. It remains benchmark-only; no
runtime selector, threshold relaxation or layout/geometry rescue follows.

### Bounded Fixed-Width Address Specialization

The independent [address/loop audit](experiments/2026-10-25-maxwell-fixed-k-ffn-hypothesis.md)
motivates `--ffn-f32-fixed-k` in the shared FFN benchmark. It exposes Qwen's
K=2048 as a compile-time constant, retaining original separate planes,
one-column/128-thread/four-value mapping, FP32 arithmetic and reduction.
Other widths fail explicitly, before allocation or launch. It does not
combine the below-gate joint layout or select a new generation runtime.

`CudaContext::ffn_q8_split_f32_k2048_async` has the same unsafe span,
alignment, lifetime and optional capture contract as the dynamic split API,
with K exactly 2048. Both arms reuse the same prepared pair and immutable
down/vocabulary. Existing numerical, micro, quality and HTTP gates apply.
The [completed screen](experiments/2026-10-25-maxwell-fixed-k-ffn-screen.md)
preserves bits but gains only 1.24-1.32% complete one-row throughput, below
5%. Keep it benchmark-only without another unroll/width/geometry search.

### Bounded Native QKV Reduction Probe

`q8_execution_bench --qkv-f32-warp --ffn-inputs ... --ffn-capture-report ...`
compares the original raw-Q8 split4 fused QKV with one fixed 256-thread
exact-tree warp finish. It reuses original Q/K/V weights and biases, with
no execution-weight preparation for QKV or generation selector. See the
[pre-implementation audit and gates](experiments/2026-10-25-maxwell-qkv-warp-hypothesis.md).

The mode uses frozen actual normalized FFN vectors as projection surrogates,
not captured attention-norm inputs. Rows 1/4 are one or four distinct
single-token requests, one fused launch per row, not a batched GEMM.
Reports include source identities, CPU/bitwise checks, guarded immutable
spans, actual enqueue-operation counters and paired complete-QKV latency.
Whole-model actual-token quality and original HTTP gates remain mandatory.

`CudaContext::qkv_f32xq8_0_gguf_warp_async` enqueues raw Q8_0 on an explicit
stream with caller-owned spans. It preserves split4 math and optional FP32
biases while removing repeated reduction barriers. It neither allocates,
transfers nor synchronizes. The unsafe contract requires aligned, disjoint
live outputs and properly ordered input/weight/bias producers.
The [completed screen](experiments/2026-10-25-maxwell-qkv-warp-screen.md)
preserves bits and improves complete one-request QKV throughput 27.47-27.76%.
It passes the stage gate but not the prespecified 3% standalone E2E ceiling;
retain it benchmark-only pending a separately qualified combined candidate.

### Bounded Native Output Reduction Probe

`q8_execution_bench --output-f32-warp --qkv-warp-report ...` uses the same
projection probe and frozen FFN input surrogates, with real output weights.
The peer must match the prespecified QKV report hash and model/capture
provenance. One original four-column/split4/256-thread kernel is compared
with its original-tree warp finish; geometry and storage are unchanged.
`CudaContext::gemm_f32xq8_0_gguf_coltile4_warp_async` enqueues on an explicit
stream without allocation, copies or synchronization. Caller-owned raw
weights and aligned FP32 spans must remain live and properly ordered.

The [completed output screen](experiments/2026-10-25-maxwell-output-warp-screen.md)
preserves control bits and gains **30.42-31.57%** at one request. The
prespecified combined QKV/output optimistic ceiling is **4.68-4.72%**,
clearing feasibility but not measured runtime qualification. Next use an
immutable opt-in combined choice with unchanged split-FFN/SGEMM control,
then actual-512/frozen quality and the original HTTP/Ollama gates.
Neither projection primitive selects a new generation default.

The combined runtime is opt-in with `M40LLM_Q8_ATTN_PROJECTIONS=warp-256`.
Generic model loading captures that choice once; explicit backend/reference
constructors stay native unless passed the new typed backend. It affects
only one-row fused QKV and attention-output projection, always on decode.
Multi-row prefill, prepared gate/up/down/vocabulary and SGEMM are unchanged.
Raw QKV/output device identities and shapes are validated at loading and
dispatch; prepared QKV/output families, non-Q8 models and unsupported CUDA
builds fail explicitly rather than falling back. Use `native` to disable.
Model-local state has no per-token environment lookup, new device allocation
or activation quantization. In-memory reference fixtures initialize native
state and must not replace a model's state during workspace/session use.
The [actual-model quality checkpoint](experiments/2026-10-25-maxwell-projection-runtime-quality.md)
now preserves all 512 actual IDs, all twelve task outputs and sampled full
logits against current split-FFN/SGEMM. All 109 existing prepared tensors
are identical. The subsequent
[incremental HTTP gate](experiments/2026-10-25-maxwell-projection-incremental-http.md)
passes at 47.323 versus 44.950 tok/s, +5.279%, with original quality and
timing rules intact. The subsequent
[corrected fresh official comparison](experiments/2026-10-25-maxwell-projection-ollama-http.md)
measures 47.379 versus 43.097 tok/s, +9.937%, with actual IDs and warp
dispatch verified. Both original goals remain unmet: 53.06 and 56.026
HTTP tok/s for 1.30x the fresh official median. Defaults remain unchanged.
The subsequent [selected-runtime profile](experiments/2026-10-25-maxwell-projection-stage-profile.md)
preserves actual IDs with 1.01% instrumentation overhead and locates 39.43%
gate/up, 23.33% prepared down and 3.80% attention. Its next bounded hypothesis
tests the existing prepared projection's reduction only, retaining qualified
controls, original math/layout/geometry and all promotion gates.
The resulting [prepared projection screen](experiments/2026-10-25-maxwell-prepared-warp-screen.md)
preserves bits but misses the 5% complete-FFN gate at 2.44-2.60%. Vocabulary
gains 6.54%, with only 0.474% optimistic standalone E2E scope. Keep both
benchmark-only; no runtime integration or geometry expansion follows.
The independent [unfused FFN screen](experiments/2026-10-25-maxwell-unfused-ffn-screen.md)
also preserves bits but regresses complete one-row throughput 14.14-14.27%.
Keep `286cb9a` benchmark-only, without a column/thread rescue. The pinned
Ollama unfused dispatch does not qualify unfusion of our different arithmetic.
The subsequent [L2-only weight-cache screen](experiments/2026-10-25-maxwell-cache-ffn-screen.md)
also preserves bits with distinct compiled loads but gains only 0.095-0.184%
complete one-row throughput. Keep `0cff8ab` benchmark-only without a cache
operator or geometry rescue. No defaults or authoritative HTTP claims change.
The [exact byte-conversion screen](experiments/2026-10-25-maxwell-byte-ffn-screen.md)
also misses the 5% gate at 0.138-0.295% complete one-row throughput gain,
despite bit equality and unchanged 32-register usage. Keep `d5a3350`
benchmark-only without conversion tuning. Actual compiled QKV/output
kernels still have scalar weight/activation loads, an independent bounded
load-vectorization opportunity rather than a reason to revive failed FFN
experiments.

The [packed QKV/output load screens](experiments/2026-10-25-maxwell-packed-projection-screen.md)
now pass against qualified scalar-load warp controls: +46.87-51.97% QKV and
+40.02-46.78% output throughput, unchanged bits and 32/40 registers. Both
independent stage gates and 5.23-5.33% combined optimistic scope pass.
Runtime integration `c264c89` and shared-tool support `43ea00a` now expose
the explicit model-local `packed-warp-256` backend. Its
[actual-model quality gate](experiments/2026-10-25-maxwell-packed-projection-runtime-quality.md)
preserves 512 actual IDs, twelve frozen outputs, sampled full logits and all
109 prepared identities. The [incremental HTTP gate](experiments/2026-10-25-maxwell-packed-projection-incremental-http.md)
passes twenty fresh confirmation pairs at 49.082 versus 47.186 tok/s,
+4.018%. The [fresh official comparison](experiments/2026-10-25-maxwell-packed-projection-ollama-http.md)
then measures 49.146 versus 43.106 HTTP tok/s, +14.014%. No prepared weight
delta or runtime default change is introduced. Both original goals remain
unmet: 53.06 and 56.037 for 1.30x that fresh official median. Next re-profile
the selected configuration rather than infer remaining stage shares from
the older scalar-load runtime.

The shared `model_quality` and `compare_ollama` tools accept
`--q8-runtime-attention warp-256` together with `split-f32x4-128` and
`dequant-sgemm`. Use `--q8-runtime-control qualified-split-sgemm` for this
incremental comparison. This control includes the existing gate/up cache,
so both models must preserve every gate/up/down/vocabulary key and payload;
there is no new preparation delta. The older native/sequential and
packed128/SGEMM controls remain explicitly named for historical work.
Browser quality records carry the actual model projection backend and
one-row launch counts. HTTP preflight rejects missing/wrong dispatch or
backend evidence; merely relabeling a previous quality report is not enough.

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

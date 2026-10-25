# Bounded Q8 Prefill SGEMM Experiment

This is the independent M07 branch of the
[Maxwell Q8 execution investigation](maxwell_q8_execution.md). Decode and
prepared down/vocabulary remain the control. There is no default change or
prefill speed claim.

## Temporary Weight Contract

`Q8PrefillWeightShape` checks one logical GGUF `[input_dim, output_dim]`
Q8_0 tensor with complete 32-value input blocks. The FP32 payload cannot
exceed 128 MiB; zero dimensions, incomplete blocks, arithmetic overflow and
larger projections fail. Both Qwen FFN orientations need 90,177,536 bytes,
exactly 86 MiB. The vocabulary matrix exceeds the budget and is rejected.

`CudaContext::dequantize_q8_0_to_f32_colmajor_async` consumes native 34-byte
blocks and writes `output[input * output_dim + column]`, column-major
`[output_dim, input_dim]`. This is the N-leading layout used by the existing
async `gemm_f32xf32_f32_stream_async` wrapper, not native GGUF linear order.
The FP16 scale is widened, then multiplied by the signed eight-bit value.
There is no additional quantization or FP16 output rounding.

Conversion enqueues one kernel on prefill into caller-owned temporary storage.
It adds no allocation/free, host transfer, cuBLAS call or synchronization.
The unsafe API requires disjoint spans, two-byte source/four-byte output
alignment, sufficient reported capacity and explicit source-producer ordering.
The storage must survive the last SGEMM that reads it. Capacity/alignment/null
errors fail before launch. It is not a persistent FP32 tensor cache.

The fixed Maxwell kernel uses a 32x8-thread block and a bank-padded 32x33
FP32 shared tile to transpose contiguous Q8 reads into contiguous FP32 writes.
Release NVCC 12.8.93 sm_52 inspection reports 30 registers, 4,224 bytes shared,
zero stack and zero local bytes. No Tensor Core, DP4A, graph or scheduling
feature is needed. This kernel is independently written; no new upstream
code is copied. Static resources are not measured occupancy or latency.

## Verification

`tests/q8_prefill_dequant.rs` has a CPU bounds test and explicit hardware
checks. Release M40 results with cuBLAS enabled pass all four tests:

- Every converted FP32 bit matches independent CPU decoding for tiny/tail
  shapes and both real Qwen FFN orientations. Signed bytes, zero scales,
  two-byte bases, immutable source bytes and output guards are covered.
- Repeated conversion reuses the allocation and records one launch with no
  allocation, copy or synchronization inside enqueue.
- A queued decode-stream source copy, explicit prefill wait, conversion and
  SGEMM match independent CPU projections for genuine rows 4/64 and distinct
  inputs. The pair records one conversion launch and one cuBLAS call.
- Invalid buffers/capacity fail before device work; tracked allocations return
  to their initial value on cleanup.

The bounds test passes in the warning-as-error non-CUDA build. CUDA/server
all-target Clippy and formatting pass. Run hardware checks serially with
escalation after sourcing `scripts/dev-env.sh`:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --release --features cuda --test q8_prefill_dequant \
  -- --include-ignored --nocapture --test-threads=1
```

With cuBLAS disabled, both conversion hardware checks and the CPU shape check
pass. The SGEMM-specific test is compile-gated out rather than reporting a
successful no-cuBLAS skip.

## Runtime Selection and Ownership

`M40LLM_Q8_PREFILL_BACKEND=native|dequant-sgemm` is captured at model load.
Native remains the default. The experimental backend requires compiled
CUDA/cuBLAS, sm_52, head64/head128, uploaded Q8_0 transformer projections and
the 128 MiB per-weight bound. Unsupported requests give a native-mode remedy,
not a silent algorithm change.

Use the existing `M40LLM_PREFILL_CHUNK_SIZE` bound to admit a packed prompt;
for the canonical 65-token prompt, use 128 to cover the whole prompt while
prefilling only its 64-token prefix. An explicitly selected SGEMM backend
errors if the requested dense/exact packed prefix is disabled or over bound.
Errors during experimental prefill propagate rather than falling back to
sequential. Native retains its existing fallback policy.

Library callers can select prefill per request through
`GenerateOptions::prompt_prefill`: `PromptPrefill::Sequential`,
`PromptPrefill::PackedPrefix { max_prompt_tokens: 128 }`, or the default
`PromptPrefill::Environment`. Explicit choices ignore the ambient prefill
bound. Explicit packed requests fail if unsupported, over bound or unable to
complete; they never silently become sequential. This lets paired experiments
alternate strategies without changing process-wide environment variables.
`AppState::prompt_prefill` applies the same choice to serialized buffered HTTP
generation. An explicit override rejects streaming and scheduler routes, which
do not implement this request contract. Ordinary CLI/server defaults remain
environment-driven.

API migration: full `GenerateOptions` and `AppState` struct literals must
initialize `prompt_prefill`; use `PromptPrefill::Environment` to preserve
existing behavior. Constructors and `GenerateOptions::default()` already do.

One RAII temporary weight buffer belongs to the forward workspace. It is
allocated before the packed forward, reused for all seven projections in each
layer and ordered with SGEMM on prefill. Existing Q/K/V bias, split-half RoPE,
absolute-position append, attention/masks and stream waits remain in place.
The vocabulary and ordinary one-row decode do not use the buffer. Prepared
down/vocabulary stay resident and unchanged; the prefill temporary does not
become a full-model FP32 cache. One-row workspace setup releases it after
packed prefill has drained. Allocation/conversion cost belongs in prompt/E2E
timing, not outside the measured request.

`DecodeSession::prefill_packed_prefix_ids` leaves the final prompt token on
the normal logits or GPU-greedy path. The independent native-path fix in
`eb39200` addresses an older integration gap: GPU-greedy generation previously
ignored an explicit packed-prefill bound. Its regression
[fails before](experiments/2026-10-25-greedy-prefill-before.txt) and
[passes after](experiments/2026-10-25-greedy-prefill-after.txt) that fix without
the SGEMM runtime integration present.
Native sequential with bound zero is unchanged. No implicit packed admission
is added to the native GPU-greedy path without an explicit bound. Generation
reports `packed-prefix-dequant-sgemm` and the actual
`q8_prefill_temporary_weight_bytes`, retained after decode releases scratch.

`from_gguf_with_q8_backends` accepts independent execution-weight, FFN and
prefill choices. The explicit existing weight-cache constructor keeps both
FFN and prefill native regardless of ambient experiment flags. In-memory
models initialize `q8_prefill: Q8PrefillBackend::Native`; direct workspace
callers pass zero prefill weight bytes for decode. Multi-request prefill,
experimental graphs and model-changing research combinations are rejected.

## Runtime Correctness Checkpoint

Release M40 runtime tests compare sequential, native packed and SGEMM packed
prefill for two-layer head64 and Qwen-shaped head128 GQA fixtures. They retain
the existing 2e-3 absolute tolerance for final logits and decoded FP16 K/V.
Device sequence lengths and K/V at every tested absolute position agree.
The Qwen case includes Q/K/V biases and split-half RoPE. There is exactly one
additional allocation, fourteen conversions and fourteen cuBLAS calls for
two layers, not one weight allocation per projection. Repeated prefix calls
reuse the same pointer/capacity. Subsequent decode allocates/frees nothing
after one-row setup, performs no temporary conversion and leaves prepared
cache statistics unchanged.

Generation tests verify actual conversion dispatch with both CPU and GPU
greedy sampling, identical fixture token IDs, actual temporary capacity and
clear rejection when the prefix bound is disabled. The new synchronous
`KVCache::debug_sequence_length` exposes device metadata only for diagnostics;
it is not part of warm decode. Warning-as-error non-CUDA tests, CUDA/server
all-target Clippy, attention/forward/prefill/ownership regressions and
no-cuBLAS rejection checks pass. These are fixture correctness checks, not
full-model continuation equality.

### Real Qwen Dispatch Smoke

The original Qwen2.5-3B Q8_0 model, prepared down/vocabulary control and dense
KV/context 1024 were run serially with GPU-greedy enabled, eight maximum
generated tokens and the prompt `What is 2+2? Answer with one digit.`.
The rendered prompt has twenty tokens; packed runs process nineteen prefix
rows. Launch logging is enabled, so this is not a throughput measurement.

| Path | Output | Temporary conversions / SGEMM calls |
| --- | --- | --- |
| Parent `5d7472d`, native sequential | `2` | 0 / 0 |
| Runtime integration, native sequential | `2` | 0 / 0 |
| Runtime integration, native packed | `4` | 0 / 0 |
| Runtime integration, SGEMM packed | `4` | 252 / 252 |

The sequential result predates this integration, verified by compiling the
parent source in an isolated temporary snapshot and rerunning the identical
command. This is not evidence of a new sequential regression. Native packed
and SGEMM packed agree on this answer, but they do **not** preserve the
sequential answer. Do not call the smoke exact-output qualification, a quality
improvement of the inference engine, or evidence of a speedup. The fixed
512-token benchmark equality and dense-valid task gates remain mandatory.

The [diagnostic JSONL and linked compressed launch logs](experiments/2026-10-25-maxwell-q8-prefill-smoke.jsonl)
preserve the original model hash, execution configuration, runtime binary
hash and source-patch digest. The parent binary hash was not retained; its
source revision, [release build log](experiments/2026-10-25-maxwell-q8-prefill-parent-build.txt),
output and full launch log are recorded. The parent used `git archive
5d7472d Cargo.toml Cargo.lock build.rs src cuda benches tests` into a temporary
source tree, followed by an offline locked release build. Its initially shared
target directory caused a later stale-library compile failure, not an inference
result; package outputs were cleared and all integrated regressions rerun from
the current source. Follow the [separate-target rule](controlled_benchmark.md#build-isolation)
for future cross-revision checks. The GPU process list was empty before and
after each serial smoke group. No Ollama inference overlapped these runs.

M08 must compare sequential, existing packed and temporary-SGEMM prefill
independently before combining with FFN candidates. Include every conversion
and allocation in prompt timing. Do not materialize vocabulary, promote from
microbenchmarks, or weaken exact-output acceptance to accommodate drift.

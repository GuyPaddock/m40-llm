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

## Remaining Integration

The converter is not yet selected by runtime prefill. M07 still needs a
model-local opt-in configuration, one RAII temporary buffer reused projection
by projection, complete Qwen bias/RoPE/KV parity and unchanged decode dispatch.
Include every conversion in per-prompt timings. Do not retain a full FP32
model, materialize the vocabulary, silently fall back or infer an E2E gain
from these correctness checks. M08 must compare sequential, existing packed
and temporary-SGEMM prefill independently before combining with FFN candidates.

# Maxwell FFN Runtime Integration

M06 connects the candidates that passed the
[M03/M04 complete-FFN screen](2026-10-25-maxwell-q81-ffn-screen.md)
to model-local, opt-in single-row runtime dispatch. This checkpoint adds no
performance measurement or default promotion. Whole-model 512-token equality,
Q8_1 task/loss evaluation and HTTP qualification remain M08 work.

## Configuration and Ownership

`M40LLM_Q8_FFN_KERNEL` defaults to `native`. `packed-f32` chooses the
256-thread FP32 packed/warp candidate; `q8-1` chooses the 128-thread numerical
candidate. Typed constructors also expose the other screened geometry.
Loading captures the choice once, independently of execution-weight layouts.
The explicit existing weight-cache constructor retains native FFN even if
the environment later changes.

Prepared down/vocabulary are unchanged. Single-row forward dispatches the
candidate; packed-prefix projection rows remain native and only the final
prompt token uses the experimental FFN. Q8_1 activation storage belongs to
the RAII forward workspace, is shared across layers under its existing lease,
and is reused across decode calls. Capacity is 72 bytes for the D=64 test
fixture and 2304 bytes for Qwen D=2048. Packed FP32 needs no activation buffer.

Tests check the actual operation counters, not environment strings alone.
No allocation/free, host transfer or stream synchronization occurs inside
candidate enqueue. Q8_1 adds one quantization launch per layer. Both candidates
use decode, preserving waits for other projections that still use prefill.
Unsupported models, missing/short/aliased scratch, graphs and batched decode
fail rather than silently switching implementations. Frozen activation
capture and model-changing research retain native-only guards.

## Correctness Results

The release M40 runtime checks cover:

- Model-local selection despite subsequent environment changes; native frozen
  references remain native and prepared cache statistics remain unchanged.
- Independent CPU-reference FFN results for packed FP32 and reconstructed
  Q8_1 activations at both 128/256-thread geometries.
- Two-layer, three-step sessions under both Q8 producer-stream schedules.
  Packed256 preserves every final-logit bit and every checked K/V byte for
  five absolute positions in both layers. Q8_1 produces finite logits but is
  not required or claimed to equal native arithmetic.
- Stable Q8_1 pointer/capacity and zero decode allocation/free across tokens.
- Packed-prefix followed by the final prompt step, with exactly two candidate
  FFN launches for the two layers rather than one per prefix token. Packed256
  preserves final-logit and KV bits against native packed-prefix construction.
- Rejected configurations enqueue no CUDA work; tracked device payload returns
  to its initial value after successful and rejected model construction.

The strengthened mixed-flag test exposed an older logits stream bug, fixed
separately in `d211726`. An unrelated F16 stream flag had caused Q8 logits to
report decode while its projection ran on prefill. The isolated regression
failed before the fix and passes after it for both Q8 schedules and native /
prepared vocabulary. The qualified control already uses Q8 decode throughout;
its schedule does not change. These fixture results are not whole-model token
qualification or evidence that the earlier controlled timings were wrong.

## Validation

All commands source `scripts/dev-env.sh`. Hardware tests run with escalation,
serially, without another GPU compute process. Release CUDA regressions pass
with `M40LLM_ENABLE_NVCC=1` and both `M40LLM_ENABLE_CUBLAS=1` and `0`:

```bash
cargo test --release --features cuda \
  --test q8_ffn_runtime --test q8_logits_stream \
  --test q8_ffn_fp32 --test q8_ffn_q81 --test q8_activation \
  --test q8_native_mlp_cache --test q8_family_selection \
  --test ffn_capture_cuda --test proj_wrappers --test mlp_wrappers \
  --test forward_with_layer_smoke --test attention_parity_cuda_grid \
  --test cuda_transfer_order \
  -- --include-ignored --nocapture --test-threads=1
```

The full warning-as-error non-CUDA suite passes:

```bash
RUSTFLAGS='-D warnings' cargo test --no-default-features --locked
```

The warning-as-error CUDA/server all-target check with runtime NVCC uploads
disabled also passes. `M40LLM_ENABLE_NVCC=0` is a runtime flag, not proof of
an NVCC-free CUDA build; `build.rs` still detects and compiles with NVCC.
Configuration-only tests pass separately in that build. Formatting and
CUDA/server all-target Clippy pass. The shared complete-FFN benchmark's nine
regression tests pass in release with cuBLAS enabled. Hardware-only tests use
explicit ignored markers and were requested with `--include-ignored`, not
counted as passing from an unexecuted default test run.

## API Migration

In-memory model literals explicitly initialize `q8_ffn` to native. The unsafe
fused-FFN wrapper gains an optional scratch argument; existing native callers
pass `None`. Forward workspace sizing/allocation accepts the same model-local
configuration. No compatibility aliases or process-global selection are added.

Next: independent bounded prefill SGEMM, then full-model logits, actual
512-token comparisons, Q8_1 quality and controlled E2E pairs. Both original
exact-output throughput targets remain unmet.

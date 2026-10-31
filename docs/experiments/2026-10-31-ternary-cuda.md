# Ternary screening CUDA checkpoint

`DeviceTernaryScreen` implements the two kernels fixed in the
[screening protocol](2026-10-31-ternary-screen-protocol.md). The first applies
the signed normalized 128-coordinate Walsh transform and selects 80 absolute
coordinates, with ties resolved by index. The second reads coordinate-major
gate/up ternary codes, accumulates FP32 projections, applies SwiGLU and weights
its magnitude by the supplied down-column norm. It writes four planes for
validation: gate, up, estimated activation and score.

The research owner retains both sketches, signs and norms. Rotation and
selection scratch has guards and is reused. Calls accept explicitly ordered
streams and disjoint input/output spans. Runtime dispatch does not use this
API. No channel top-k or recovery is implemented here.

Both hardware tests pass with cuBLAS enabled and disabled. Shapes are
128 by 5, 256 by 35, and 2048 by 11008. They check independent FP64 rotation,
coordinate order, all four output planes, changed input at the same address,
partial channel tiles, both producer/consumer stream directions, deterministic
reuse, guards, immutable data, and complete tracked allocation cleanup. Invalid
norms, signs, shapes, pointers and aliases fail before allocation or launch as
applicable. Each valid enqueue records two kernels and zero allocations,
copies, waits or synchronizations inside its boundary.

Regression checks pass: selected Q4 FFN 2, compound runtime/forward/ownership
3, native Q4 projections 2, and attention grid 11. The warning-as-error
non-CUDA suite, CUDA/server all-target Clippy and formatting pass. The known
tiny legacy forward fixture limitation remains documented in the
[earlier checkpoint](2026-10-30-q4-selected-cuda.md); the affected compound
forward tests use an explicit dense reference and pass.

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/channel-selection/target
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --test ternary_screen_cuda \
  -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=0 cargo test --features cuda --test ternary_screen_cuda \
  -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test q4_selected_ffn --test q8_compound_runtime \
  -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --test gemm_mixed q4_0 \
  -- --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test attention_parity_cuda_grid -- --nocapture --test-threads=1
```

Raw logs use the prefix `2026-10-31-ternary-cuda-` and suffix `.log.gz` beside
this report. These are correctness checks, with no performance or model-quality
claim. Next connect the kernels to the retained source-bound activation
capture and measure complete screening cost before channel top-k or recovery.

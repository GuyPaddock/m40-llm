# CUDA parity grid and KV cache layout usage

This note summarizes how to:
- configure and run the CUDA attention parity grid on Tesla M40 (sm_52)
- enable/disable cuBLAS and verify parity across both paths
- explicitly control KV cache head layout in tests/examples

## Environment prerequisites

- Build with the `cuda` feature: `cargo test --features=cuda`
- NVCC available to compile kernels for sm_52; build.rs will expose `cfg(nvcc)`
- Optional cuBLAS: enable only when both the cuBLAS header and shared library are present, and when the environment variable `M40LLM_ENABLE_CUBLAS=1` is set.

Notes:
- We compile for `sm_52` and embed `compute_52` PTX for forward compatibility.
- The project supports runtime auto-selection of Tesla M40 when `CudaContext::new(-1)` is used.

## Running the attention CUDA parity grid

The parity grid compares the CUDA attention implementation against a CPU reference over a small coverage grid of shapes.

- Fallback (no cuBLAS for GEMMs used in projections):
  ```bash
  M40LLM_ENABLE_CUBLAS=0 cargo test --features=cuda -- --nocapture tests/attention_parity_cuda_grid.rs
  ```
- With cuBLAS enabled (when available):
  ```bash
  M40LLM_ENABLE_CUBLAS=1 cargo test --features=cuda -- --nocapture tests/attention_parity_cuda_grid.rs
  ```

The test `attention_last_token_cuda_parity_grid` will print any mismatches and fail on exceedances. The current tolerance target is ~1e-3.

## Wrappers parity tests for projections and MLP

Targeted parity smoke tests exist for Q/K/V projections, the output projection, and MLP gates/down-proj. Run them with:

```bash
# Fallback kernels
M40LLM_ENABLE_CUBLAS=0 cargo test --features=cuda -- --nocapture tests/proj_wrappers.rs
M40LLM_ENABLE_CUBLAS=0 cargo test --features=cuda -- --nocapture tests/mlp_wrappers.rs

# With cuBLAS (when available)
M40LLM_ENABLE_CUBLAS=1 cargo test --features=cuda -- --nocapture tests/proj_wrappers.rs
M40LLM_ENABLE_CUBLAS=1 cargo test --features=cuda -- --nocapture tests/mlp_wrappers.rs
```

These tests validate row‑major conventions and numeric parity to within ~1e-3 against a CPU reference.

## Explicit KV cache layout in tests/examples

To ensure the KV cache layout matches model dimensions (num_heads × head_dim = d_model), use the explicit layout allocation API available on `LoadedModel` from `infer`:

```rust
use m40_llm::infer::LoadedModel;

// Example values
let max_seq_len: u32 = 8;
let max_batch_size: u32 = 1;
let num_heads: u32 = 4;
let head_dim: u32 = 16; // implies d_model = 64

let mut lm = LoadedModel::from_gguf(gg, weights, -1)?; // auto-select device
lm.allocate_kv_cache_with_layout(max_seq_len, max_batch_size, num_heads, head_dim)?;
```

This avoids relying on implicit defaults and reduces the chance of layout mismatches during attention. The forward smoke test `tests/forward_with_layer_smoke.rs` demonstrates this usage in practice.

## Troubleshooting

- If the CUDA tests are skipped, verify you built with `--features=cuda` and that NVCC is available (build.rs emits `cfg(nvcc)` when detected).
- If cuBLAS paths are not active when expected, ensure both `cublas_v2.h` and a `libcublas.so` are present and set `M40LLM_ENABLE_CUBLAS=1`.
- For sm_52 devices like Tesla M40, you may see NVCC deprecation warnings about older architectures; these are documented as benign for this project’s purposes.

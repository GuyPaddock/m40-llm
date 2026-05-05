# Minimal Forward Path

Status: full-layer CUDA decode is functional for TinyLlama on Tesla M40. This
document is kept as a compact orientation note for the current one-token forward
path and its guardrails.

## What Runs On Device

- Embedding row load to an FP32 device buffer.
- Weighted RMSNorm, RoPE, residual adds, and SwiGLU/gated MLP activation.
- Q/K/V, output, MLP gate/up, MLP down, and lm_head projection GEMMs.
- FP32 K/V projection append into the FP16 KV cache.
- Last-token attention against the KV cache.
- Reusable CUDA forward workspace for per-layer scratch and full-layer ping-pong buffers.

## Current Limits

- Batch size is still effectively one request for the optimized decode path.
- GGUF F16 projection weights use the dedicated GGUF-layout CUDA GEMM kernel; set
  `M40LLM_GEMM_LOG=1` to print backend selection.
- The attention GQA microbenchmark shows the current attention kernel is the next
  bottleneck to optimize.

## Guardrails

- `tests/forward_parity_toy.rs` validates toy forward parity against a CPU reference.
- `tests/forward_with_layer_smoke.rs` checks CUDA forward execution and workspace reuse.
- `tests/attention_parity_cuda_grid.rs` protects last-token attention and GQA behavior.
- `tests/server_smoke.rs` exercises the full-layer `/generate` server path.

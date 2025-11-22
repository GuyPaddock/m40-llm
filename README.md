# m40-llm

Tesla M40–optimized Rust + CUDA LLM server/runtime. FP16 weights, FP32 compute via cuBLAS. GGUF loader and stable C FFI (`m40llm_*`). Goal: be much faster than ollama on the M40.

## What it is
- Single-GPU server for Maxwell Tesla M40 (sm_52)
- FP16 storage / FP32 compute (cuBLAS/cuBLASLt as available)
- GGUF loader; C FFI symbols `m40llm_*` for embedding
- Small, explicit codebase focused on M40 performance

## Who it’s for
- M40 owners who want maximum throughput/low latency on this specific card
- Tinkerers/researchers who want Maxwell-specific hacks, not generic portability
- Users who find vLLM hard/unsupported on M40 and llama.cpp too slow there

## How it compares
- vs ollama: we compete head‑on for M40. Expect higher throughput/lower latency from Maxwell‑specific kernels/layouts, FP16‑storage/FP32‑compute, and decode‑path tricks (graphs, persistent kernel, warp micro‑batching).
- vs vLLM: excellent on modern GPUs but impractical on M40 (sm_52). m40‑llm is designed to be M40‑first and actually set up/run there.
- vs llama.cpp: very portable, but most speed paths target newer GPUs. On M40 it tends to run without its big speed tricks; m40‑llm focuses on sm_52‑specific performance instead of broad portability.

## Performance strategy on M40
- FP16 storage, FP32 compute tiles: load FP16 to shared, convert to FP32, compute in registers
- Tuned GEMM with cuBLAS/cuBLASLt; explicit row/col layouts; layout tests included
- CUDA Graphs + persistent decode kernel to minimize launch overhead
- Warp-level micro-batching (e.g., one warp per sequence) for decode
- Optimized KV cache: FP16 or INT8 per-head; contiguous per-head layout; pinned host staging
- Streams/Hyper‑Q: high‑priority decode stream, concurrent lower‑priority prefill
- Read‑only (`__ldg`) and texture caches for non-GEMM ops (norms, embeddings)


## Features
- `--features cuda` enables the GPU path
- NVCC auto-detect; stub build when missing (works with conda headers/libs)
- cuBLAS header gating; tests and GEMM enabled only when available

## Build
- Non-CUDA: `cargo build --no-default-features; cargo test --no-default-features`
- CUDA, no NVCC (conda headers/libs): `cargo build --features cuda; cargo test --features cuda`
- CUDA, with NVCC: `cargo build --features cuda; cargo test --features cuda`

## Tests
- Non-CUDA tests run by default
- CUDA tests require `--features cuda` and will run when NVCC/headers are present

## Server (feature = server)
```
cargo run \
  --no-default-features \
  --features server \
  -- run \
  --model path/to.gguf \
  --addr 0.0.0.0:58439
```

## Contributing
See `CONTRIBUTING.md` for guidelines.


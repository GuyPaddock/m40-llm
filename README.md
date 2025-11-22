# m40-llm

Rust + CUDA inference runtime for Tesla M40 (sm_52). FP16 weights, FP32 compute via cuBLAS. GGUF loader and stable C FFI (`m40llm_*`). Optional tiny HTTP server behind a feature flag.

What it is
- Target: Maxwell Tesla M40 (sm_52)
- FP16 storage / FP32 compute (cuBLAS)
- GGUF loader; C FFI symbols `m40llm_*` for embedding
- Minimal, tested paths (GEMM, KV cache) to start

How it differs
- Not ollama: no packaging/orchestration; low-level engine for M40
- Not vLLM: no tensor-core batching focus; correctness on Maxwell first
- Not llama.cpp: not a general runtime; narrow, sm_52-focused, small codebase

## Features
- --features cuda enables the GPU path
- Auto-detect NVCC; if missing, build uses a stub library (headers from conda supported)
- Auto-gate cuBLAS: if cublas_v2.h is found, link libcublas and enable GEMM; otherwise GEMM is a safe no-op

## Build
- Non-CUDA: cargo build --no-default-features; cargo test --no-default-features
- CUDA, no NVCC (conda headers/libs): cargo build --features cuda; cargo test --features cuda
- CUDA, with NVCC: cargo build --features cuda; cargo test --features cuda

Notes
- Target GPU: sm_52 (Tesla M40) via -gencode=arch=compute_52,code=sm_52
- NVCC and cuBLAS headers are detected by build.rs; tests that require them are skipped when unavailable

## Tests
- Non-CUDA tests run by default
- CUDA tests require --features cuda and will run only when NVCC/headers are present

## Server (feature = server)
- Minimal Axum-based server: cargo run --no-default-features --features server -- run --model path/to.gguf --addr 0.0.0.0:58439

## Contributing
See CONTRIBUTING.md for guidelines (Conventional Commits, hooks, CI matrix).


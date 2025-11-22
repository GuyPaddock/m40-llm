# m40-llm

A small, focused Rust + CUDA LLM inference runtime for the Tesla M40 (Maxwell, sm_52). It stores weights in FP16 and performs compute in FP32 via cuBLAS. The crate keeps a GGUF loader, exposes a C FFI with `m40llm_*` symbols for embedding into other apps, and optionally provides a tiny HTTP server behind a feature flag.

Goals
- Make an older Maxwell GPU (sm_52) useful for local inference
- FP16 storage / FP32 compute with cuBLAS as the first, well-tested path
- Maintain a simple GGUF loader and stable `m40llm_*` FFI for interoperability
- Keep non-CUDA builds green for development and CI; gate CUDA/NVCC paths cleanly
- Prefer correctness, explicit layouts, and tests over breadth of features; iterate from GEMM/KV cache upward
- Provide a minimal feature-gated server to experiment with model endpoints

How this differs from ollama, vLLM, and llama.cpp
- ollama: packaging and orchestration for many models and backends. m40-llm is a low-level, purpose-built engine for M40; no model zoo, no large packaging layer, and the C FFI is the primary integration surface.
- vLLM: high-throughput serving (PagedAttention, batching) aimed at modern GPUs with tensor cores. m40-llm targets Maxwell without tensor cores and focuses on correctness, layout tests, and a compact codebase rather than cluster-scale throughput.
- llama.cpp: a mature, portable runtime with wide model/backends support (CPU, Metal, CUDA). m40-llm is not a competitor; it’s a narrow, educational/experimentation runtime optimized for sm_52, keeping GGUF compatibility and stable FFI while staying small and explicit.

Build and test matrix — why run without CUDA or NVCC?
Even though the end goal is GPU inference on an M40, there are real reasons to build/test without CUDA or without NVCC:
- Developer ergonomics: contributors can work on CPUs/laptops without a CUDA toolchain. The crate compiles and tests core logic (parsing, server scaffolding, FFI shims) with `--no-default-features`.
- CI gating: the matrix validates both portable and CUDA builds. CUDA tests run only when NVCC and cuBLAS headers are available; otherwise they’re skipped to keep CI green.
- Packaging/bindings: consumers may depend on the crate for the GGUF loader or FFI types without needing GPU code at compile time.
- Cross-build environments: some builders have CUDA headers/libs (e.g., via conda) but not NVCC; the project supports that path and conditionally links cuBLAS when headers are present.

## Features
- CUDA feature flag (`--features cuda`) to enable GPU path
- NVCC auto-detection in build.rs; stub library used if CUDA feature is set but NVCC missing
- cuBLAS auto-gating: build.rs detects `cublas_v2.h` and defines `M40LLM_HAVE_CUBLAS` when present; GEMM falls back to no-op if headers unavailable
- Persistent cuBLAS handle in CUDA context (created/destroyed once; stream bound to prefill by default)

## Build

- Non-CUDA (portable):
  - cargo build --no-default-features
  - cargo test --no-default-features

- CUDA without NVCC (headers/libs from conda recommended):
  - Ensure CONDA_PREFIX points to an environment providing CUDA headers/libs (e.g., via mamba install -c conda-forge -c nvidia cuda-cudart-dev cuda-toolkit)
  - cargo build --features cuda
  - cargo test --features cuda

- CUDA with NVCC installed:
  - Install NVCC (e.g., NVIDIA CUDA Toolkit). On Ubuntu, avoid dated distro packages if you need newer CUDA.
  - cargo build --features cuda
  - cargo test --features cuda

Notes:
- cuBLAS header gating: if `cublas_v2.h` is found (CONDA_PREFIX/include or /usr/local/cuda/include), build.rs defines `have_cublas_header` and `M40LLM_HAVE_CUBLAS`, and links libcublas. Otherwise GEMM is compiled to a safe no-op, and tests that depend on cuBLAS are skipped.
- NVCC gating: when NVCC is available, build.rs emits `cfg(nvcc)` to enable kernels/tests compiled with the CUDA toolchain.
- The device target is sm_52 (Tesla M40) via `-gencode=arch=compute_52,code=sm_52`.

## Tests

- Non-CUDA tests run by default with `--no-default-features`.
- CUDA tests are gated behind both `--features cuda` and presence of NVCC (build.rs emits cfg(nvcc)).
- GEMM layout test (FP16 storage, FP32 compute) validates row-major conventions when cuBLAS is available.

## Server (feature = server)

- Simple Axum-based server behind `--features server`.
- To run a placeholder server (no CUDA requirement):
  - cargo run --no-default-features --features server -- run --model path/to.gguf --addr 0.0.0.0:58439
  - Ensure to set CORS/iframe if you wrap behind a front-end.

## Git hygiene
- Conventional Commits enforced via commitlint.
- Pre-commit git hook (cargo fmt) recommended.

## Environment variables
- CONDA_PREFIX: used by build.rs to locate CUDA headers/libs; when set and valid, build.rs includes and links accordingly.
- LD_LIBRARY_PATH: ensure the runtime can find cudart and cublas from your conda environment when running CUDA binaries.


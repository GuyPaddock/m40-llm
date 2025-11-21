# m40-llm

Rust + CUDA LLM inference targeting Tesla M40 (sm_52). FP16 storage / FP32 compute. GGUF loader skeleton. CUDA interop with FFI names `m40llm_*`. Optional HTTP server behind `server` feature flag.

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
  - Ensure CONDA_PREFIX points to an environment providing CUDA headers/libs (e.g. via mamba install -c conda-forge -c nvidia cuda-cudart-dev cuda-toolkit)
  - cargo build --features cuda
  - cargo test --features cuda

- CUDA with NVCC installed:
  - Install nvcc (e.g. apt install nvidia-cuda-toolkit or NVIDIA toolkit)
  - cargo build --features cuda
  - cargo test --features cuda

Notes:
- If `cublas_v2.h` is found (CONDA_PREFIX/include or /usr/local/cuda/include), build.rs defines M40LLM_HAVE_CUBLAS and links libcublas. Otherwise GEMM is compiled to a safe no-op.
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


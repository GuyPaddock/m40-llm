# m40-llm

Tesla M40–optimized Rust + CUDA LLM server/runtime. FP16 weights, FP32 compute via cuBLAS. GGUF loader and stable C FFI (`m40llm_*`). Goal: be much faster than ollama on the M40.

## What it is
- Single-GPU server for Maxwell Tesla M40 (sm_52)
- FP16 storage / FP32 compute (cuBLAS/cuBLASLt as available)
- GGUF loader; C FFI symbols `m40llm_*` for embedding
- Small, explicit codebase focused on M40 performance
- Optional HTTP server (enable with `--features server`)

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

## Build features (Cargo)
This project uses Cargo feature flags to switch between CPU‑only and GPU‑accelerated builds, and to include an optional HTTP server.

- `cuda`: Enables the CUDA backend. When set:
  - If `nvcc` is available, we compile CUDA kernels for sm_52 and link against CUDA runtime. If the cuBLAS header (`cublas_v2.h`) is found, we also link cuBLAS and enable GEMM paths and tests.
  - If `nvcc` is not available, we build and link a stub library that provides the required symbols so the crate still compiles. This lets you develop on machines with only CUDA headers/libs (e.g., conda) or no CUDA toolchain.
- `server`: Includes the HTTP server binary routes so you can run `m40-llm run ...`.

Build script behavior:
- When `nvcc` is present, compiles kernels for `sm_52` and also embeds PTX for `compute_52` so newer GPUs can JIT from PTX if needed.
- Always exposes `cfg(have_cublas_header)` when the cuBLAS header is detected so tests can gate accordingly.
- Always exposes `cfg(nvcc)` when `nvcc` is present so code/tests can detect a real CUDA toolchain.

## Build
Build the project in one of these modes:

- CPU only (no CUDA):
  - Build: `cargo build --no-default-features`
  - Test: `cargo test --no-default-features`
- CUDA enabled, without nvcc (e.g., conda headers/libs available):
  - Build: `cargo build --features cuda`
  - Test: `cargo test --features cuda`
- CUDA enabled, with nvcc installed:
  - Build: `cargo build --features cuda`
  - Test: `cargo test --features cuda`

## Tests
- CPU‑only mode: `cargo test --no-default-features` runs all non‑CUDA tests.
- CUDA mode (`--features cuda`): CUDA smoke and GEMM tests run when the environment has CUDA headers, and additional GEMM/cuBLAS tests run when the build detects `cublas_v2.h`. If `nvcc` is present, tests will also be able to detect `cfg(nvcc)` paths.


## CUDA device selection and cuBLAS
- Auto‑select M40: CudaContext::new(-1) will pick a Tesla M40 (sm_52) if one is visible. If none is visible, it falls back to device 0.
- Force selection: set M40LLM_FORCE_M40=1 to force runtime selection of an sm_52 device even when a specific device_id is passed.
- Respect CUDA_VISIBLE_DEVICES: device enumeration respects CUDA_VISIBLE_DEVICES. The auto‑picker searches only among visible devices and selects the first sm_52 it finds.
- cuBLAS control: by default, we do not link cuBLAS even if headers are present. Set M40LLM_ENABLE_CUBLAS=1 to enable cuBLAS integration if both the header (cublas_v2.h) and a shared library (e.g., libcublas.so.11) are detected. Otherwise, fallback CUDA kernels are used.
- Test gating: build.rs exposes cfg(nvcc) when a real CUDA toolchain is present and cfg(have_cublas_header) when the cuBLAS headers are detected; CUDA tests use these to gate cuBLAS‑specific coverage. Some CUDA tests also use require_sm52() to skip gracefully when not on an sm_52 device.

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


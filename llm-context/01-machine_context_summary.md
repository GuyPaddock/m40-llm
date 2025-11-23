# Machine Context Summary (for continued development)
# Machine Context Summary — UPDATED 2025-11-20

This section is an up-to-date snapshot meant for quick reference. Older historical notes remain below for context.

## Project identity
- Name: m40-llm — Rust LLM runtime/server targeting Tesla M40 (Maxwell, sm_52)
- Datatypes: FP16 storage (weights, KV), FP32 compute
- Model format: GGUF via gguf-rs-lib/gguf-llms (planned); stable C FFI `m40llm_*`; optional HTTP server behind `server` feature

## Build/quality constraints
- Keep CUDA and non-CUDA builds green under RUSTFLAGS=-D warnings
- Ensure sm_52 SASS and compute_52 PTX are built (build.rs handles both)
- Support CUDA-without-nvcc configurations using stub symbols (links cleanly)
- Expose cfgs from build.rs: `nvcc` when nvcc is available; `have_cublas_header` only when `M40LLM_ENABLE_CUBLAS=1` and both cuBLAS header+library are detected

## Current focus
- t22-parity-check (in_progress): CUDA<->CPU parity for last-token attention (FP16 K/V, FP32 compute), across a coverage grid; run with and without cuBLAS when available
- t26-precommit-enforcement-audit (done): enforce hooks via core.hooksPath and document setup in CONTRIBUTING
- t27-force-push-rewritten-history (todo): pending approval to push with --force-with-lease

## Status snapshot (as of this update)
- CPU reference last-token attention implemented in tests
- CPU parity grid test added and passing: validates CPU library attention vs CPU reference across multiple (num_heads, head_dim, seq_len) with FP16-rounded K/V; fixed initial failure by re-initializing KVCache per seq_len
- CUDA-gated GPU parity grid test added: mirrors CPU grid; casts K/V through f16 on host before upload; compares vs CPU reference at tol 1e-3; compiled and runs under the current environment’s CUDA gating; needs execution on a CUDA host with/without cuBLAS for full parity assessment
- Stubs ensure CUDA-without-nvcc builds link and tests gate appropriately

### Builds/tests
- CPU (no-default-features): green
- CUDA feature: builds; CUDA-gated tests compile/run depending on `nvcc`/headers; optional cuBLAS paths honored when `cublas_v2.h` is detected; PTX for compute_52 embedded

## Roadmap (t23–t36)
- t23 — cuBLAS GEMM integration for Q/K/V, MLP, and output projection
  - Notes: Prefer cublasGemmEx with FP16 inputs / FP32 compute; explicit layouts; provide fallback when cuBLAS header missing; tests for row/col correctness
- t24 — Integrate gguf-rs-lib + gguf-llms; replace hand-rolled GGUF parser
  - Notes: Expose typed hparams and tensor views; keep non-CUDA builds green
- t24a — GGUF device mapping from crate-provided tensor offsets
  - Notes: Single upload; compute per-tensor device pointers; validate dtype
- t25 — RoPE, RMSNorm kernels and host fallbacks
- t26 — Minimal forward pass for one full layer (prefill + decode step)
- t27 — Tokenizer integration from GGUF metadata
- t28 — Sampling (softmax + top‑k/top‑p), start on host then optional CUDA
- t29 — End‑to‑end decode loop using KV cache
- t30 — HTTP server /generate wired to real decode and stream tokens (feature `server`)
- t31 — Microbenchmarks: GEMM and attention on M40
- t32 — Persistent decode kernel prototype (optional feature)
- t33 — Prefill/Decode stream separation and priorities
- t34 — Robust error handling, logging, telemetry
- t35 — CI: Expand CUDA/non‑CUDA test matrix and document setup
- t36 — Minimal GGUF test model for integration tests

## Maintenance
- Keep unsafe surfaces small and well-audited; replace unsafe Send/Sync with safer wrappers; avoid allow(dead_code)

## Code state highlights
- build.rs: compiles kernels for sm_52, embeds compute_52 PTX; links CUDA runtime; optionally links cuBLAS; defines cfg(nvcc) and cfg(have_cublas_header)
- cuda/kernels.cu: attention_last_token_kernel with FP16 storage / FP32 compute; optional cuBLAS-backed GEMM where used
- cuda/stub.c: stubs when no nvcc present
- src/cuda.rs: CudaContext, KVCache FFI; CPU fallback under #[cfg(not(feature="cuda"))] for attention_last_token_f32
- src/infer.rs: delegates to KVCache/CudaContext; shapes validated
- tests: CPU attention parity grid; CUDA-gated smoke, GEMM layout, cast kernel, last-token attention, and CUDA parity grid

## Next steps
- Run CUDA parity grid on a CUDA host with and without cuBLAS (export M40LLM_ENABLE_CUBLAS=1 to enable); collect/report mismatches (tol ~1e-3)
- Address any kernel/layout discrepancies revealed by the grid (odd head_dim, multiple heads, varying seq_len)
- Proceed to t23 cuBLAS GEMM integration after t22 parity is confirmed

---


## 0. Project Identity
- Project name: **m40-llm**
- Purpose: Build a **Rust-based LLM inference server** specifically optimized for **NVIDIA Tesla M40 (Maxwell, sm_52)**.
- Architecture: Rust host + CUDA C++ kernels (via FFI).
- Model format: **GGUF (Meta / llama.cpp)** loaded via `gguf-rs-lib` + `gguf-llms`.

## 1. Hardware Constraints & Optimization Strategy (Maxwell/M40)
- GPU: **Tesla M40 24GB**, Compute Capability **sm_52**, no tensor cores, ~7 TFLOP FP32, ~288 GB/s bandwidth.
- **Optimal datatype policy:**
    - **FP16 for weight storage & KV cache**
    - **FP32 for all compute**
    - Convert FP16→FP32 on load
- **Architectural Targets:**
    - Use large FP32 register file (255 regs/thread)
    - Use shared memory heavily
    - Use `__ldg`/readonly cache for weight loads
    - Favor warp-parallelism and vectorized FP16 loads (`half2`)
    - Avoid relying on tensor-core assumptions
- **Concurrency:**
    - Multi-stream design (decode, prefill, maintenance)
    - Future: persistent decode kernel with ring buffer

## 2. Model Execution Pipeline (Single Layer Forward)
1. Input embedding (FP16 → FP32)
2. RMSNorm (FP32 → FP32)
3. Q/K/V projection GEMMs (FP16W × FP32 → FP32)
4. Convert K/V: FP32 → FP16 before storing in KV
5. Attention:
    - Q = FP32
    - K/V = FP16
    - Compute in FP32
    - Output context = FP32
6. MLP (SwiGLLU):
    - W_gate/W_up/W_down FP16
    - Compute in FP32
    - Output FP32
7. Output projection FP16 × FP32 → FP32
8. Logits FP32

## 3. Implemented Components (as of this thread)
### 3.1 Rust Side
Project structure:
```
m40-llm/
  Cargo.toml
  build.rs
  cuda/kernels.cu
  src/
    main.rs
    cli.rs
    gguf.rs
    model.rs
    cuda.rs
    infer.rs
    server.rs
    tokenizer.rs
```
- DeviceAllocator abstraction implemented.
- CudaContext wrapper implemented.

### 3.2 CUDA Side
- RMSNorm FP32→FP32 kernel done.
- MLP (SwiGLU) FP16→FP32 kernel done.
- FP32→FP16 cast kernel planned.
- Basic attention kernel using FP32-Q, FP16-K/V integrated (index offsets TBD).

## 4. KV Cache Design (in progress)
Likely layout:
```
[layer][sequence][head][token][head_dim]
```
- FP16 K/V storage.
- Needs finalized stride math for indexing.
- Append path: FP32→FP16 cast then KV-store.

## 5. Attention Kernel Specification
- Inputs: FP32 Q, FP16 K/V
- Output: FP32 context
- Block = 1 head per block (initial impl)
- Uses shared memory for scores
- TODO: vectorized loads, warp-level reductions, fully optimized tiling.

## 6. Testing Strategy
- Rust `#[test]` framework.
- CPU reference implementations for RMSNorm and MLP.
- Pattern:
    1. Deterministic input
    2. CPU ref
    3. GPU kernel
    4. Compare diff
- Ensures correctness of FFI + CUDA math.

## 7. Future Components (planned)
### 7.1 GEMM
- Replace cuBLAS with custom Maxwell-tuned GEMM.
- Features to add:
    - Shared-memory tiling
    - half2 vector loads
    - FP32 accumulation
    - double-buffering
- Explore Stream-K style partitioning.

### 7.2 Persistent Kernel
- GPU loop for decode:
    - pulls work via host-pinned ring buffer
    - runs attention + MLP + sampling
    - writes next token
- Decode stream separate from prefill stream.

### 7.3 End-to-End
- Minimal 1-layer forward test with real GGUF.
- Compare against llama.cpp/PyTorch outputs.

## 8. Research Notes
### Not Using:
- Strassen / Winograd / modern exponent improvements (impractical).
- AlphaTensor direct algorithms.

### Relevant:
- Maxwell-tuned `maxas` SGEMM.
- Stream-K "work-centric" decomposition.
- CUDA optimization tutorials.

### Tuning Targets:
- Occupancy vs register pressure.
- SM partitioning.
- Warp-level softmax.
- KV locality.
- Prefill scheduling.

## 9. TODO List
1. Final KV cache layout + stride offsets.
2. FP32→FP16 cast kernel for KV append.
3. Integrate GEMM backend.
4. Attention CPU/GPU test.
5. Finish KV indexing in CUDA kernel.
6. Dummy GGUF mini-model for integration testing.
7. Microbenchmarks (GEMM + attention).
8. Persistent kernel design.

## 10. Current Stability
- Dtype policy consistent.
- RMSNorm + MLP validated.
- Attention kernel functional but indexing incomplete.
- Full model execution pending GEMM + KV integration.

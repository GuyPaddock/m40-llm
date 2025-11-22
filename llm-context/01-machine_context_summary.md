# Machine Context Summary (for continued development)
# Machine Context Summary — UPDATED 2025-11-20

This section is an up-to-date snapshot meant for quick reference. Older historical notes remain below for context.

## Project identity
- Name: m40-llm — Rust LLM runtime/server targeting Tesla M40 (Maxwell, sm_52)
- Datatypes: FP16 storage (weights, KV), FP32 compute
- Model format: GGUF via gguf-rs-lib/gguf-llms (planned); stable C FFI

## Build/quality constraints
- Keep CUDA and non-CUDA builds green under RUSTFLAGS=-D warnings
- Ensure sm_52 SASS and compute_52 PTX are built
- Support CUDA-without-nvcc configurations using stub symbols

## Current focus (t22)
- Last-token attention with FP16 K/V, FP32 compute
- Keep server feature build clean; keep test matrices green

## Status snapshot (as of this update)
- t21 — done: Host API -> CUDA KV append FP32→FP16 cast, with CUDA-gated test
- t22 — in progress: Last-token attention (FP32 compute, FP16 K/V)
  - t22-1 — partial/done: API/layout implied by run_attention(d_q, d_out, seq_id, seq_len, dim, num_heads, head_dim); KV layout [seq][token][head][dim]
  - t22-2 — done (initial): CUDA kernel attention_last_token_kernel and C wrapper m40llm_attention_last_token_f32 added (naive per-head single-thread baseline)
  - t22-3 — done: Safe Rust wrapper KVCache::attention_last_token_f32; infer::LoadedModel::run_attention calls it under cfg(feature="cuda")
  - t22-4 — todo: CPU reference implementation for tests
  - t22-5 — todo: CUDA-gated tests comparing CUDA vs CPU ref
  - t22-6 — done: Stub symbol added in cuda/stub.c to keep CUDA-without-nvcc builds linking

### Builds/tests
- Non-CUDA: green (no attention tests yet)
- CUDA with nvcc: compiles; attention kernel is naive but correct for small cases
- CUDA without nvcc: links thanks to new stub symbol

## Roadmap (t23–t36)
- t23 — todo: cuBLAS GEMM integration for Q/K/V, MLP, and output projection
  - Notes: Use cublasGemmEx with FP16 inputs / FP32 compute; handle row/col layouts; ensure rectangular GEMM tests pass; provide fallback when cuBLAS header missing.
- t24 — todo: Integrate gguf-rs-lib + gguf-llms; replace hand-rolled GGUF parser
  - Notes: Update Cargo.toml; implement wrapper exposing typed hparams (LLaMA/Mistral) and tensor views; delete legacy parser paths; keep non-CUDA builds green.
- t24a — todo: GGUF device mapping from crate-provided tensor offsets
  - Notes: Upload tensor data block once; compute per-tensor device pointers from d_data_base + offset; validate dtype (expect F16 where needed).
- t25 — todo: RoPE, RMSNorm kernels and host fallbacks
  - Notes: Implement RoPE (device) and RMSNorm (device); CPU fallbacks; compare tests.
- t26 — todo: Minimal forward pass for one full layer (prefill + decode step)
  - Notes: Wire embeddings → RMSNorm → QKV GEMMs → attention → MLP → residuals; return logits. Compare against small GGUF reference if available.
- t27 — todo: Tokenizer integration (SentencePiece/BPE) from GGUF metadata
  - Notes: Load tokenizer from GGUF; implement encode/decode; unit tests.
- t28 — todo: Sampling (softmax + top‑k/top‑p), start on host then optional CUDA
  - Notes: Numerically stable softmax; top‑k/top‑p; temperature; tests. CUDA gated.
- t29 — todo: End‑to‑end decode loop: prefill, then iterative decode using KV cache
  - Notes: Append K/V each step; last-token attention; logits → sample → next token; stop on eos/max tokens; smoke test on toy model.
- t30 — todo: HTTP server /generate wired to real decode and stream tokens
  - Notes: Replace dummy output; streaming via SSE or chunked JSON; allow CORS/iframing per runtime guidance.
- t31 — todo: Microbenchmarks: GEMM and attention on M40
  - Notes: Benchmark typical LLaMA/Mistral shapes; record TFLOPs and latency; document expectations and regressions.
- t32 — todo: Persistent decode kernel prototype (optional feature)
  - Notes: Design ring buffer; launch persistent kernel; host↔device queues; feature-gated; smoke test.
- t33 — todo: Prefill/Decode stream separation and priorities
  - Notes: Two CUDA streams with priorities; validate overlap; add tracing logs.
- t34 — todo: Robust error handling, logging, telemetry
  - Notes: Surface nvcc/cuBLAS detection, device info, memory usage; structured logs; simple metrics.
- t35 — todo: CI: Expand CUDA/non‑CUDA test matrix and document setup
  - Notes: Ensure gates for have_cublas_header and nvcc work on CI; document conda/non‑nvcc setup and nvcc path.
- t36 — todo: Minimal GGUF test model for integration tests
  - Notes: Tiny redistributable GGUF or download-at-test with checksum.

## Maintenance
- t20x-maint-1..4 — done (as previously)
- t20x-maint-5 — pending: Replace unsafe Send/Sync with safer wrappers; tighten allow(dead_code)

## Code state highlights
- cuda/kernels.cu: attention_last_token_kernel and m40llm_attention_last_token_f32 added; launches 1 block/head; syncs decode_stream
- cuda/stub.c: stub for m40llm_attention_last_token_f32 to keep CUDA-without-nvcc builds green
- src/cuda.rs: FFI declaration added; safe KVCache::attention_last_token_f32 wrapper
- src/infer.rs: LoadedModel::run_attention validates shapes and calls CUDA wrapper when enabled

## Next steps
- Implement CPU reference attention and CUDA-vs-CPU compare tests (t22-4/5)
- Parameterize KV cache allocation to avoid head_dim/head count mismatch
- Proceed to t23 cuBLAS GEMM integration once t22 tests land

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

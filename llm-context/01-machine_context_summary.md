# Machine Context Summary (for continued development)

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

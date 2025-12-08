# Machine Context Summary — UPDATED 2025-11-23

Purpose: concise, actionable snapshot to resume m40-llm development fast.

## 1. Project Identity
- Project name: m40-llm — Rust LLM runtime/server for NVIDIA Tesla M40 (Maxwell, sm_52)
- Precision: FP16 storage (weights, KV), FP32 compute (max accuracy on M40)
- Model format: GGUF (via gguf-rs-lib + gguf-llms planned)
- Architecture: Rust host + CUDA C/C++ kernels behind stable C FFI; optional HTTP server (feature "server")

## 2. KV Cache Design (reference)
Likely layout:
```
[layer][sequence][head][token][head_dim]
```
- Storage: FP16 for K and V
- Strides: finalize and document per-axis strides for correct indexing
- Append path: convert FP32→FP16 before storing into KV
- Debug: ffi_debug_read_kv_token to validate contents and layout

## 3. Attention Kernel Specification (reference)
- Inputs: Q in FP32; K/V in FP16 (from KV cache)
- Compute: FP32; Output context: FP32
- Initial mapping: 1 head per block (simple, correct baseline)
- Optimizations (later): vectorized loads, warp-level reductions, shared-memory tiling

## 4. Testing Strategy (reference)
- Use Rust #[test] with CUDA-gated tests
- Pattern:
  1) Build deterministic inputs
  2) CPU reference implementation
  3) Run GPU kernel via FFI
  4) Compare with tolerance (single-token ~1e-3; two-token minimal forward 5e-3)
- Cover: RMSNorm, MLP, attention last-token, projection shapes/layouts, FP32↔FP16 casts

## 5. Future Components (planned)
### 5.1 GEMM
- Keep cuBLAS as baseline; consider Maxwell-tuned custom GEMM only if needed
- Techniques: shared-memory tiling, half2 vector loads (for FP16 storage), FP32 accumulation, double-buffering

### 5.2 Persistent Kernel
- GPU-resident decode loop pulling jobs from a host-pinned ring buffer
- Runs attention + MLP + sampling per token; writes results back
- Complements multi-stream design and reduces launch overhead

### 5.3 End-to-End
- Minimal one-layer forward test on toy weights
- Compare against CPU reference / known small models for sanity

## 6. Research Notes
- Not using: Strassen/Winograd, AlphaTensor-derived algorithms (impractical here)
- Relevant references: Maxwell-tuned maxas SGEMM; Stream-K decomposition; CUDA optimization guides
- Tuning targets: occupancy vs register pressure, SM partitioning, warp-level softmax, KV locality, prefill scheduling

State of the repo (HEAD)
- CUDA parity grid for last-token attention is green on Tesla M40 with cuBLAS off/on
- cuBLAS GEMM wrappers integrated; runtime toggle via M40LLM_ENABLE_CUBLAS=0|1
- Minimal forward validated: one- and two-token prefill+decode parity vs CPU; two-token tol=5e-3 due to accumulated rounding; KV allocated via allocate_kv_cache_with_layout
- Clippy -D warnings clean for CUDA and non-CUDA builds
- Docs added: docs/cuda_parity_and_kv_layout.md (how to run CUDA grid, cuBLAS toggling, KV layout API)

How to run key tests quickly
- Attention parity grid (fallback):
  M40LLM_ENABLE_CUBLAS=0 cargo test --features=cuda -- --nocapture tests/attention_parity_cuda_grid.rs
- Attention parity grid (cuBLAS):
  M40LLM_ENABLE_CUBLAS=1 cargo test --features=cuda -- --nocapture tests/attention_parity_cuda_grid.rs
- Projection/MLP wrappers:
  cargo test --features=cuda -- --nocapture tests/proj_wrappers.rs tests/mlp_wrappers.rs
- Forward smokes:
  cargo test --features=cuda -- --nocapture tests/forward_with_layer_smoke.rs
- Reference doc: docs/cuda_parity_and_kv_layout.md

CUDA/cuBLAS toggles and build
- build.rs detects NVCC, compiles sm_52 SASS and embeds compute_52 PTX
- cuBLAS enabled only if header+lib detected and M40LLM_ENABLE_CUBLAS=1
- Runtime device: use -1 to auto-select; tests prefer ctx_m40(); M40LLM_FORCE_M40=1 optional

Datatype policy (Maxwell-optimized)
- Storage: FP16 weights and KV
- Compute: FP32 everywhere (QKV GEMMs, attention, MLP, output proj)
- Q: FP32; K/V: FP16 in cache; convert as needed at boundaries

Important APIs and files
- src/infer.rs: allocate_kv_cache_with_layout(max_seq, batch, heads, head_dim); forward_one_token_minimal/_with_layer; projection wrappers; attention helpers; guards in map_standard_layer
- src/cuda.rs: CUDA FFI (context, GEMMs, KV ops); CPU fallbacks; ffi_debug_read_kv_token
- build.rs: NVCC/cuBLAS detection, sm_52 + compute_52
- tests/: attention_parity_cuda_grid.rs; proj_wrappers.rs; mlp_wrappers.rs; forward_with_layer_smoke.rs; KV cast tests

Task tracker snapshot (grouped, current)
- In progress
  - t23-6-autoselect-m40: runtime auto-select + guardrails; tests use ctx_m40()
  
  - t23-5e-attn-green: keep attention parity grid green during integration
  - t26-3-guards: FP16 + shape invariants in map_standard_layer
  - t26-5-docs: document minimal forward wiring/limits; KV layout + CUDA grid usage
- Done
  - t23-5a-qkv: integrate GEMM into Q/K/V projections; CUDA smoke passing
  - t23-5b-mlp: integrate GEMM into MLP projections; parity smoke passing
  - t23-5c-out-proj: integrate GEMM into output projection; smoke passing
  - t23-5d-parity: numeric parity tests for Q/K/V/MLP/out-proj (~1e-3)
  - t22-parity-check: CUDA<->CPU parity for last-token attention grid (M40) green
  - t23-cublas-gemm: cuBLAS-enabled GEMM wrappers; tests pass with cuBLAS off/on
  - t26-run-tests: non-CUDA and CUDA suites passing on M40 sm_52
- t26-min-forward: minimal forward validated (prefill + decode) – Done
- Todo (near-term P1 focus)
  - t24-gguf-integration: integrate gguf-rs-lib + gguf-llms (typed hparams/tensors)
  - t24a-gguf-device-mapping: single upload; device pointers; dtype validation
  - t25-rope-rmsnorm: verify RMSNorm; implement RoPE host+CUDA with tests

  - t27-tokenizer: from GGUF metadata; tests on small vocabs
  - t28-sampling: host softmax + top‑k/top‑p; optional CUDA later
  - t29-e2e-decode: end‑to‑end decode loop with KV cache
- Todo (supporting P2/P3)
  - t31b-microbench-attn; t33 stream separation; t34 robustness/logging; t35 CI matrix; t36 minimal GGUF test model; t30 server feature; t23-6-docs/benches-ctx/warnings

Roadmap (condensed)
1) Keep parity green while finishing t26 minimal forward (prefill + decode); validate numerics
2) Wire real GGUF (t24/t24a) and RoPE/RMSNorm (t25)
3) Add tokenizer (t27) and host sampling (t28) to emit tokens
4) Build e2e decode loop (t29); then expand performance (streams, microbenches, persistent decode)

Quick resume checklist
- cargo clippy --all-features -- -D warnings
- Run attention parity grid with cuBLAS off/on; keep green
- Use allocate_kv_cache_with_layout in tests/examples to prevent KV mismatches
- Progress t26 minimal forward with small, verifiable increments

See also
- llm-context/00-index.md: curated summaries for all design/context notes
- docs/cuda_parity_and_kv_layout.md: how to run CUDA parity grid and use KV layout API

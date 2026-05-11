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

## Building

### Standard (non-CUDA)
```bash
cargo build --no-default-features
```

### CUDA-enabled (requires CUDA 12.x toolkit)
```bash
cargo build --features cuda  # With NVCC installed
```

Recommended micromamba toolchain setup (x86_64, tested on this branch):

```bash
# install micromamba if needed
curl -Ls https://micro.mamba.pm/install.sh | bash
source ~/.bashrc

# CUDA 12.4 toolchain with cuBLAS headers + libs
micromamba create -y -n m40-llm -c conda-forge -c nvidia/label/cuda-12.4.1 \
  cuda-nvcc=12.4.99 cuda-cudart=12.4.99 cuda-cudart-dev=12.4.99 \
  libcublas=12.4.5.8 libcublas-dev=12.4.5.8

# Build/link with cuBLAS enabled
micromamba run -n m40-llm env M40LLM_ENABLE_CUBLAS=1 cargo build --features cuda
```

CI verifies two configurations:
1. `noncuda`: No CUDA dependencies
2. `cuda-with-nvcc`: Full CUDA+NVCC toolchain

---

## Performance strategy on M40
- FP16 storage, FP32 compute tiles: load FP16 to shared, convert to FP32, compute in registers
- Tuned GEMM with cuBLAS/cuBLASLt; explicit row/col layouts; layout tests included
- CUDA Graphs + persistent decode kernel to minimize launch overhead
- Warp-level micro-batching (e.g., one warp per sequence) for decode
- Optimized KV cache: FP16 or INT8 per-head; contiguous per-head layout; pinned host staging
- Streams/Hyper‑Q: high‑priority decode stream, concurrent lower‑priority prefill
- Read‑only (`__ldg`) and texture caches for non-GEMM ops (norms, embeddings)

Current optimization order is measurement and ownership first:

1. Split warm/cold benchmark modes and add launch/sync/copy/cuBLAS counters.
2. Serialize generation and introduce request-owned decode/session scratch.
3. Make device scratch RAII-safe and fix KV addressing to separate layer and sequence IDs.
4. Harden the materialized FP32 cuBLAS fast-fits backend with budget/fallback logs.
5. Remove per-kernel synchronization through async enqueue APIs, then evaluate fusion and CUDA Graphs.
6. Integrate packed varlen server scheduling only after ownership and KV addressing are correct.

Keep `__ldg`/texture experiments opt-in and off by default unless profiling
identifies a specific read-cache bottleneck.

## Build features (Cargo)
This project uses Cargo feature flags to switch between CPU‑only and GPU‑accelerated builds, and to include an optional HTTP server.

- `cuda`: Enables the CUDA backend. When set:
  - Requires `nvcc` on PATH; CUDA builds fail fast if the toolchain is missing.
  - Compiles CUDA kernels for sm_52 (plus compute_52 PTX) and links against the CUDA runtime. If the cuBLAS header (`cublas_v2.h`) is found and `M40LLM_ENABLE_CUBLAS=1` is set, we also link cuBLAS and enable GEMM paths and tests.
- `server`: Includes the HTTP server binary routes so you can run `m40-llm run ...`.

Build script behavior:
- Compiles kernels for `sm_52` and also embeds PTX for `compute_52` so newer GPUs can JIT from PTX if needed.
- Exposes `cfg(nvcc)` when a real CUDA toolchain is present.
- Exposes `cfg(have_cublas)` when cuBLAS headers and libraries are found and `M40LLM_ENABLE_CUBLAS=1`.

## Build
Build the project in one of these modes:

- CPU only (no CUDA):
  - Build: `cargo build --no-default-features`
  - Test: `cargo test --no-default-features`
- CUDA enabled (requires nvcc on PATH):
  - Build: `cargo build --features cuda`
  - Test: `cargo test --features cuda`

## CLI Generate
Run one local decode loop without starting the HTTP server:

```bash
M40LLM_ENABLE_CUBLAS=1 cargo run \
  --features cuda \
  -- generate path/to.gguf "Hello" \
  --max-tokens 2 \
  --top-k 1 \
  --require-sm52
```

The command prints generated text only; it does not echo the prompt. It uses the
same non-streaming decode helper as `POST /generate`.

## Tests
- CPU‑only mode: `cargo test --no-default-features` runs all non‑CUDA tests.
- CUDA mode (`--features cuda`): CUDA smoke and GEMM tests run when the environment has CUDA headers, and additional GEMM/cuBLAS tests run when the build detects `cublas_v2.h`. Tests rely on `nvcc` being present because the build fails without it when CUDA is enabled.
- Minimal forward parity: see docs/minimal_forward.md and tests/forward_parity_toy.rs for CUDA‑gated toy coverage. The real server path now runs full-layer TinyLlama F16 `/generate` on M40. The CUDA forward path keeps residual adds and SwiGLU activation on device; remaining optimization work should be measured before making performance claims.


## CUDA device selection and cuBLAS
- Auto‑select M40: CudaContext::new(-1) will pick a Tesla M40 (sm_52) if one is visible. If none is visible, it falls back to device 0.
- Force selection: set M40LLM_FORCE_M40=1 to force runtime selection of an sm_52 device even when a specific device_id is passed.
- Respect CUDA_VISIBLE_DEVICES: device enumeration respects CUDA_VISIBLE_DEVICES. The auto‑picker searches only among visible devices and selects the first sm_52 it finds.
- cuBLAS control: by default, we do not link cuBLAS even if headers are present. Set M40LLM_ENABLE_CUBLAS=1 to enable cuBLAS integration if both the header (cublas_v2.h) and a shared library (e.g., libcublas.so.11) are detected. Otherwise, fallback CUDA kernels are used.
- Test gating: build.rs exposes cfg(nvcc) when a real CUDA toolchain is present and cfg(have_cublas) when cuBLAS is enabled; CUDA tests use these to gate cuBLAS‑specific coverage. Some CUDA tests also use require_sm52() to skip gracefully when not on an sm_52 device.
- Allocation tracing: set `M40LLM_ALLOC_LOG=1` to print per-allocation
  `device_malloc`/`device_free` traces. Add `M40LLM_ALLOC_BT=1` to include
  backtraces with those allocation logs.
- Tensor view tracing: set `M40LLM_TENSOR_VIEW_LOG=1` to print each GGUF
  tensor-to-device pointer mapping during model load.
- GEMM backend tracing: set `M40LLM_GEMM_LOG=1` to print one backend
  selection line per GEMM wrapper. With `M40LLM_ENABLE_CUBLAS=1`, hot GGUF F16
  projection weights are materialized into FP32 device buffers and routed through
  `cublasSgemm`; set `M40LLM_MATERIALIZE_F32_WEIGHTS=0` to force the dedicated
  GGUF-layout CUDA fallback. Set `M40LLM_MATERIALIZE_F32_BUDGET_MB=<mb>` to cap
  cached FP32 materialized weights; over-budget tensors fall back to the GGUF F16
  kernel and are logged when `M40LLM_GEMM_LOG=1`. Cache keys include the source
  pointer plus tensor identity metadata when a GGUF tensor view is available.
- Timing tracing: set `M40LLM_TIMING_LOG=1` to print per-token and per-layer
  decode timing. This is intentionally verbose and intended for profiling runs.
- Stream tracing: set `M40LLM_STREAM_LOG=1` to print prefill/decode stream
  creation details and best-effort priority selection.
- Launch/sync/copy counters: set `M40LLM_LAUNCH_LOG=1` to log kernel launch
  and cuBLAS counter events, `M40LLM_SYNC_LOG=1` to log stream synchronization
  counter events, and `M40LLM_COPY_LOG=1` to log H2D/D2H copy counter events.
  Set `M40LLM_PROFILE_LOG=1` to print lower-noise per-operation counter deltas
  around forward-pass timing regions.

## Server (feature = server)
```
M40LLM_ENABLE_CUBLAS=1 cargo run \
  --features cuda,server \
  -- run path/to.gguf \
  --addr 0.0.0.0:58439
```

`POST /generate` accepts JSON such as:

```bash
curl -sS -X POST http://127.0.0.1:58439/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":1,"temperature":1.0,"top_k":1}'
```

The `output` field contains generated text only; it does not echo the prompt.
`max_tokens` counts generated token IDs, not decoded characters, so a single token
may decode to multiple characters.
Generation requests are serialized at the server state level for now so buffered
and streaming `/generate` calls cannot share the same CUDA forward workspace
concurrently; the intended replacement is a per-session workspace pool.
The CUDA decode path uses a shared `DecodeSession` for CLI and server generation,
reusing per-request `d_x` and `d_out` device scratch instead of allocating those
buffers for every token.
Forward workspace and decode-session scratch buffers are RAII-owned, so partial
allocation failures clean up any buffers that were already allocated.
Full-layer decode uses explicit model-level KV addressing
`KV[layer][sequence][position][kv_head][head_dim]`. Physical KV slots are mapped
sequence-major as `sequence_id * block_count + layer_id`; current `/generate`
requests still default to `sequence_id=0` and remain serialized until the server
scheduler leases sequence/workspace slots per active request.

Current M40 validation target:
- Model: `TinyLlama-1.1B-Chat-v1.0.f16.gguf`
- Expected log evidence: `full-layer forward enabled layers=22`
- CLI evidence: `m40-llm generate ... --require-sm52` prints decoded text
  while exercising the same full-layer CUDA path as non-streaming `/generate`.
- CUDA hot path: parallel RMSNorm, RoPE, residual adds, and SiLU/gated MLP
  activation run on device in the forward path.
- Fresh M40 GEMM, attention, and TinyLlama `/generate` baselines: see
  `docs/perf_baselines.md`.
- Workspace reuse removes repeated forward scratch allocation.
- Materialized FP32 projection weights route steady-state decode projection GEMMs
  through `cublasSgemm`.
- The GQA last-token attention path has an optimized `head_dim=64` CUDA kernel;
  set `M40LLM_ATTN_LOG=1` to print attention backend selection.
- Current timed CLI profiles point at batching, launch overhead, and prefill
  shape handling as the next performance targets before stream separation.
- Variable-length batch metadata is available through `m40_llm::infer` for
  packed token/Q/KV offsets and length buckets; CUDA kernels consume this path
  incrementally to avoid padded-token prefill work.
- Batched last-token GQA attention supports packed Q/output buffers with
  per-sequence `seq_len` metadata for mixed-KV-length decode batches on
  `head_dim=64`.
- Packed prefill GQA attention supports mixed query/KV lengths for
  `[total_q_tokens, q_heads, 64]` Q/output and
  `[total_kv_tokens, kv_heads, 64]` K/V buffers, with a CPU-reference parity
  test and initial M40 benchmark baseline.
- Prefill/decode attention can be enqueued on separate non-blocking CUDA
  streams for benchmarked overlap experiments; the default CLI/server decode
  path remains synchronous until a request scheduler can use this safely.
- Persistent decode has an experimental synthetic worker prototype with Rust
  lifecycle wrappers and a benchmark. It is not wired into CLI/server generation
  yet; use it only to evaluate launch-overhead reduction candidates.
- Read-only cache experiments are opt-in. `M40LLM_CACHE_EXPERIMENT=ldg` enables
  the first `__ldg` experiment for weighted RMSNorm; current measurements keep
  the default kernel unchanged.
- `M40LLM_CACHE_EXPERIMENT=ldg_kv` enables an opt-in KV-cache `__ldg`
  experiment for head-dim-64 GQA attention; M40 measurements also keep the
  default attention kernels unchanged.

Variable-length batching is inspired by Zhang and Lu's SC25 research poster,
"An Efficient GEMM Acceleration Method for LLM Inference with Variable-Length
Sequences" ([PDF](https://sc25.supercomputing.org/proceedings/posters/poster_files/post167s2-file2.pdf)).
The M40 implementation adapts the high-level ideas to conventional CUDA FP32
accumulation on Maxwell sm_52, without Tensor Core / WMMA assumptions.

## Contributing
See `CONTRIBUTING.md` for guidelines.

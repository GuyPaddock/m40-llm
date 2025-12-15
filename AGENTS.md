# Agent Guidelines for m40-llm

This file replaces `llm-context/01-machine_context_summary.md` as the single source of task-level instructions for this repository.

## Introduction
You are continuing development of m40-llm—a Rust LLM runtime/server targeting NVIDIA Tesla M40 (sm_52). Work efficiently and keep changes focused.

## How to Start Each Task
1. Review `CONTRIBUTING.md` and set up the git hooks (pre-commit and commit message hooks).
2. Ensure Cocogitto (`cog`) is installed and working.
3. Ensure `nvcc` is available. If missing, install CUDA Toolkit compatible with Tesla M40 and modern `gcc`/`g++`.

## Commit Discipline
- Commit early and often.
- Use Conventional Commits.
- Every commit must include both a summary and a descriptive body.

## Context Files
- `llm-context/00-index.md` points to key design notes; consult it when resuming work.
- CUDA and KV cache guidance: `docs/cuda_parity_and_kv_layout.md`.

## Project Snapshot (from prior context)
- Precision policy: FP16 storage (weights and KV), FP32 compute; Q in FP32, K/V stored as FP16.
- Model format: GGUF; integration planned via `gguf-rs-lib` and `gguf-llms`.
- CUDA build: `build.rs` detects NVCC and embeds sm_52 SASS plus compute_52 PTX; cuBLAS toggled via `M40LLM_ENABLE_CUBLAS` when headers/libs are present.
- KV cache API: `allocate_kv_cache_with_layout` in `src/infer.rs`; `ffi_debug_read_kv_token` in `src/cuda.rs` helps validate layout.
- Key tests to keep green (with `--features=cuda`):
  - Attention parity grid: `tests/attention_parity_cuda_grid.rs` (set `M40LLM_ENABLE_CUBLAS=0|1`).
  - Projection/MLP wrappers: `tests/proj_wrappers.rs`, `tests/mlp_wrappers.rs`.
  - Forward smoke: `tests/forward_with_layer_smoke.rs`.

## Phase Ordering
- **Phase 1 (Functional, critical path):** Goal is to load a real GGUF and print decoded tokens on Tesla M40 with no silent CPU fallback.
  - Current in-progress tasks include: shape/dtype guards (t26-3-guards), minimal forward (t26-min-forward), CUDA↔CPU parity grid (t22-parity-check), cuBLAS integration for minimal forward (t23-5-integration), M40 auto-select/build guards (t23-6-autoselect-m40), tokenizer init (t27-tokenizer), host sampling (t28-sampling), and minimal GGUF test model (t36-min-gguf-model).
  - Acceptance highlights: fast-fail invalid GGUFs, logits parity across CUDA/CPU, deterministic sampler, and a CLI decode loop that visibly exercises the GPU.
- **Phase 2 (Fast, M40 optimization):** Optimize cuBLAS wiring and attention while keeping parity grid green.
  - Focus areas: cuBLAS GEMM wrappers (t23-cublas-gemm), full projection wiring (t23-5-integration continuation), attention parity guardrail (t23-5e-attn-green), microbenchmarks (t31b-microbench-attn), stream separation (t33-stream-sep), and persistent decode prototype (t32-persistent-kernel).
- **Phase 3 (Infrastructure, QA, polish):** Stabilize CI, logging, and server polish.
  - Focus areas: clippy cleanups (t23-6-warnings), CUDA benchmark unification (t23-6-benches-ctx), expanded CI matrix (t35-ci-matrix), better errors/telemetry (t34-robustness), `/generate` server parity (t30-server), prefill-vs-decode validation plan (t26-tests-plan), and completing `llm-context` indexing (t41-lc-index).

**Task selection rule:** when choosing work, pick the lowest-numbered unfinished task in the lowest-numbered phase.

## Near-Term Focus (from previous roadmap)
- Maintain CUDA attention parity grid on M40.
- Finish and document minimal forward path (prefill + decode) using KV cache helpers.
- Integrate GGUF loading, RoPE/RMSNorm, tokenizer, and host sampling toward end-to-end decode.


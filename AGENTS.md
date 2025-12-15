# Agent Guidelines for m40-llm

This file replaces `llm-context/01-machine_context_summary.md` as the single source of task-level instructions for this repository.

## Introduction
You are continuing development of m40-llm—a Rust LLM runtime/server targeting NVIDIA Tesla M40 (sm_52). Work efficiently and keep changes focused.

## How to Start Each Task
1. Review `CONTRIBUTING.md` and set up the git hooks (pre-commit and commit message hooks).
2. Ensure Cocogitto (`cog`) is installed and working.
3. Ensure `nvcc` is available. If missing, install CUDA Toolkit compatible with Tesla M40 (up to `cuda-nvcc=12.4.*`, `cuda-cudart=12.4.*`, and `cuda-cudart-dev=12.4.*`) and modern `gcc`/`g++`.
4. If you are running in OpenHands (not Codex): Use micromamba to install packages, not apt-get.

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
  - Acceptance highlights: fast-fail invalid GGUFs, logits parity across CUDA/CPU, deterministic sampler, and a CLI decode loop that visibly exercises the GPU.
- **Phase 2 (Fast, M40 optimization):** Optimize cuBLAS wiring and attention while keeping parity grid green.
  - Focus areas: cuBLAS GEMM wrappers (t23-cublas-gemm), full projection wiring (t23-5-integration continuation), attention parity guardrail (t23-5e-attn-green), microbenchmarks (t31b-microbench-attn), stream separation (t33-stream-sep), and persistent decode prototype (t32-persistent-kernel).
- **Phase 3 (Infrastructure, QA, polish):** Stabilize CI, logging, and server polish.
  - Focus areas: clippy cleanups (t23-6-warnings), CUDA benchmark unification (t23-6-benches-ctx), expanded CI matrix (t35-ci-matrix), better errors/telemetry (t34-robustness), `/generate` server parity (t30-server), prefill-vs-decode validation plan (t26-tests-plan), and completing `llm-context` indexing (t41-lc-index).

## Selecting Work
- **Task selection rule:** when choosing work, pick the lowest-numbered unfinished task in the lowest-numbered phase.
- Update this file (`AGENTS.md`) as you complete high-level tasks. 

## Current Tasks
```
[
{
  "phase": 1,
  "name": "Functional (Critical Path)",
  "goal": "Load a real GGUF model and print tokens using the Tesla M40 GPU (host sampling allowed).",
  "success_definition": "CLI run prints decoded text while exercising the Tesla M40 GPU path (no silent CPU fallback).",
  "tasks": [
    {
      "id": "t28-sampling",
      "priority": 5,
      "status": "todo",
      "title": "Host-side sampling (softmax, top-k, top-p)",
      "rationale": "Host sampling keeps the GPU MVP simple and correct.",
      "scope": [
        "Softmax implementation.",
        "Temperature scaling.",
        "Top-k and top-p sampling.",
        "Deterministic RNG."
      ],
      "acceptance": [
        "Greedy and top-k/top-p sampling paths work.",
        "Given logits, sampler returns a token ID deterministically."
      ]
    },
    {
      "id": "t26-min-forward",
      "priority": 6,
      "status": "in_progress",
      "title": "Minimal forward pass (prefill + decode)",
      "rationale": "This is the minimal computation needed to produce logits.",
      "scope": [
        "Single-layer forward correctness.",
        "KV cache write/read path.",
        "Uses real GGUF weights."
      ],
      "acceptance": [
        "Forward produces last-token logits.",
        "Works for both prefill and decode paths."
      ]
    },
    {
      "id": "t25-rope-rmsnorm",
      "priority": 7,
      "status": "todo",
      "title": "RMSNorm and RoPE (GPU-first)",
      "rationale": "Keeping activations on the GPU avoids catastrophic perf loss.",
      "scope": [
        "CUDA kernels for RMSNorm and RoPE.",
        "Host fallback only behind feature or debug flag."
      ],
      "acceptance": [
        "Parity vs CPU on small shapes.",
        "Integrated without host round-trips in hot path."
      ]
    },
    {
      "id": "t22-parity-check",
      "priority": 8,
      "status": "in_progress",
      "title": "CUDA ↔ CPU parity checks",
      "rationale": "Protects correctness while wiring GPU math.",
      "scope": [
        "Last-token attention parity grid.",
        "Run with M40LLM_ENABLE_CUBLAS=0 and =1."
      ],
      "acceptance": [
        "Parity grid stays green within ~1e-3 tolerance.",
        "Coverage includes kv_head_dim variants."
      ]
    },
    {
      "id": "t23-6-autoselect-m40",
      "priority": 9,
      "status": "in_progress",
      "title": "Runtime auto-selection of Tesla M40",
      "rationale": "Prevents accidental arch mismatches on Maxwell.",
      "scope": [
        "Compile for sm_52 at build time.",
        "Prefer Tesla M40 at runtime.",
        "Graceful CPU fallback if CUDA unavailable."
      ],
      "acceptance": [
        "Logs confirm selected device: Tesla M40 (sm_52).",
        "Build fails fast on incompatible CUDA arch."
      ]
    },
    {
      "id": "t23-5-integration",
      "priority": 10,
      "status": "in_progress",
      "title": "GPU minimal forward using cuBLAS (FP32)",
      "rationale": "M40 stability requires FP32 compute with cuBLAS.",
      "scope": [
        "FP16 used only for storage.",
        "Upcast weights to FP32 on device.",
        "Use cublasSgemm and StridedBatched GEMM.",
        "Document and validate layout/transposition rules."
      ],
      "acceptance": [
        "Prefill and decode compute logits fully on GPU.",
        "Numeric parity vs CPU for tiny configs.",
        "No reliance on half arithmetic on sm_52."
      ]
    },
    {
      "id": "t29-e2e-decode",
      "priority": 11,
      "status": "todo",
      "title": "End-to-end decode loop (prints tokens)",
      "rationale": "This is the visible success milestone.",
      "scope": [
        "Prefill initializes device KV cache.",
        "Decode loop runs device forward, copies logits to host, samples next token.",
        "CLI entry point."
      ],
      "acceptance": [
        "CLI run prints decoded text.",
        "GPU path is exercised (confirmed via logs).",
        "No silent CPU fallback."
      ]
    },
    {
      "id": "t36-min-gguf-model",
      "priority": 12,
      "status": "todo",
      "title": "Minimal GGUF test model",
      "rationale": "Fast deterministic tests unblock iteration and CI.",
      "scope": [
        "1–2 layers.",
        "Tiny dimensions.",
        "Deterministic outputs."
      ],
      "acceptance": [
        "cargo test runs prefill + 1–2 decode steps deterministically.",
        "Used by CUDA-gated tests."
      ]
    }
  ]
},
{
  "phase": 2,
  "name": "Fast (M40 Optimization)",
  "goal": "Improve performance on Tesla M40 while preserving numerical correctness and parity guarantees.",
  "success_definition": "End-to-end decode runs significantly faster on M40 with cuBLAS enabled and all parity tests passing.",
  "tasks": [
    {
      "id": "t23-cublas-gemm",
      "priority": 1,
      "status": "in_progress",
      "title": "cuBLAS GEMM wrappers and validation",
      "rationale": "All performance gains depend on correct and well-understood GEMM behavior on Maxwell.",
      "scope": [
        "Harden cuBLAS wrapper APIs.",
        "Validate row-major Rust layouts vs column-major cuBLAS expectations.",
        "Document lda/ldb/ldc and transposition contracts.",
        "Basic performance sanity checks on Tesla M40."
      ],
      "acceptance": [
        "GEMM wrappers are documented and reused consistently.",
        "No silent shape or stride mismatches.",
        "Measured performance is reasonable for M40-class hardware."
      ]
    },
    {
      "id": "t23-5-integration",
      "priority": 2,
      "status": "in_progress",
      "title": "Full GEMM wiring for model layers",
      "rationale": "Moving all major projections onto GPU is required for meaningful speedups.",
      "scope": [
        "Wire Q/K/V projections to device GEMMs.",
        "Wire MLP gate/up/down projections.",
        "Wire output projection.",
        "Keep CPU fallback paths for debugging."
      ],
      "acceptance": [
        "All projections run on GPU when cuBLAS is enabled.",
        "CPU fallback remains functional.",
        "Parity tests pass within tolerance."
      ]
    },
    {
      "id": "t23-5e-attn-green",
      "priority": 3,
      "status": "in_progress",
      "title": "Maintain attention parity grid during optimization",
      "rationale": "Attention bugs are subtle and catastrophic if unchecked.",
      "scope": [
        "Continuously run attention parity grid during integration.",
        "Cover multiple head_dim and kv_head_dim combinations."
      ],
      "acceptance": [
        "Parity grid remains green throughout Phase 2 work.",
        "Failures are caught immediately during refactors."
      ]
    },
    {
      "id": "t31b-microbench-attn",
      "priority": 4,
      "status": "todo",
      "title": "Attention microbenchmarks on M40",
      "rationale": "Optimization requires measurement, not guesswork.",
      "scope": [
        "Microbenchmarks for attention forward pass.",
        "Measure prefill vs decode costs.",
        "Capture baseline vs optimized timings."
      ],
      "acceptance": [
        "Benchmarks run reproducibly on M40.",
        "Results are documented and tracked over time."
      ]
    },
    {
      "id": "t33-stream-sep",
      "priority": 5,
      "status": "todo",
      "title": "Prefill and decode stream separation",
      "rationale": "Separate CUDA streams allow better overlap and latency hiding.",
      "scope": [
        "Introduce distinct CUDA streams for prefill and decode.",
        "Tune stream priorities where applicable."
      ],
      "acceptance": [
        "Prefill and decode run on separate streams.",
        "No correctness regressions.",
        "Measured latency improvement or neutrality."
      ]
    },
    {
      "id": "t32-persistent-kernel",
      "priority": 6,
      "status": "todo",
      "title": "Persistent decode kernel prototype",
      "rationale": "Optional advanced optimization to reduce kernel launch overhead.",
      "scope": [
        "Prototype persistent kernel for decode loop.",
        "Limit scope to experimentation and benchmarking."
      ],
      "acceptance": [
        "Prototype builds and runs behind a feature flag.",
        "Performance impact is measured and documented."
      ]
    },
    {
      "id": "t26-3-impl",
      "priority": 7,
      "status": "todo",
      "title": "Remove remaining host fallbacks in forward path",
      "rationale": "Host fallbacks in hot paths negate GPU gains.",
      "scope": [
        "CUDA RMSNorm and residual paths.",
        "Remove unnecessary host round-trips."
      ],
      "acceptance": [
        "Forward path runs fully on GPU in normal operation.",
        "Parity tests remain green."
      ]
    }
  ]
},
{
  "phase": 3,
  "name": "Infrastructure, QA, and Product Polish",
  "goal": "Stabilize the system, improve developer experience, and make the project production-ready.",
  "success_definition": "Clean CI, clear diagnostics, documented behavior, and a usable server interface.",
  "tasks": [
    {
      "id": "t23-6-warnings",
      "priority": 1,
      "status": "in_progress",
      "title": "Eliminate warnings and keep clippy clean",
      "rationale": "Warnings hide real problems and reduce signal quality.",
      "scope": [
        "Remove unused CudaContext imports.",
        "Keep clippy -D warnings green across feature combinations."
      ],
      "acceptance": [
        "No warnings in default or CUDA builds.",
        "CI treats warnings as errors."
      ]
    },
    {
      "id": "t23-6-benches-ctx",
      "priority": 2,
      "status": "todo",
      "title": "Unify benchmarks on ctx_m40",
      "rationale": "Benchmarks must reflect real target hardware.",
      "scope": [
        "Update benches to use cuda_env::ctx_m40().",
        "Remove ad-hoc device selection logic."
      ],
      "acceptance": [
        "All CUDA benchmarks use ctx_m40.",
        "Bench results are comparable across runs."
      ]
    },
    {
      "id": "t35-ci-matrix",
      "priority": 3,
      "status": "todo",
      "title": "Expand CI build and test matrix",
      "rationale": "Prevent regressions across feature combinations.",
      "scope": [
        "CUDA vs non-CUDA builds.",
        "Different nvcc / feature flag combinations.",
        "Document CI expectations."
      ],
      "acceptance": [
        "CI covers all supported build modes.",
        "Failures clearly indicate configuration."
      ]
    },
    {
      "id": "t34-robustness",
      "priority": 4,
      "status": "todo",
      "title": "Robust error handling and telemetry",
      "rationale": "Clear errors and logs reduce debugging time dramatically.",
      "scope": [
        "Improve error messages in inference and loading paths.",
        "Add basic telemetry and timing logs."
      ],
      "acceptance": [
        "Common failure modes have actionable messages.",
        "Logs show device, timing, and execution path."
      ]
    },
    {
      "id": "t30-server",
      "priority": 5,
      "status": "todo",
      "title": "HTTP server wired to real decode path",
      "rationale": "Server interface is required for real-world usage.",
      "scope": [
        "Wire /generate endpoint to real decode loop.",
        "Support streaming output and CORS."
      ],
      "acceptance": [
        "/generate produces valid streamed or buffered output.",
        "Uses the same decode path as CLI."
      ]
    },
    {
      "id": "t26-tests-plan",
      "priority": 6,
      "status": "todo",
      "title": "Formal prefill vs decode validation plan",
      "rationale": "Decode paths often diverge subtly from prefill.",
      "scope": [
        "Define expected invariants between prefill and decode.",
        "Add CUDA-gated tests where appropriate."
      ],
      "acceptance": [
        "Tests catch prefill/decode divergence.",
        "Documented validation strategy exists."
      ]
    },
    {
      "id": "t41-lc-index",
      "priority": 7,
      "status": "in_progress",
      "title": "Finalize llm-context index and cross-links",
      "rationale": "Contextual documentation improves long-term maintainability.",
      "scope": [
        "Finalize relevance tags.",
        "Cross-link related design notes."
      ],
      "acceptance": [
        "llm-context index is complete and consistent.",
        "Key design decisions are discoverable."
      ]
    }
  ]
}
]
```

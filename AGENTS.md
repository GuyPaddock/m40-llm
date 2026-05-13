# Agent Guidelines for m40-llm

This file replaces `llm-context/01-machine_context_summary.md` as the single source of task-level instructions for this repository.

## Introduction
You are continuing development of m40-llm—a Rust LLM runtime/server targeting NVIDIA Tesla M40 (sm_52). Work efficiently and keep changes focused.

## How to Start Each Task
1. Review `CONTRIBUTING.md` and set up the git hooks (pre-commit and commit message hooks).
2. Ensure Cocogitto (`cog`) is installed and working.
3. Ensure `nvcc` is available. If missing, install a CUDA Toolkit that still supports compiling sm_52 kernels, plus cuBLAS development headers and modern `gcc`/`g++`.
   - The micromamba CUDA **12.4** command in `README.md` is the reproducible fallback; the build also detects local `/opt/cuda` installs.
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

## Active Performance Plan
The current plan still prioritizes measurement, ownership, and correctness before deeper scheduling and backend expansion.
Packed varlen attention integration is now moving into scheduler execution after request/session
ownership and KV addressing are established; larger experiments should wait for a validated
batched decode path before touching persistent decode or large-model fused-dequant.

## Selecting Work
- **Task selection rule:** use the strict reconciled task order below. The historical
  task JSON remains useful context, but this strict order takes precedence when the
  two conflict.
- Update this file (`AGENTS.md`) as you complete high-level tasks. 

## Current Progress
- Completed: warm/cold benchmark mode names, launch/sync/allocation/copy counter
  instrumentation, per-forward-operation profile counter deltas, HTTP generation
  serialization, and a shared CUDA `DecodeSession` for CLI/server decode with
  reusable `d_x`/`d_out` scratch. `ForwardWorkspace` and decode-session scratch
  now use RAII `DeviceBuffer` cleanup. Full-layer forward now uses explicit
  model-level KV layer/sequence addressing instead of passing `layer as seq_id`
  directly. FP32 materialized projection weights have budget reporting and
  over-budget fallback logging plus tensor identity metadata in the cache key.
  `DecodeSession` now also owns reusable `d_logits` and optional
  `d_norm_hidden` scratch for CUDA logits. Hot CUDA wrappers now expose async
  enqueue variants while preserving existing sync wrappers for tests/simple
  callers. Warm second-token launch/sync profiling after async wrapper cleanup
  is recorded in `docs/perf_baselines.md`. The forward path now fuses K RoPE
  with FP32-to-FP16 KV append while leaving Q RoPE separate; profiling reduced
  RoPE/KV operation groups from 66 to 44 launches/syncs per steady TinyLlama
  token. Full-layer forward now enqueues the already-fused SwiGLU kernel
  asynchronously and uses an explicit decode-to-prefill stream wait before the
  MLP down projection, removing 22 explicit SwiGLU stream synchronizations per
  steady token. Full-layer forward now also uses async enqueue paths for RMSNorm,
  Q RoPE, fused K RoPE + KV append, GQA attention, and residual adds; steady
  TinyLlama profiling shows non-GEMM forward ops now contribute zero stream
  synchronizations, while the remaining forward syncs are cuBLAS GEMM wrappers.
  CUDA Graph capture/instantiate/launch/destroy infrastructure is in place and
  validated with fixed-pointer decode-style async elementwise work on M40;
  whole-token capture is still blocked by synchronous cuBLAS wrappers and
  host-side KV sequence length updates. Packed varlen decode scheduling now has
  a request-state `DecodeBatchPlan`, device metadata upload, and CUDA dispatch
  through the batched GQA decode attention primitive; HTTP `/generate` remains
  serialized. Model-level KV addressing now maps logical
  `KV[layer][sequence]` onto sequence-major physical slots, and the server has a
  small decode sequence lease pool. With `M40LLM_SERVER_BATCH_DECODE=1`,
  `/generate` requests lease per-request KV sequence slots and skip whole-cache
  resets; buffered scheduler ticks can now use the packed batched GQA decode
  attention path for head_dim=64 models. Materialized
  FP32 cuBLAS projection calls now have async enqueue wrappers, and full-layer
  decode uses those wrappers with explicit stream waits; steady TinyLlama
  profiling shows projection groups issue 154 cuBLAS calls with zero stream
  synchronizations inside full-layer forward. KV append now has explicit-position
  and device-position async APIs; production full-layer decode uses the
  explicit-position path, which removes the prior host-side KV length D2H read
  and updates `seq_map` on device. A one-layer-shaped CUDA Graph prototype now
  captures and replays seven async materialized-FP32 cuBLAS projection GEMMs on
  M40. A cross-stream graph prototype also captures the production dependency
  topology: decode-stream elementwise work, prefill-stream async cuBLAS, and a
  return wait back to decode-stream elementwise work. A production one-layer
  graph smoke now captures a warmed `forward_one_token_with_layer` call using
  real model/workspace pointers, KV append, attention, and projection wrappers.
  Graph-compatible device-parameter wrappers now cover Q RoPE position and GQA
  attention sequence length, complementing the existing device-position KV
  append API. `DecodeSession` can now cache and replay an opt-in full-token
  decode CUDA Graph with `M40LLM_DECODE_GRAPH=1`, including multi-layer models.
  The async materialized-GEMM path briefly regressed generation by letting
  SwiGLU read MLP gate/up outputs before the prefill-stream cuBLAS work
  completed; `mlp_gate_up_to_swiglu` now restores the decode-stream wait, and a
  TinyLlama token canary covers the reported prompt. Full-token graph replay was
  benchmarked against the normal async path across three TinyLlama trials:
  graph replay reduces host-side `forward_all_layers` enqueue time, but
  end-to-end steady token latency regresses because logits/output-norm absorbs
  much larger GPU completion time. Keep `M40LLM_DECODE_GRAPH=1` experimental and
  off by default. With `M40LLM_SERVER_BATCH_DECODE=1`, buffered `/generate`
  requests now route through a queued decode scheduler that owns request state,
  leases distinct KV sequence slots, steps active requests round-robin, and
  builds `DecodeBatchPlan` snapshots for active mixed-length requests. The
  scheduler keeps the server generation lock around each CUDA token step to
  protect shared workspace use across scheduler and streaming paths. The
  scheduler now steps all active requests every scheduler tick and can execute
  row-batched full-layer decode with packed GQA attention for compatible
  head64 models. `M40LLM_DECODE_GRAPH_DIAG_SYNC=1`
  now synchronizes graph replay immediately after launch and reports CUDA-event
  GPU elapsed time; this showed the graph replay itself is slow, while
  logits/output-norm was previously absorbing graph completion time. The server
  batch scheduler now has an opt-in batched full-layer decode path for
  head_dim=64 models: it packs active request rows into the shared forward
  workspace, runs row-batched projections/MLP, uses the existing packed batched
  GQA decode attention primitive, and scatters per-request outputs back into
  `DecodeSession` scratch. It falls back to the prior per-request path when the
  batch is size 1 or the model cannot use the head64 batched attention kernel.
  TinyLlama concurrent buffered `/generate` benchmarking shows
  `M40LLM_SERVER_BATCH_DECODE=1` is neutral for batch size 1 and improves
  throughput by 1.18x for batch size 2, 1.69x for mixed batch size 4, and 1.61x
  for skewed batch size 4, with all requests returning HTTP 200. Results and
  validation commands are recorded in `docs/perf_baselines.md`.
- Packed varlen prefill is now available behind
  `M40LLM_SERVER_BATCH_PREFILL=1` for compatible head_dim=64 buffered scheduler
  batches; TinyLlama benchmarking shows neutral batch-1 behavior, 1.12x batch-2
  speedup, 1.88x mixed batch-4 speedup, and 2.51x skewed batch-4 wall-time
  speedup with all HTTP requests successful.
- Experimental KV compression modes are available for CLI decode attention:
  `block-select-exact` keeps old exact KV while sparsifying attention, and
  `block-summary` / `block-select-lossy` now use a physical compressed CUDA
  sidecar for CLI decode with a recent exact ring plus old-block mean K/V
  summaries. M40 attention microbenchmarks at 4K/8K/16K/32K are recorded in
  `docs/perf_baselines.md`. The retrieval quality harness now requires
  `M40LLM_LONG_CONTEXT_RETRIEVAL_MODEL` instead of scanning cache trees, probes
  metadata, and reports pass/fail/inconclusive rows. CUDA logits support
  compatible tied F16 output embeddings in `[d_model, vocab]` GGUF layout, which
  unblocks full-layer decode for the cached Llama 3.2 1B Instruct F16 GGUF.
  Llama 3 GGUF tokenizer metadata now routes through the `tiktoken` Llama 3
  encoding, auto prompt formatting applies the Llama 3 chat control-token
  wrapper, and stop criteria include both end-of-turn and end-of-text tokens.
  The GGUF loader now honors aligned tensor data offsets, including the default
  32-byte alignment when the aligned payload fits. This fixes the Llama 3.2
  repeated-token-0/NaN-logits failure. The KV retrieval harness now defaults to
  bounded 64/512 old/recent smoke targets, supports explicit
  `M40LLM_KV_QUALITY_TARGETS`, gates per-token decode-session diagnostics behind
  `M40LLM_DECODE_SESSION_LOG=1`, and emits JSONL rows with actual prompt token
  counts, generated token counts, pass/fail status, full output, and
  prompt/decode/total timing breakdowns. `M40LLM_PREFILL_CHUNK_SIZE` now enables
  an experimental CLI/test packed-prefix prefill path for dense `off` prompts
  within the configured bound; it pre-fills the prompt prefix with packed varlen
  prefill and computes the final prompt token through the normal one-token path.
  Dense 512-token prefill improves from roughly 56-59 s to roughly 10 s, while
  KV-compressed modes deliberately fall back to sequential prefill because
  packed prefill is not yet equivalent for those cache modes. The quiet 64-token
  and 512-token old/recent smokes pass for dense `off`, `block-select-exact`,
  `block-summary`, and `block-select-lossy`; results are recorded in
  `docs/perf_baselines.md`. `M40LLM_KV_COMPRESSED_PREFILL_CHUNK_SIZE` now
  enables an explicit compressed-aware chunked prefill path for KV-compressed
  quality runs. It preserves sequential token order and compressed sidecar
  updates while skipping prefix-token logits. CUDA parity tests compare final
  logits and compressed-KV debug snapshots for `block-summary` and
  `block-select-lossy`. The retrieval harness passes 64-token old/recent cases
  with `M40LLM_KV_COMPRESSED_PREFILL_CHUNK_SIZE=32` and 512-token old/recent
  cases with `M40LLM_KV_COMPRESSED_PREFILL_CHUNK_SIZE=64` for
  `block-select-exact`, `block-summary`, and `block-select-lossy`. The chunked
  path improves compressed 64-token prefill to roughly 2.1 s and compressed
  512-token prefill by roughly 6-7 s, but 512-token compressed rows still take
  roughly 51-59 s of prefill. `M40LLM_KV_PACKED_THEN_COMPRESS_PREFILL=1` now
  enables a faster experimental path for `block-summary` and
  `block-select-lossy`: packed dense prefill builds a temporary dense KV cache,
  a compression pass constructs the final compressed sidecar, and decode
  continues on the compressed cache. CUDA parity tests compare final logits and
  compressed snapshots for both modes. The 512-token old/recent rows pass with
  roughly 9.8-10.3 s prefill, and 1024-token old/recent rows pass with roughly
  38-39 s prefill. `block-select-exact` remains sequential and is now the slow
  quality row. `M40LLM_KV_QUALITY_LOSSY_PACKED_SWEEP=1` now runs a bounded
  lossy-only sweep over dense `off`, `block-summary`, and `block-select-lossy`
  with 1024/2048/4096 old/recent targets and reports final compressed KV bytes,
  dense-equivalent KV bytes, and temporary dense KV bytes. The sweep passes at
  1024 for lossy modes, but 2048 and 4096 lossy retrieval currently fail even
  though dense `off` passes; this is now documented as a quality limitation, not
  a runtime failure. The compressed sidecar now supports opt-in exact
  representatives per old block for `block-summary` and `block-select-lossy`.
  `--kv-compress-representatives` defaults to 0 and activates physical
  representative K/V storage when set above 0; `last` and `stride` selection
  policies are implemented. Sequential and packed-then-compress debug snapshot
  parity passes for representative storage. A 64-token representative quality
  spot-check passed the harness but was inconclusive because dense `off` missed
  the exact needle in that short packed-prefix case. The quality harness now
  streams each JSONL row as soon as it completes. The representative matrix
  shows 1024-token old/recent retrieval passes for `last` and `stride`, but
  2048-token old/recent retrieval still fails for `block-summary` and
  `block-select-lossy` at reps 0/1/2/4 with `last`, while dense `off` passes.
  4096 representative rows were skipped because 2048 already fails decisively.
  `M40LLM_KV_QUALITY_EXACT_SELECTION_SWEEP=1` now enables a diagnostic sweep
  with block-selection telemetry. JSONL rows include `needle_block_index`,
  `selected_block_indices`, `needle_block_selected`, `needle_block_rank`,
  `total_old_blocks`, and `top_blocks`. A 2048-token top-4 diagnostic on
  Llama-3.2-1B shows dense `off` and `block-select-exact` pass, while
  `block-summary` and `block-select-lossy` fail even when the old needle block
  is selected at rank 0; for recent needles, lossy modes fail despite the target
  living in the exact recent window. Treat this as evidence that current lossy
  summaries/reps are the bottleneck rather than sparse block selection.
- Next: do not increase representative count further. Investigate alternative
  lossy summary designs or attention weighting/masking behavior before
  expanding compressed KV into server scheduling.

## Strict Reconciled Task Order
1. Add warm/cold benchmark split.
2. Add launch/sync/allocation/copy count instrumentation.
3. Profile warm steady second-token decode with per-kernel launch counts.
4. Add request-level generation serialization to protect shared workspace.
5. Extract shared `DecodeSession` used by CLI and server.
6. Move `d_x` and `d_out` into `DecodeSession` scratch.
7. Make `ForwardWorkspace` / device allocations RAII-safe.
8. Fix KV cache API/addressing to distinguish `layer_id` from `sequence_id`.
9. Add FP32 materialization memory-budget reporting and fallback logging.
10. Improve materialized weight cache key with tensor identity/offset/name.
11. Move `d_logits` and `d_norm_hidden` into `DecodeSession` scratch.
12. Add async enqueue variants for hot CUDA kernels; keep sync wrappers.
13. Re-profile launch/sync counts after async/session cleanup.
14. Fuse RoPE + KV append if still visible.
15. Treat SwiGLU as already fused; remove sync/graph it before deeper fusion.
16. Prototype CUDA Graph capture for warm one-token decode.
17. Integrate packed varlen decode attention into the server scheduler.
18. Integrate packed varlen prefill after decode batching is correct.
19. Add fast-fits vs large-model backend selection.
20. Start large-model fused-dequant projection backend.
21. Leave `__ldg`/texture experiments off by default unless profiling reopens them.

## Reconciled Phase Details
- **Phase 0 (Measurement):** add warm/cold benchmark modes and per-token counts for
  CUDA launches, cuBLAS calls, stream synchronizations, allocations/frees, H2D
  copies, and D2H copies. Use env gates such as `M40LLM_LAUNCH_LOG=1`,
  `M40LLM_SYNC_LOG=1`, and `M40LLM_ALLOC_LOG=1`.
- **Phase 1 (Ownership):** serialize generation first, then introduce
  `DecodeSession`, reusable token scratch, RAII `DeviceBuffer`, and explicit
  `KV[layer][sequence][position][kv_head][head_dim]` addressing.
- **Phase 2 (Fast-fits):** preserve materialized FP32 projection weights plus
  cuBLAS as the TinyLlama-class backend, add materialization budget/fallback
  logging, improve cache keys, and split sync wrappers from async enqueue wrappers.
- **Phase 3 (Graphs):** prototype CUDA Graph capture only after stable session
  scratch, async wrappers, stable workspace pointers, and explicit KV addressing
  exist.
- **Phase 4 (Scheduler):** integrate packed varlen decode first, packed prefill
  second, and mixed prefill/decode overlap last.
- **Phase 5 (Backends):** add fast-fits vs large-model selection before any
  large-model fused-dequant projection backend.

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
      "status": "done",
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
      "status": "done",
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
      "status": "done",
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
      "status": "done",
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
      "status": "done",
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
      "status": "done",
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
      "status": "done",
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
      "status": "done",
      "title": "cuBLAS GEMM wrappers and validation",
      "rationale": "All performance gains depend on correct and well-understood GEMM behavior on Maxwell.",
      "scope": [
        "Harden cuBLAS wrapper APIs.",
        "Validate row-major Rust layouts vs column-major cuBLAS expectations.",
        "Document lda/ldb/ldc and transposition contracts.",
        "Basic performance sanity checks on Tesla M40.",
        "Materialize hot GGUF F16 projection weights into FP32 device buffers for cublasSgemm."
      ],
      "acceptance": [
        "GEMM wrappers are documented and reused consistently.",
        "No silent shape or stride mismatches.",
        "Measured performance is reasonable for M40-class hardware.",
        "Steady-state TinyLlama projection timings improve materially on M40."
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
      "status": "done",
      "title": "Attention microbenchmarks on M40",
      "rationale": "Optimization requires measurement, not guesswork.",
      "scope": [
        "Microbenchmarks for attention forward pass.",
        "Measure prefill vs decode costs.",
        "Capture baseline vs optimized timings."
      ],
      "acceptance": [
        "Benchmarks run reproducibly on M40 (`cargo bench --features cuda --bench attention`).",
        "Results are documented and tracked in `docs/perf_baselines.md`."
      ]
    },
    {
      "id": "t31c-optimize-attn-kernel",
      "priority": 5,
      "status": "done",
      "title": "Optimize GQA last-token attention kernel",
      "rationale": "The new attention benchmark shows current GQA attention dominates decode latency at practical context lengths.",
      "scope": [
        "Improve `m40llm_attention_last_token_f32_gqa` for M40.",
        "Keep attention parity grid green.",
        "Remeasure attention and TinyLlama `/generate` latency."
      ],
      "acceptance": [
        "Attention benchmark improves materially for seq_len 128+.",
        "CUDA attention parity tests pass.",
        "Updated measurements are recorded in `docs/perf_baselines.md`."
      ]
    },
    {
      "id": "t31d-optimize-rmsnorm",
      "priority": 6,
      "status": "done",
      "title": "Optimize RMSNorm decode kernels",
      "rationale": "Short-context decode profiles showed serial per-row RMSNorm consumed roughly 17 ms per layer.",
      "scope": [
        "Replace one-thread-per-row RMSNorm with parallel per-row reductions.",
        "Keep weighted and unweighted RMSNorm parity tests green.",
        "Remeasure TinyLlama CLI decode latency."
      ],
      "acceptance": [
        "Norm latency is no longer a dominant short-context decode cost.",
        "CUDA RMSNorm/RoPE tests pass.",
        "Updated measurements are recorded in `docs/perf_baselines.md`."
      ]
    },
    {
      "id": "t33-stream-sep",
      "priority": 7,
      "status": "done",
      "title": "Prefill and decode stream separation",
      "rationale": "Separate CUDA streams allow better overlap and latency hiding after projection and norm costs are reduced.",
      "scope": [
        "Introduce distinct CUDA streams for prefill and decode.",
        "Tune stream priorities where applicable.",
        "Add async enqueue variants for independent prefill/decode attention benchmarks."
      ],
      "acceptance": [
        "Prefill and decode streams are non-blocking, with best-effort decode priority.",
        "Async prefill and decode attention paths have CUDA parity coverage.",
        "Measured overlap benchmark improved from ~47.07 ms sequential to ~45.75 ms async final-sync on M40."
      ]
    },
    {
      "id": "t32-persistent-kernel",
      "priority": 8,
      "status": "done",
      "title": "Persistent decode kernel prototype",
      "rationale": "Optional advanced optimization to reduce kernel launch overhead.",
      "scope": [
        "Prototype persistent kernel for decode loop.",
        "Limit scope to experimentation and benchmarking.",
        "Use a synthetic decode-style vector command before attempting full transformer integration."
      ],
      "acceptance": [
        "Prototype builds and runs behind Rust lifecycle wrappers.",
        "CUDA lifecycle test covers start, submit, poll, and idempotent stop.",
        "Synthetic M40 benchmark improved from ~32.3 us launch-based work to ~28.2 us persistent-worker work."
      ]
    },
    {
      "id": "t26-3-impl",
      "priority": 9,
      "status": "todo",
      "title": "Remove remaining host fallbacks in forward path",
      "rationale": "Host fallbacks in hot paths negate GPU gains.",
      "scope": [
        "Audit remaining host round-trips in embedding/logits and debug fallback paths.",
        "Keep normal full-layer decode on device except logits copyback for host sampling."
      ],
      "acceptance": [
        "Forward path runs fully on GPU in normal operation.",
        "Parity tests remain green."
      ]
    },
    {
      "id": "t31e-varlen-batch",
      "priority": 10,
      "status": "in_progress",
      "title": "Variable-length batched prefill and attention",
      "rationale": "Batched serving should avoid padded-token computation for mixed-length requests.",
      "scope": [
        "Add batch metadata with valid lengths and packed offsets.",
        "Add length buckets before fully packed variable-length attention.",
        "Add M40-safe variable-length attention kernels without Tensor Core assumptions.",
        "Benchmark padded, bucketed, and packed variants."
      ],
      "acceptance": [
        "Batch metadata exposes valid lengths, packed offsets, and deterministic length buckets.",
        "Batched last-token GQA attention supports mixed KV lengths for decode.",
        "Mixed-length batches avoid full max_seq padding work where possible.",
        "Packed prefill GQA attention supports mixed query/KV lengths.",
        "Benchmarks cover padded, packed, and bucketed prefill dispatch for skewed, 0.6*max_seq average, and near-uniform length distributions.",
        "Remaining work: tune variable tile selection and integrate packed prefill above the kernel/benchmark layer."
      ]
    },
    {
      "id": "t31f-readonly-cache-experiments",
      "priority": 11,
      "status": "in_progress",
      "title": "Read-only cache experiments for non-GEMM paths",
      "rationale": "M40 read-only cache and texture paths may help read-heavy kernels, but only measured wins should become defaults.",
      "scope": [
        "Benchmark opt-in `__ldg` and texture-cache variants for embeddings, RoPE constants, dequant tables, KV reads, norms, and activation LUTs.",
        "Keep experiments gated until parity and performance justify default use."
      ],
      "acceptance": [
        "Weighted RMSNorm has an opt-in `M40LLM_CACHE_EXPERIMENT=ldg` path with parity coverage.",
        "RMSNorm default-vs-`__ldg` benchmark results are recorded.",
        "KV-cache GQA attention has an opt-in `M40LLM_CACHE_EXPERIMENT=ldg_kv` path with parity coverage.",
        "KV-cache default-vs-`ldg_kv` benchmark results are recorded.",
        "Remaining work: defer texture objects until profiling identifies a stronger read-cache target."
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
      "status": "done",
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
      "status": "in_progress",
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

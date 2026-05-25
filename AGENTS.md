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
  Projection GEMMs now have explicit `auto`, `fast-fits`, and `large-model`
  backend selection. Auto estimates uploaded weights, materialized FP32 cache,
  one-row forward workspace, and resident KV against
  `M40LLM_FAST_FITS_BUDGET_MB`; `large-model` currently means the compact GGUF
  fallback path without full FP32 materialization, while fused dequant
  projection kernels are starting with an opt-in GGUF Q8_0 projection primitive
  that computes `f32 x Q8_0 -> f32` and dequantizes inside the CUDA kernel.
  Q8_0 projection dispatch now uses an `M=1,K%32=0` decode-tiled kernel for
  single-token decode, a shared-activation kernel for prefill-shaped multi-row
  work, and a block-loop kernel for smaller non-decode shapes; scalar Q8_0
  remains as a benchmark/debug baseline. Benchmarks show shared activation
  brings Qwen prefill64 Q/O below the F16 fallback, while materialized FP32
  cuBLAS remains much faster when fast-fits is available. A CUDA-only
  `q8_generation_canary` test now probes an explicit
  `M40LLM_Q8_GENERATION_MODEL` and runs bounded full generation only when the
  supplied Q8_0 GGUF has supported LLaMA/Qwen-style metadata and standard Q8_0
  projection coverage; unsupported models are reported as coverage gaps rather
  than kernel failures. The Ollama `qwen2.5:3b-instruct-q8_0` blob now passes
  that canary after enabling Q8_0 tied output embeddings for logits; the current
  prompt asks `What is 2+2? Answer with one digit.` and generated `4` with
  fused Q8_0 projection launches.
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
  head64/head128 dense-KV models. `M40LLM_DECODE_GRAPH_DIAG_SYNC=1`
  now synchronizes graph replay immediately after launch and reports CUDA-event
  GPU elapsed time; this showed the graph replay itself is slow, while
  logits/output-norm was previously absorbing graph completion time. The server
  batch scheduler now has an opt-in batched full-layer decode path for
  head_dim=64/128 models: it packs active request rows into the shared forward
  workspace, runs row-batched projections/MLP, uses the existing packed batched
  GQA decode attention primitive, and scatters per-request outputs back into
  `DecodeSession` scratch. It falls back to the prior per-request path when the
  batch is size 1 or the model cannot use the head64/head128 batched attention
  kernel.
  TinyLlama concurrent buffered `/generate` benchmarking shows
  `M40LLM_SERVER_BATCH_DECODE=1` is neutral for batch size 1 and improves
  throughput by 1.18x for batch size 2, 1.69x for mixed batch size 4, and 1.61x
  for skewed batch size 4, with all requests returning HTTP 200. Results and
  validation commands are recorded in `docs/perf_baselines.md`.
- Packed varlen prefill is now available behind
  `M40LLM_SERVER_BATCH_PREFILL=1` for compatible head64 dense buffered scheduler
  batches; TinyLlama benchmarking shows neutral batch-1 behavior, 1.12x batch-2
  speedup, 1.88x mixed batch-4 speedup, and 2.51x skewed batch-4 wall-time
  speedup with all HTTP requests successful. The dense server scheduler now
  partitions each tick into pending-prefill, decode-ready, and complete groups
  so mixed ticks can run packed prefill and packed decode in the same scheduler
  cycle. `M40LLM_SERVER_BATCH_LOG=1` reports queue size, per-tick composition,
  and packed-decode fallback reasons. Preferred compressed KV
  (`block-select-exact`, direct FP16-K/q4-V exact-old attention, top-k) now
  participates in the queued server scheduler with distinct logical KV
  sequence leases. The scheduler admits that preferred compressed runtime,
  allocates multi-sequence compressed KV slots, uses packed-prefix prefill for
  the verified head64 compressed path, and routes compatible compressed decode
  rows through a batched direct FP16-K/q4-V exact-old attention kernel. The
  CUDA parity test compares batched compressed attention against individual
  direct compressed decode for `head_dim=64` and `head_dim=128`; server smoke
  asserts the preferred compressed scheduler launches the batched compressed
  attention op. Unsupported compressed batching configs fail or log clear
  fallback reasons instead of silently routing through dense batched attention.
  `scripts/bench_server_batch_decode.sh` now passes
  dense `--kv-compress-mode off` by default and a 2026-05-24 single-trial
  TinyLlama check shows packed prefill plus batched decode remains the useful
  path for short mixed/skewed batches. The script now accepts
  `CARGO_RUN_ARGS="--release"`; release TinyLlama confirmation shows
  decode+prefill batching reaches 6.35x to 9.41x wall-time speedup on short
  batch-4 mixed/skewed prompts, while decode-only batching remains neutral.
  `CASES="..."` filters the benchmark matrix; a bounded release
  `MAX_TOKENS=16` run shows decode-only batching becomes useful once decode work
  is visible, improving batch2 by 1.51x and batch4 mixed by 1.21x, while
  decode+prefill remains strongest. Dense scheduler tests now assert
  scheduler-level profile events for batched prefill/decode ticks in addition
  to the underlying CUDA kernel launches, making scheduler decisions visible
  through `profile::snapshot()` and `M40LLM_LAUNCH_LOG=1`. Dense server
  batched decode now also admits `head_dim=128` models through a dedicated
  batched GQA head128 CUDA specialization and lifted scheduler/forward guards.
  `--max-context-tokens` / `MAX_CONTEXT_TOKENS` is available for bounded dense
  KV server benchmarking; uncapped Qwen2.5 dense batching can exceed M40 VRAM
  when eight full-context slots and materialized FP32 weights are both present.
  CUDA parity, full-layer forward smoke, and server smoke tests cover head128
  batched decode and preferred compressed-KV batched decode. Qwen-shaped
  head128 packed-prefix and batched-prefill parity tests now pass with Q/K/V
  biases and split-half RoPE. Server packed prefill now uses prompt-prefix
  semantics, leaving the final prompt token on the normal one-token path.
  Head128/Qwen server admission now uses multi-row packed-prefix prefill in
  scheduler ticks for dense and preferred compressed KV paths. A bounded
  Qwen2.5 release run with `MAX_CONTEXT_TOKENS=512` shows batch4 mixed output
  parity and a 4.32x wall-time speedup versus dense serial.
  Preferred compressed-KV packed-prefix prefill is now also admitted for
  head128/Qwen-shaped server requests. CUDA parity compares Qwen-shaped
  compressed packed-prefix prefill against sequential compressed prefill using
  final logits and compressed-KV snapshots with Q/K/V biases and split-half
  RoPE. A server smoke confirms two concurrent compressed head128 requests
  record `server_scheduler_compressed_packed_prefill_tick`. A bounded Qwen2.5
  release sanity check with `MAX_CONTEXT_TOKENS=512` and
  `--kv-recent-window 256` returns HTTP 200 for compressed top8 batch2, but is
  slightly slower than dense off on that tiny short-prompt row; treat it as
  admission evidence, not a long-context compression speed claim.
  The Qwen-like parity test covers `head_dim=128`, `q_heads=16`, `kv_heads=2`,
  Q/K/V biases, split-half RoPE, and compressed KV snapshots. A bounded real
  Qwen2.5 release run with `MAX_CONTEXT_TOKENS=512` improves compressed top8
  mixed batch4 wall time to 1104 ms / 7.246 tok/s. The release-only real
  Qwen mixed-length blocker is now isolated and fixed: a standalone
  Qwen-shaped packed-prefill attention parity test reproduces the old NaN
  failure and now passes, and the real Qwen release diagnostic shows
  mixed-length multi-row prefill matching the safe single-row packed-prefix
  path for prefix hidden vectors and generated tokens. The varlen prefill
  metadata plan is kept alive until final stream drain, packed prefill drains
  both streams before releasing shared workspace ownership, stream waits use
  per-wait CUDA events, and the packed-prefill attention kernel uses a fixed
  shared-memory score region separate from reduction scratch. Mixed-length
  head128/Qwen dense and preferred compressed multi-row prefill are now
  admitted when `M40LLM_SERVER_BATCH_PREFILL=1`. A follow-up bounded Qwen
  release run after dense admission shows dense `off` and compressed top8 are
  effectively tied on short 512-context prompts: batch4 mixed is 1132 ms /
  7.067 tok/s dense versus 1136 ms / 7.042 tok/s compressed, with matching
  outputs.
  Staggered server scheduler coverage now records
  `server_scheduler_mixed_prefill_decode_tick` whenever prompt-prefill rows and
  decode rows share a tick. `scripts/bench_server_batch_decode.sh` supports
  `STAGGER_MS` plus `staggered_mixed` / `staggered_skewed` cases and enables
  scheduler tick logs by default. A bounded TinyLlama release run with
  `staggered_mixed` shows packed prefill plus decode scheduling improving wall
  time from 1362 ms dense serial to 398 ms, while a wider-stagger run confirms
  mixed ticks with both `prefill_rows` and `decode_rows` nonzero.
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
  `recent-only` is now available as a diagnostic compressed KV mode that attends
  only the exact recent ring. The quality harness also records first-captured
  attention group telemetry: recent mass, selected old exact mass, summary mass,
  representative mass, needle-block mass, top attended entries, and logit
  max/mean by group. The 2048 telemetry shows summary probability mass is near
  zero, not excessive, and `recent-only` still fails for the 2048 recent needle
  while dense and `block-select-exact` pass. A dense recent-window diagnostic
  confirmed that compressed recent-only matches dense recent-window under
  matching sequential semantics; the recent ring path is numerically sound and
  long-context retrieval needs exact old context. `block-select-exact` is now
  the architectural diagnostic for summary-indexed exact-block retrieval. The
  staged exact-block prototype validates the data flow by gathering selected old
  exact K/V plus the exact recent window into a compact working set before
  attention. `DecodeSession` now allocates reusable exact-block staging
  workspaces when `M40LLM_KV_EXACT_BLOCK_STAGING=1` and
  `block-select-exact` is active; JSONL rows report
  `staged_workspace_reused`, workspace capacity/bytes, and one allocation per
  session. The 2048 reusable-staged sweep preserves the known pass/fail pattern:
  `top_blocks=1` fails and `top_blocks>=2` passes for old/recent needles.
  `M40LLM_KV_EXACT_OLD_BACKING=q8` now uses a hybrid compressed/exact
  `block-select-exact` layout instead of dense KV plus a rebuilt q8 sidecar:
  dense `off` remains the FP16 reference, recent tokens stay in an exact FP16
  ring, old tokens are quantized incrementally as they age out of the ring, and
  selected old blocks are dequantized into the reusable staging workspace. A
  2048 Llama-3.2-1B sweep preserves the same quality pattern (`top_blocks=1`
  fails, `top_blocks>=2` passes) while reducing final KV allocation from
  4.00 GiB dense-equivalent to 2.53 GiB for q8 exact-block rows. An opt-in
  `M40LLM_KV_EXACT_OLD_ATTENTION=q8-direct` backend now skips the old-K/V FP16
  staging round trip and dequantizes q8 old K/V inside the attention kernel
  while preserving staged FP16 rounding semantics. A bounded 2048 sweep with
  `top_blocks=1,2,4,8,16` preserves the same quality pattern and improves
  passing direct-q8 decode rows versus staged-q8 from roughly 7.7-8.0 s to
  5.5-5.8 s for the smallest passing rows. A 4096 sweep shows direct-q8 is not
  ready as a default: old-needle retrieval requires `top_blocks>=8`, and
  recent-needle retrieval is non-monotonic (`top_blocks=4` and `16` pass while
  `8` fails). The 4096 direct-q8 diagnostics now record score-ranked selected
  blocks, block scores/ranges, candidate-order flags, selected-block attention
  mass, active attended KV size, and dense-vs-compressed prompt/first-decode
  logit drift. The diagnostic q8 scorer now bulk-copies q8 old K/scales to host
  instead of dereferencing device pointers. Results show the old-needle block is
  selected at rank 0 even when top_blocks=4 fails, while recent top_blocks=8
  fails after the first answer token despite first-token logits matching dense
  at top-token level and more than 99.99% first-token attention mass on the
  recent ring. `M40LLM_KV_SELECTED_BLOCK_ORDER=score|chronological` now tests
  whether selected old blocks should be laid out by score or absolute position,
  and `M40LLM_KV_LOGIT_TRACE=1` records per-generated-token dense-vs-compressed
  logit traces. A 4096 score-vs-chronological sweep showed identical pass/fail
  behavior: old top_blocks=4 still fails, old top_blocks=8/16 pass, recent
  top_blocks=4/16 pass, and recent top_blocks=8 still fails. The trace shows
  recent top_blocks=8 diverges at generated step 2, while old top_blocks=4
  tracks dense through `ZXQ-NEEDLE` and then repeats; selected-block ordering is
  not the primary root cause. `M40LLM_KV_Q8_DRIFT_DIAG=1` now compares dense
  `off`, FP16 exact selected blocks, staged-q8, and direct-q8 for the known 4096
  failures. KV cache reallocation now drops the previous cache before allocating
  the replacement, avoiding transient double-allocation when the harness
  switches backends. The 4096 diagnostic shows old/top_blocks=4 passes with
  FP16 exact selected blocks but fails with staged-q8 and direct-q8, isolating
  that failure to q8 quantization/dequant drift. Recent/top_blocks=8 fails with
  FP16 exact, staged-q8, and direct-q8, isolating that failure to
  selected-context sensitivity at that top-block count. Direct-q8 matches
  staged-q8 in both rows, so it is not currently worse than staged q8 in these
  diagnostics. `M40LLM_KV_Q8_PRECISION_SPLIT_DIAG=1` now adds a diagnostic
  dense-shadow path for q8 exact-block rows so old K and old V precision can be
  split independently. The 4096 old/top_blocks=4 precision split shows FP16
  K+FP16 V passes, q8 K+q8 V fails, FP16 K+q8 V passes, and q8 K+FP16 V fails.
  Treat this as evidence that q8 K/scoring precision is the primary old/top4
  failure source; V quantization alone did not fail in that diagnostic. The
  dense shadow is diagnostic overhead, not a deployable memory-saving backend.
  `M40LLM_KV_Q4_V_DIAG=1` now tests FP16-K/q4-V staged exact-block retrieval
  with packed signed q4 old V plus FP32 per-token/per-head V scales. The 2048
  old/recent top_blocks=2 regressions pass for FP16-K/FP16-V, FP16-K/q8-V, and
  FP16-K/q4-V. The fragile 4096 old/top_blocks=4 case also passes with
  FP16-K/q4-V, but q4 V shows materially larger logit drift than q8 V. The
  4096 recent top_blocks=4/8/16 matrix shows q4 V matches the FP16
  selected-context pass/fail pattern: top_blocks=4 and 16 pass for FP16 V, q8
  V, and q4 V, while top_blocks=8 fails for all three and emits EOT at generated
  step 2. Treat that top_blocks=8 row as selected-context sensitivity rather
  than a V-quantization-specific failure. `M40LLM_KV_EXACT_OLD_BACKING=fp16-k-q4-v`
  now implements the deployable mixed exact-old layout without q8 old K/V or
  dense-shadow allocations: recent K/V stay FP16, old K is FP16, old V is packed
  signed q4 with FP32 per-token/per-head scales, and selected blocks dequantize
  into the reusable staging workspace. The 2048 old/recent top_blocks=2 matrix
  and 4096 old/top4 plus recent/top4/top8/top16 matrix pass with the expected
  FP16 selected-context pass/fail pattern. Final KV allocation drops from 4.00
  GiB dense-equivalent to 2.97 GiB for the Llama-3.2-1B max-context allocation.
  `M40LLM_KV_EXACT_OLD_ATTENTION=fp16-k-q4-v-direct` now enables an
  experimental direct mixed attention backend that keeps old K in FP16 and
  unpacks packed q4 old V inside attention instead of materializing selected old
  V into FP16 staging. CUDA attention parity covers staged-vs-direct
  FP16-K/q4-V. A bounded 2048 old/top_blocks=2 quality row passed and improved
  decode time from 8.97 s staged-q4 to 4.94 s direct-q4 while keeping final KV
  allocation at 2.97 GiB. A filtered 4096 direct-only sweep also preserves the
  expected selected-context pattern: old/top4 passes in 7.35 s, recent/top4
  passes in 7.36 s, recent/top8 fails with `ZXQ` like the FP16 baseline, and
  recent/top16 passes in 8.91 s. The harness now supports
  `M40LLM_KV_QUALITY_MODES` and `M40LLM_KV_EXACT_BACKEND_VARIANTS` to keep
  expensive exact-block sweeps bounded. `M40LLM_KV_BLOCK_SELECT_POLICY` now
  adds diagnostic selected-block promotion policies (`topk`, `neighbors`,
  `threshold`, `anchor`, and `anchor-neighbors`) for direct exact-old attention.
  JSONL rows report base selected blocks, policy-added blocks, final selected
  blocks, and policy knobs. A focused 4096 recent/top8 direct FP16-K/q4-V
  neighbor-policy row still failed with `ZXQ`, while first-token attention
  remained dominated by the exact recent ring; simple neighbor promotion is not
  enough to fix the known non-monotonic row. `M40LLM_KV_POLICY_DIAG=1` now runs
  a bounded 4096 recent/top8 policy matrix with dense `off`, direct
  FP16-K/q4-V `block-select-exact`, generated-step attention capture, and
  per-row `block_policy_case` labels. In that matrix, small score-threshold
  deltas did not add blocks and still failed with `ZXQ`; anchor block 0
  recovered the needle content but not exact formatting (`QXZNEEDLE41729`);
  anchor-plus-neighbor promotion passed with `ZXQ-NEEDLE-41729` using 16
  selected old blocks and 48.0 MiB active attended KV across all layers.
  `M40LLM_KV_ANCHOR_NEIGHBOR_VALIDATE=1` now runs the broader 4096 old/top4 and
  recent/top4/top8/top16 matrix for dense `off` plus direct FP16-K/q4-V
  `block-select-exact`. That matrix confirms anchor-neighbors fixes recent/top8
  and preserves old/top4 plus recent/top16, but it regresses the previously
  passing recent/top4 row to `QXZNEEDLE41729`. Do not make anchor-neighbors the
  preferred policy as-is. `M40LLM_KV_FALLBACK_DIAG=1` now runs a bounded 4096
  fallback matrix for direct FP16-K/q4-V exact-block retrieval. It keeps top-k
  as the primary policy and retries fragile rows with top16 only when a
  diagnostic gate fires. Stable old/top4, recent/top4, and recent/top16 rows do
  not trigger retries and keep their original active KV. The known recent/top8
  failure still emits `ZXQ` with top-k, but the oracle, EOT-anomaly,
  low-margin, and score-spread gates all recover `ZXQ-NEEDLE-41729` by retrying
  with top16, raising active attended KV from 40.0 MiB to 48.0 MiB while keeping
  final allocated KV at 2.97 GiB. The oracle gate is answer-aware and
  diagnostic-only. `M40LLM_KV_FALLBACK_MULTITASK_DIAG=1` now runs a bounded
  multi-task validation suite for answer-agnostic fallback gates outside the
  single-needle benchmark. It compares dense `off`, top-k only, EOT-anomaly,
  low-margin, score-spread, and combined top16 fallback rows across
  single-needle, multi-needle, distractor-code retrieval, early-fact QA,
  early-fact summary, and normal long-chat smoke tasks. The default target is
  1024 tokens because the 4096 multi-task matrix is expensive; set
  `M40LLM_KV_QUALITY_TARGETS=4096` for the heavier version. The 1024 suite
  produced no fallback regressions. EOT-anomaly and low-margin can trigger on
  already-passing rows but preserved the answer in this run; score-spread was
  conservative and did not trigger. The summary task failed for dense `off` too,
  so it is not compressed-KV-specific evidence. The harness now supports
  `M40LLM_KV_MULTITASK_TASKS` and `M40LLM_KV_MULTITASK_FALLBACK_CASES` filters
  for expensive runs. A focused 4096 multi-task run over single-needle,
  multi-needle, distractor-code, early-fact QA, and long-chat smoke tasks showed
  that score-spread and combined fallback triggered on every compressed row. The
  multi-needle row is decisive policy evidence: dense `off` failed, top-k
  exact-block retrieval passed with `ALPHA-13579, BRAVO-24680`, but both
  score-spread and combined top16 retries regressed it to
  `Alfa-13579, Bravo-24680`. Keep fallback diagnostic/opt-in.
  `M40LLM_KV_TOPK_MULTITASK_DIAG=1` now characterizes top-k robustness across
  the same multi-task prompt family without fallback. The 1024 smoke passed all
  tasks and top-block counts. The 4096 matrix shows top4 is sufficient for
  single-needle, distractor retrieval, early-fact QA, and long-chat smoke, while
  multi-needle is non-monotonic: top4 fails, top8 passes exactly, and top16
  fails. Dense `off` also fails multi-needle, so treat that row as task/model
  capability evidence rather than pure compression failure. Direct
  FP16-K/q4-V remains the preferred experimental backend and plain top-k
  remains the preferred selection policy. In the current 4096 matrix, top4 is
  the best efficiency setting, top8 is useful for multi-fact retrieval but not
  universally superior, and top16 is not a safe robustness default because
  quality is non-monotonic. `M40LLM_KV_TOPK_SENSITIVITY_DIAG=1` now focuses the
  top-k suite on the 4096 multi-needle row and emits richer selected-block,
  attention-mass, and generated-logit trace telemetry. The focused run
  reproduced the non-monotonic result: top4 fails with `[94,80,77,91]`, top8
  passes after adding `[92,78,89,87]`, and top16 fails after adding
  `[88,57,90,71,46,53,73,93]`. First-captured attention remains dominated by
  the recent ring. `M40LLM_KV_TOPK_ABLATION_DIAG=1` now adds diagnostic
  explicit include/exclude selected-block controls and a score-cluster policy.
  The 4096 multi-needle ablation shows top4 cannot be repaired by any single
  top8-delta block, top8 still passes when any one selected block is removed,
  and top8 still passes when any one top16-extra block is added. The top16
  regression is therefore a combined support-set/distribution-shift effect, not
  a single toxic block. Failing rows diverge at generated step 6.
  `M40LLM_KV_ABLATION_CASES` now filters expensive ablation runs, and
  `score-cluster-adaptive` honors min/max block caps. The filtered 4096
  support-shape run shows top4 plus tested pairs still fails; top4 plus the full
  top8 delta also fails through the older explicit-include path; top8 plus
  tested pairs/quartets all pass; and score-cluster-adaptive with min8 recovers
  exactly the passing top8 set. `M40LLM_KV_ORDER_EQUIV_DIAG=1` now adds
  explicit score-order selection and `M40LLM_KV_SELECTED_BLOCK_ORDER=descending`
  so same-set top8 can be compared in score, ascending, and descending
  materialized order. The corrected 4096 multi-needle order-equivalence run
  shows same-set top8 passes under all three tested orders, while top16 still
  fails. Candidate ordering is therefore not the root cause for this prompt;
  the top16 regression remains best explained as cumulative distribution shift
  from the full tail of extra blocks. Score-cluster-adaptive min8/max12 and
  min8/max16 still select the passing top8 core and pass, but remain candidate
  policies only. `top8-plus-tail1` through `top8-plus-tail8` now run
  cumulative top16-tail prefix ablations. The 4096 multi-needle run shows
  tail1 `[88]` and tail2 `[88,57]` pass, while tail3 `[88,57,90]` is the first
  failing transition and every larger prefix also fails. Tail attention mass is
  small in absolute terms, so this is not just the tail taking all probability;
  the dense-reference token logit margin shifts at the failing transition.
- Next: avoid 8192 and server integration until exact-block quality is more
  stable. Treat direct FP16-K/q4-V as the recommended experimental mixed
  attention backend, but keep it opt-in. Top-k should remain the preferred
  exact-old selection policy for now, but do not blindly raise the default to
  top16. Fallback gates are too false-positive prone at 4096. The next policy
  step should validate score-cluster-adaptive on other prompt types and context
  shapes: single-needle 4096, multi-needle 2048/4096/8192 if practical,
  distractor-heavy prompts, answers near block boundaries, far-apart relevant
  blocks, and high-score-plus-weak-support cases. Qwen2/Qwen2.5 GGUF metadata
  and prompt/tokenizer detection are now wired as the next cross-model target
  before attempting Mistral-7B F16. `Qwen2.5-3B-Instruct-f16.gguf` downloaded
  successfully and completed a one-token CUDA generation smoke on Tesla M40;
  dense packed prefill and direct FP16-K/q4-V exact-old attention now admit
  head_dim=128 so Qwen long-context KV quality validation can proceed. The older
  staged/q8 exact-old and summary/lossy compressed CUDA paths remain head_dim=64
  only. Qwen/head128 microbenchmark hooks now show raw packed prefill attention
  is milliseconds at 512 tokens, while a 256-token full quality row still timed
  out after 180 s even with minimal telemetry. A 64-token Qwen first-token
  smoke localizes the issue: the cold dense row spent roughly 88.6 s in prompt
  prefill while materializing FP32 projection weights, the following compressed
  top4 row reused those weights and prefilled in roughly 0.9 s, and disabling
  materialization was slower for both dense and compressed rows. The next speed
  target should separate cold materialization from warm steady Qwen quality
  timings rather than disabling the materialized FP32 cuBLAS path.
  `M40LLM_KV_QUALITY_WARMUP_MATERIALIZATION=1` now records a dense warmup
  generation before top-k multitask rows, but a same-prompt Qwen warmup still
  did not produce measured rows within a 180 s bound; add more explicit
  materialized-cache timing/inspection before longer Qwen sweeps. Generated
  quality rows now include materialized FP32 cache before/after-prompt/final
  totals, added entry/byte deltas, and a `materialized_f32_warm_row` label for
  that investigation. Warmup stderr logging also prints cache before/after/add
  totals because Qwen can still time out before measured rows are emitted. A
  longer Qwen 64-token warm-row diagnostic completed: warmup materialized
  253 FP32 entries / 12.34 GB in roughly 89.2 s; the measured dense row added
  zero materialized bytes but still spent roughly 87.2 s in prompt prefill,
  while the direct FP16-K/q4-V row took roughly 0.9 s. Timing logs show the
  dense-row delay is paid at `logits.output_norm`, which is the first
  synchronization point after dense packed-prefix work, not an actual 86 s norm
  kernel. `M40LLM_PREFILL_SYNC_DIAG=1` now adds an opt-in two-stream CUDA-event
  sync diagnostic after packed-prefix prefill and is verified on a short
  packed-prefix parity test. KV quality JSONL rows now include those sync
  timings. The Qwen 64-token warm-row JSONL result shows dense `off` and direct
  FP16-K/q4-V both spend only roughly 0.73-0.74 s in packed-prefix sync; dense
  still spends roughly 87 s total while compressed is roughly 1.47 s. This
  rules out packed-prefix prefill as the dense-only delay.
  `M40LLM_FORWARD_SYNC_DIAG=1` now adds an opt-in two-stream CUDA-event
  checkpoint around the prompt final-token forward pass before logits. A
  repeated Qwen 64-token run shows dense `off` spends roughly 86.5 s in that
  final-token forward sync, while direct FP16-K/q4-V spends roughly 0.13 s.
  Treat the Qwen delay as dense full-context final-token forward work, most
  likely dense attention over the full prompt/KV range, rather than
  packed-prefix prefill or output-norm itself. Dense last-token GQA now has a
  shared-score `head_dim=128` specialization analogous to the existing head64
  path. The Qwen dense attention microbenchmark reports roughly 991 us at
  512 tokens and 3.924 ms at 2048 tokens. The bounded 64-token Qwen warm-row
  diagnostic now drops dense `off` prompt prefill from roughly 87.3 s to
  roughly 0.86 s and final-token forward sync from roughly 86.5 s to
  roughly 0.117 s; dense and direct FP16-K/q4-V are both roughly 1.47-1.48 s
  total for that one-token row. Longer Qwen quality sweeps are now unblocked
  for bounded targets. A bounded Qwen cross-model checkpoint now runs 256/512
  top-k multitask targets and confirms the harness emits rows in practical
  time after the head128 fix; the top-k multitask diagnostic now honors every
  requested `M40LLM_KV_QUALITY_TARGETS` entry instead of only the first.
  Qwen2/Qwen2.5 tokenization now uses the real `tiktoken` Qwen2 encoding for
  normal text and special-token paths, and standard layer mapping accepts
  optional Q/K/V F32 attention biases such as `blk.N.attn_q.bias`,
  `blk.N.attn_k.bias`, and `blk.N.attn_v.bias`. Full-layer decode applies those
  biases after async Q/K/V projection before RoPE/KV append. The direct Qwen2.5
  CUDA canary for `Hello, please answer with the word OK.` now generates `OK`,
  whereas it previously emitted nonsensical text. The quality harness supports
  `M40LLM_KV_RETRIEVAL_PROMPT_STYLE=default|qwen-strict|qwen-fewshot|qwen-natural`
  for Qwen prompt validation, emits `retrieval_prompt_style`,
  `dense_reference_passed`, and `quality_conclusion` in JSONL rows, and marks
  compressed multitask rows inconclusive when dense `off` fails the same prompt.
  Optional Q/K/V attention bias handling is now shared and
  applied by sequential one-token decode, graph-parameter decode, batched decode,
  and packed-prefix prefill. A synthetic CUDA packed-prefill parity test with
  QKV biases now matches sequential logits, and the direct Qwen `OK` canary
  still passes. The multitask harness now supports
  `M40LLM_KV_MULTITASK_PREFILL_MODE=packed|sequential`, and JSONL multitask
  rows include the actual generated `prefill_mode`. `qwen-natural` provides a
  Qwen retrieval canary using an unambiguous `answer key is BLUE` fact. An
  Ollama oracle comparison with the same `Qwen2.5-3B-Instruct-f16.gguf` showed
  the default synthetic `ZXQ-NEEDLE-41729` canary should pass under raw
  prompting. The remaining Qwen blocker was an architecture-specific RoPE
  layout bug: m40-llm was using adjacent-pair RoPE, while Qwen needs
  split-half/NeoX pairing. CUDA RoPE and fused K-RoPE+KV-append wrappers now
  accept explicit layout selection, and `qwen*` architectures select NeoX RoPE.
  After the fix, raw and auto-formatted Qwen CLI canaries return
  `ZXQ-NEEDLE-41729`, and the 64/256-token default single-needle quality rows
  pass for dense `off` and direct FP16-K/q4-V top4. A Qwen cross-model
  top-k multitask checkpoint now validates direct FP16-K/q4-V exact-old
  retrieval on targets 256, 512, and 1024 across single-needle, multi-needle,
  distractor-needle, and early-fact QA tasks. Top4, top8, and top16 all pass
  whenever dense `off` is given enough decode tokens to pass; the only 512
  multi-needle inconclusive row was a 16-token answer truncation and passes
  with `M40LLM_KV_MULTITASK_MAX_TOKENS=24`. Treat Qwen as a useful second-model
  checkpoint for the backend, while keeping top4 as the efficiency default and
  higher top-k values diagnostic/task-driven. The bounded Qwen 2048 checkpoint
  also passes dense `off`, top4, and top8 for single-needle, multi-needle,
  distractor-needle, and early-fact QA. Top16 was not needed because top4 and
  top8 agree. The Qwen 2048 run used minimal telemetry, so selected block
  indices/scores were not emitted; rerun without minimal telemetry only if
  future Qwen failures require selected-block diagnosis. The Llama 4096
  selection-anatomy diagnostic now confirms the multi-needle top16 regression
  is a cumulative support-set/distribution-shift effect rather than a single
  toxic block: top8 passes, top8 minus any one block still passes, top8 plus any
  single top16-extra block still passes, and the cumulative tail first fails at
  `[88,57,90]`. `M40LLM_KV_SCORE_CLUSTER_VALIDATE=1` now runs a bounded
  validation suite over dense `off`, top4/top8/top16, and
  score-cluster-adaptive min8/max12 plus min8/max16 across 2048/4096 prompt
  shapes, including boundary and far-apart retrieval prompts. The 2048 suite
  passed all dense-valid rows. The completed 4096 dense-valid rows also passed,
  while multi-needle and far-apart rows remain inconclusive because dense
  `off` itself fails. A Qwen2.5 2048 score-cluster validation checkpoint now
  passes dense `off`, top4, top8, top16, score-cluster min8/max12, and
  score-cluster min8/max16 for single-needle, multi-needle, distractor-needle,
  and early-fact QA. Score-cluster did not regress any Qwen dense-valid row and
  matched top8/top16 quality. A later telemetry fix extended debug selection to
  `head_dim=128`, so Qwen score-cluster rows now emit selected block sets and
  support buckets; the corrected Qwen 2048 multi/distractor confirmation shows
  score-cluster min8/max12 selects top8-sized support, matching plain top8.
  The Llama 4096 dense-valid single/distractor/boundary confirmation shows the
  same top8-sized score-cluster behavior with no regressions. Keep
  score-cluster-adaptive opt-in/candidate only; direct FP16-K/q4-V with plain
  top-k remains the preferred experimental path. Targeted
  `M40LLM_KV_SCORE_CLUSTER_DIFF_VALIDATE=1` prompts now exercise clustered
  distractors, boundary distractors, and weak-support multi-needle retrieval
  while recording selected support size/bucket, active KV, decode time, and
  `selection_elapsed_ms`. The first Llama/Qwen differentiation checkpoint did
  not find a dense-valid top9-top12 score-cluster case: dense-valid targeted
  rows collapsed to top8-sized support and passed, while the Llama clustered
  distractor row was formally inconclusive because dense `off` chose a decoy.
  `M40LLM_KV_REALISTIC_PROMPT_VALIDATE=1` now runs a realistic prompt suite
  over long chat QA, early/middle/late document QA, multifact distractor
  extraction, and code/config lookup. The Llama 2048 checkpoint passed dense
  plus top4/top8 for chat and document QA; the multifact row is inconclusive
  because dense chose the wrong region; and the config row is dense-valid
  policy evidence where top4 chose the archived value `M40_BATCH_LIMIT=73` but
  top8/top16 recovered the active setting `M40_BATCH_LIMIT=37`. Compressed KV
  is now the project default when supported: `block-select-exact`,
  `fp16-k-q4-v` exact-old backing, `fp16-k-q4-v-direct` attention, plain
  score-ranked top-k, and `top_blocks=8`. Dense `off` remains the explicit
  reference/compatibility mode through `--kv-compress-mode off` and
  `KvCompressionConfig::dense_reference()`. Top4 is the efficiency/YMMV
  override; top16 is diagnostic/escalation-only. Score-cluster-adaptive,
  fallback, and anchor-neighbor policies remain opt-in diagnostics. The
  `KvCompressionConfig::default()` audit intentionally leaves runtime
  generation/server defaults compressed while moving dense correctness tests
  and quality dense rows to `dense_reference()` or explicit per-mode configs.
  Long-generation diagnostics now include `M40LLM_LONG_DECODE_LOG=1|N`, and
  generation failures report generated-token count, sequence length, remaining
  context, KV mode, `top_blocks`, exact-old backing, and exact-old attention.
  CLI generation and server startup now emit a best-effort `nvidia-smi` warning
  when other compute processes are using the selected GPU because parallel Qwen
  release smokes produced token-0 GGUF GEMM failures while sequential reruns
  passed. Default compressed top8 long-generation validation now covers the
  reported Qwen relationship prompt at 512 and 1024 generated tokens, and the
  original 3000-token request stopped naturally after 1024 tokens without a
  CUDA error. Llama-3.2-1B default compressed top8 was also validated with 512
  and 1024 token caps, stopping naturally after 256 tokens.
  Server smoke coverage now includes default compressed non-streaming,
  compressed top4/top16 non-streaming overrides, and default compressed
  streaming. Real Qwen2.5 and Llama-3.2 server `/generate` smokes return HTTP
  200 with default compressed top8; Qwen dense/top4/top16 server overrides also
  return HTTP 200. Unsupported Qwen q8-direct exact-old server startup fails
  clearly for head_dim=128 and suggests dense `off`.
  Do not increase representative count, tune pure summary modes, run 8192, or
  expand compressed KV into deeper server scheduling yet.

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

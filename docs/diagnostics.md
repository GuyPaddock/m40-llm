# Diagnostics and Profiling

Most diagnostics are disabled by default because they are verbose or can change
timing. Enable only the signal needed for a profiling or correctness run.

## Device and Tensor Mapping

- `M40LLM_FORCE_M40=1`: require a visible sm_52 device.
- `M40LLM_TENSOR_VIEW_LOG=1`: print each GGUF tensor-to-device pointer mapping
  during model load.

See [`device_selection.md`](device_selection.md) for selection behavior.

## Allocation and Copy Logs

- `M40LLM_ALLOC_LOG=1`: print `device_malloc` and `device_free` traces.
- `M40LLM_ALLOC_BT=1`: include backtraces with allocation logs.
- `M40LLM_COPY_LOG=1`: log H2D/D2H copy counter events.

Allocation tracking is also used by KV compression quality reports for final
compressed KV, dense-equivalent KV, and temporary dense KV byte accounting.

## Launch, Sync, and Timing Logs

- `M40LLM_LAUNCH_LOG=1`: log kernel launch and cuBLAS counter events.
- `M40LLM_SYNC_LOG=1`: log stream synchronization counter events.
- `M40LLM_PROFILE_LOG=1`: print lower-noise per-operation counter deltas around
  forward-pass timing regions.
- `M40LLM_TIMING_LOG=1`: print verbose per-token and per-layer decode timing.
- `M40LLM_PREFILL_SYNC_DIAG=1`: after CLI/test packed-prefix prefill, record
  CUDA events on both decode and prefill streams, synchronize the stop events,
  and print `sync_diag.wall`, `sync_diag.decode_gpu`, and
  `sync_diag.prefill_gpu` timing labels. KV quality JSONL rows also include
  `packed_prefill_sync_wall_ms`, `packed_prefill_sync_decode_gpu_ms`, and
  `packed_prefill_sync_prefill_gpu_ms`. This changes timing and is intended
  only for attribution diagnostics.
- `M40LLM_FORWARD_SYNC_DIAG=1`: for the prompt final-token forward pass,
  record CUDA events on both decode and prefill streams before forward starts
  and synchronize them immediately before logits. KV quality JSONL rows include
  `prompt_forward_sync_wall_ms`, `prompt_forward_sync_decode_gpu_ms`, and
  `prompt_forward_sync_prefill_gpu_ms`. Use this with
  `M40LLM_PREFILL_SYNC_DIAG=1` when separating packed-prefix prefill time from
  final-token dense attention or KV-cache work.
- `M40LLM_STREAM_LOG=1`: print prefill/decode stream creation details and
  best-effort priority selection.
- `M40LLM_SERVER_BATCH_LOG=1`: print dense server scheduler queue/tick
  composition and packed-decode fallback reasons. The scheduler also records
  profile events named `server_scheduler_batched_prefill_tick`,
  `server_scheduler_batched_decode_tick`,
  `server_scheduler_sequential_prefill_tick`, and
  `server_scheduler_sequential_decode_tick`; inspect them with
  `profile::snapshot()` in tests or `M40LLM_LAUNCH_LOG=1` during diagnostics.
- `M40LLM_SERVER_BATCH_PREFILL_LOG=1`: print packed-prefill fallback reasons
  and avoided padded-token counts for compatible dense server prefill batches.
- `M40LLM_DECODE_SESSION_LOG=1`: print verbose decode-session token logs.
- `M40LLM_LONG_DECODE_LOG=1`: print bounded long-generation progress at the
  first token, every 64 generated tokens, and the configured generation cap.
  Set `M40LLM_LONG_DECODE_LOG=N` to log every `N` generated tokens instead.
  Progress rows include sequence length, remaining context, sampled token, KV
  compression mode, selected exact-old backend, exact-old attention backend, and
  `top_blocks`. Generation failures also include this state in the error
  context even when the progress logger is disabled.
- `M40LLM_GPU_BUSY_WARN=0`: suppress the best-effort `nvidia-smi` warning when
  CLI generation or server startup sees other compute processes on the selected
  GPU. The warning is non-fatal because `nvidia-smi` may be unavailable or
  permission-limited, but concurrent large generations can still cause CUDA
  allocation or GEMM failures on the M40.

Use `docs/perf_baselines.md` to record repeatable benchmark results rather than
leaving one-off timing notes in the README.

## GEMM and Materialization

- `M40LLM_GEMM_LOG=1`: print GEMM backend selection lines.
- `M40LLM_ENABLE_CUBLAS=1`: enable cuBLAS integration when headers/libraries
  are detected.
- `M40LLM_PROJECTION_BACKEND=auto|fast-fits|large-model`: choose the projection
  backend policy. `auto` is the default. `fast-fits` uses materialized FP32
  projection weights with cuBLAS when allowed. `large-model` keeps compact GGUF
  F16/quantized weights and skips full FP32 materialization; this is the
  current fallback path, not the future fused-dequant projection backend.
- `M40LLM_FAST_FITS_BUDGET_MB=<mb>`: total memory budget used by
  `M40LLM_PROJECTION_BACKEND=auto` when deciding whether materialized FP32
  projections fit. The estimate includes uploaded model weights, estimated FP32
  materialized projection cache, one-row forward workspace, and the resident KV
  cache. The default is `22528` MiB to leave headroom on a 24 GiB M40.
- GGUF Q8_0 projection tensors now route through a fused CUDA
  `f32 x Q8_0 -> f32` path that dequantizes inside the projection kernel. This
  is the first large-model fused-dequant primitive; it is not yet a complete
  replacement for the fast-fits materialized FP32 backend.
- `M40LLM_MATERIALIZE_F32_WEIGHTS=0`: force the dedicated GGUF-layout CUDA
  fallback instead of materialized FP32 projection weights.
- `M40LLM_MATERIALIZE_F32_BUDGET_MB=<mb>`: cap cached FP32 materialized
  weights; over-budget tensors fall back to the GGUF F16 kernel and log when
  `M40LLM_GEMM_LOG=1`.

Materialized cache keys include the source pointer plus tensor identity metadata
when a GGUF tensor view is available.

## CUDA Graph Diagnostics

CUDA Graph decode remains opt-in:

```bash
M40LLM_DECODE_GRAPH=1
```

Diagnostic controls:

- `M40LLM_DECODE_GRAPH_DIAG_SYNC=1`: synchronize the decode stream immediately
  after replay and report CUDA-event elapsed time.
- `M40LLM_DECODE_GRAPH_DIAG_MAX_MS=<float>`: with diagnostic sync enabled,
  disable graph replay automatically if timed replay exceeds the threshold.

Graph mode is experimental. If explicit replay timing regresses token latency,
keep graph mode off and prioritize scheduler or batching work.

## Correctness Diagnostics

- `M40LLM_FORWARD_FINITE_LOG=1`: synchronously sample intermediate CUDA forward
  tensors and report non-finite counts. This is very verbose and intended only
  for correctness debugging.
- `M40LLM_ATTN_LOG=1`: print attention backend selection.
- `M40LLM_CACHE_EXPERIMENT=ldg`: opt into the weighted RMSNorm `__ldg`
  experiment.
- `M40LLM_CACHE_EXPERIMENT=ldg_kv`: opt into the KV-cache `__ldg` attention
  experiment.

Read-only cache experiments are off by default unless measurements justify a
default change.

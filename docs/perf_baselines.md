# Performance Baselines

This file tracks measured CUDA baselines before M40-specific optimization work.

## 2026-05-04: Tesla M40 GEMM Baseline

Environment:

- GPU: Tesla M40 24GB, sm_52, driver 580.126.09
- Also visible: NVIDIA RTX A4000, sm_86
- Command prefix: `source scripts/dev-env.sh && M40LLM_ENABLE_NVCC=1`
- Criterion sample size: 10
- Cargo features: `cuda`

Commands:

```bash
cargo bench --features cuda --bench gemm -- --sample-size 10
cargo bench --features cuda --bench gemm_fallback -- --sample-size 10
```

Results:

| Benchmark | Shape | Time estimate | Throughput estimate |
| --- | ---: | ---: | ---: |
| `gemm_f16xf16_f32` | 64x64x64 | 11.834 us | 2.5787 GiB/s |
| `gemm_f16xf16_f32` | 128x128x128 | 19.030 us | 6.4146 GiB/s |
| `gemm_f16xf16_f32` | 256x256x256 | 96.821 us | 5.0431 GiB/s |
| `gemm_f16xf16_f32` | 512x512x512 | 686.99 us | 2.8430 GiB/s |
| `gemm_f16_storage_f32_compute` | 64x64x64 | 11.862 us | 1.9295 GiB/s |
| `gemm_f16_storage_f32_compute` | 128x128x128 | 19.123 us | 4.7875 GiB/s |
| `gemm_f16_storage_f32_compute` | 256x256x256 | 96.932 us | 3.7780 GiB/s |
| `gemm_f16_storage_f32_compute` | 512x512x512 | 687.13 us | 2.1318 GiB/s |

Notes:

- The two GEMM benchmark paths currently have nearly identical latency, especially at larger shapes.
- Treat these as pre-optimization baselines only; they do not yet represent end-to-end token latency.

## 2026-05-04: TinyLlama `/generate` Baseline

Environment:

- GPU: Tesla M40 24GB, sm_52, driver 580.126.09
- Model: `/mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf`
- Server command prefix: `source scripts/dev-env.sh && M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1`
- Server command: `cargo run --features cuda,server -- run <model> --addr 127.0.0.1:52150 --require-sm52`
- Expected server log evidence observed: `full-layer forward enabled layers=22`

Requests:

```bash
curl -sS -w '\nTIME_TOTAL=%{time_total}\n' \
  -X POST http://127.0.0.1:52150/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":1,"temperature":1.0,"top_k":1}'

curl -sS -w '\nTIME_TOTAL=%{time_total}\n' \
  -X POST http://127.0.0.1:52150/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":2,"temperature":1.0,"top_k":1}'
```

Results:

| Prompt | Generated tokens | Output | Total latency |
| --- | ---: | --- | ---: |
| `Hello` | 1 | `,` | 3.513541 s |
| `Hello` | 2 | `, World` | 5.524597 s |

Notes:

- This is a development-build baseline, not a release-build latency target.
- The measurement includes HTTP handling, prompt tokenization, full-layer prefill/decode work, logits copyback, greedy sampling, and UTF-8 decoding.
- Approximate second-token incremental latency from this two-request sample is 2.01 s.

## 2026-05-04: Attention GQA Decode Baseline

Environment:

- GPU: Tesla M40 24GB, sm_52, driver 580.126.09
- Command prefix: `source scripts/dev-env.sh && M40LLM_ENABLE_NVCC=1`
- Command: `cargo bench --features cuda --bench attention -- --sample-size 10`
- Shape: `q_heads=32`, `kv_heads=4`, `head_dim=64`

Results:

| Sequence length | Time estimate | Throughput estimate |
| ---: | ---: | ---: |
| 1 | 234.50 us | 4.2644 Kelem/s |
| 16 | 3.5461 ms | 4.5121 Kelem/s |
| 128 | 33.568 ms | 3.8131 Kelem/s |
| 512 | 145.51 ms | 3.5187 Kelem/s |
| 1024 | 293.24 ms | 3.4921 Kelem/s |

Notes:

- The current GQA attention kernel scales roughly linearly with context length and is a clear decode bottleneck.
- Next attention work should target this kernel before stream separation or persistent decode experiments.

## 2026-05-04: Post-Workspace Reuse Baseline

Changes since earlier baselines:

- `LoadedModel` now keeps a reusable CUDA forward workspace for per-layer Q/K/V, norm, residual, MLP, and full-layer ping-pong scratch buffers.
- Same-shape forward calls reuse tracked device memory instead of allocating/freeing scratch inside each token/layer call.

GEMM refresh:

| Benchmark | Shape | Time estimate | Throughput estimate |
| --- | ---: | ---: | ---: |
| `gemm_f16xf16_f32` | 64x64x64 | 11.928 us | 2.5586 GiB/s |
| `gemm_f16xf16_f32` | 128x128x128 | 19.253 us | 6.3403 GiB/s |
| `gemm_f16xf16_f32` | 256x256x256 | 98.426 us | 4.9609 GiB/s |
| `gemm_f16xf16_f32` | 512x512x512 | 690.55 us | 2.8284 GiB/s |
| `gemm_f16_storage_f32_compute` | 64x64x64 | 12.153 us | 1.8833 GiB/s |
| `gemm_f16_storage_f32_compute` | 128x128x128 | 19.433 us | 4.7111 GiB/s |
| `gemm_f16_storage_f32_compute` | 256x256x256 | 98.520 us | 3.7171 GiB/s |
| `gemm_f16_storage_f32_compute` | 512x512x512 | 688.76 us | 2.1268 GiB/s |

TinyLlama `/generate` refresh:

| Prompt | Generated tokens | Output | Total latency |
| --- | ---: | --- | ---: |
| `Hello` | 1 | `,` | 3.571916 s |
| `Hello` | 2 | `, World` | 5.476869 s |

Notes:

- Workspace reuse reduced allocation churn but did not materially improve end-to-end latency.
- The two-token sample improved slightly, while the one-token sample was effectively flat within run-to-run noise.
- The measured attention cost is large enough to explain more of the decode latency than scratch allocation overhead.

## 2026-05-04: Optimized GQA Attention Kernel

Changes since earlier baselines:

- `m40llm_attention_last_token_f32_gqa` now routes `head_dim=64` requests through
  a shared-score CUDA kernel when `seq_len <= 8192`.
- The generic GQA attention kernel remains the fallback for other head dimensions
  and longer contexts.
- Set `M40LLM_ATTN_LOG=1` to print which attention backend was selected.

Attention refresh:

| Sequence length | Previous estimate | Optimized estimate |
| ---: | ---: | ---: |
| 1 | 234.50 us | 10.639 us |
| 16 | 3.5461 ms | 40.015 us |
| 128 | 33.568 ms | 259.89 us |
| 512 | 145.51 ms | 1.0961 ms |
| 1024 | 293.24 ms | 2.2153 ms |

TinyLlama `/generate` refresh:

| Prompt | Generated tokens | Output | Previous latency | Optimized latency |
| --- | ---: | --- | ---: | ---: |
| `Hello` | 1 | `,` | 3.571916 s | 2.999206 s |
| `Hello` | 2 | `, World` | 5.476869 s | 4.449623 s |

Notes:

- The attention microbenchmark improved by roughly two orders of magnitude at
  practical context lengths.
- Development-build `/generate` latency improved, but remaining token latency is
  still dominated by full-layer projection work, synchronization, launch overhead,
  and host sampling/logits copyback.

## 2026-05-05: TinyLlama CLI Decode Timing Profile

Environment:

- GPU: Tesla M40 24GB, sm_52
- Model: `/mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf`
- Command prefix: `source scripts/dev-env.sh && M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 M40LLM_TIMING_LOG=1`
- Command: `cargo run --features cuda -- generate <model> Hello --max-tokens 1 --top-k 1 --require-sm52`
- Expected log evidence observed: `full-layer forward enabled layers=22`

Selected results:

| Region | Time |
| --- | ---: |
| `cli.generate_text_total` | 2969.878 ms |
| `cli.decode_loop` | 2950.514 ms |
| Prompt token 0 `forward_all_layers` | 1433.026 ms |
| Prompt token 1 `forward_all_layers` | 1424.404 ms |
| Prompt token 0 `logits` | 29.873 ms |
| Prompt token 1 `logits` | 33.612 ms |
| `logits.copy_d2h` | 0.054-0.064 ms |

Steady per-layer short-context timings:

| Operation | Typical time per layer |
| --- | ---: |
| `mlp_gate_up` | ~17.6 ms |
| `qkv_project` | ~13.1 ms |
| `mlp_down` | ~12.1 ms |
| `attn_norm` | ~8.5 ms |
| `ffn_norm` | ~8.5 ms |
| `out_project` | ~4.4 ms |
| `attention` | ~0.05-0.08 ms |

Notes:

- Projection and norm operations dominate short-context decode after the
  optimized GQA attention kernel.
- Logits host copyback is currently negligible compared with full-layer forward.
- Stream separation should follow another projection/norm optimization pass,
  unless a future profile shows synchronization overhead has become dominant.

## 2026-05-09: Parallel RMSNorm Decode Optimization

Changes since earlier baselines:

- `m40llm_rms_norm_f32` and `m40llm_rms_norm_f32_weighted` now use one CUDA
  block per row and parallel reduction across the hidden dimension.
- The build script now detects CUDA/cuBLAS installs under `/opt/cuda`, including
  `/opt/cuda/targets/x86_64-linux/{include,lib}`.
- `M40LLM_ENABLE_CUBLAS` is now a build-script invalidation key, so toggling it
  forces Cargo to re-evaluate `cfg(have_cublas)`.

TinyLlama CLI timing refresh:

| Region | Previous profile | After parallel RMSNorm |
| --- | ---: | ---: |
| `cli.generate_text_total` | 2969.878 ms | 2202.864 ms |
| `cli.decode_loop` | 2950.514 ms | 2184.234 ms |
| Prompt token 0 `forward_all_layers` | 1433.026 ms | 1059.508 ms |
| Prompt token 1 `forward_all_layers` | 1424.404 ms | 1053.145 ms |
| Prompt token 0 `logits` | 29.873 ms | 21.261 ms |
| Prompt token 1 `logits` | 33.612 ms | 21.290 ms |

Steady per-layer short-context timings:

| Operation | Previous typical time | After parallel RMSNorm |
| --- | ---: | ---: |
| `attn_norm` | ~8.5 ms | ~0.05 ms |
| `ffn_norm` | ~8.5 ms | ~0.04-0.05 ms |
| `mlp_gate_up` | ~17.6 ms | ~17.6 ms |
| `qkv_project` | ~13.1 ms | ~13.1 ms |
| `mlp_down` | ~12.1 ms | ~12.1 ms |
| `out_project` | ~4.4 ms | ~4.4 ms |

Notes:

- Norm latency is no longer a meaningful short-context bottleneck.
- cuBLAS is now correctly detected from the local `/opt/cuda` install, but the
  current GGUF F16 projection path remains effectively projection-bound at the
  same timings. The next optimization target should be projection GEMM layout and
  FP32 materialization/transposition for `cublasSgemm` on sm_52.

## 2026-05-09: Materialized FP32 Projection Weights

Changes since the parallel RMSNorm baseline:

- Hot GGUF F16 projection weights can now be materialized once into FP32
  column-major-transposed device buffers.
- The projection path uses `cublasSgemm` for materialized FP32 weights and keeps
  the original GGUF-layout CUDA kernel as fallback.
- `M40LLM_MATERIALIZE_F32_WEIGHTS=0` disables the materialized path; the default
  is enabled when cuBLAS is available.

TinyLlama CLI timing refresh:

| Region | After parallel RMSNorm | After materialized FP32 projections |
| --- | ---: | ---: |
| `cli.generate_text_total` | 2202.864 ms | 640.907 ms |
| `cli.decode_loop` | 2184.234 ms | 609.506 ms |
| Prompt token 0 `forward_all_layers` | 1059.508 ms | 492.489 ms |
| Prompt token 1 `forward_all_layers` | 1053.145 ms | 37.243 ms |
| Prompt token 0 `logits` | 21.261 ms | 33.198 ms |
| Prompt token 1 `logits` | 21.290 ms | 4.273 ms |

Steady per-layer second-token timings:

| Operation | After parallel RMSNorm | After materialized FP32 projections |
| --- | ---: | ---: |
| `qkv_project` | ~13.1 ms | ~0.20-0.24 ms |
| `out_project` | ~4.4 ms | ~0.12-0.14 ms |
| `mlp_gate_up` | ~17.6 ms | ~0.49-0.52 ms |
| `mlp_down` | ~12.1 ms | ~0.27 ms |
| `attn_norm` / `ffn_norm` | ~0.04-0.05 ms | ~0.04-0.06 ms |
| `attention` | ~0.07-0.08 ms | ~0.07-0.09 ms |

Notes:

- First-token latency includes one-time FP32 materialization for each projection
  tensor; later tokens reuse cached device buffers.
- The measured device allocation total rose to roughly 6.34 GB for TinyLlama,
  which is acceptable on the 24 GB M40 and should be guarded for larger models.
- Remaining short-context steady-state costs are now mostly launch overhead,
  KV append, RoPE, norms, and logits/sampling overhead rather than projection
  math.

## 2026-05-10: Variable-Length Batched Attention Benchmarks

Benchmark scaffolding now includes `attention_last_token_f32_gqa_batched_varlen`
with three mixed-length decode distributions:

- `avg_0p6_max`: average length near 0.6 * max sequence length.
- `skewed`: short, medium, and long KV lengths in one batch.
- `near_uniform`: lengths close to max sequence length.

Each distribution compares individual per-sequence dispatch against the packed
batched variable-length GQA decode kernel.

```bash
cargo bench --features cuda --bench attention -- --sample-size 10
```

Measured on Tesla M40:

| Distribution | Lengths | Individual dispatch | Batched varlen | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `avg_0p6_max` | 384, 512, 640, 768 | 4.6016 ms | 1.5933 ms | 2.89x |
| `skewed` | 16, 64, 256, 1024 | 2.8305 ms | 2.1325 ms | 1.33x |
| `near_uniform` | 896, 960, 1000, 1024 | 8.6244 ms | 2.4545 ms | 3.51x |

Notes:

- The current batched kernel uses one grid launch for all batch entries and
  skips invalid KV regions via per-sequence lengths.
- Packed prefill now has a separate baseline using
  `attention_prefill_f32_gqa_varlen`.

Prefill dispatch distributions:

| Distribution | Query/KV lengths | Padded max | Packed varlen | Bucketed varlen |
| --- | ---: | ---: | ---: | ---: |
| `avg_0p6_max` | 384/384, 512/512, 640/640, 768/768 | 296.31 ms | 177.91 ms | 178.68 ms |
| `skewed` | 16/16, 64/64, 256/256, 1024/1024 | 525.79 ms | 141.25 ms | 141.66 ms |
| `near_uniform` | 896/896, 960/960, 1000/1000, 1024/1024 | 526.12 ms | 473.59 ms | 473.83 ms |
| `prefix_query` | 16/512, 32/640, 64/768, 128/1024 | 123.86 ms | 50.564 ms | 51.276 ms |

Packed prefill notes:

- The first prefill kernel is correctness-first: one CUDA block handles one
  sequence/query-head/query-token and skips invalid KV regions through
  per-sequence query and KV lengths.
- The prefix-query case demonstrates the intended savings when query tokens are
  much fewer than cached KV tokens.
- Bucketed dispatch is currently neutral to slightly slower than packed dispatch
  in this microbenchmark because the kernel already consumes true per-sequence
  lengths and extra bucket launches dominate.
- Remaining `t31e-varlen-batch` work should tune tile choices for M40 occupancy
  and shared-memory limits, then integrate the packed path into higher-level
  batched prefill instead of leaving it as an exposed kernel/benchmark.

## 2026-05-10: Read-Only Cache Experiment Baseline

Weighted RMSNorm now has an opt-in `__ldg` read-only cache experiment selected
with `M40LLM_CACHE_EXPERIMENT=ldg`. The default path is unchanged.

Command:

```bash
cargo bench --features cuda --bench rmsnorm -- --sample-size 10
```

Measured on Tesla M40:

| Shape | Default | `__ldg` experiment | Result |
| --- | ---: | ---: | --- |
| rows=1, dim=2048 | 13.677 us | 14.599 us | slower |
| rows=4, dim=2048 | 13.748 us | 13.594 us | neutral/slightly faster |
| rows=1, dim=4096 | 17.167 us | 18.778 us | slower |
| rows=4, dim=4096 | 17.063 us | 17.807 us | slower |

Notes:

- Keep the `__ldg` RMSNorm path experimental; the first measurements do not
  justify changing the default kernel.
- The next read-only cache target should be KV-cache attention reads, where the
  same K/V rows are revisited across score and value passes.

## 2026-05-10: KV-Cache `__ldg` Attention Experiment

Last-token GQA attention now has an opt-in `__ldg` KV-cache read experiment
selected with `M40LLM_CACHE_EXPERIMENT=ldg_kv`. The default path is unchanged.

Command:

```bash
cargo bench --features cuda --bench attention -- attention_last_token_f32_gqa --sample-size 10
```

Single-sequence decode:

| Sequence length | Default | `ldg_kv` experiment | Result |
| ---: | ---: | ---: | --- |
| 1 | 10.679 us | 10.712 us | neutral/slower |
| 16 | 40.075 us | 40.148 us | neutral/slower |
| 128 | 260.00 us | 260.37 us | neutral/slower |
| 512 | 1.0969 ms | 1.0976 ms | neutral/slower |
| 1024 | 2.2133 ms | 2.2136 ms | neutral/slower |

Batched mixed-length decode:

| Distribution | Default batched | `ldg_kv` batched | Result |
| --- | ---: | ---: | --- |
| `avg_0p6_max` | 1.5727 ms | 1.5753 ms | neutral/slower |
| `skewed` | 2.1113 ms | 2.1129 ms | neutral/slower |
| `near_uniform` | 2.4353 ms | 2.4351 ms | neutral/noise |

Notes:

- Keep `ldg_kv` experimental; it does not justify changing the default attention
  kernel on these M40 measurements.
- Future cache work should avoid more `__ldg` duplication unless a profile shows
  a stronger read-cache bottleneck. Texture-object experiments should remain
  deferred until there is a more promising target.

## 2026-05-10: Prefill/Decode Stream Separation

CUDA contexts now create separate non-blocking prefill and decode streams. The
decode stream uses best-effort higher priority when the driver reports a useful
priority range. Set `M40LLM_STREAM_LOG=1` to print the selected priorities.

Async enqueue variants were added for independent variable-length prefill
attention and batched last-token decode attention. The default CLI/server decode
path remains synchronous; this benchmark isolates the potential overlap benefit
without changing request scheduling semantics.

Command:

```bash
cargo bench --features cuda --bench stream_overlap -- --sample-size 10
```

Measured on Tesla M40:

| Workload | Time estimate |
| --- | ---: |
| `sequential_sync` | 47.066 ms |
| `split_async_final_sync` | 45.746 ms |

Notes:

- The benchmark uses independent prefill and decode attention buffers, enqueues
  prefill on the prefill stream and decode on the decode stream, then
  synchronizes both streams at the end.
- This is a small isolated win, not an end-to-end server scheduling change.
  Keep the normal generate path synchronous until a batched scheduler can avoid
  shared KV/workspace hazards.

## 2026-05-10: Persistent Decode Prototype

Persistent decode now has an experimental synthetic worker path. The prototype
keeps one CUDA block resident, polls a mapped host command slot, applies a small
decode-style vector transform to device buffers, and reports completion through
the same command slot. It is intentionally not wired into CLI/server generation.

Command:

```bash
cargo bench --features cuda --bench persistent_decode -- --sample-size 10
```

Measured on Tesla M40:

| Workload | Time estimate | Throughput estimate |
| --- | ---: | ---: |
| `launch_residual_add/2048` | 32.305 us | 63.397 Melem/s |
| `persistent_worker/2048` | 28.239 us | 72.524 Melem/s |

Notes:

- This shows a small launch-overhead reduction for a synthetic workload, enough
  to keep the persistent path as a candidate for future decode scheduling work.
- The prototype should remain isolated until the remaining host fallbacks and
  shared workspace/KV hazards are cleaned up.

## 2026-05-10: Ownership Hardening and Materialization Budget Refresh

This refresh was taken after request-level server serialization, shared
`DecodeSession` scratch, RAII `DeviceBuffer` cleanup, explicit model-level KV
layer/sequence addressing, and FP32 materialization budget/key hardening.

Commands:

```bash
cargo bench --features cuda --bench gemm -- --sample-size 10
cargo bench --features cuda --bench attention -- attention_last_token_f32_gqa --sample-size 10
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 M40LLM_TIMING_LOG=1 \
  M40LLM_GEMM_LOG=1 cargo run --features cuda -- generate \
  /mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf \
  Hello --max-tokens 1 --top-k 1 --require-sm52
```

GEMM microbench, measured on Tesla M40:

| Shape | Time estimate | Throughput |
| --- | ---: | ---: |
| `64x64x64` | 8.868 us | 3.441 GiB/s |
| `128x128x128` | 13.161 us | 9.275 GiB/s |
| `256x256x256` | 21.266 us | 22.961 GiB/s |
| `512x512x512` | 97.404 us | 20.052 GiB/s |

Last-token GQA attention guardrail, measured on Tesla M40:

| Sequence length | Default | `ldg_kv` experiment | Result |
| ---: | ---: | ---: | --- |
| 1 | 11.166 us | 11.151 us | neutral/slower vs prior baseline |
| 16 | 40.644 us | 40.709 us | neutral/slower vs prior baseline |
| 128 | 261.58 us | 261.25 us | within noise |
| 512 | 1.1015 ms | 1.1015 ms | within noise |
| 1024 | 2.2223 ms | 2.2198 ms | within noise |

Batched mixed-length decode guardrail:

| Distribution | Individual dispatch | Batched varlen | `ldg_kv` batched |
| --- | ---: | ---: | ---: |
| `avg_0p6_max` | 4.5803 ms | 1.5820 ms | 1.5823 ms |
| `skewed` | 2.8137 ms | 2.1178 ms | 2.1140 ms |
| `near_uniform` | 8.5964 ms | 2.4362 ms | 2.4363 ms |

TinyLlama CLI timing, measured on Tesla M40:

| Mode | `generate_text_total` | `token.0.forward_all_layers` | `token.1.forward_all_layers` | Final tracked device bytes |
| --- | ---: | ---: | ---: | ---: |
| Materialized FP32 default | 624.282 ms | 499.420 ms | 36.047 ms | 6.338 GB |
| `M40LLM_MATERIALIZE_F32_WEIGHTS=0` | 2210.434 ms | 1056.148 ms | 1055.661 ms | 2.200 GB |
| `M40LLM_MATERIALIZE_F32_BUDGET_MB=0` | 2218.500 ms | 1058.227 ms | 1057.396 ms | 2.200 GB |

Notes:

- The shared `DecodeSession` and RAII cleanup did not introduce an obvious
  regression in the steady materialized path.
- A follow-up `M40LLM_ALLOC_LOG=1` CLI run on 2026-05-11 confirmed
  `decode_session:logits_f32` and `decode_session:logits_norm_hidden_f32`
  allocate once at session start and free once at session teardown, instead of
  reallocating in each token's logits path.
- The materialized FP32 path remains the fast-fits backend: steady second-token
  full-layer forward was about 29x faster than the GGUF F16 fallback.
- The default run logged an estimated 4.400 GB of F16 2D tensors eligible for
  materialization and materialized 4.138 GB of projection/output weights for
  this short prompt.
- The zero-budget run confirms the new memory-budget fallback behaves like the
  explicit disabled-materialization mode: it preserves correctness and memory
  headroom, but is not the fast path.

## 2026-05-11: Async Wrapper Launch/Sync Decode Profile

This profile was taken after adding native and Rust async enqueue wrappers for
hot CUDA kernels while keeping the normal generate path on sync compatibility
wrappers.

Command:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  M40LLM_LAUNCH_LOG=1 M40LLM_SYNC_LOG=1 M40LLM_PROFILE_LOG=1 \
  M40LLM_TIMING_LOG=1 cargo run --features cuda -- generate \
  /mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf \
  Hello --max-tokens 1 --top-k 1 --require-sm52
```

TinyLlama CLI timing, measured on Tesla M40:

| Region | Time |
| --- | ---: |
| `cli.token.1.forward_all_layers` | 48.024 ms |
| `cli.token.1.logits` | 4.313 ms |
| `cli.token.1.total` | 52.446 ms |
| `cli.generate_text_total` | 652.373 ms |

Steady second-token aggregate over 22 layers:

| Operation group | Launches | cuBLAS calls | Stream syncs | Elapsed |
| --- | ---: | ---: | ---: | ---: |
| `qkv_project` | 0 | 66 | 66 | 5.803 ms |
| `mlp_gate_up` | 0 | 44 | 44 | 11.467 ms |
| `mlp_down` | 0 | 22 | 22 | 6.460 ms |
| `out_project` | 0 | 22 | 22 | 3.051 ms |
| `rope_qk` | 44 | 0 | 44 | 1.443 ms |
| `kv_append` | 22 | 0 | 22 | 1.056 ms |
| `attention` | 22 | 0 | 22 | 1.698 ms |
| `rms_norm_weighted` | 44 | 0 | 44 | 2.111 ms |
| `residual_add` | 44 | 0 | 44 | 0.886 ms |
| `swiglu` | 22 | 0 | 22 | 0.450 ms |

Notes:

- The sync compatibility path still performs one stream synchronization for
  each hot wrapper call. The steady token had 352 stream syncs inside
  `forward_all_layers`: 154 from materialized-FP32 cuBLAS projection calls and
  198 from non-GEMM kernels.
- RoPE plus KV append is visible: 66 launches, 66 stream syncs, and roughly
  2.50 ms per steady token. This supports the next strict task, fusing K RoPE
  with KV append, while keeping Q RoPE separate for now.
- cuBLAS synchronization is the largest remaining sync source, so after the
  RoPE/KV fusion task the larger scheduling lever is an async full-layer decode
  path that reduces sync boundaries across GEMM and elementwise operations.

## 2026-05-11: Fused K RoPE + KV Append Decode Profile

This profile was taken after replacing the forward path's separate K RoPE plus
KV append with a fused K-RoPE/f32-to-f16 KV append kernel. Q RoPE remains a
separate in-place operation because attention consumes Q directly.

Command:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  M40LLM_LAUNCH_LOG=1 M40LLM_SYNC_LOG=1 M40LLM_PROFILE_LOG=1 \
  M40LLM_TIMING_LOG=1 cargo run --features cuda -- generate \
  /mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf \
  Hello --max-tokens 1 --top-k 1 --require-sm52
```

TinyLlama CLI timing, measured on Tesla M40:

| Region | Time |
| --- | ---: |
| `cli.token.1.forward_all_layers` | 49.907 ms |
| `cli.token.1.logits` | 4.749 ms |
| `cli.token.1.total` | 54.769 ms |
| `cli.generate_text_total` | 692.045 ms |

Steady second-token aggregate over 22 layers:

| Operation group | Launches | cuBLAS calls | Stream syncs | Elapsed |
| --- | ---: | ---: | ---: | ---: |
| `qkv_project` | 0 | 66 | 66 | 5.896 ms |
| `mlp_gate_up` | 0 | 44 | 44 | 11.765 ms |
| `mlp_down` | 0 | 22 | 22 | 6.507 ms |
| `out_project` | 0 | 22 | 22 | 3.171 ms |
| `rope_q` | 22 | 0 | 22 | 0.809 ms |
| `kv_append_rope_k` | 22 | 0 | 22 | 1.600 ms |
| `attention` | 22 | 0 | 22 | 1.752 ms |
| `rms_norm_weighted` | 44 | 0 | 44 | 2.175 ms |
| `residual_add` | 44 | 0 | 44 | 0.934 ms |
| `swiglu` | 22 | 0 | 22 | 0.487 ms |

Notes:

- The fused path reduced RoPE/KV operation groups from 66 launches/syncs to 44
  launches/syncs per steady token. The measured RoPE/KV elapsed time moved from
  about 2.50 ms to about 2.41 ms in this run, which is a small win and within
  expected M40 run-to-run noise.
- The fused kernel uses one thread per RoPE pair and half2 stores for K/V cache
  writes. A scalar first version reduced launch count but was slower, so it was
  replaced before landing.
- The remaining dominant synchronization source is still the sync compatibility
  path around cuBLAS and elementwise kernels. The next strict task should focus
  on removing sync boundaries from already-fused SwiGLU and preparing graph or
  async full-layer scheduling rather than adding more local micro-fusions.

## 2026-05-11: Async SwiGLU Stream-Wait Decode Profile

This profile was taken after changing full-layer forward to enqueue SwiGLU on
the decode stream asynchronously, then make the prefill stream wait on that
work before the MLP down-projection GEMM consumes `dhid`.

Command:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  M40LLM_LAUNCH_LOG=1 M40LLM_SYNC_LOG=1 M40LLM_PROFILE_LOG=1 \
  M40LLM_TIMING_LOG=1 cargo run --features cuda -- generate \
  /mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf \
  Hello --max-tokens 1 --top-k 1 --require-sm52
```

TinyLlama CLI timing, measured on Tesla M40:

| Region | Time |
| --- | ---: |
| `cli.token.1.forward_all_layers` | 52.390 ms |
| `cli.token.1.logits` | 5.573 ms |
| `cli.token.1.total` | 58.050 ms |
| `cli.generate_text_total` | 650.119 ms |

Steady second-token aggregate over 22 layers:

| Operation group | Launches | cuBLAS calls | Stream syncs | Stream waits | Elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qkv_project` | 0 | 66 | 66 | 0 | 6.179 ms |
| `mlp_gate_up` | 0 | 44 | 44 | 0 | 11.800 ms |
| `mlp_down` | 0 | 22 | 22 | 22 | 13.477 ms |
| `out_project` | 0 | 22 | 22 | 0 | 3.394 ms |
| `rope_q` | 22 | 0 | 22 | 0 | 0.832 ms |
| `kv_append_rope_k` | 22 | 0 | 22 | 0 | 1.537 ms |
| `attention` | 22 | 0 | 22 | 0 | 1.752 ms |
| `rms_norm_weighted` | 44 | 0 | 44 | 0 | 2.210 ms |
| `residual_add` | 44 | 0 | 44 | 0 | 1.076 ms |
| `swiglu` | 22 | 0 | 0 | 0 | 0.339 ms |

Notes:

- SwiGLU explicit stream synchronizations dropped from 22 to 0 per steady token.
  The required dependency is now represented as 22 stream waits in `mlp_down`,
  because cuBLAS GEMM still runs on the prefill stream.
- This is primarily a scheduling prerequisite for CUDA Graph capture rather than
  a standalone latency win. The per-run timing was mixed: `swiglu` elapsed fell,
  while `mlp_down` now includes the event wait plus normal GEMM sync cost.
- The next strict task is to prototype CUDA Graph capture for warm one-token
  decode now that the hot path has stable scratch buffers and explicit stream
  dependencies.

## 2026-05-11: CUDA Graph Capture Prototype

This checkpoint added CUDA Graph capture/instantiate/launch/destroy plumbing and
validated it with fixed-pointer decode-style async elementwise work on the M40.

Validation command:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test cuda_elementwise -- --nocapture --test-threads=1
```

Observed result:

| Test | Result |
| --- | --- |
| `cuda_graph_replays_decode_elementwise_work` | pass |
| `async_elementwise_wrappers_match_cpu` | pass |
| `stream_wait_allows_prefill_gemm_to_consume_decode_swiglu` | pass |

Notes:

- The prototype captures and replays a decode-stream graph containing fixed
  device pointers and async kernel enqueues. This validates the CUDA Graph
  lifecycle on Tesla M40 without changing the production decode path yet.
- Whole-token graph capture is not ready to enable: the hot path still has sync
  compatibility wrappers around cuBLAS and other kernels, and KV append still
  reads the sequence length through a host-side `cudaMemcpy`.
- The next graph-specific step, when it becomes the priority again, should be to
  make a one-layer decode subgraph fully async and device-parameterized. The
  immediate strict-plan task now moves to reducing the remaining production
  decode sync boundaries before packed variable-length decode scheduling.

## 2026-05-11: Async Full-Layer Decode Boundary Profile

This profile was taken after changing full-layer forward to enqueue RMSNorm,
Q RoPE, fused K RoPE + KV append, GQA attention, and residual adds
asynchronously. Dependencies between decode-stream kernels and prefill-stream
cuBLAS GEMMs are now represented as explicit stream waits. The stream-wait FFI
uses one event per dependency so alternating waits cannot reuse and overwrite a
single bridge event.

Command:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  M40LLM_LAUNCH_LOG=1 M40LLM_SYNC_LOG=1 M40LLM_PROFILE_LOG=1 \
  M40LLM_TIMING_LOG=1 cargo run --features cuda -- generate \
  /mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf \
  Hello --max-tokens 1 --top-k 1 --require-sm52
```

TinyLlama CLI timing, measured on Tesla M40:

| Region | Time |
| --- | ---: |
| `cli.token.1.forward_all_layers` | 53.197 ms |
| `cli.token.1.logits` | 4.559 ms |
| `cli.token.1.total` | 57.857 ms |
| `cli.generate_text_total` | 691.421 ms |

Steady second-token aggregate over 22 layers:

| Operation group | Launches | cuBLAS calls | Stream syncs | Stream waits | Elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qkv_project` | 0 | 66 | 66 | 22 | 6.142 ms |
| `mlp_gate_up` | 0 | 44 | 44 | 22 | 11.767 ms |
| `mlp_down` | 0 | 22 | 22 | 22 | 6.711 ms |
| `out_project` | 0 | 22 | 22 | 22 | 3.409 ms |
| `rope_q` | 22 | 0 | 0 | 22 | 0.500 ms |
| `kv_append_rope_k` | 22 | 0 | 0 | 0 | 0.839 ms |
| `attention` | 22 | 0 | 0 | 0 | 0.333 ms |
| `attn_norm` | 22 | 0 | 0 | 0 | 0.415 ms |
| `ffn_norm` | 22 | 0 | 0 | 0 | 0.290 ms |
| `attn_residual` | 22 | 0 | 0 | 22 | 0.507 ms |
| `mlp_residual` | 22 | 0 | 0 | 22 | 0.501 ms |
| `swiglu` | 22 | 0 | 0 | 0 | 0.325 ms |

Notes:

- Converted non-GEMM forward operations now contribute zero stream
  synchronizations in the steady token. Remaining forward synchronizations are
  the 154 materialized-FP32 cuBLAS projection calls.
- Explicit stream waits increased because each prefill-stream GEMM boundary now
  waits for decode-stream producers, and decode-stream consumers wait for
  prefill-stream GEMM outputs. This is graph/scheduling groundwork, not an
  immediate latency win.
- Whole-token graph capture is still blocked by synchronous cuBLAS wrappers and
  host-side KV sequence length updates. The next narrow step is to add async
  cuBLAS enqueue wrappers or a one-layer graph capture path that can keep GEMM
  dependencies inside capture without host stream synchronizations.

## 2026-05-11: Async Materialized cuBLAS Decode Profile

This profile was taken after adding an async cuBLAS enqueue wrapper for
materialized FP32 GGUF projection weights and routing full-layer decode
projections through explicit stream waits instead of per-GEMM stream
synchronizations. Synchronous wrappers remain available for tests and simple
callers.

Command:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  M40LLM_LAUNCH_LOG=1 M40LLM_SYNC_LOG=1 M40LLM_PROFILE_LOG=1 \
  M40LLM_TIMING_LOG=1 cargo run --features cuda -- generate \
  /mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf \
  Hello --max-tokens 1 --top-k 1 --require-sm52
```

TinyLlama CLI timing, measured on Tesla M40:

| Region | Time |
| --- | ---: |
| `cli.token.1.forward_all_layers` | 24.090 ms |
| `cli.token.1.logits` | 4.985 ms |
| `cli.token.1.total` | 29.140 ms |
| `cli.generate_text_total` | 619.592 ms |

Steady second-token aggregate over 22 layers:

| Operation group | Launches | cuBLAS calls | Stream syncs | Stream waits | Elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qkv_project` | 0 | 66 | 0 | 22 | 4.060 ms |
| `mlp_gate_up` | 0 | 44 | 0 | 22 | 2.838 ms |
| `mlp_down` | 0 | 22 | 0 | 22 | 2.179 ms |
| `out_project` | 0 | 22 | 0 | 22 | 1.769 ms |
| `rope_q` | 22 | 0 | 0 | 22 | 0.791 ms |
| `kv_append_rope_k` | 22 | 0 | 0 | 0 | 0.611 ms |
| `attention` | 22 | 0 | 0 | 0 | 0.235 ms |
| `attn_norm` | 22 | 0 | 0 | 0 | 0.307 ms |
| `ffn_norm` | 22 | 0 | 0 | 0 | 0.199 ms |
| `attn_residual` | 22 | 0 | 0 | 22 | 0.743 ms |
| `mlp_residual` | 22 | 0 | 0 | 22 | 0.771 ms |
| `swiglu` | 22 | 0 | 0 | 0 | 0.239 ms |

Notes:

- Materialized projection groups still issue 154 cuBLAS calls per steady token,
  but they now contribute zero stream synchronizations in the full-layer
  forward profile.
- The remaining observed synchronizations are host boundaries: output norm still
  uses its sync compatibility wrapper, logits explicitly synchronizes before
  D2H copyback for host sampling, and CLI shutdown synchronizes streams.
- This clears the per-GEMM sync blocker for one-layer CUDA Graph experiments.
  The next graph blocker is host-managed KV position/length state.

## 2026-05-11: Packed Varlen Decode Scheduler Foundation

This checkpoint adds a scheduler-facing decode batch plan for mixed-length
last-token attention. The plan filters active requests, preserves per-request
sequence IDs, builds `BatchMetadata` with `query_len=1`, uploads sequence IDs
and KV lengths to device metadata buffers, and dispatches the existing batched
GQA decode attention primitive.

Validation target:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test attention_batched_varlen -- --nocapture --test-threads=1
```

Coverage:

| Path | Shape | Expected result |
| --- | --- | --- |
| Direct batched decode attention | `seq_lens=[1,3,5]` | matches individual decode |
| Async batched decode attention | `seq_lens=[1,3,5]` | matches sync batched decode |
| Scheduler-built CUDA plan | active requests with `kv_len=[1,3,5]` | matches direct batched decode |
| Opt-in `ldg_kv` cache experiment | `seq_lens=[1,3,5]` | matches default batched decode |

Notes:

- `/generate` remains serialized by default. Setting
  `M40LLM_SERVER_BATCH_DECODE=1` enables leased per-request KV sequence slots,
  and the later server scheduler checkpoint adds packed batched decode attention
  for compatible buffered requests.
- Model-level KV ownership supports sequence-major physical slots for
  `KV[layer][sequence]`; multi-request generation can use separate logical
  sequences.
- Packed prefill should wait until decode batching has real request/session
  ownership above this scheduler foundation.

## 2026-05-11: Explicit-Position KV Append

This checkpoint adds graph-friendly KV append APIs for fused K RoPE plus FP32 to
FP16 KV storage:

- `m40llm_kvcache_append_token_f32_rope_k_at_async` takes an explicit token
  position and updates `seq_map[seq_id]` on device.
- `m40llm_kvcache_append_token_f32_rope_k_position_dev_async` reads the token
  position from a device pointer so a captured graph can bind a stable parameter
  address.
- Full-layer decode now uses the explicit-position path because the Rust decode
  loop already knows `pos = seq_len - 1`.

Validation:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test kv_f32_to_f16_append -- --nocapture --test-threads=1

M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test forward_with_layer_smoke -- --nocapture --test-threads=1
```

TinyLlama profile spot-check:

```text
forward.layer.N.seq_len.2.kv_append_rope_k:
  op=kvcache_append_token_f32_rope_k_at
  launches=1 syncs=0 h2d=0(0 bytes) d2h=0(0 bytes)
```

This removes the previous host-side `cudaMemcpyDeviceToHost` length read from
the production KV append path. The next graph-specific step is to capture a
one-layer decode segment that includes async cuBLAS and a stable device position
parameter.

## 2026-05-12: One-Layer cuBLAS Graph Prototype

This checkpoint validates CUDA Graph capture with materialized FP32 cuBLAS
projection work on the Tesla M40. The test captures a one-layer-shaped prefill
stream graph containing seven async `cublasSgemm` calls:

- Q, K, V projections
- attention output projection
- MLP gate and up projections
- MLP down projection

The test warms cuBLAS before capture, captures the async GEMM sequence, launches
the graph, synchronizes once at the graph boundary, and compares all projection
outputs against CPU references.

Validation:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test cuda_elementwise \
  -- cuda_graph_replays_one_layer_projection_gemms --nocapture --test-threads=1
```

Result on M40: pass.

Notes:

- This proves async materialized cuBLAS calls can participate in CUDA Graph
  capture on the target GPU/toolchain.
- The prototype is still projection-only and single-stream. The next production
  step is to capture a true one-layer decode segment that also includes
  decode-stream elementwise/attention/KV work and cross-stream dependencies.

## 2026-05-12: Cross-Stream Decode Graph Prototype

This checkpoint validates a graph segment with the same stream topology as the
warm decode path:

1. decode stream enqueues elementwise work,
2. prefill stream waits and runs async materialized cuBLAS,
3. decode stream waits and enqueues a follow-up elementwise op.

Validation:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test cuda_elementwise \
  -- cuda_graph_replays_cross_stream_decode_gemm_segment --nocapture --test-threads=1
```

Result on M40: pass.

Notes:

- CUDA graph capture can now cover the production stream dependency pattern,
  not just isolated prefill-stream cuBLAS work.
- This is still synthetic. The next step is to capture a real one-layer decode
  slice using model/workspace pointers, KV append, attention, and projection
  wrappers.

## 2026-05-12: Production One-Layer Decode Graph Smoke

This checkpoint captures a real `forward_one_token_with_layer` call after
warming the model's materialized weights and forward workspace. The graph covers
the production one-layer decode path, including:

- RMSNorm
- async materialized projection GEMMs
- Q RoPE
- fused K RoPE + KV append
- GQA attention
- residual adds
- SwiGLU and MLP projections

Validation:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test forward_with_layer_smoke \
  -- cuda_graph_replays_forward_one_token_with_layer --nocapture --test-threads=1
```

Result on M40: pass.

Notes:

- The test compares graph replay output against a normal one-layer forward on a
  separate KV sequence.
- This proves a warmed, fixed-shape, fixed-pointer one-layer decode graph can be
  captured and replayed. The next step is deciding how to cache and launch this
  in production sessions, then expanding from one layer toward full-token graph
  coverage.

## 2026-05-12: Device-Parameter Graph Wrappers

This checkpoint adds graph-compatible wrappers for per-token values that need to
vary between graph launches:

- Q RoPE can read `past_len` from a device `u32`.
- GQA last-token attention can read `seq_len` from a device `u32`.
- KV append already has a device-position variant from the prior checkpoint.

Validation:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test rmsnorm_rope -- --nocapture --test-threads=1

M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test attention_last_token -- --nocapture --test-threads=1
```

Result on M40: pass.

Notes:

- Device-seq-len attention intentionally starts with the generic GQA kernel.
  This keeps graph replay correctness simple; a tuned head64 device-parameter
  kernel can be added after graph-mode performance is measured.

## 2026-05-12: Opt-In DecodeSession One-Layer Graph Cache

`DecodeSession` now caches and replays a warmed one-layer decode CUDA Graph when
`M40LLM_DECODE_GRAPH=1`. The graph binds stable session scratch/workspace
pointers and uses device-resident `position` and `seq_len` parameters for Q
RoPE, fused K RoPE + KV append, and GQA attention. Multi-layer sessions log once
and continue using the normal async decode path.

Validation:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test forward_with_layer_smoke -- --nocapture --test-threads=1
```

Result on M40: pass, including `decode_session_uses_one_layer_graph_when_enabled`
with observed `cuda_graph_launch` profile events.

Notes:

- First use warms the normal one-layer path before capture so lazy workspace
  allocation and FP32 weight materialization do not occur inside stream capture.
- The current graph parameter update uses two small host-to-device copies before
  graph launch. That keeps the graph topology stable; a future device-side token
  counter can remove those copies if profiling shows they matter.
- This is intentionally not enabled for TinyLlama-class 22-layer sessions yet.
  The next graph step is expanding from one layer to a full-token graph once
  pointer stability and replay behavior are validated.

## 2026-05-12: Async MLP Stream-Order Regression Canary

The async materialized-GEMM path briefly regressed TinyLlama generation because
`swiglu_f32_async` read `dgate`/`dup` on the decode stream before the async MLP
gate/up cuBLAS calls on the prefill stream completed. The fix adds an explicit
decode-stream wait named `mlp_gate_up_to_swiglu`.

Validation:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test forward_with_layer_smoke -- --nocapture --test-threads=1

M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test tinyllama_generation_canary -- --nocapture --test-threads=1
```

Result on M40: pass. The TinyLlama canary uses the stock-quotes prompt, disables
graph mode, and asserts the exact deterministic generated token sequence:

```text
[13, 13, 29896, 29889, 22402, 385, 1409, 310, 10961, 29879,
 411, 1009, 5829, 29892, 1024, 29892, 322, 8666, 472, 278]
```

Corrected warm decode profile command:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 M40LLM_DECODE_GRAPH=0 \
  M40LLM_LAUNCH_LOG=1 M40LLM_SYNC_LOG=1 M40LLM_PROFILE_LOG=1 \
  M40LLM_TIMING_LOG=1 cargo run --features cuda -- generate \
  /mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf \
  Hello --max-tokens 1 --top-k 1 --require-sm52
```

TinyLlama CLI timing, measured on Tesla M40:

| Region | Time |
| --- | ---: |
| `cli.token.1.forward_all_layers` | 28.887 ms |
| `cli.token.1.logits` | 4.511 ms |
| `cli.token.1.total` | 33.482 ms |
| `cli.generate_text_total` | 604.047 ms |

Steady second-token notes:

- Full-layer forward still has zero stream synchronizations inside the 22-layer
  body, but now has the required 22 additional `mlp_gate_up_to_swiglu` stream
  waits.
- Expected steady-token projection count remains 154 async cuBLAS calls
  (7 projections x 22 layers).
- The corrected profile confirms the restored wait is visible in every
  `forward.layer.N.seq_len.2.swiglu` timing region.

## 2026-05-12: Opt-In Full-Token Decode Graph Capture

`DecodeSession` graph mode now captures the full all-layer decode token instead
of only one layer. Capture still stays behind `M40LLM_DECODE_GRAPH=1` and warms
the normal async path first so workspace allocation and FP32 weight
materialization happen outside CUDA stream capture.

Implementation notes:

- `forward_one_token_all_layers_for_sequence_graph_params` mirrors the normal
  all-layer forward loop but passes device-resident `position` and `seq_len`
  through every layer.
- The shared tiny GGUF fixture can now generate multi-layer test models, and the
  CUDA smoke covers both one-layer and two-layer `DecodeSession` graph replay.
- TinyLlama graph smoke captured all 22 layers with:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 M40LLM_DECODE_GRAPH=1 \
  cargo run --features cuda -- generate \
  /mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf \
  Hello --max-tokens 1 --top-k 1 --require-sm52
```

Observed log evidence:

```text
[cli] captured full-token decode CUDA graph layers=22
```

Validation:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test forward_with_layer_smoke -- --nocapture --test-threads=1
```

Result on M40: pass.

Next graph work should benchmark `M40LLM_DECODE_GRAPH=1` versus the normal async
path for steady TinyLlama decode. If graph replay does not materially reduce
latency, move the strict plan forward to packed varlen decode scheduling instead
of adding more graph complexity.

## 2026-05-12: Full-Token Decode Graph Benchmark

Benchmark command, run three times for each graph setting on Tesla M40:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 M40LLM_TIMING_LOG=1 \
  M40LLM_DECODE_GRAPH={0,1} cargo run --features cuda -- generate \
  /mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf \
  Hello --max-tokens 4 --top-k 1 --require-sm52
```

Both modes generated the expected text:

```text
, World!
```

Median timing across steady tokens 2-4 and three trials:

| Mode | `forward_all_layers` | `logits` | token total | `generate_text_total` |
| --- | ---: | ---: | ---: | ---: |
| `M40LLM_DECODE_GRAPH=0` | 8.386 ms | 21.295 ms | 29.833 ms | 744.079 ms |
| `M40LLM_DECODE_GRAPH=1` | 0.095 ms | 713.868 ms | 714.058 ms | 3341.133 ms |

Interpretation:

- Graph replay greatly reduces host-side `forward_all_layers` enqueue time, which
  confirms the graph launches rather than re-enqueueing every layer operation.
- End-to-end token latency regresses by roughly 24x for steady tokens because
  the following logits/output-norm region absorbs much larger GPU completion
  time.
- Keep `M40LLM_DECODE_GRAPH=1` experimental and off by default.
- Do not expand graph coverage yet. Either investigate graph replay stream
  completion/accounting against logits, or move the strict plan forward to real
  packed varlen decode scheduling.

## 2026-05-12: Server Decode Scheduler Skeleton

`M40LLM_SERVER_BATCH_DECODE=1` now routes buffered `/generate` requests through a
queued decode scheduler instead of serializing each whole request behind the
HTTP handler. The scheduler owns per-request `DecodeSession` state, holds leased
KV sequence slots for the request lifetime, steps active requests round-robin,
builds `DecodeBatchPlan` snapshots from active mixed-length requests, and still
uses the server generation lock around each CUDA token step to protect shared
workspace use across scheduler and streaming paths.

Current boundary:

- Superseded by the packed batched decode attention checkpoint below.
- Streaming `/generate` remains on the previous serialized path for this slice.

Validation:

```bash
cargo fmt --all -- --check
cargo test --no-default-features --locked
cargo test --no-default-features --features server --locked
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda,server --test server_smoke -- --nocapture --test-threads=1
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test attention_batched_varlen -- --nocapture --test-threads=1
```

The CUDA server smoke showed two concurrent buffered requests assigned sequence
IDs 0 and 1 and stepped alternately by the scheduler. The follow-up checkpoint
below replaces the per-request attention region with packed batched GQA decode
attention while keeping the same request/session ownership model.

## 2026-05-12: Packed Batched Decode Attention in Server Scheduler

`M40LLM_SERVER_BATCH_DECODE=1` now has a real batched full-layer decode path for
head_dim=64 models. Each scheduler tick prepares one token per active request,
packs request hidden rows into a row-aware forward workspace, runs row-batched
projection and MLP GEMMs, applies per-row RoPE/KV append for mixed positions,
uses `attention_last_token_f32_gqa_batched_async` for the shared decode
attention step, and scatters rows back into each request's `DecodeSession`
scratch before host sampling.

Current boundary:

- Batch size 1 and non-head64 models fall back to the previous per-request path.
- Buffered `/generate` is covered; streaming `/generate` remains on the
  serialized path.
- This is a correctness/scheduler integration checkpoint; TinyLlama concurrency
  latency measurements are still pending.

Validation:

```bash
source scripts/dev-env.sh && M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda,server --test server_smoke -- --nocapture --test-threads=1

source scripts/dev-env.sh && M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test forward_with_layer_smoke -- --nocapture --test-threads=1

source scripts/dev-env.sh && M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test attention_batched_varlen -- --nocapture --test-threads=1
```

## 2026-05-12: Decode Graph Replay Diagnostic Sync

`M40LLM_DECODE_GRAPH_DIAG_SYNC=1` adds a diagnostic-only graph replay path:

- records CUDA events before and after `cudaGraphLaunch`,
- synchronizes the graph stream immediately after launch,
- logs graph replay GPU elapsed time, and
- inserts an explicit decode-stream event dependency before logits consumes the
  graph output buffer.

Diagnostic command on Tesla M40:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  M40LLM_DECODE_GRAPH=1 M40LLM_DECODE_GRAPH_DIAG_SYNC=1 \
  M40LLM_TIMING_LOG=1 cargo run --features cuda -- generate \
  /mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf \
  Hello --max-tokens 4 --top-k 1 --require-sm52
```

Observed graph replay GPU elapsed time:

| Token | Graph replay GPU elapsed | `forward_all_layers` | `logits` | token total |
| --- | ---: | ---: | ---: | ---: |
| 2 | 539.041 ms | 539.133 ms | 3.947 ms | 543.176 ms |
| 3 | 709.210 ms | 709.280 ms | 3.979 ms | 713.334 ms |
| 4 | 883.275 ms | 883.375 ms | 3.995 ms | 887.439 ms |

Interpretation:

- The previous graph benchmark's large `logits` timing was stream-completion
  accounting: logits was waiting for slow graph replay to finish.
- With explicit post-launch synchronization, logits/output norm returns to the
  expected small timing, but graph replay itself is far slower than the normal
  async path.
- Keep `M40LLM_DECODE_GRAPH=1` experimental and off by default.
- Do not expand graph coverage until graph replay performance is understood;
  move the main strict plan forward to packed varlen decode scheduling.

## 2026-05-12: Attention Bench Rebaseline (Latest Run)

Command:

```bash
source scripts/dev-env.sh
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo bench --features cuda --bench attention -- --sample-size 10
```

Observations:

- Full benchmark set completed successfully on Tesla M40.
- The first few `attention_last_token_f32_gqa` cases showed unusually large variance;
  this run is still useful for relative packed-batched comparisons, but individual
  single-case absolute values for very short sequence lengths should be re-run in a
  standalone bench if absolute confidence is required.

### Last-token attention

| Case | Throughput (ns) | Notes |
| --- | ---: | --- |
| `s1` | 2.6589 ms (baseline) | includes no `ldg` |
| `s1_ldg_kv` | 1.2593 ms (median) | high variance |
| `s16` | 1.9324 ms | no `ldg` |
| `s16_ldg_kv` | 623.92 µs (median) | high variance |
| `s128` | 3.4103 ms | no `ldg` |
| `s128_ldg_kv` | 3.0305 ms | slight improvement vs no-ldg |
| `s512` | 3.7727 ms | no `ldg` |
| `s512_ldg_kv` | 1.2235 ms (median) | best path here |
| `s1024` | 2.230 ms | no `ldg` |
| `s1024_ldg_kv` | 2.2322 ms | similar |

### Batched last-token decode attention (`attention_last_token_f32_gqa_batched_varlen`)

All times are median reported by Criterion.

| Distribution | Baseline (individual) | Packed batched | `ldg_kv` packed | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `avg_0p6_max` | 4.594 ms | 1.584 ms | 1.585 ms | 2.9x |
| `skewed` | 2.822 ms | 2.122 ms | 2.120 ms | 1.3x |
| `near_uniform` | 8.601 ms | 2.445 ms | 2.444 ms | 3.5x |

### Prefill attention (`attention_prefill_f32_gqa_varlen`)

Times are median over 10 samples for each case.

| Distribution | Padded max | Packed varlen | Bucketed varlen | Best |
| --- | ---: | ---: | ---: | --- |
| `avg_0p6_max` | 297.29 ms | 178.34 ms | 179.08 ms | Packed 1.67x |
| `skewed` | 526.91 ms | 141.66 ms | 142.05 ms | Packed 3.72x |
| `near_uniform` | 527.27 ms | 474.53 ms | 474.59 ms | Packed 1.11x |
| `prefix_query` | 123.97 ms | 50.61 ms | 51.35 ms | Packed 2.45x |

Key interpretation:

- Padded-varlen prefill remains the baseline; packed/bucketed variants materially
  improve throughput on mixed and skewed workloads.
- For near-uniform sequences, packed varlen narrows toward padded behavior, as
  expected.
- `ldg_kv` is now only used in last-token batched decode and provides no strong
  win or loss relative to default in this run.

## 2026-05-12: TinyLlama Buffered Server Batch-Decode Benchmark

This checkpoint compares buffered `/generate` with and without the queued
batched decode scheduler. CUDA Graph replay stayed disabled because graph replay
diagnostics showed it is slower than the normal async path.

Environment:

- GPU: Tesla M40 24GB, sm_52
- Model: `/mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf`
- Features: `cuda,server`
- Command prefix: `source scripts/dev-env.sh && M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1`
- Benchmark script: `TRIALS=3 MAX_TOKENS=2 PORT_BASE=52680 scripts/bench_server_batch_decode.sh`
- Log directory: `/tmp/m40llm_batch_decode_bench_20260512_183735`

Each case starts a fresh server, performs one warmup request, then sends the case
requests concurrently. Times below are means across three trials.

| Case | Batch decode | Requests | Mean wall | Mean avg request latency | Mean tokens/s | HTTP |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `batch1_hello` | off | 1 | 174.0 ms | 0.164952 s | 11.495 | 3/3 ok |
| `batch1_hello` | on | 1 | 176.0 ms | 0.167599 s | 11.364 | 3/3 ok |
| `batch2_same` | off | 2 | 338.7 ms | 0.247982 s | 11.811 | 6/6 ok |
| `batch2_same` | on | 2 | 287.3 ms | 0.274962 s | 13.924 | 6/6 ok |
| `batch4_mixed` | off | 4 | 1537.0 ms | 0.913495 s | 5.205 | 12/12 ok |
| `batch4_mixed` | on | 4 | 909.3 ms | 0.679747 s | 8.798 | 12/12 ok |
| `batch4_skewed` | off | 4 | 2456.0 ms | 1.653452 s | 3.257 | 12/12 ok |
| `batch4_skewed` | on | 4 | 1523.7 ms | 0.966136 s | 5.250 | 12/12 ok |

Speedups from enabling `M40LLM_SERVER_BATCH_DECODE=1`:

| Case | Wall-time speedup | Throughput speedup |
| --- | ---: | ---: |
| `batch1_hello` | 0.99x | 0.99x |
| `batch2_same` | 1.18x | 1.18x |
| `batch4_mixed` | 1.69x | 1.69x |
| `batch4_skewed` | 1.61x | 1.61x |

Correctness checks:

- All benchmarked requests returned HTTP 200.
- Batch scheduler logs showed distinct leased sequence IDs for concurrent
  requests.
- `cargo test --features cuda,server --test server_smoke -- --nocapture --test-threads=1`
  passed.
- `M40LLM_TINYLLAMA_CANARY_MODEL=<TinyLlama path> cargo test --features cuda
  --test tinyllama_generation_canary -- --nocapture --test-threads=1` passed.

Interpretation:

- The batched decode scheduler is neutral for batch size 1 and improves
  concurrent buffered request throughput for batch sizes 2 and 4.
- The mixed and skewed prompt cases are the important validation signal: they
  show the scheduler can share decode work across active requests with different
  prompt/KV lengths while preserving per-request outputs.
- This is enough to keep moving the strict plan forward. The next task is packed
  varlen prefill integration, starting behind an opt-in server flag or internal
  scheduler path until correctness and perf are characterized.

## 2026-05-12: TinyLlama Opt-In Packed Prefill Server Benchmark

This checkpoint integrates packed variable-length prompt prefill into the
buffered server scheduler behind `M40LLM_SERVER_BATCH_PREFILL=1`. The comparison
below keeps `M40LLM_SERVER_BATCH_DECODE=1` enabled and toggles only packed
prefill.

Environment:

- GPU: Tesla M40 24GB, sm_52
- Model: `/mnt/array-fastest/home/guyep/.cache/m40-llm/models/TinyLlama-1.1B-Chat-v1.0.f16.gguf`
- Features: `cuda,server`
- Command prefix: `source scripts/dev-env.sh && M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1`
- Benchmark script: `BATCH_DECODE_MODES=1 PREFILL_MODES="0 1" TRIALS=3 MAX_TOKENS=2 PORT_BASE=53180 scripts/bench_server_batch_decode.sh`
- Log directory: `/tmp/m40llm_batch_decode_bench_20260512_191709`

Times below are means across three trials.

| Case | Packed prefill | Requests | Mean wall | Mean avg request latency | Mean tokens/s | HTTP |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `batch1_hello` | off | 1 | 175.7 ms | 0.167317 s | 11.386 | 3/3 ok |
| `batch1_hello` | on | 1 | 176.0 ms | 0.167416 s | 11.364 | 3/3 ok |
| `batch2_same` | off | 2 | 286.0 ms | 0.274211 s | 13.987 | 6/6 ok |
| `batch2_same` | on | 2 | 255.7 ms | 0.243096 s | 15.646 | 6/6 ok |
| `batch4_mixed` | off | 4 | 904.3 ms | 0.676625 s | 8.846 | 12/12 ok |
| `batch4_mixed` | on | 4 | 481.7 ms | 0.412279 s | 16.609 | 12/12 ok |
| `batch4_skewed` | off | 4 | 1535.7 ms | 0.975667 s | 5.211 | 12/12 ok |
| `batch4_skewed` | on | 4 | 610.7 ms | 0.545692 s | 13.112 | 12/12 ok |

Speedups from enabling `M40LLM_SERVER_BATCH_PREFILL=1` while batched decode is
already enabled:

| Case | Wall-time speedup | Throughput speedup |
| --- | ---: | ---: |
| `batch1_hello` | 1.00x | 1.00x |
| `batch2_same` | 1.12x | 1.12x |
| `batch4_mixed` | 1.88x | 1.88x |
| `batch4_skewed` | 2.51x | 2.52x |

Validation:

- All benchmarked requests returned HTTP 200.
- `forward_batched_prefill_uses_varlen_attention` compares packed prefill final
  hidden output against sequential token prefill on a tiny CUDA model.
- `server_smoke` now enables `M40LLM_SERVER_BATCH_PREFILL=1` and asserts that
  server batching launches `attention_prefill_f32_gqa_varlen_head64`.
- `attention_prefill_varlen`, `forward_with_layer_smoke`, and CUDA/server Clippy
  passed after the integration.

Interpretation:

- Packed prefill is neutral for batch size 1 because the opt-in path falls back.
- Mixed and skewed prompts benefit the most because the scheduler avoids running
  all prompt attention at the largest prompt length.
- The next scheduler task is mixed prefill/decode overlap or broader prefill
  compatibility; keep this opt-in until more server workloads are characterized.

## 2026-05-12: Experimental KV Compression Attention Benchmark

This checkpoint adds experimental compressed-KV decode attention modes inspired
by block summary and block-selection ideas from DeepSeek-AI's DeepSeek-V4 work
on efficient million-token context intelligence. It does not attempt to
reproduce that architecture exactly. The implementation deliberately separates
sparse selection from lossy compression:

- `off`: dense exact KV attention.
- `block-select-exact`: keeps exact old KV and attends selected exact old
  blocks plus the exact recent window.
- `block-summary`: attends old block mean K/V summaries plus the exact recent
  window.
- `block-select-lossy`: scores old summaries and attends selected summaries
  plus the exact recent window.

Historical limitation: this first pass kept dense KV as the backing store for
all modes so selection and lossy summary quality could be validated before old
KV was physically discarded. The memory columns below therefore show both
physical dense KV bytes and compressed-equivalent K/V bytes for the summary
modes.

Environment:

- GPU: Tesla M40 24GB, sm_52
- Features: `cuda`
- Command: `source scripts/dev-env.sh && M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo bench --features cuda --bench attention -- attention_kv_compression_modes --sample-size 10`
- Benchmark shape: `q_heads=32`, `kv_heads=4`, `head_dim=64`
- Config: `recent_window=1024`, `block_size=32`, `top_blocks=16`

| Context | Mode | Mean time | Tokens/sec | Dense physical KV | Compressed-equivalent KV |
| --- | --- | ---: | ---: | ---: | ---: |
| 4K | dense | 8.944 ms | 111.81 | 4.00 MiB | 4.00 MiB |
| 4K | block-select-exact | 5.395 ms | 185.36 | 4.00 MiB | 4.00 MiB |
| 4K | block-summary | 4.639 ms | 215.56 | 4.00 MiB | 1.09 MiB |
| 4K | block-select-lossy | 4.619 ms | 216.50 | 4.00 MiB | 1.02 MiB |
| 8K | dense | 17.897 ms | 55.87 | 8.00 MiB | 8.00 MiB |
| 8K | block-select-exact | 8.147 ms | 122.74 | 8.00 MiB | 8.00 MiB |
| 8K | block-summary | 7.454 ms | 134.16 | 8.00 MiB | 1.22 MiB |
| 8K | block-select-lossy | 7.282 ms | 137.32 | 8.00 MiB | 1.02 MiB |
| 16K | dense | 4.983 s | 0.20 | 16.00 MiB | 16.00 MiB |
| 16K | block-select-exact | 13.672 ms | 73.14 | 16.00 MiB | 16.00 MiB |
| 16K | block-summary | 13.294 ms | 75.22 | 16.00 MiB | 1.47 MiB |
| 16K | block-select-lossy | 12.819 ms | 78.01 | 16.00 MiB | 1.02 MiB |
| 32K | dense | 10.205 s | 0.10 | 32.00 MiB | 32.00 MiB |
| 32K | block-select-exact | 24.318 ms | 41.12 | 32.00 MiB | 32.00 MiB |
| 32K | block-summary | 24.725 ms | 40.45 | 32.00 MiB | 1.97 MiB |
| 32K | block-select-lossy | 23.367 ms | 42.80 | 32.00 MiB | 1.02 MiB |

Validation:

- `attention_block_select_exact_matches_dense_when_all_old_blocks_selected`
  verifies exact selection matches dense attention when all old blocks are
  selected.
- `attention_block_summary_lossy_is_finite_and_deterministic` verifies lossy
  summary/select outputs are finite and deterministic.
- `long_context_needle_retrieval_quality_smoke` adds an env-gated retrieval
  quality harness using `M40LLM_LONG_CONTEXT_RETRIEVAL_MODEL`; no long-context
  model was configured for this benchmark run, so retrieval quality was not
  measured here.

Interpretation:

- `block-select-exact` is the useful validation baseline: it isolates whether
  cheap block scoring selects a good sparse exact read set before lossy summaries
  are trusted.
- The dense 16K/32K baseline falls back to the generic attention path, so the
  speedups there compare against an intentionally poor current dense kernel.
- A later checkpoint adds a physical compressed sidecar allocation/update path
  so lossy modes reduce actual allocated KV memory instead of only reporting
  compressed-equivalent bytes.

## 2026-05-13: Physical Compressed KV Sidecar

This checkpoint changes `block-summary` and `block-select-lossy` from
dense-backed approximation modes into physical compressed KV sidecar modes for
CLI decode. The sidecar keeps a recent exact FP16 ring buffer, evicts older
tokens into fixed-size block mean K/V summaries, and stores FP32 summary
accumulators so summaries can be updated incrementally during decode.
`off` and `block-select-exact` remain dense-backed because they are exact
validation modes.

Environment:

- GPU: Tesla M40 24GB, sm_52
- Features: `cuda`
- Command: `source scripts/dev-env.sh && M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo bench --features cuda --bench attention -- attention_kv_compression_modes --sample-size 10`
- Benchmark shape: `q_heads=32`, `kv_heads=4`, `head_dim=64`
- Config: `recent_window=1024`, `block_size=32`, `top_blocks=16`

| Context | Mode | Mean time | Tokens/sec | Dense equivalent KV | Actual allocated KV |
| --- | --- | ---: | ---: | ---: | ---: |
| 4K | dense | 9.866 ms | 101.36 | 4.00 MiB | 4.00 MiB |
| 4K | block-select-exact | 5.679 ms | 176.09 | 4.00 MiB | 4.00 MiB |
| 4K | block-summary | 3.101 ms | 322.43 | 4.00 MiB | 1.28 MiB |
| 4K | block-select-lossy | 3.306 ms | 302.43 | 4.00 MiB | 1.28 MiB |
| 8K | dense | 19.697 ms | 50.77 | 8.00 MiB | 8.00 MiB |
| 8K | block-select-exact | 8.431 ms | 118.61 | 8.00 MiB | 8.00 MiB |
| 8K | block-summary | 3.527 ms | 283.55 | 8.00 MiB | 1.66 MiB |
| 8K | block-select-lossy | 3.964 ms | 252.28 | 8.00 MiB | 1.66 MiB |
| 16K | dense | 5.184 s | 0.19 | 16.00 MiB | 16.00 MiB |
| 16K | block-select-exact | 13.855 ms | 72.18 | 16.00 MiB | 16.00 MiB |
| 16K | block-summary | 4.316 ms | 231.70 | 16.00 MiB | 2.41 MiB |
| 16K | block-select-lossy | 5.166 ms | 193.59 | 16.00 MiB | 2.41 MiB |
| 32K | dense | 10.388 s | 0.10 | 32.00 MiB | 32.00 MiB |
| 32K | block-select-exact | 24.492 ms | 40.83 | 32.00 MiB | 32.00 MiB |
| 32K | block-summary | 5.953 ms | 168.00 | 32.00 MiB | 3.91 MiB |
| 32K | block-select-lossy | 7.436 ms | 134.49 | 32.00 MiB | 3.91 MiB |

Validation:

- `cargo test --features cuda --test attention_parity_cuda_grid -- --nocapture --test-threads=1`
  passed on M40, including the compressed recent-window parity test.
- TinyLlama CLI smoke with `--kv-compress-mode block-summary` generated one
  token through the GPU path and logged `actual_allocated_bytes=25234264` versus
  `dense_equivalent_bytes=46137344` for the selected context.

Interpretation:

- The physical sidecar now reduces allocated KV memory for lossy modes. The
  reported allocation includes recent exact KV, summary FP16 K/V, FP32 summary
  accumulators, block counts, and sequence metadata.
- `block-select-lossy` currently allocates the same sidecar as `block-summary`;
  it reduces attended summary entries but does not yet shrink the stored summary
  table.
- The dense 16K/32K baseline still falls back to the generic attention path, so
  those speedups compare against a known weak dense kernel rather than a tuned
  long-context dense baseline.
- Representative-token storage is still deferred; `--kv-compress-representatives`
  is accepted but not yet used by the sidecar.

## 2026-05-13: Compressed KV Retrieval Quality Harness

This checkpoint upgrades the env-gated retrieval smoke into a diagnostic quality
harness. It probes an explicit GGUF model path before loading weights and
reports per-mode retrieval outcomes without forcing lossy modes to pass. The
harness no longer scans cache trees; set `M40LLM_LONG_CONTEXT_RETRIEVAL_MODEL`
to choose the model and `M40LLM_KV_QUALITY_FULL=1` only for the broad context
sweep.

Environment:

- GPU: Tesla M40 24GB, sm_52
- Features: `cuda`
- Command: `source scripts/dev-env.sh && M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 M40LLM_LONG_CONTEXT_RETRIEVAL_MODEL=/mnt/array-fastest/home/guyep/.cache/m40-llm/models/Llama-3.2-1B-Instruct-f16.gguf M40LLM_KV_QUALITY_SMOKE_TOKENS=64 cargo test --features cuda --test kv_compression_long_context -- --nocapture --test-threads=1`

Selected model:

| Model | Size | Context | Reason |
| --- | ---: | ---: | --- |
| Llama-3.2-1B-Instruct-f16.gguf | 2.31 GiB | 131072 | Explicit model path; supported after tied-output embedding support |

Candidate probe summary:

- Cache discovery was removed from this harness to keep test startup bounded and
  explicit. Use `M40LLM_LONG_CONTEXT_RETRIEVAL_MODEL` to select the model.
- The selected Llama 3.2 1B F16 model has LLaMA metadata, 128K context,
  GQA head_dim=64, and F16/F32 tensors.

Retrieval results:

| Target | Needle | Mode | Status | Notes |
| ---: | --- | --- | --- | --- |
| 64 | old | off | pass | Generated `ZXQ-NEEDLE-41729` |
| 64 | old | block-select-exact | pass | Generated `ZXQ-NEEDLE-41729` |
| 64 | old | block-summary | pass | Generated `ZXQ-NEEDLE-41729` |
| 64 | old | block-select-lossy | pass | Generated `ZXQ-NEEDLE-41729` |
| 64 | recent | off | pass | Generated `ZXQ-NEEDLE-41729` |
| 64 | recent | block-select-exact | pass | Generated `ZXQ-NEEDLE-41729` |
| 64 | recent | block-summary | pass | Generated `ZXQ-NEEDLE-41729` |
| 64 | recent | block-select-lossy | pass | Generated `ZXQ-NEEDLE-41729` |

Interpretation:

- The earlier dense `!!!!!!!!` failure was caused by reading GGUF weights from an
  unaligned tensor data offset. After applying the GGUF default 32-byte tensor
  data alignment, Llama 3.2 logits are finite and the short retrieval smoke is
  a valid compression-quality baseline.
- `block-select-exact`, `block-summary`, and `block-select-lossy` all match the
  dense baseline on this 64-token old/recent smoke. This is not enough evidence
  to trust lossy compression at long context; use `M40LLM_KV_QUALITY_FULL=1`
  for the broader sweep.
- The harness now uses `M40LLM_KV_QUALITY_MAX_TOKENS` with a default of 16 so
  exact-code answers are not falsely marked failed due to truncation.

Follow-up 512-token retrieval validation:

- Command: `source scripts/dev-env.sh && M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 M40LLM_LONG_CONTEXT_RETRIEVAL_MODEL=/mnt/array-fastest/home/guyep/.cache/m40-llm/models/Llama-3.2-1B-Instruct-f16.gguf M40LLM_KV_QUALITY_SMOKE_TOKENS=512 M40LLM_KV_QUALITY_REPORT=/tmp/m40llm_kv_quality_llama32_512.jsonl cargo test --features cuda --test kv_compression_long_context -- --nocapture --test-threads=1`
- Result: passed on M40 in 530.99 s; report contained 8 JSONL rows.

| Target | Prompt tokens | Needle | Mode | Status | Elapsed | Output |
| ---: | ---: | --- | --- | --- | ---: | --- |
| 512 | 469 | old | off | pass | 60.434 s | `ZXQ-NEEDLE-41729` |
| 512 | 469 | old | block-select-exact | pass | 61.572 s | `ZXQ-NEEDLE-41729` |
| 512 | 469 | old | block-summary | pass | 66.835 s | `ZXQ-NEEDLE-41729` |
| 512 | 469 | old | block-select-lossy | pass | 66.863 s | `ZXQ-NEEDLE-41729` |
| 512 | 484 | recent | off | pass | 62.582 s | `ZXQ-NEEDLE-41729` |
| 512 | 484 | recent | block-select-exact | pass | 64.766 s | `ZXQ-NEEDLE-41729` |
| 512 | 484 | recent | block-summary | pass | 70.251 s | `ZXQ-NEEDLE-41729` |
| 512 | 484 | recent | block-select-lossy | pass | 70.252 s | `ZXQ-NEEDLE-41729` |

Interpretation:

- Dense `off` passes at 512 tokens, so this run is a valid reference-capability
  check for the compression modes.
- `block-select-exact` passing old/recent cases means sparse block selection did
  not lose the needle at this context size.
- `block-summary` and `block-select-lossy` passing old/recent cases means lossy
  summaries are not obviously broken at this bounded context size.
- The full 4K+ sweep remains deferred. This 512-token run already takes almost
  nine minutes because the harness currently processes prompt tokens
  token-by-token and emits verbose per-token diagnostics. Quieting the harness
  and/or adding a faster prefill path should happen before making full-sweep
  quality runs routine.

## 2026-05-12: Quiet KV Quality Harness Timing

This checkpoint keeps the same retrieval behavior but makes the quality harness
practical to run repeatedly. Per-token `DecodeSession` diagnostics are now gated
behind `M40LLM_DECODE_SESSION_LOG=1`, JSONL rows include prompt/decode/total
timing fields, and `M40LLM_KV_QUALITY_TARGETS` can run bounded target lists
without using the full sweep.

Environment:

- GPU: Tesla M40 24GB, sm_52
- Features: `cuda`
- Model: `/mnt/array-fastest/home/guyep/.cache/m40-llm/models/Llama-3.2-1B-Instruct-f16.gguf`
- 64-token command: `source scripts/dev-env.sh && M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 M40LLM_LONG_CONTEXT_RETRIEVAL_MODEL=/mnt/array-fastest/home/guyep/.cache/m40-llm/models/Llama-3.2-1B-Instruct-f16.gguf M40LLM_KV_QUALITY_TARGETS=64 M40LLM_KV_QUALITY_REPORT=/tmp/m40llm_kv_quality_quiet_64.jsonl cargo test --features cuda --test kv_compression_long_context -- --nocapture --test-threads=1`
- 512-token command: `source scripts/dev-env.sh && M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 M40LLM_LONG_CONTEXT_RETRIEVAL_MODEL=/mnt/array-fastest/home/guyep/.cache/m40-llm/models/Llama-3.2-1B-Instruct-f16.gguf M40LLM_KV_QUALITY_TARGETS=512 M40LLM_KV_QUALITY_REPORT=/tmp/m40llm_kv_quality_quiet_512.jsonl cargo test --features cuda --test kv_compression_long_context -- --nocapture --test-threads=1`

Results:

| Target | Prompt tokens | Generated tokens | Needle | Mode | Status | Prefill | Decode | Total | Output |
| ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 64 | 58 | 10 | old | off | pass | 3.349 s | 0.530 s | 5.900 s | `ZXQ-NEEDLE-41729` |
| 64 | 58 | 10 | old | block-select-exact | pass | 2.861 s | 0.546 s | 5.417 s | `ZXQ-NEEDLE-41729` |
| 64 | 58 | 10 | old | block-summary | pass | 2.920 s | 0.568 s | 5.509 s | `ZXQ-NEEDLE-41729` |
| 64 | 58 | 10 | old | block-select-lossy | pass | 2.928 s | 0.570 s | 5.531 s | `ZXQ-NEEDLE-41729` |
| 64 | 58 | 10 | recent | off | pass | 2.789 s | 0.531 s | 5.363 s | `ZXQ-NEEDLE-41729` |
| 64 | 58 | 10 | recent | block-select-exact | pass | 2.859 s | 0.546 s | 5.430 s | `ZXQ-NEEDLE-41729` |
| 64 | 58 | 10 | recent | block-summary | pass | 2.919 s | 0.568 s | 5.522 s | `ZXQ-NEEDLE-41729` |
| 64 | 58 | 10 | recent | block-select-lossy | pass | 2.920 s | 0.568 s | 5.529 s | `ZXQ-NEEDLE-41729` |
| 512 | 479 | 10 | old | off | pass | 56.059 s | 1.748 s | 59.825 s | `ZXQ-NEEDLE-41729` |
| 512 | 479 | 10 | old | block-select-exact | pass | 57.509 s | 1.809 s | 61.355 s | `ZXQ-NEEDLE-41729` |
| 512 | 479 | 10 | old | block-summary | pass | 62.683 s | 2.008 s | 66.728 s | `ZXQ-NEEDLE-41729` |
| 512 | 479 | 10 | old | block-select-lossy | pass | 62.613 s | 2.012 s | 66.656 s | `ZXQ-NEEDLE-41729` |
| 512 | 494 | 10 | recent | off | pass | 58.549 s | 1.794 s | 62.356 s | `ZXQ-NEEDLE-41729` |
| 512 | 494 | 10 | recent | block-select-exact | pass | 60.590 s | 1.840 s | 64.423 s | `ZXQ-NEEDLE-41729` |
| 512 | 494 | 10 | recent | block-summary | pass | 65.975 s | 2.052 s | 70.058 s | `ZXQ-NEEDLE-41729` |
| 512 | 494 | 10 | recent | block-select-lossy | pass | 65.971 s | 2.056 s | 70.023 s | `ZXQ-NEEDLE-41729` |

Interpretation:

- Default output is now bounded to candidate/model lines, one line per mode run,
  and the compact summary table. The previous per-token IDs and per-token memory
  diagnostics are off by default.
- The JSONL report now records actual post-prompt-format prompt token counts,
  generated token counts, pass/fail status, full output, and timing breakdowns.
- `attention_compression_elapsed_ms` is currently `null`; the runtime does not
  expose a per-case attention/compression counter yet.
- Prompt prefill dominates runtime at 512 tokens. The existing packed prefill
  path is server/scheduler-oriented, so this checkpoint does not risk wiring it
  into `generate_text`; the next performance task is a safe bounded/batched
  prefill entrypoint for CLI quality runs.

## 2026-05-12: CLI/Test Packed-Prefix Prefill

This checkpoint adds an opt-in CLI/test packed-prefix prefill path via
`M40LLM_PREFILL_CHUNK_SIZE`. The path uses the existing packed varlen prefill
primitive for the prompt prefix, then runs the final prompt token through the
normal one-token decode path to preserve final-logits behavior. It is currently
enabled only for dense `off`; KV-compressed modes fall back to sequential
prefill because packed prefill is not yet equivalent for those cache modes.

Validation:

- `cargo test --features cuda --test forward_with_layer_smoke -- --nocapture --test-threads=1`
  passed, including dense packed-prefix-vs-sequential final-logit parity.
- `cargo test --features cuda --test attention_parity_cuda_grid -- --nocapture --test-threads=1`
  passed.
- `cargo fmt --all -- --check` and `cargo clippy --features cuda,server --all-targets -- -D warnings`
  passed.

Quality harness results:

| Target | Chunk | Needle | Mode | Status | Prefill mode | Prefill | Prefill tok/s | Decode | Decode tok/s | Total |
| ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | unset | old | off | pass | sequential | 3.342 s | 17.35 | 0.529 s | 18.90 | 5.896 s |
| 64 | 64 | old | off | pass | packed-prefix | 0.872 s | 66.51 | 0.530 s | 18.87 | 3.330 s |
| 64 | 64 | old | block-select-exact | pass | sequential-kv-compressed | 2.850 s | 20.35 | 0.545 s | 18.35 | 5.324 s |
| 64 | 64 | old | block-summary | pass | sequential-kv-compressed | 2.913 s | 19.91 | 0.568 s | 17.61 | 5.424 s |
| 64 | 64 | old | block-select-lossy | pass | sequential-kv-compressed | 2.912 s | 19.92 | 0.567 s | 17.64 | 5.412 s |
| 64 | unset | recent | off | pass | sequential | 2.800 s | 20.71 | 0.531 s | 18.83 | 5.350 s |
| 64 | 64 | recent | off | pass | packed-prefix | 0.319 s | 181.82 | 0.529 s | 18.90 | 2.785 s |
| 64 | 64 | recent | block-select-exact | pass | sequential-kv-compressed | 2.852 s | 20.34 | 0.545 s | 18.35 | 5.338 s |
| 64 | 64 | recent | block-summary | pass | sequential-kv-compressed | 2.917 s | 19.88 | 0.568 s | 17.61 | 5.420 s |
| 64 | 64 | recent | block-select-lossy | pass | sequential-kv-compressed | 2.922 s | 19.85 | 0.568 s | 17.61 | 5.419 s |
| 512 | 512 | old | off | pass | packed-prefix | 9.958 s | 48.10 | 1.753 s | 5.70 | 13.731 s |
| 512 | 512 | old | block-select-exact | pass | sequential-kv-compressed | 57.893 s | 8.27 | 1.827 s | 5.47 | 61.735 s |
| 512 | 512 | old | block-summary | pass | sequential-kv-compressed | 62.719 s | 7.64 | 2.009 s | 4.98 | 66.740 s |
| 512 | 512 | old | block-select-lossy | pass | sequential-kv-compressed | 62.714 s | 7.64 | 2.008 s | 4.98 | 66.732 s |
| 512 | 512 | recent | off | pass | packed-prefix | 9.989 s | 49.45 | 1.793 s | 5.58 | 13.777 s |
| 512 | 512 | recent | block-select-exact | pass | sequential-kv-compressed | 61.030 s | 8.09 | 1.876 s | 5.33 | 64.903 s |
| 512 | 512 | recent | block-summary | pass | sequential-kv-compressed | 66.367 s | 7.44 | 2.056 s | 4.86 | 70.506 s |
| 512 | 512 | recent | block-select-lossy | pass | sequential-kv-compressed | 66.047 s | 7.48 | 2.061 s | 4.85 | 70.085 s |

Interpretation:

- Dense 512-token prefill improves materially: old-needle dense `off` drops from
  56.059 s sequential to 9.958 s packed-prefix, and recent-needle dense `off`
  drops from 58.549 s to 9.989 s.
- The full 512 matrix still takes 437.25 s because the KV-compressed modes are
  intentionally sequential. Earlier experiments showed packed prefill can change
  retrieval output for compressed/exact sparse modes, so those modes are not
  accelerated until a compressed-aware packed/chunked prefill design is added.
- A broader 1K/2K/4K quality sweep was not run in this checkpoint because the
  compressed rows still dominate runtime.

## 2026-05-13: Compressed-Aware Chunked Prefill Parity

This checkpoint adds `M40LLM_KV_COMPRESSED_PREFILL_CHUNK_SIZE` as an explicit
CLI/test opt-in for KV-compressed modes. Unlike the dense packed-prefix path, it
does not run packed varlen prefill over compressed modes. It processes prompt
tokens in bounded chunks while preserving sequential one-token forward order,
absolute positions, RoPE positions, recent-ring evictions, block counts, and
summary accumulators. Prefix-token logits are skipped; the final prompt token
still uses the normal logits path.

Validation:

- `cargo test --features cuda --test forward_with_layer_smoke -- --nocapture --test-threads=1`
  passed on M40.
- The new parity coverage compares sequential compressed prefill against
  chunked compressed prefill for `block-summary` and `block-select-lossy`.
- The tests compare final logits within the existing tolerance and compare CUDA
  compressed-KV debug snapshots, including sequence length, recent-ring buffers,
  block counts, summary accumulators, and finalized summaries.

Quality harness results:

| Target | Compressed chunk | Needle | Mode | Status | Prefill mode | Prefill | Prefill tok/s | Decode | Decode tok/s | Total |
| ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 32 | old | block-select-exact | pass | chunked-kv-compressed | 2.065 s | 28.09 | 0.549 s | 18.21 | 4.655 s |
| 64 | 32 | old | block-summary | pass | chunked-kv-compressed | 2.125 s | 27.29 | 0.572 s | 17.48 | 4.748 s |
| 64 | 32 | old | block-select-lossy | pass | chunked-kv-compressed | 2.133 s | 27.19 | 0.571 s | 17.51 | 4.767 s |
| 64 | 32 | recent | block-select-exact | pass | chunked-kv-compressed | 2.089 s | 27.76 | 0.583 s | 17.15 | 5.546 s |
| 64 | 32 | recent | block-summary | pass | chunked-kv-compressed | 2.150 s | 26.98 | 0.578 s | 17.30 | 5.282 s |
| 64 | 32 | recent | block-select-lossy | pass | chunked-kv-compressed | 2.126 s | 27.28 | 0.570 s | 17.54 | 4.741 s |
| 512 | 64 | old | block-select-exact | pass | chunked-kv-compressed | 51.065 s | 9.38 | 1.840 s | 5.43 | 55.806 s |
| 512 | 64 | old | block-summary | pass | chunked-kv-compressed | 56.109 s | 8.54 | 2.040 s | 4.90 | 61.048 s |
| 512 | 64 | old | block-select-lossy | pass | chunked-kv-compressed | 56.159 s | 8.53 | 2.043 s | 4.89 | 61.095 s |
| 512 | 64 | recent | block-select-exact | pass | chunked-kv-compressed | 53.852 s | 9.17 | 1.894 s | 5.28 | 58.271 s |
| 512 | 64 | recent | block-summary | pass | chunked-kv-compressed | 59.369 s | 8.32 | 2.090 s | 4.78 | 64.396 s |
| 512 | 64 | recent | block-select-lossy | pass | chunked-kv-compressed | 59.262 s | 8.34 | 2.093 s | 4.78 | 64.271 s |

Interpretation:

- 64-token compressed prefill improves from roughly 2.85-2.92 s to roughly
  2.07-2.15 s while preserving retrieval output.
- 512-token compressed prefill improves only modestly compared with the prior
  sequential rows: exact sparse rows drop by roughly 6-7 s, and lossy rows drop
  by roughly 6-7 s but remain around 56-59 s of prefill.
- This confirms the chunked path is behavior-preserving, but not sufficient for
  routine 1K/2K/4K sweeps. A true compressed-aware packed prefill or scheduler
  path is still needed for larger contexts.

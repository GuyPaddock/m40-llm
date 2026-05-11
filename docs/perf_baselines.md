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

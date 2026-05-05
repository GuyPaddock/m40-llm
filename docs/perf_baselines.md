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

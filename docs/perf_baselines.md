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

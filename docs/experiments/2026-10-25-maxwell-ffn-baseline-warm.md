# Warm Complete FFN Baseline

## Result

M02 is complete. Clean release `29dee55df0f757165cdf41d5dad625f71ac3ab23`
measures all six real Qwen2.5-3B Q8_0 FFN cases after bounded active warmup.
Every primary boundary sample reports P0, 1113 MHz SM and 3004 MHz memory.
All numerical checks, single-row frozen SwiGLU bits, guards, immutable inputs,
enqueue accounting and allocation cleanup pass.

One-row complete FFN is about 0.457 ms/layer, with gate/up/SwiGLU about
0.327 ms and prepared down about 0.131 ms. Four-row gate/up/SwiGLU takes
about 4.15 ms and complete FFN 4.67-4.72 ms. The existing multi-row path is
not automatically faster per row. These are projection baselines, not a
candidate improvement, token throughput or a speculative verification result.

## Five-Trial GPU Medians

Milliseconds per full operation. Each trial repeats its stage ten times,
dividing measured GPU elapsed time by ten. Rows4 is one genuine four-row
matrix containing distinct captured inputs, not four repeated inputs.

| Original Layer | Rows | Gate/Up/SwiGLU | Isolated Prepared Down | Complete FFN | Complete MAD/Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.327930 | 0.131539 | 0.456864 | 0.116% |
| 0 | 4 | 4.159037 | 0.551885 | 4.720525 | 0.418% |
| 17 | 1 | 0.327443 | 0.131523 | 0.457514 | 0.136% |
| 17 | 4 | 4.146896 | 0.554304 | 4.709558 | 0.033% |
| 35 | 1 | 0.326698 | 0.131312 | 0.456579 | 0.023% |
| 35 | 4 | 4.141533 | 0.537862 | 4.671059 | 0.488% |

The [initial run](2026-10-25-maxwell-ffn-baseline.md) remains preserved,
including its 949-to-1113 MHz ramp and early outliers. Do not pool it with this
confirmation or present the warmer clock as a kernel optimization. Candidate
screening must pair each candidate with a fresh native control, not use a
historical denominator. Stage sums need not equal complete latency exactly.

## Protocol and Numerical Evidence

- Original GGUF SHA-256
  `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
  Frozen calibration inputs from original layers 0, 17 and 35, as documented
  in [the input checkpoint](2026-10-25-maxwell-ffn-inputs.md).
- Native fused one-row gate/up/SwiGLU with FP32 activations, four-column/
  256-thread prepared down. Genuine rows4 uses the existing multi-row Q8
  projections, decode SwiGLU and four-column/128-thread prepared down.
  No projection family, arithmetic, model weight or runtime default changes.
- After CPU validation, each case performs at least 1000 ms of measured full
  FFN warmup, in 32-iteration batches capped at 256 batches. One-row cases
  need 67/69/69 batches; four-row cases need seven each. Actual warmup is
  1005.7-1062.6 GPU ms. Warmup records remain separate from primary trials.
- Five trials per case, alternating stage order. CUDA events cover each
  causal producer/consumer path, with host enqueue-plus-completion time and
  operation counters reported separately. All 90 primary intervals are
  non-overlapping; no concurrent tests, builds, inference or MemPalace work.
- Per-iteration launches are 1/1/2 for one-row gate/down/complete and 3/1/4
  for rows4; the latter has 2/1/3 explicit stream waits. Timed enqueue adds
  no allocation/free, H2D/D2H copy, cuBLAS call or stream synchronization.
  A single event completion wait follows the repeated enqueue interval.
- Scalar CPU Q8 dequantization and FP64 references compare every gate/up,
  SwiGLU and down output within `1e-3 + 2e-5 * abs(reference)`. The largest
  CPU-reference difference is 0.000295679. Single-row SwiGLU is bit-identical
  to frozen captures at all three layers. Rows4 has the existing bounded
  arithmetic drift, at most 0.000328064 versus sequential captured SwiGLU.

GPU boundary temperature is 34-38 C. Only this benchmark PID appears during
the primary trials, and compute-process queries are empty before/after the
process. Boundary samples are not per-kernel frequency or DRAM counters.

## Memory and Limits

Original weights occupy 3,279,519,744 requested device bytes; immutable split
down/vocabulary preparation adds 1,192,935,424. Vocabulary remains resident
but is not executed. Guarded reusable scratch occupies 201,152 bytes for one
row and 803,264 for four. Tracked allocations return zero to zero. The CUDA
context's transient post-release residency is not a leaked model allocation.

No KV, attention, embedding, RMSNorm or vocabulary work is timed here. Setup,
model hashing, preparation, CPU references, allocation/upload, readback and
warmup are excluded. This is not a peak-VRAM or E2E result. FP32 control has
no activation quantization operation, so its quantization cost is zero.

## Reproduction and Artifacts

Use [the complete-FFN command](../ffn_benchmark_inputs.md#complete-ffn-baseline)
with the exact frozen capture environment and new output paths. This run uses
`target/research-fixtures/m02-ffn-baseline-warm.jsonl`. Rust release 1.95.0,
NVCC 12.8.93, g++ 14.3.1; CUDA `-O3`, sm_52 SASS and compute_52 PTX, no fast
math. Executable SHA-256 is
`b219e5aff2c51e40c8b44cdfe60af11128942e76e0a2b1eadab595bbf9b56dd9`.

- [Raw JSONL](2026-10-25-maxwell-ffn-baseline-warm.jsonl): manifest, six
  validations, six separate warmups, 30 timing records and completion.
- [Raw stderr](2026-10-25-maxwell-ffn-baseline-warm.stderr.log).
- [Summary and audit](2026-10-25-maxwell-ffn-baseline-warm-summary.json):
  individual values, ranges, MADs, host times, warmups and clock samples.

All audit checks pass, including the warmup bound, primary clock observations,
numerical/capture parity, absence of timed enqueue churn, interval isolation and
cleanup. Benchmark CPU/M40 tests, cuBLAS-on/off pipeline checks, warning-as-error
non-CUDA tests, CUDA/server Clippy and formatting pass. GPU-less checks explicitly
ignore hardware assertions; explicitly requested hardware without a device fails.

Next is M03's bounded FP32 load/reduction ablation, with independent M04 Q8_1
numerical work and M07 prefill SGEMM after this baseline. Keep down/vocabulary
fixed and require fresh paired complete-FFN improvement before runtime
evaluation. Both original exact-output E2E goals remain unmet.

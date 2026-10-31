# Maxwell FP32 FFN Screen

## Decision

M03's fixed real-input screen is complete. Packed loads plus warp reduction
pass the one-row complete-FFN gate at original layers 0, 17 and 35.
The 256-thread candidate improves FFN throughput by 7.75-8.27% while preserving
native SwiGLU and completed down output bits on these real inputs. The sole
128-thread candidate improves it by 12.18-12.63%, but changes arithmetic
association and has small nonzero output drift.

Neither is a runtime recommendation yet. No generation policy, project KV
default or prepared down/vocabulary path changed. Whole-model final-logit and
512-token parity plus controlled HTTP gains remain required. M04's independent
Q8_1 numerical experiment and the M05 combined screen still remain. Do not
substitute a microbenchmark result for either original E2E objective.

Packed/shared256 gains only 1.40-1.57% and fails the 5% screen.
Scalar/warp256 is 10.38-11.38% slower than native; scalar/shared256 is
25.56-25.77% slower. Preserve those losing implementations and raw measurements.
No additional geometry was tried to rescue them.

## One-Row Complete FFN

Milliseconds are five-trial GPU medians, each repeating the operation ten
times. Throughput gain is native latency / candidate latency minus one, not
percent latency reduction. The 95% intervals resample contemporaneous pairs
10,000 times with seed zero. Every candidate has a fresh native denominator.

| Layer | Variant | Native ms | Candidate ms | Throughput Gain | Paired 95% Gain Interval | Micro Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | scalar-shared256 | 0.456474 | 0.614960 | -25.77% | -25.90 to -25.69% | Fail |
| 0 | packed-shared256 | 0.456675 | 0.450384 | 1.40% | 1.21 to 1.53% | Fail |
| 0 | scalar-warp256 | 0.457002 | 0.509933 | -10.38% | -11.80 to -9.46% | Fail |
| 0 | packed-warp256 | 0.456627 | 0.423789 | 7.75% | 7.34 to 9.10% | Pass |
| 0 | packed-warp128 | 0.457018 | 0.407398 | 12.18% | 11.89 to 12.64% | Pass |
| 17 | scalar-shared256 | 0.457296 | 0.614480 | -25.58% | -25.67 to -25.49% | Fail |
| 17 | packed-shared256 | 0.457363 | 0.451050 | 1.40% | 1.27 to 1.70% | Fail |
| 17 | scalar-warp256 | 0.457072 | 0.515766 | -11.38% | -12.67 to -10.75% | Fail |
| 17 | packed-warp256 | 0.457331 | 0.423981 | 7.87% | 6.86 to 8.50% | Pass |
| 17 | packed-warp128 | 0.457693 | 0.406755 | 12.52% | 11.37 to 13.32% | Pass |
| 35 | scalar-shared256 | 0.457616 | 0.614707 | -25.56% | -25.66 to -25.53% | Fail |
| 35 | packed-shared256 | 0.457325 | 0.450246 | 1.57% | 1.48 to 1.68% | Fail |
| 35 | scalar-warp256 | 0.457584 | 0.513155 | -10.83% | -11.20 to -10.16% | Fail |
| 35 | packed-warp256 | 0.457523 | 0.422570 | 8.27% | 7.55 to 8.80% | Pass |
| 35 | packed-warp128 | 0.457453 | 0.406163 | 12.63% | 12.18 to 12.80% | Pass |

Packed/warp256 gate/up/SwiGLU medians are 0.2915-0.2930 ms versus native
0.3265-0.3276 ms. Packed/warp128 is 0.2749-0.2764 ms versus 0.3267-0.3274 ms.
Isolated prepared down stays about 0.131 ms. Stage sums need not equal
complete timing because the isolated stage has different cache history.

The new scalar/shared kernel is a fixed-size implementation, not a binary copy
of native. Its reduction unrolls in release, while native retains a loop.
Both preserve native work assignment and FP32 association at 256 threads,
but equal math does not establish equal performance. The four template
variants form the load/reduction ablation; the original native kernel remains
the runtime comparison control. Do not attribute these results to measured
DRAM bandwidth or occupancy. Neither was measured.

## Genuine Four-Row Diagnostic

Each matrix contains four distinct frozen inputs, not repetitions of one row.
These candidates fuse gate/up/SwiGLU for four rows, while native uses two
multi-row projection launches and a separate SwiGLU. The gains include that
algorithm/fusion change and must not be described as pure load/reduction gains
or autoregressive decode throughput.

| Layer | Variant | Native Complete ms | Candidate Complete ms | Ratio |
| --- | --- | ---: | ---: | ---: |
| 0 | scalar-shared256 | 4.695747 | 2.527782 | 1.858x |
| 0 | packed-shared256 | 4.644695 | 1.834595 | 2.532x |
| 0 | scalar-warp256 | 4.699830 | 2.123702 | 2.213x |
| 0 | packed-warp256 | 4.708121 | 1.798163 | 2.618x |
| 0 | packed-warp128 | 4.692381 | 1.728176 | 2.715x |
| 17 | scalar-shared256 | 4.704682 | 2.526163 | 1.862x |
| 17 | packed-shared256 | 4.644873 | 1.835686 | 2.530x |
| 17 | scalar-warp256 | 4.637143 | 2.049965 | 2.262x |
| 17 | packed-warp256 | 4.705286 | 1.788397 | 2.631x |
| 17 | packed-warp128 | 4.716202 | 1.745213 | 2.702x |
| 35 | scalar-shared256 | 4.706074 | 2.537843 | 1.854x |
| 35 | packed-shared256 | 4.710080 | 1.914701 | 2.460x |
| 35 | scalar-warp256 | 4.771232 | 2.183942 | 2.185x |
| 35 | packed-warp256 | 4.774614 | 1.865875 | 2.559x |
| 35 | packed-warp128 | 4.757594 | 1.773021 | 2.683x |

Prepared four-row down remains the qualified four-column/128-thread path.
This is not a batched verifier, packed-prefix integration, or a server result.

## Protocol and Correctness

Clean release source `bedea89af2bd54a8039aa1904f1e81ad103e2570` follows
kernel commit `38e14a0`. Executable SHA-256:
`f7b6cae4cca6f817b95480f25863886ac4a8ae3c5faeed4ebbd69c36ef854f18`.
Rust 1.95.0, NVCC 12.8.93, g++ 14.3.1; CUDA -O3, sm_52 SASS plus compute_52
PTX, no fast math. Reproduce with the
[bounded FFN screen command](../ffn_benchmark_inputs.md#bounded-fp32-ffn-screen),
exact capture environment and a new output path.

- Original Qwen2.5-3B Q8_0 model SHA-256:
  `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
  Frozen calibration inputs and source tensor hashes come from the
  [M02 capture](2026-10-25-maxwell-ffn-inputs.md).
- Three original layers, rows 1/4, five fixed variants, five alternating pairs
  per case: 150 pairs and 900 non-overlapping primary stage intervals.
  Arm and stage order alternate. All expensive CPU validation precedes warmup.
- Each candidate gets bounded 1000 ms native and 100 ms candidate GPU warmup.
  Warmup is separate from primary trials and changes no clock setting.
  All 300 primary boundary samples are P0, SM 1113 MHz, memory 3004 MHz.
  Temperatures are 35-49 C. Only benchmark PID 632412 appears; compute-process
  queries are empty before and after the run.
- Every timed enqueue has zero device allocation/free, H2D/D2H bytes/copies,
  cuBLAS calls and stream synchronizations. One event completion wait follows
  each repeated stage. Launch/wait counts match the actual pipeline.
  Native M4 gate/down/complete uses 3/1/4 launches and 2/1/3 waits; fused M4
  uses 1/1/2 launches and 1/1/2 waits. M1 uses 1/1/2 launches with zero waits.
- All 36 native/candidate validation rows pass full-output CPU-reference
  tolerance, immutable-input checks and guards. Gate/up oracle projections
  are checked separately; candidate fused SwiGLU and completed down outputs
  are compared to scalar Q8 dequantization and FP64 CPU references.
  All initial 256-thread variants match fused row-control bits, and one-row
  results also match frozen native hidden and completed output bits.
- Packed/warp128 one-row hidden drift versus native is at most 0.000015259,
  completed output drift 0.000030518. This is not whole-model logit or token
  parity. All CPU-reference errors remain below the existing
  `1e-3 + 2e-5 * abs(reference)` bound.
- Release sm_52 inspection of this executable reports 32 registers and zero
  stack/local bytes for all five variants and native. Dynamic shared storage
  is 2048 bytes at 256 threads and 1024 at 128. Packed variants issue aligned
  halfword payload loads; warp variants have one barrier and ten shuffle sites.
  Static barrier-site counts are not dynamic counts for native's loop.

## Memory and Scope

Original weights remain 3,279,519,744 requested device bytes, with
1,192,935,424 bytes of qualified prepared down/vocabulary. Total requested
weight payload is 4,472,455,168 bytes. Vocabulary is resident but not executed.
Guarded scratch is 201,152 bytes at M1 and 803,264 at M4, reused across arms.
No KV is allocated. Tracked allocations return zero to zero. CUDA context and
driver residency are separate from model payload accounting.

This measures FFN primitives only, not embedding, norms, attention, KV, logits,
or HTTP. Model loading, hashing, preparation, validation, allocation/upload,
readback and warmup are outside primary timing. FP32 activations require no
quantizer; its measured cost is zero because that operation does not exist.
No tests, builds, MemPalace work or other GPU jobs overlap this timed run.

## Artifacts and Next Step

- [Raw JSONL](2026-10-25-maxwell-fp32-ffn-screen.jsonl), SHA-256
  `d8240c5d13b444b9165cb593c47902ac74ce8e666924f15ffe7efd57267460e3`.
- [Raw stderr](2026-10-25-maxwell-fp32-ffn-screen.stderr.log).
- [Summary and audit](2026-10-25-maxwell-fp32-ffn-summary.json), SHA-256
  `e79803db87b44fdcef6056b6c3fe252971e044d35b66de49944c9ce28352404f`.

Every recorded audit check passes. Kernel parity tests and all eight benchmark
tests pass in release on M40; affected CUDA parity/forward/ownership tests and
the benchmark pass with cuBLAS disabled. CUDA/server Clippy, formatting, full
non-CUDA warning-as-error tests, and NVCC-disabled warning-as-error all-target
compilation pass. GPU-less checks are not hardware parity evidence.

Next run the independent bounded M04 Q8_1 experiment, then compare all complete
FFN evidence at M05 before conditional M06 runtime integration. M07 prefill
SGEMM is independent of decode success. Keep packed/warp256 as the bit-preserving
FP32 candidate and 128 as a numerical candidate pending full-model tests.
Both original exact-output E2E targets remain unmet.

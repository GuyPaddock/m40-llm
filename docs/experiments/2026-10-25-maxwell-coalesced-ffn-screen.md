# Adjacent-Pair FP32 FFN Screen

## Decision

Keep this candidate benchmark-only. Complete one-row FFN throughput improves
**1.84-1.98%** versus qualified packed/warp128, with positive paired confidence
intervals, but misses the fixed 5% micro gate. Four-row medians improve only
0.70-1.70% with intervals crossing zero. No runtime integration, whole-model
quality run or HTTP promotion is justified by this screen. Do not add geometry
configurations to rescue it.

Implementation `aa767eb` and these measurements preserve the experiment.
The last qualified runtime remains typed packed FP32/128 plus dequant SGEMM,
prepared down/vocabulary and explicit dense reference KV. Its previous
41.331 HTTP tok/s measurement is unchanged, not remeasured here. Both original
targets remain unmet: 53.06 and 55.829 tok/s for 1.30x fresh Ollama. No runtime
default, backend selection or acceptance threshold changes.

## Complete FFN Results

Milliseconds are CUDA-event medians of five alternating pairs, with ten
repetitions per stage. Gain is control latency / candidate latency minus one.
Confidence intervals use 10,000 paired bootstrap resamples, seed zero.

| Layer | Rows | Packed128 Control ms | Adjacent-Pair ms | Gain | Paired 95% Gain Interval | Micro Gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 1 | 0.407008 | 0.399094 | +1.98% | +1.57 to +2.43% | Fail |
| 17 | 1 | 0.407088 | 0.399411 | +1.92% | +1.17 to +2.07% | Fail |
| 35 | 1 | 0.406464 | 0.399123 | +1.84% | +1.26 to +2.03% | Fail |
| 0 | 4 | 1.778128 | 1.765712 | +0.70% | -2.85 to +2.97% | Fail |
| 17 | 4 | 1.739610 | 1.715594 | +1.40% | -0.46 to +4.59% | Fail |
| 35 | 4 | 1.749955 | 1.720627 | +1.70% | -3.30 to +4.26% | Fail |

One-row gate/up/SwiGLU falls from 0.27726-0.27787 ms to
0.26793-0.26818 ms. Isolated prepared down remains about 0.131-0.132 ms.
The complete-FFN gain is not an E2E result. Its one-row confidence bounds
remain well below 5%; this is not a borderline screen requiring more trials.
Four-row timing is noisier and is not autoregressive decode throughput.

## Request Pattern and Limits

Only lane ownership and loads change. One lane handles two adjacent values,
loads one aligned `float2` activation and one halfword payload from each Q8
matrix, and keeps the original per-weight FP32 scaling. The kernel retains
128 threads, one shared publication, the same final-warp reduction and fused
SwiGLU. Weights and activation precision are unchanged. The different
summation ownership still requires numerical and eventual token checks.

The manifest records a CPU address model of 32-byte sectors touched per warp
memory instruction, summed for one output column and one input row at K=2048:

| Source-Level Footprint | Packed128 | Adjacent Pairs |
| --- | ---: | ---: |
| FP32 input sector requests | 2048 | 256 |
| Gate/up payload sector requests, weight phase 0 | 576 | 192 |
| Gate/up payload sector requests, weight phase 2 | 584 | 192 |
| Gate/up scale sector requests | 128 | 128 |

These are **predictions, not measured cache misses, physical transactions or
DRAM bytes**. Repeated requests can hit cached data. The model assumes the
benchmark's 32-byte-aligned input and considers both legal halfword phases.
Tests also cover every even weight-base phase modulo 32. Generated sm_52 code
confirms 64-bit activation loads in the candidate versus scalar FP32 loads in
control. Both use 32 registers, no stack/local storage, 1024 bytes of dynamic
shared storage, one barrier site and ten shuffle sites.

Lower sector footprint does not imply proportionally less instruction work.
At source level, the two-value mapping issues 64 gate/up scale-load groups
per output column instead of 16, because it processes four times as many
smaller tasks. Its input groups fall from 64 to 32; payload groups remain 64.
This analytical accounting does not include compiler control-flow transforms.
The compiler unrolls candidate work and emits a prologue; whole-function static
instruction-site counts are not dynamic counts. Neither these estimates nor
the 2% gain prove a bandwidth, occupancy or specific latency bottleneck.

[Legacy CUPTI counters](2026-10-25-maxwell-counter-capability.md) remain
unsupported on this stack even under an administrator retry. Enumerated
metric names are not measurements. No counter activity overlaps this run.

## Protocol and Validation

- Clean source revision `aa767eb363739f822424f2d6e1d1d304990aa38f`, executable
  SHA-256 `fbbbf872c9bcfa04563e296877517286f089d4271cb49c2d18bc002edac58a2f`.
  The manifest lists only its newly created report as untracked. Rust 1.95.0,
  NVCC 12.8.93, CUDA `-O3`, sm_52 SASS plus compute_52 PTX, no fast math,
  cuBLAS enabled. Use the [bounded follow-up command](../ffn_benchmark_inputs.md#bounded-adjacent-pair-load-follow-up)
  with the exact frozen environment and a new output path.
- Original Qwen2.5-3B Q8_0 model SHA-256
  `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
  Frozen real calibration inputs and original layers 0/17/35 are unchanged.
  Rows 1/4 are genuine distinct inputs. The old native path is a numerical
  reference only; timing control is qualified packed/warp128, never the
  failed scale-factor candidate.
- All eighteen native/control/candidate numerical validations pass. Maximum
  candidate CPU-reference absolute errors are 1.64e-6 gate, 1.55e-6 up,
  2.43e-5 SwiGLU and 5.67e-5 completed down, within the unchanged
  `1e-3 + 2e-5 * abs(reference)` bound. Candidate intermediate capture and
  ordinary SwiGLU bits match. Maximum native differences are 0.00031281
  hidden and 0.00042725 completed output. These are not model-logit or
  generated-token parity results.
- Thirty pairs contain 180 non-overlapping primary intervals. Arm and stage
  order alternate. Bounded GPU warmup is 1000 ms control / 100 ms candidate.
  All sixty primary boundaries report P0, SM 1113 MHz, memory 3004 MHz and
  36-42 C. Only benchmark PID 1149420 appears. No tests, builds, Ollama,
  MemPalace embeddings or other inference jobs overlap timing. Process exit
  is zero and final GPU compute-process queries are empty.
- Timed enqueue has no allocation/free, host/device transfer, cuBLAS call or
  stream synchronization. Per ten repetitions, launches are 10/10/20 for
  gate-up/isolated-down/complete; stream waits are 0/0/0 at one row and
  10/10/20 at four rows. One event completion wait follows each repeated
  stage. Required cross-stream dependencies remain in complete timing.
- Requested device payload is unchanged: original weights 3,279,519,744
  bytes plus prepared down/vocabulary 1,192,935,424, total 4,472,455,168.
  There is no KV or quantized activation buffer. Both arms reuse guarded
  scratch of 201,152 / 803,264 bytes at rows 1/4. Tracked payload returns
  zero to zero; driver context residency is accounted separately.
- Release CUDA primitive 2/2 and benchmark 11/11 tests pass with cuBLAS on
  and off. Enabled affected attention/forward/projection/MLP/runtime tests
  pass 46/46. Warning-as-error non-CUDA tests pass 208/208. CUDA/server
  all-target Clippy and formatting pass. Input alignment, signed extremes,
  zero scales, tails, guards, capture parity and scratch reuse are covered.

## Artifacts and Next Experiment

- [Raw JSONL](2026-10-25-maxwell-coalesced-ffn-screen.jsonl),
  [stderr](2026-10-25-maxwell-coalesced-ffn-screen.log.gz) and
  [audited summary](2026-10-25-maxwell-coalesced-ffn-audit.json).
- [Reusable audit](../../scripts/audit_ffn_followup.jq) accepts this and the
  preserved scale-factor report and rejects wrong control, timed allocation
  and fabricated median negative controls. It validates pairs, coverage,
  numerical records, clocks/processes, medians, gate computation and timing
  isolation. Invoke it with source `aa767eb363739f822424f2d6e1d1d304990aa38f`,
  candidate `packed-warp128-coalesced-pairs`, report path and PID `1149420`.
- [Resources](2026-10-25-maxwell-coalesced-ffn-resources.txt),
  [SASS](2026-10-25-maxwell-coalesced-ffn-sass.txt.gz),
  [static instruction sites](2026-10-25-maxwell-coalesced-ffn-instructions.txt),
  [CUDA tests](2026-10-25-maxwell-coalesced-ffn-cuda-tests.log.gz),
  [cuBLAS-disabled tests](2026-10-25-maxwell-coalesced-ffn-no-cublas.log.gz),
  [affected regressions](2026-10-25-maxwell-coalesced-ffn-regressions.log.gz),
  [non-CUDA tests](2026-10-25-maxwell-coalesced-ffn-noncuda.log.gz),
  [Clippy](2026-10-25-maxwell-coalesced-ffn-clippy.log.gz) and
  [artifact hashes](2026-10-25-maxwell-coalesced-ffn.sha256).

Stop native two-value mapping tuning. The next bounded hypothesis is one
four-value-per-lane fused gate/up kernel using the **existing** separate
scale/value execution-weight representation. Aligned 32-bit payload and
`float4` activation loads could reduce instruction groups as well as sector
footprint. This is not a demonstrated improvement. Prototype it only in the
existing FFN benchmark, retain prepared down/vocabulary, count additional
prepared storage/setup explicitly, and keep the same complete-FFN gate.
Do not add another format, geometry sweep, activation quantization, unrelated
scheduling change or runtime/default promotion before that evidence exists.

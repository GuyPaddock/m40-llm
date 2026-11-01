# Maxwell Q8_1 FFN Screen and M05 Decision

## Result

M04's quantization-inclusive real-input FFN screen is complete. Both fixed
Q8_1 geometries pass the one-row micro gate at original layers 0, 17 and 35.
128 threads improves complete-FFN throughput by 27.26-27.41%; 256 threads
improves it by 21.50-21.71%. Every complete candidate iteration includes a
fresh quantization and uses unchanged prepared FP32-activation down.

This is a separate numerical track. At the sampled layer-35 one-row input,
completed output differs from native FP32 by up to 0.538361. CPU checks using
the encoded activation values pass strict tolerance; that verifies the
implementation, not retention of original-model output or task quality.
There is no runtime recommendation, whole-model quality claim, E2E gain or
satisfaction of the original exact-output goal.

The result improves on the older scalar-int8 prototype's organization at
matched real FFN shapes. It does not explain by itself how many milliseconds
of the Ollama gap come from activation quantization, packing or reductions.

## Five-Pair GPU Medians

Milliseconds per full operation. Each trial repeats a stage ten times.
Gate/up/SwiGLU and complete totals already include fresh quantization; do not
add the separate quantizer measurement again. Gain is native/candidate minus
one. Confidence intervals resample contemporaneous pairs 10,000 times,
seed zero.

| Layer | Rows | Threads | Quantize ms | Gate/Up/SwiGLU ms | Native Complete ms | Candidate Complete ms | Throughput Gain | Paired 95% Gain Interval |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | threads128 | 0.003498 | 0.227568 | 0.456890 | 0.358586 | 27.41% | 27.07 to 27.67% |
| 0 | 1 | threads256 | 0.003456 | 0.244893 | 0.456883 | 0.375379 | 21.71% | 21.04 to 21.89% |
| 0 | 4 | threads128 | 0.024339 | 0.979024 | 4.748982 | 1.603805 | 196.11% | 190.74 to 203.04% |
| 0 | 4 | threads256 | 0.022410 | 1.036378 | 4.698179 | 1.617811 | 190.40% | 189.00 to 194.65% |
| 17 | 1 | threads128 | 0.003427 | 0.227728 | 0.456950 | 0.359078 | 27.26% | 26.33 to 27.37% |
| 17 | 1 | threads256 | 0.003434 | 0.245254 | 0.457670 | 0.376698 | 21.50% | 21.00 to 21.90% |
| 17 | 4 | threads128 | 0.024835 | 0.975686 | 4.715120 | 1.587923 | 196.94% | 196.24 to 214.42% |
| 17 | 4 | threads256 | 0.024598 | 1.044099 | 4.715754 | 1.627872 | 189.69% | 185.70 to 192.85% |
| 35 | 1 | threads128 | 0.003594 | 0.227674 | 0.457190 | 0.358826 | 27.41% | 26.75 to 27.64% |
| 35 | 1 | threads256 | 0.003389 | 0.244858 | 0.457050 | 0.375738 | 21.64% | 21.09 to 21.66% |
| 35 | 4 | threads128 | 0.024118 | 0.991571 | 4.764512 | 1.627200 | 192.80% | 191.72 to 206.29% |
| 35 | 4 | threads256 | 0.023853 | 1.053734 | 4.753821 | 1.667523 | 185.08% | 182.96 to 191.74% |

Single-row quantization costs 0.00339-0.00359 ms, versus
0.22757-0.22773 ms complete gate/up/SwiGLU at 128 threads and
0.24486-0.24525 ms at 256. Prepared one-row down remains about 0.131 ms.
The four-row standalone quantizer measurement includes its per-iteration
decode/prefill dependency, not a pure kernel-only cost. Stage sums need not
equal complete timing because isolated stages have different cache histories.

The four-row matrices contain distinct frozen inputs, not repeated single
rows. Native uses two multi-row projections and SwiGLU; Q8_1 uses a quantizer
and fused gate/up/SwiGLU. Four-row gains include fusion/dispatch changes and
must not be presented as autoregressive decode or full-model batched speed.

## Numerical Evidence

All 18 native/candidate validation rows pass guard, immutable-input and
strict CPU-reference checks. All candidate activation bytes match the pinned
independent CPU encoder. Every gate/up/SwiGLU value is checked against scalar
Q8_0 dequantization and FP64 accumulation using reconstructed Q8_1 inputs.
Completed down is checked independently on the actual candidate SwiGLU,
and isolated down still uses the unmodified frozen captured hidden vector.

Largest one-row errors versus quantized-input CPU references are
0.000001146 gate, 0.000002529 up, 0.000015191 SwiGLU and 0.000043145 down.
Across four-row inputs they are 0.000001271, 0.000003393, 0.000031739 and
0.000058404. All are inside the existing
`1e-3 + 2e-5 * abs(reference)` gate. That gate was not relaxed.

Original-FP32 differences are reported separately rather than required to be
zero. The table shows maximum absolute errors, not relative model-quality
scores:

| Layer | Threads | Activation Reconstruction | Hidden vs Native GPU | Completed Output vs Native GPU |
| --- | --- | ---: | ---: | ---: |
| 0 | threads128 | 0.005877 | 0.016356 | 0.003834 |
| 0 | threads256 | 0.005877 | 0.016356 | 0.003834 |
| 17 | threads128 | 0.024409 | 0.024580 | 0.021173 |
| 17 | threads256 | 0.024409 | 0.024579 | 0.021174 |
| 35 | threads128 | 0.145818 | 0.664528 | 0.538361 |
| 35 | threads256 | 0.145818 | 0.664536 | 0.538361 |

The four-row maximum completed-output difference is 0.563324. Native one-row
control hidden bits still match the frozen capture exactly. Q8_1 first logit
divergence, actual 512-token continuations, dense-valid task scores and
held-out loss/perplexity are **not measured yet**. They belong to conditional
runtime/evaluation work and block recommendation until verified.

## Protocol, Memory and Code Generation

Clean release source `6431295181cca67afca2325bd9490b53ccf54bef`, following
encoder `5e0598d` and consumer `cb5b12a`. Executable SHA-256:
`df2a8c35fb687e0602b882a7f411b6c13b43f36676a26892bed3224cac0b34bb`.
Rust 1.95.0, NVCC 12.8.93, g++ 14.3.1, CUDA -O3, sm_52 SASS and compute_52
PTX, no fast math.

- Original Qwen2.5-3B Q8_0 model SHA-256:
  `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
  Real layer 0/17/35 tensors and first 1/4 distinct activation rows come from
  the [frozen M02 capture](2026-10-25-maxwell-ffn-inputs.md).
- Exactly two geometries, five alternating pairs per case, 60 pairs and
  420 non-overlapping primary intervals. Arm/stage order alternates. All
  expensive CPU validation precedes bounded 1000 ms native / 100 ms candidate
  GPU warmup. No geometry expansion or search for favorable timing.
- All 120 primary boundary samples show P0, SM 1113 MHz, memory 3004 MHz,
  temperature 35-42 C and only benchmark PID 682859. GPU compute queries are
  empty before/after the process. No tests, builds, inference or MemPalace
  embedding overlap timing.
- Timed enqueue has zero allocation/free, H2D/D2H copies/bytes, cuBLAS calls
  and stream synchronizations. Event completion waits follow the repeated
  interval. Candidate quantizer/gate/down/complete has 1/2/1/3 launches.
  One-row has no explicit stream waits; four-row has 1/1/1/2. Native retains
  its prior pipeline and counts. No cached quantization across repetitions.
- Original uploaded weights occupy 3,279,519,744 requested device bytes and
  qualified prepared down/vocabulary adds 1,192,935,424, total 4,472,455,168.
  Vocabulary is resident but not executed. No KV is allocated.
- Reusable Q8_1 payload is 2304 bytes at M1 and 9216 at M4, plus 32 guard
  bytes. Total guarded pipeline scratch is 203,488 / 812,512 bytes. Both
  arms retain the same allocations; native does not use the quantizer.
  Tracked allocations return zero to zero. Driver/context RSS is separate.
- Release sm_52 inspection reports 17 quantizer registers and 32 for each
  consumer, with zero stack/local bytes. Dynamic consumer shared memory is
  1/2 KiB at 128/256 threads. The inspected 128-thread consumer has 64 XMAD,
  ten shuffle and one barrier instruction sites; no native DP4A/Tensor use.
  Static code-generation counts are not measured occupancy or DRAM traffic.

Loading, source hashing, preparation, allocation/upload, CPU references,
readback and warmup are excluded from primary timing. These are FFN-only
measurements, with no embedding/norm/attention/KV/logits/HTTP work.

## M05 Screening Decision

M03 and M04 now have the required bounded matched-shape complete-FFN screens.
The [FP32 evidence](2026-10-25-maxwell-fp32-ffn-screen.md) and this report
preserve every implementation and measured loser. This closes M05's
microbenchmark decision, not the plan's runtime or quality acceptance.

| Candidate | One-Row Micro Decision | Required Next Gate |
| --- | --- | --- |
| Scalar/shared256 | Reject: 25.56-25.77% slower than native | Preserve; no rescue geometry or runtime integration |
| Packed/shared256 | Reject: only 1.40-1.57% gain | Preserve; below 5% screen |
| Scalar/warp256 | Reject: 10.38-11.38% slower | Preserve; no runtime integration |
| Packed/warp256 FP32 | Advance: 7.75-8.27%, sampled native bits | Actual full-model logits/512 IDs, frozen dense-valid tasks, matched HTTP gain |
| Packed/warp128 FP32 | Advance conditionally: 12.18-12.63%, arithmetic drift | Test actual 512 IDs rather than assuming small drift preserves or breaks continuation |
| Q8_1 threads128 | Advance numerical research: 27.26-27.41% | Logit/continuation drift, task scores and held-out loss/perplexity; remains opt-in |
| Q8_1 threads256 | Advance numerical research: 21.50-21.71% | Same quality gates; no automatic promotion |

M03 and M04 are separate paired experiments, each against its own native
control. Their numerical tracks differ; absolute micro latencies are not a
contemporaneous candidate-to-candidate HTTP comparison. Both Q8_1 variants
advance the laboratory gate, not unchanged-output qualification.

Next M06 must capture choices immutably at model construction, retain reusable
scratch and dependencies, default to native, fail unsupported explicit
configurations clearly, and preserve qualified down/vocabulary. Benchmark-only
launch variants do not need generation CLI switches. Compare the qualifying
FP32 geometries at actual continuation equality before choosing the runtime
exact candidate. Q8_1 stays a separately labeled numerical experiment.

M07's bounded reusable prefill dequantization/SGEMM is independent. M08 must
evaluate decode-only, prefill-only and a qualifying combination before M09's
fresh controlled Ollama comparison. Neither 53.06 HTTP tok/s nor 1.30x fresh
Ollama has been achieved.

## Reproduction and Artifacts

Use [the quantization-inclusive command](../q8_activation.md#quantization-inclusive-benchmark)
with the frozen capture environment, `--ffn-q81`, release, rows 1/4 and new
output paths. No runtime generation flags select this path yet.

- [Raw JSONL](2026-10-25-maxwell-q81-ffn-screen.jsonl), SHA-256
  `52ba0df6beffe9a7aafb47949807199ea5c9c09ca7ffd3fe5474f4232f74c295`.
- [Raw stderr](2026-10-25-maxwell-q81-ffn-screen.stderr.log).
- [Summary and audit](2026-10-25-maxwell-q81-ffn-summary.json), SHA-256
  `d53ff1b991642f65bb10c41c343bcec60208c01badea832bd9142b6bfa56d82f`.
- [Pinned contract and MIT provenance](../q8_activation.md), including the
  complete upstream license shipped in `licenses/llama.cpp-MIT.txt`.

Every audit check passes. Release encoder/consumer/FP32 CUDA regressions and
all nine shared runner tests pass on M40. Affected attention/projection/MLP,
forward and allocation tests pass with cuBLAS disabled; the runner also passes
without cuBLAS. Full warning-as-error non-CUDA tests, NVCC-disabled CUDA/server
all-target checking, CUDA/server Clippy and formatting pass. Explicit hardware
tests remain ignored in GPU-less CI, not counted as hardware evidence.

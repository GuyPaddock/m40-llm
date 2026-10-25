# Bounded FP32 Scale-Factoring Screen

## Decision

Do not integrate the scale-factored kernel into generation. It passes the
existing numerical checks but fails the complete-FFN performance gate at all
six layer/row shapes. One-row throughput is 0.07-0.28% slower than the
qualified packed/warp128 control. Four-row gains are 1.59-2.96%, below 5%.
No geometry expansion or runtime quality/HTTP run is justified by this result.

Implementation `498699a` and these measurements preserve the failed experiment.
The last qualified exact runtime remains typed packed FP32/128 plus dequant
SGEMM, prepared down/vocabulary and dense reference KV. Its previously measured
41.331 HTTP tok/s is not remeasured here. Both original targets, 53.06 and
55.829 tok/s against fresh Ollama, remain unmet. Runtime defaults are unchanged.

## Results

These are complete-FFN CUDA-event medians in milliseconds. Gain is control
latency / candidate latency minus one. Each row has five contemporaneous
alternating pairs with ten repetitions per stage; confidence intervals use
10,000 paired bootstrap resamples, seed zero.

| Layer | Rows | Packed128 Control ms | Factored Candidate ms | Gain | Paired 95% Gain Interval | Micro Gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 1 | 0.407005 | 0.407299 | -0.07% | -0.23 to +0.26% | Fail |
| 17 | 1 | 0.406970 | 0.408016 | -0.26% | -0.52 to -0.03% | Fail |
| 35 | 1 | 0.406502 | 0.407638 | -0.28% | -0.56 to +0.74% | Fail |
| 0 | 4 | 1.685821 | 1.657546 | +1.71% | +0.32 to +2.64% | Fail |
| 17 | 4 | 1.738058 | 1.688096 | +2.96% | -0.74 to +3.17% | Fail |
| 35 | 4 | 1.714669 | 1.687824 | +1.59% | -0.21 to +5.46% | Fail |

One-row gate/up/SwiGLU medians are 0.27588-0.27784 ms for control and
0.27611-0.27684 ms for candidate. Isolated down remains 0.1313-0.1318 ms.
There is no consistent one-row improvement hidden by the unchanged down path.
Four-row timing is noisier, and its isolated down also varies despite identical
code. Do not attribute that variation to scale factoring. Stage sums need not
equal complete timing because their cache histories differ.

The screen requires at least 5% complete-FFN gain and a positive paired lower
bound at one row before runtime evaluation. This is not a borderline result
against that threshold; extra trials cannot be used to rescue it. No E2E gain
or actual 512-token parity claim follows from these microbenchmarks.

## Hypothesis and Implementation

The [fresh selected-runtime profile](2026-10-25-maxwell-selected-stage-profile.md)
puts about 43% of middle-token GPU stage time in gate/up/SwiGLU. This experiment
keeps the current 128-thread geometry, split4 work distribution, alignment-safe
two-byte weight loads, one shared publication and final-warp reduction. It
accumulates eight FP32 activation/int8 products before applying their common
Q8 scale, instead of scaling each weight separately. Activations remain FP32;
weights, model semantics, attention and prepared down/vocabulary are unchanged.
FP32 association differs, so numerical checks remain required.

Release sm_52 inspection confirms fewer static multiply sites: FMUL goes from
18 to 2, while plain FFMA goes from 26 to 28. These whole-function static
counts include activation code and are not dynamic instruction counts.
Both variants use 32 registers, zero stack/local storage, one barrier site,
ten shuffle sites and 1024 bytes of dynamic shared storage. Weight/load and
conversion site counts remain unchanged. Optional gate/up stores belong to
the same candidate kernel and are enabled only for validation.

Fewer multiplies did not improve useful throughput. This result does not prove
a memory-bandwidth, occupancy or dependency bottleneck; none was measured.
Before another kernel change, investigate the qualified kernel's packed-load
transaction pattern and warp work distribution, using Maxwell-supported
counters if available. Do not expand this experiment into a geometry search.

## Protocol and Correctness

- Model SHA-256: `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
  Original Qwen2.5-3B Q8_0 tensors and frozen real calibration inputs from
  layers 0/17/35 are reused. Genuine rows 1/4 are not repeated copies.
- Source `498699a8b7131cb766c08776e5799099c9b59807`; executable SHA-256
  `64ddf2872b38b8d8af8c52607fd7847e116108ef3b6ad40467433da90ec8dc84`.
  Source was clean before starting; the manifest lists only its newly created
  JSONL output as untracked. Rust 1.95.0, NVCC 12.8.93, CUDA `-O3`, sm_52
  SASS plus compute_52 PTX, no fast math, cuBLAS enabled.
- Native remains an independent numerical reference, but **packed/warp128**
  is the timing control. All eighteen reference/control/candidate validations
  precede each shape's warmup and primary trials. Both timed arms are fused
  at rows 1 and 4, so this is not a new four-row fusion gain.
- Thirty primary pairs contain 180 non-overlapping stage intervals. Arm and
  stage order alternate. Bounded warmup is 1000 ms control / 100 ms candidate.
  All sixty clock boundaries report P0, 1113 MHz SM and 3004 MHz memory; only
  benchmark PID 1114652 appears. No tests, builds, MemPalace embeddings, Ollama
  or other inference jobs overlap the timing run. GPU queries are empty before
  and after the process, separately from its final context-only residency.
- Timed enqueue has zero allocation/free, host/device copies, cuBLAS calls or
  stream synchronization. Launches per ten repetitions are 10 for gate/up or
  isolated down and 20 for complete FFN. Stream waits are zero at one row;
  at four rows they are 10/10/20, covering scratch reuse and down dependencies.
  One event completion wait follows each repeated stage.
- Candidate gate/up, SwiGLU and completed down outputs pass the unchanged
  `1e-3 + 2e-5 * abs(reference)` CPU-reference tolerance. Maximum absolute
  errors across all six shapes are respectively 2.03e-6, 3.87e-6, 2.92e-5
  and 4.80e-5. Capture and ordinary candidate SwiGLU bits match. Maximum
  difference versus native is 0.00032044 hidden / 0.00042725 completed output.
  These are primitive checks, not whole-model logit or generation parity.
- Original weight payload is 3,279,519,744 bytes, prepared down/vocabulary
  1,192,935,424, total 4,472,455,168. No KV or quantized activations are
  allocated. Guarded scratch remains 201,152 bytes at one row and 803,264
  at four rows; both arms reuse it. Tracked allocations return zero to zero.

The first invocation was rejected before GPU work because an extra
`M40LLM_STAGE_PROFILE=0` variable violated the exact frozen-environment guard.
The successful run restored only the recorded environment; the guard was not
relaxed. Reproduce with the [bounded follow-up command](../ffn_benchmark_inputs.md#bounded-fp32-scale-factoring-follow-up)
and a new output filename.

## Validation and Artifacts

Release CUDA primitive tests pass 2/2 and benchmark tests 9/9 with cuBLAS both
enabled and disabled. Enabled affected regressions pass 46/46 across attention,
forward, projection, MLP and runtime ownership. Warning-as-error non-CUDA tests
pass 208/208; CUDA/server all-target Clippy and formatting pass. These results
do not substitute for the skipped runtime gate of this losing candidate.

- [Raw JSONL](2026-10-25-maxwell-factored-ffn-screen.jsonl),
  [stderr](2026-10-25-maxwell-factored-ffn-screen.log.gz),
  [audit source](2026-10-25-maxwell-factored-ffn-audit.jq) and
  [audit output](2026-10-25-maxwell-factored-ffn-audit.json).
  The audit also rejects changed control, timed allocation and fabricated
  median negative controls.
- [sm_52 resources](2026-10-25-maxwell-factored-ffn-resources.txt),
  [selected-function SASS](2026-10-25-maxwell-factored-ffn-sass.txt.gz) and
  [static instruction counts](2026-10-25-maxwell-factored-ffn-instructions.txt).
- [CUDA tests](2026-10-25-maxwell-factored-ffn-cuda-tests.log.gz),
  [cuBLAS-disabled tests](2026-10-25-maxwell-factored-ffn-no-cublas.log.gz),
  [affected regressions](2026-10-25-maxwell-factored-ffn-regressions.log.gz) and
  [non-CUDA tests](2026-10-25-maxwell-factored-ffn-noncuda.log.gz).
- [Artifact hashes](2026-10-25-maxwell-factored-ffn.sha256).

Keep the implementation as a benchmark-only negative result. No generation
selection, default, acceptance threshold or original objective changes.

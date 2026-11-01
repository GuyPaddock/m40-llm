# Native QKV Reduction Hypothesis

## Evidence Before Implementation

The qualified split-FFN/SGEMM configuration spends 2.522 ms / 11.72% of
middle-token GPU time in QKV, according to its
[stage profile](2026-10-25-maxwell-split-stage-profile.md). Its cache contains
gate/up, down and vocabulary, not QKV. Fused QKV therefore uses the original
raw-GGUF split4 kernel at 256 threads, not the earlier losing prepared QKV
consumer. Frozen `M40LLM_Q8_DECODE_SPLIT_QBLOCK=4` selects this path.

The original `qkv_f32xq8_0_gguf_decode_split4_kernel` has a publication
barrier and a shared-memory reduction loop over strides 128 through 1.
At the fixed 256-thread launch this is nine block-wide barriers. Emitted
sm_52 code contains the publication site at `0608`, loop barrier at `0688`
and backward branch at `0698`. Two static sites do not mean two dynamically
executed barriers. The extracted control function SHA-256 is
`b40f55cf110eeff0d125b246b1e8fc54fd517e12cd98d1ca51677f7d9dca509f`.

## One Candidate And Measurement Scope

Keep raw 34-byte Q8_0 blocks, original scalar FP32-activation math, split4
task assignment, 256 threads, one output column per block, Q/K/V routing
and optional biases. After publishing the same partials, use the existing
`reduce_published<256>` tree inside the first warp. It combines the same
128/64/32 strides before shuffle strides 16 through 1, preserving FP32
association. Do not change loads, layouts, warp count, attention, scheduling,
FFN, prepared down/vocabulary or runtime defaults. Preserve the original
control body and its emitted instructions.

Extend `q8_execution_bench`, not a new performance framework. Reuse frozen
M02 model/environment/source guardrails, calibration vectors, guarded
storage and paired bootstrap. Use real layer 0/17/35 Q/K/V weights and
biases. Disclose that the frozen vectors are actual normalized **FFN**
inputs used as QKV projection surrogates, not newly captured attention-norm
inputs. Full-model quality remains mandatory after any micro qualification.
Rows 1/4 mean one or four distinct single-token requests with one fused
launch each, not a batched GEMM or a prefill speed claim.

Five alternating pairs, ten repetitions, GPU completion events and explicit
enqueue counters. Validate CPU outputs, bit equality, nonuniform/tail widths,
biases, signed extremes, zero scales, guards, streams and RAII cleanup before
timing. No packing or quantized activation scratch is added.

This independent stage experiment cannot use complete-FFN timing as its
screen: FFN is unchanged. Require >=5% **complete fused QKV** gain with a
positive paired interval at one request on all three layers. Also require
the optimistic profile-weighted ceiling to reach the original 3% E2E gate:
`1 / (1 - 0.1172 + 0.1172 / (1 + stage_gain)) - 1 >= 0.03`.
This is only a reason to attempt runtime measurement, not an E2E prediction.
No FFN acceptance threshold is changed, and a kernel-only win cannot qualify
generation. Give a borderline result one fresh twenty-pair confirmation;
do not retune geometry to rescue a loser.

## Atomic Sequence And Runtime Gate

1. `docs(perf): scope native QKV reduction`: this pre-implementation audit
   and independent stage-specific measurement boundary.
2. `feat(cuda): test native QKV warp reduction`: one benchmark-only CUDA
   primitive, behavioral tests and a mode in the shared runner.
3. `docs(perf): record native QKV reduction`: source/executable/SASS hashes,
   paired evidence, memory/cleanup audits and measured proceed/stop decision.

Only qualifying candidates proceed to immutable opt-in runtime integration,
actual 512-token equality, frozen dense-valid task quality, incremental
HTTP comparison against current split-FFN/SGEMM, and fresh official Ollama.
Keep the original E2E timing gates and both original goals: 53.06 HTTP tok/s
and >=1.30x fresh Ollama, currently 56.055. The last qualified result remains
45.075 versus 43.119. No numerical/model-changing method substitutes for
those targets.

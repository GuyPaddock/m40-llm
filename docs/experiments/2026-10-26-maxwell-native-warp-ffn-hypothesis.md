# Native 32-Partial FP32 FFN Reduction

## Different Arithmetic Ownership

Qualified gate/up remains the largest middle-token stage, about 41.11%.
The closed virtual-warp experiment emulates 128 original partials using
four gate and four up accumulators per lane. It needs 56 registers and
regresses complete FFN. Do not retune that emulation or change its contract.

Test a separate **32-partial arithmetic mapping**, not another implementation
of the original 128-partial tree. Each lane keeps only one gate/up accumulator
pair and visits four-value tasks at stride 32. Five warp shuffle strides
finish each output. This changes FP32 association. It is not bit-identical
primitive arithmetic, Q8_1 activation quantization or a model modification.
The original actual-512-token equality and frozen dense-valid gates remain
mandatory, as with the previously qualified FP32/128 reassociation.

## One Fixed Kernel

- Four independent output warps in one 128-thread CTA, ceil(N/4) blocks per
  row. Warp w owns output block*4+w; entire out-of-range warps return.
- Preserve original split FP16 scale and signed-byte payload planes, float4
  activation loads, per-weight FP32 scale/FMA and precise SiLU expression.
  Each lane visits task lane, lane+32, etc. Support dynamic complete Q8
  blocks and positive odd N without padding or fixed-K specialization.
- Reduce 32 actual partials, not four emulated partials per lane. No shared
  storage or block barrier. Do not combine closed packet/padding/scale/cache/
  byte-conversion variants. No thread/warp-count, unroll or register-cap sweep.
- Reuse existing typed split owners, raw-span validation, output/capture
  scratch and decode stream. No per-token allocation, copy or synchronization.
  The closed 128-partial emulation and qualified control remain untouched.

This tests whether a different association avoids the register cost of
emulation while increasing independent output ownership. It does not claim
measured occupancy, memory traffic, idle time or attribution to Ollama.
The pinned Ollama dense MMVQ dispatch still combines four warps per output;
it is not evidence that Ollama uses this proposed native-warp mapping.

## Validation And Acceptance

Use the shared captured-input FFN benchmark, actual M02 layers 0/17/35,
genuine rows1/4, original split-f32x4-128 control and immutable prepared
down/vocabulary. Report gate/up/SwiGLU, isolated down and complete FFN.
Record control drift, intermediate/capture parity, guards, immutable weights,
scratch, actual dispatch/counters, setup costs and GPU process/clocks.

Test independent full-output CPU references under the existing finite
`1e-3 + 2e-5 * abs(reference)` tolerance. Do not weaken this after seeing
results. Control-bit equality is not required for the new mapping and must
not be claimed. Ordinary/capture outputs must still be bit-identical.
Cover signed extremes, zero/negative-zero scales, multiple dynamic K,
odd/tail outputs, distinct rows, stream producers, reuse, cleanup and early
span/alignment/overflow/alias rejection. Keep previous strict-control tests
strict. Run CUDA/cuBLAS on/off, attention/forward/projection/MLP regressions,
non-CUDA warning-as-error, Clippy and formatting.

Commit implementation before a clean release screen. Inspect actual sm_52
resources, original control/down words and realized load/reduction mapping.
Use the unchanged bounded warmup and five alternating pairs, ten iterations
per stage. Require **>=5% complete one-row FFN with positive paired lower
bounds at all three layers**. Four-row gains cannot bypass that gate.
Fresh confirmation is allowed only under the existing borderline rules;
no pooling or geometry rescue. A failure closes this new arithmetic mapping.

Only a qualifying screen and sufficient independent profile scope permit
separate immutable runtime selection, actual512/frozen task/logit validation
and repeated HTTP qualification. Changed actual tokens or dense-valid
regressions block exact-output qualification. Both **53.06 HTTP tok/s** and
**1.30x fresh Ollama** remain required. Runtime defaults stay unchanged.

## Atomic Delivery

1. `docs(research): bound native warp FFN study`: this contract.
2. `feat(cuda): probe native warp FFN reduction`: fixed primitive, shared
   benchmark selector, ownership/counter and numerical tests; depends on 1.
3. `docs(perf): record native warp FFN screen`: clean measured evidence,
   compiler resources, audited result and decision; depends on tested 2.
4. Only if all applicable gates pass, separately plan runtime/quality/HTTP
   work. Preserve failed code before any later removal.

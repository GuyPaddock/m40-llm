# Role-Parallel Gate/Up FFN Study

## Independent Ownership Hypothesis

The qualified split-f32x4 gate/up stage remains about 41.11% of sampled
middle-token GPU time. Each of its 128 threads owns both a gate and an up
accumulator. Test one alternative: 256 physical threads for one output,
with 128 threads computing gate and 128 computing up. This separates tensor
roles rather than outputs. It is not the closed two-output cohort mapping,
unfused multi-launch path, or native/virtual output-warp experiment.

Each role retains the original task indices, stride 128, per-weight FP32
scale/FMA, original 128-partial reduction and precise SiLU. There is one
accumulator per physical thread instead of two. Both roles read activations;
this increases logical activation loads. Twice as many threads per output
may also hurt resource use. Register, occupancy or latency improvements
are hypotheses, not measured facts. No attribution to Ollama is claimed.

## Fixed Scope

- One output per CTA, 256 threads, N CTAs per row, 1024 shared bytes.
- Original split FP16 scales and signed-byte payload, float4 activations,
  dynamic complete Q8 blocks and positive odd N. No new weight layout.
- Gate uses threads 0-127, up uses threads 128-255, with logical task IDs
  modulo 128. Publish one partial per thread, then one block barrier.
  The first warp reduces the same two original 128-partial trees.
- Preserve the qualified control and prepared down/vocabulary unchanged.
  Reuse typed owners, session scratch, span checks and decode stream.
  No per-token allocation, transfer, additional synchronization or selector.
- No geometry, role-size, cache, unroll, register-cap or layout rescue.
  Do not combine closed probes or change activation precision/semantics.

## Validation And Decision

Require bit-identical gate, up, SwiGLU and completed FFN results versus the
qualified control as well as the unchanged independent CPU tolerances.
Cover capture/ordinary parity, signed extremes, zero/negative-zero scales,
dynamic K, odd outputs, distinct rows, guards, producer ordering, immutable
weights, reuse, cleanup and rejected overlap/alignment/overflow spans.
Run affected CUDA tests with cuBLAS on/off, attention/forward/projection/MLP
regressions, non-CUDA warning-as-error, Clippy and formatting.

Commit the implementation before clean release timing. Inspect actual sm_52
instructions and resources; compare control/prepared encoded instructions.
Use the shared frozen M02 layers 0/17/35, genuine rows1/4, five alternating
pairs and ten iterations per stage. Include complete gate/up/SwiGLU,
isolated unchanged down and complete FFN. Require at least 5% complete
one-row FFN improvement and a positive paired lower bound at all three
layers. Apply existing fresh confirmation rules without pooling. A failure
closes this fixed mapping; four-row gains do not bypass the one-row gate.

Only a qualifying screen with enough independently supported E2E scope
permits a separate immutable runtime selector, unchanged actual512/frozen
dense-valid quality checks and controlled HTTP qualification. Both original
53.06 HTTP tok/s and 1.30x fresh official Ollama remain required. No runtime
or KV default changes are authorized by this experiment.

## Atomic Delivery

1. `docs(research): bound role-parallel FFN study`: this contract.
2. `feat(cuda): probe role-parallel FFN work`: primitive, shared benchmark
   and behavioral tests, dependent on 1.
3. `docs(perf): record role-parallel FFN screen`: clean release evidence,
   resources, audited results and keep/close decision, dependent on 2.
4. Only if qualified, plan runtime, quality and HTTP delivery separately.
   Preserve any failed implementation before a later removal.

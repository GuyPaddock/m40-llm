# Two Physical FFN Cohorts Per CTA

## Evidence And Distinction

The qualified packed-runtime profile assigns 41.11% of middle-token GPU
stage time to gate/up/SwiGLU. Source `cuda/q8_ffn.cuh::gate_up_split_f32x4`
launches 11008 independent 128-thread CTAs per Qwen row, one output each.
Each thread owns two accumulators; each output publishes 256 floats and
its first warp finishes the original FP32 tree. The qualified sm_52 entry
uses 32 registers, 1024 dynamic shared bytes and one publication barrier.

The closed per-thread two-column probe doubled each thread's accumulators.
The closed virtual-warp probe emulated four partials per lane and increased
registers to 56. The closed reusable-CTA probe introduced a column loop,
a second barrier and 48 registers. None tests two independent **physical
128-thread groups** in a single 256-thread CTA, with two accumulators per
thread and no inter-output loop or extra barrier.

The next question is whether grouping two existing output work groups can
reduce CTA admission/retirement overhead without those resource costs. This
is a hypothesis, not measured scheduling, occupancy or bandwidth attribution.
The [recent reduction screen](2026-10-26-maxwell-decode-reductions-screen.md)
does not supply enough eligible scope; it is not combined with this test.
The current embedding event share is only 0.1455%, so a new embedding path
is not justified by the available profile either.

## One Fixed Candidate

Use exactly two 128-thread cohorts per CTA, 256 threads total. Cohort
`threadIdx.x / 128` owns output `blockIdx.x * 2 + cohort`; its local lane
is `threadIdx.x % 128`. Keep the original four-value tasks, stride 128,
per-thread gate/up FP32 accumulation order and separate scale/value planes.
Each cohort gets its own 256-float publication slice, totaling **2048 dynamic
shared bytes**. One CTA-wide publication barrier precedes the unchanged
per-cohort `reduce_published<128>` tree and original SwiGLU expression.

For an odd output width, the inactive final cohort must still participate
in the barrier, publish initialized partials, and perform no weight reads
or output writes beyond the valid span. Keep support for distinct rows 1/4.
Do not change precision, packing, cache operators, scale conversion, K
specialization, prefetching, exponential, residuals, attention or scheduling.
Keep the old control source and compiled instructions intact.

This is not a register-cap or geometry search over a rejected entry. Add a
separate kernel and leave all losing implementations intact. There is no
cohort-count/block-size/shared-layout/unroll/register-cap rescue if this fixed
mapping loses. Extra predicates or a longer CTA-wide dependency may hurt.
Inspect the actual registers and instructions rather than assume they stay
at 32 or that achieved occupancy is preserved.

## Atomic Delivery And Validation

1. `docs(research): bound physical FFN cohorts`: commit this hypothesis and
   task checkpoint before implementation.
2. `feat(cuda): probe physical FFN cohorts`: benchmark-only CUDA/API entry,
   shared ownership/argument validation, behavioral tests and one exclusive
   mode in the existing Q8 execution benchmark. Depends on step 1.
3. `docs(perf): record physical FFN cohort screen`: clean release executable,
   compiled sm_52 instructions/resources, raw pairs, independent audit and
   measured eligibility decision. Depends on verified step 2.

Use frozen M02 real gate/up/down tensors from layers 0/17/35 and genuine
rows 1/4. Preserve qualified prepared down/vocabulary and source identities,
and retain one guarded scratch allocation across both arms. Compare captured
gate/up, SwiGLU and complete outputs bit-for-bit against split-f32x4-128,
plus independent CPU tolerances. Cover odd/even/tiny output widths, ragged
legal K, zero/signed values, ordered producers, guards, immutable planes,
invalid arguments, enqueue side effects and RAII cleanup with cuBLAS on/off.

Use standard bounded 1000/100-ms warmup, five alternating pairs and ten
iterations per stage. Include complete FFN, not just gate/up. Record actual
dispatch and cohort geometry, scratch, clocks, isolation and serial intervals.
Use existing fresh confirmation rules for excessive variation or borderline
results. Advance only with >=5% complete one-request gain and positive paired
95% interval at all three source layers; four-row results remain separate.

Only passing gates permit immutable runtime integration, actual-512 IDs,
frozen dense-valid tasks/logits and matched incremental/fresh Ollama HTTP
qualification. Defaults remain unchanged. Completion still requires both
53.06 HTTP tok/s and 1.30x fresh official Ollama, currently 56.037 tok/s.
Numerical/model-changing methods cannot substitute for those goals.

# Prepared Projection Activation-Load Hypothesis

## Evidence And Scope

The [new packed-runtime profile](2026-10-26-maxwell-packed-stage-profile.md)
places prepared down at **24.37%** and vocabulary at **7.98%** of sampled
middle-token GPU spans. They still execute the qualified prepared projection,
not the recently packed raw QKV/output consumers. This does not establish
an instruction bottleneck or measured bandwidth.

Current source `cuda/q8_execution.cuh::q8_separate_dot` loads each eight-value
FP32 activation tile through scalar tail checks. The current executable's
[sm_52 instructions](2026-10-26-maxwell-packed-stage-prepared-control.sass.gz)
confirm eight scalar `LDG.E.CI` sites guarded by bounds predicates at
0x448/0x470/0x488/0x498/0x4c8/0x4d8/0x4f0/0x508. Its four prepared weight
loads are already `LDG.E.CI.64`; do not claim weights need scalar-to-packed
conversion. Resource use is 40 registers, zero reported stack/local storage.

Full SASS SHA-256 is
`333013085bd1e550afc35c6bf329183b675be4679e71bea69cdf994ea6673fa1`,
identical to the preserved earlier prepared-warp control. This is a new
activation-load hypothesis, not a rescue of that failed reduction experiment.

## One Fixed Candidate

For complete Q8 blocks, load the same eight FP32 values through two aligned
`float4` reads. Preserve the original per-thread tasks, activation precision,
signed conversion, per-weight scale/FMA association, all four accumulators,
column masks, weight-plane loads and **original shared reduction tree**.
Keep its block-wide barriers; do not combine the rejected warp finish,
cache hints, byte conversion, fixed-K specialization or failed FFN layouts.

Require `K % 32 == 0` and 16-byte activation alignment explicitly. Invalid
requests fail before launch, not through silent fallback. Existing public
consumers and partial-K support remain untouched. Row strides are naturally
aligned for complete blocks. No new device weights, repacking, scratch,
activation quantization or materialized FP32 cache is permitted.

Retain exactly the qualified launch settings: four output columns, 256
threads for one-row down and 128 for four-row down; vocabulary uses 128.
No thread/column/register-cap search or extra variants if this candidate
loses. Multi-row inputs must be distinct, not repeated single-row timing.

## Validation And Gates

Extend the existing prepared-projection benchmark rather than add a framework.
Use real Qwen down tensors on layers 0/17/35 and the real vocabulary tensor,
with the frozen activation fixture/report and qualified controls. Keep
current split-plane gate/up/SwiGLU unchanged when measuring complete FFN.
The raw QKV/output packed runtime also remains fixed in eventual integration.

Behavioral tests compare exact control bits, CPU tolerances and guards for
single/four rows, odd column tails, zero/scaled signed extremes and supported
input widths. Cover both streams and ordered async producers, allocation
cleanup, unchanged source/prepared data, and rejected misalignment/partial-K
requests. Require compiled evidence of changed activation loads, unchanged
weight/reduction instructions where applicable, registers and no spills.
Do not attribute speed from source spelling alone.

Screen with five alternating pairs, extending borderline findings under the
existing rules. Include isolated down, unchanged gate/up, **complete FFN**
and complete vocabulary projection timing, required launches/dependencies,
scratch/copy counts and GPU boundaries. Down requires **>=5% complete-FFN
gain with a positive paired interval**, not merely 5% isolated down.
Vocabulary requires its own >=5% complete-projection gate and positive
interval. Four-row results are separate from the single-token decision.

Only passing stages may enter a profile-weighted feasibility estimate
against the unchanged 3% E2E floor. That estimate is not runtime evidence.
No combining a below-gate candidate with another improvement to rescue it.
If feasible, explicit model-local integration must preserve all 109 prepared
identities, actual 512 IDs, frozen dense-valid quality and sampled logits,
then pass incremental HTTP and fresh official Ollama qualification.
The exact objectives stay **53.06** and **1.30x fresh Ollama**, currently
56.037 HTTP tok/s; no numerical/model-changing substitute is accepted.

## Atomic Sequence

1. `docs(perf): profile packed Q8 runtime`: measured profile, compiled control
   evidence and this bounded hypothesis. No kernel changes.
2. `feat(cuda): probe packed activation loads`: isolated CUDA/API consumer,
   behavioral tests and existing benchmark extension. Depends on step 1;
   generation defaults and qualified control remain untouched.
3. `docs(perf): record prepared activation screen`: all pairs, numerical/
   instruction proofs and measured decision. Preserve a loser before any
   separate removal; do not widen the search.
4. Only on passing gates, separate model-local runtime, quality and HTTP
   qualification commits. No automatic default promotion.

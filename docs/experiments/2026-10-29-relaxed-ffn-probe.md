# Fixed Compound ILP4 Numerical Probe

## Admission And Prior Work

Admit one benchmark-only operator after the exact portfolio decision. Current
`gate_up_joint_f32x4_shared_scales` has128 threads per output, four four-byte
tasks per thread at K2048, and one16-contribution FP32 chain for each of gate
and up. Shared scale publication and output publication each have one barrier.
The final warp already performs the original reduction. Merely replacing
block-wide reduction with shuffles would duplicate existing work.

The native32-partial warp experiment `dd78a1a` removed publication but regressed
4.72-4.77% complete FFN. Virtual-warp emulation kept four virtual thread
partials per lane; that changed physical output ownership. The task-pair
experiment changed payload access but retained serial arithmetic. The qualified
four-row consumer keeps independent rows, not independent sums for one row.
These do not implement four byte-position chains under the current compound
CTA/task/layout schedule. Do not re-run their failed configurations.

## One Fixed Change

Support the frozen single-row Qwen case K2048/N11008. Keep joint weights,
once-published scales,128 threads,1536 dynamic shared bytes, payload addressing,
FP32 per-weight scaling and precise SiLU. Keep down and vocabulary unchanged.
Maintain four gate and four up accumulators, indexed by byte0..3. Each visits
the same thread-owned tasks in order. Combine each family as
`(partial0 + partial1) + (partial2 + partial3)` before the unchanged128-partial
block reduction. No task-pair packing, fixed-K unroll, FP16, fast-math or altered
expf. An explicit shape guard is not permission to change address arithmetic.

The hypothesis is instruction-level parallelism: four chains of length4 plus
two combination levels instead of length16. Added work is six FP32 additions
and six live accumulators per thread. Loads, payload bytes, shared capacity and
barrier count do not decrease. A6-versus16 dependency-depth comparison is NOT
a throughput prediction; memory and other instructions may dominate completely.

Compiler admission requires unchanged original control/down instruction words,
no local spills, <=48 registers/thread and unchanged shared capacity. Record
actual instructions and allocation; do not assert measured occupancy or stalls.
If the compiler collapses the intended independence or exceeds resources, stop
without register caps, another accumulator count or geometry rescue.

## Numeric And Performance Gates

At middle-profile gate/up7.417376ms/token,3% request throughput needs about
0.572107ms/token saved, or7.71% gate/up latency reduction before extra costs.
Full FFN is about12.19ms/token; a5% FFN throughput improvement removes about
0.580ms/token, leaving little margin. This motivates measuring complete FFN
before any runtime integration, not claiming a forecast from chain depth.

Use frozen actual inputs/layers0/17/35 and same-shape synthetic signed/zero,
cancellation and dynamic-range cases. Record gate/up/SwiGLU/down max/mean drift,
finite behavior and independent FP64 CPU error. Keep existing absolute/relative
CPU tolerance1e-3+2e-5*abs(reference); intermediate bit equality is diagnostic
for the candidate, mandatory for unchanged controls. Include guards, streams,
alias/shape rejection, input immutability, cleanup and capture/ordinary parity.

Measure five alternating release pairs of gate/up, down and complete FFN with
the established warmup, repetitions and confirmation policy. Require >=5%
complete-FFN gain and positive paired bounds at ALL three layers. Down must
retain its original dispatch and results. Stop immediately on a failed gate;
preserve source/evidence before any removal.

Only after operator admission, add model-local load-time selection
`M40LLM_Q8_FFN_NUMERICS=qualified|approximate-ilp4`. Default remains qualified;
the approximate value requires the compound backend and supported shape.
Reject unsupported combinations, including speculative verifier use in this
task, rather than silently changing arithmetic. No per-token environment reads.
Record numerical mode in startup and every quality/performance manifest.

Freeze candidate before evaluating calibration and held-out independently.
Permit changed tokens/logits, but require no new dense-valid task failures and
<=1% relative PPL increase on EACH frozen split. Record raw full-logit drift,
outputs, disagreements and stop reasons. Dense-failed rows remain inconclusive.
The small frozen corpus is bounded evidence, not universal quality proof.

Then apply unchanged buffered HTTP gates and actual512 length-stop checks.
Compare the historical Ollama rate without rerunning it. Label results as
approximate-model versus original-model; no claim against the exact milestone.
Keep approximate mode opt-in even if it qualifies. Do not stack with the
portfolio or drafter during this experiment.

## Status

Paper mechanism/resource contract complete; operator implementation and all
hardware/quality/performance gates remain pending. No GPU hours consumed.
See [the approved track contract](2026-10-29-research-tracks-plan.md),
[native-warp negative](2026-10-26-maxwell-native-warp-ffn-screen.md) and
[current sampled dataflow](2026-10-28-dataflow-findings.md).

# Fixed CTA-Reuse FFN Experiment

## Evidence And Scope

Source base is clean `e124b8d`. The selected runtime still spends 8.045 ms,
41.11% of sampled middle-token stage time, in gate/up/SwiGLU. The qualified
kernel creates 11008 independent 128-thread CTAs per input row, one per
output. Its current sm_52 entry has 32 registers, 1024 dynamic shared bytes,
one publication barrier and the original-tree final-warp finish.

Inspecting its SwiGLU SASS shows that much of the long tail is cold
exceptional-value handling. The normal path already uses a hardware
exponential, reciprocal and correction. Static code size is not dynamic
latency evidence; no approximate exponential or fast-math experiment follows.

The independent question is whether repeatedly admitting and retiring one
CTA per output is profitable compared with reusing a bounded worker grid.
There is no measured CTA-turnover attribution yet. This is a hypothesis,
not a claim that launch count, bandwidth or occupancy has been improved.

## One Fixed Candidate

Keep one output at a time per CTA, the same 128 physical threads, four-value
task stride, FP32 accumulators, publication layout, reduction association,
ordinary expf/reciprocal and exact split planes. A CTA iterates output
columns `blockIdx.x + iteration * gridDim.x`. Reset both accumulators for
every column. Add a CTA-local barrier after the final warp has consumed
shared partials and written output, before reusing the shared storage.

Fix the grid cap at **192 CTAs per input row**, or the output count when
smaller. This conservative M40 worker budget is fixed before measurement;
it is not a claim about actual residency. Keep 128 threads and 1024 dynamic
shared bytes. There is no device-wide barrier, polling, global queue,
atomic progress counter, persistent decode worker or server scheduler change.

This differs from the closed virtual-warp mapping: each column still uses
all 128 physical threads and exactly two live accumulators per thread.
Unlike the closed two-column mapping, a thread never accumulates two output
columns simultaneously. No new layout, cache operator, conversion trick,
activation quantization, software prefetch or numerical approximation is
combined with this experiment. Extra loop/barrier cost or reduced parallelism
may make it slower.

No block-cap, warp-count, tile-size, unroll or register-cap search is allowed
if this fixed mapping loses. The existing control entry must remain intact.
Implementation is original project code with no new upstream borrowing.

## Delivery And Gates

1. Commit the prespecified hypothesis before code.
2. Add the benchmark-only primitive, shared enqueue guards, behavioral tests
   and one exclusive selector in the existing Q8 execution benchmark. Commit
   verified code before timing. No runtime/generation selector is added.
3. Commit raw pairs, compiled sm_52 resources/instructions, independent audit,
   numerical/ownership evidence and the measured decision.

Use frozen M02 real layer 0/17/35 gate/up and down tensors and distinct
captured rows 1/4. Hold prepared down/vocabulary and all source bytes fixed.
Use standard bounded warmup, five alternating pairs and ten iterations per
stage. Record gate/up/SwiGLU, unchanged isolated down and complete FFN,
actual dispatch, tracked memory, scratch and GPU process/clock boundaries.
Do not overlap timed work with inference, builds, tests or MemPalace.

Require bit-identical gate/up, SwiGLU and complete outputs versus the
qualified split control, CPU reference tolerance, immutable input/weight
planes and guards. Test output widths below/at/above the cap and multiple
worker iterations, signed extremes, zero scales, distinct rows, unsupported
arguments, stream dependencies and RAII cleanup. The candidate may not
allocate, copy, synchronize or silently fall back during enqueue.

Advance only for at least 5% complete-FFN throughput gain with positive
paired 95% lower bound at one row; four-row results remain separate.
Borderline results follow existing fresh-confirmation rules. A micro pass
only permits actual-512-ID/frozen-quality and matched HTTP qualification.
No default changes follow from this screen. Completion still requires
53.06 HTTP tok/s and 1.30x fresh controlled official Ollama, without
substituting approximate/model-changing or unrelated scheduling work.

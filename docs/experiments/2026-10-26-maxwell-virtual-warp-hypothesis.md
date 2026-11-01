# Fixed Virtual-Thread FFN Ownership Experiment

## Evidence Before Implementation

Source base is clean `2e3f60f`. The qualified packed runtime's middle-token
gate/up/SwiGLU cost is 8.045 ms, 41.11% of measured stage time. Its
`gate_up_split_f32x4` assigns 128 threads to one output, publishes two arrays
of 128 partials to 1024 shared bytes, executes a block barrier, and finishes
the original tree in the first warp. Three quarters of the threads do not
participate in that finish. This is source-level work ownership, not a
measurement of idle cycles or occupancy.

The pinned llama.cpp source is
`9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e`. `mmvq.cu:452` selects four
warps for generic one-token Q8. `calc_rows_per_block` at line 595 returns
one output unless the small-K condition holds. Q8 has `qi=8,vdr=2`;
the condition at line 1146 is `K/32 < 4*2*32/8`, false for K=2048 and
K=11008. Lines 737-830 show four warps accumulating one output, then
shared publication, one barrier and a final-warp reduction. This agrees
with the previously observed `[32,4,1]` launch. Its independent-warp MoE
path is not evidence for dense Qwen dispatch. Do not attribute the proposed
mapping below to Ollama or claim that it explains Ollama's performance.

## One Independent Hypothesis

Replace physical ownership of the existing 128 partials, not their
arithmetic. One warp owns one output. Each lane keeps four gate and four up
accumulators for virtual thread IDs `lane + {0,32,64,96}`. Each accumulator
visits exactly the control's four-value tasks at stride 128. Combine the
four virtual partials with `(p0+p2)+(p1+p3)`, then the control's five shuffle
strides. This preserves the control's partial sums and reduction association.

Pack four independent output warps into one 128-thread block. This removes
shared publication and the block barrier without adding projections,
requantization, new weight storage, activation reconstruction, or per-call
scratch. Unlike the closed two-column experiment, no thread accumulates two
output columns. The cost is eight live accumulators and longer per-lane
work; register pressure and lost parallelism may outweigh the saving.

Fix the mapping before measurement: four output warps, 128 threads, no
dynamic shared memory, existing split planes and four-value input loads.
No alternative warp count, unroll/width, register cap, cache policy, joint
layout, byte conversion or combination with rejected candidates is allowed
in this screen. This is not a rescue of the prepared activation-load probe.
Implementation is original project code; no upstream kernel is copied.

## Delivery And Gates

1. Commit this source audit and prespecified hypothesis.
2. Add a benchmark-only CUDA primitive, the shared Rust enqueue validation,
   behavioral parity/guard tests, and one selector in the existing FFN bench.
   Do not add a runtime/generation selector.
3. Commit the verified implementation before timing it. Measure clean source,
   then commit raw results, instruction/resources evidence, audit and decision.

Use the frozen M02 real layer 0/17/35 tensors and distinct captured rows 1/4,
five alternating pairs, ten iterations per stage, standard bounded warmup,
and the frozen environment. Hold the qualified prepared down/vocabulary
owners and all source bytes fixed. Include gate/up/SwiGLU and complete FFN,
report isolated down as the unchanged control, guards, intermediate/output
bits, device allocation/copy/sync counters and GPU conditions.

Require control-bit equality for gate, up, SwiGLU and complete outputs, CPU
references within the existing tolerance, valid tails, signed extremes,
zero scales, independent rows and async ordering. Inspect sm_52 SASS and
register/local/shared resources, including whether the original control
entry remains unchanged. Do not infer DRAM traffic or measured occupancy.

Advance only with at least 5% complete-FFN throughput gain and a positive
paired 95% interval at one row; report four-row results independently.
Borderline gains require fresh confirmation under the existing protocol.
A pass only permits subsequent actual-512-ID/frozen-quality and matched
HTTP qualification, not promotion. A failure closes this fixed mapping.
The exact track still needs both 53.06 HTTP tok/s and 1.30x a fresh official
Ollama median. No default, KV, model, sampling or scheduler changes follow.

# Fixed-Width FFN Address Specialization

## Evidence Before Implementation

The qualified one-column split kernel still computes addresses and loop
bounds from runtime K. Its preserved
[sm_52 SASS](2026-10-25-maxwell-joint-ffn-sass.txt.gz), function containing
`19gate_up_split_f32x4`, has **21 static XMAD sites before the first global
load**. These form row/column stride products from the runtime K parameter,
including the FP32 row stride, payload column stride K/4 and scale column
stride K/32. The compiler peels the first task, then emits two task bodies
around the backward branch at `0a90`.

For the actual Qwen gate/up shape, K is always 2048. Each of 128 threads
performs exactly four four-value tasks, in increasing original order. The
strides 2048, 512 and 64 are powers of two. Compile-time K can therefore
replace runtime products and loop control with constant shifts/offsets.
This identifies removable instructions, not a measured bottleneck fraction,
bandwidth or occupancy. The candidate must be measured to establish value.

## One Independent Candidate

Use a fixed K=2048 specialization of the current separate-plane kernel.
Keep one column, 128 threads, four-value tasks, original FP32 scaling/FMA,
accumulation order, publication and reduction tree. Permit the compiler to
unroll the four tasks. Preserve optional gate/up capture. Explicitly reject
other K values rather than silently executing a fallback.

Do not combine the below-gate joint representation, change quantization,
repack weights, alter arithmetic, tune launch geometry, or change prepared
down/vocabulary. Both arms retain the same single prepared gate/up pair.
The original qualified kernel must remain unchanged in source and emitted
instructions. Inspect new instructions and register/spill storage before
timing; fewer prologue instructions alone do not qualify an optimization.

## Atomic Work And Gates

1. `docs(perf): scope fixed-width FFN experiment`: record this hypothesis
   and the independent experiment boundary.
2. `feat(cuda): test fixed-width split FFN`: add one benchmark-only
   specialization, explicit unsupported-width handling and behavioral tests;
   reuse the existing runner and prepared ownership.
3. `docs(perf): record fixed-width FFN screen`: preserve source/executable
   hashes, sm_52 evidence, raw pairs, numerical/ownership results and the
   measured decision. Depends on the verified implementation commit.

Use original frozen inputs at layers 0/17/35, actual rows 1/4, five
alternating pairs and ten repetitions. Require bit equality against current
one-column split plus existing CPU tolerances, guards, stream ordering and
cleanup. No timed allocation/copy/synchronization or Q8_1 scratch.

The unchanged >=5% complete one-row gain with positive paired confidence
intervals gates any runtime integration. Give a borderline result one fresh
twenty-pair confirmation, never a geometry/format rescue. Only qualifying
candidates proceed through original actual 512-token/frozen-task quality,
incremental HTTP and fresh official Ollama comparison.

The selected runtime remains 45.075 HTTP tok/s; completion still requires
53.06 and 1.30x fresh official Ollama, currently 56.055 tok/s. Runtime
defaults and prepared down/vocabulary are unchanged.

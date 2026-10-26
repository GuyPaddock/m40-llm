# Gate/Up Weight Read Diagnostic

## Question And Scope

The fixed eight-value experiment improved complete one-row FFN by only
2.00-2.03%, below the existing 5% gate. Do not rescue it or combine closed
probes. Independently measure how costly reading the qualified gate/up
planes is when their inference arithmetic is replaced by verifiable integer
checksums. This is a diagnostic, not a new inference candidate.

Use the shared release Q8 benchmark, frozen M02 inputs and actual Qwen
gate/up tensors from layers 0, 17 and 35, rows 1 and 4. Keep prepared
down/vocabulary resident and unchanged. Both measured paths retain the same
lossless separate gate/up planes. No KV, attention or runtime configuration
changes are permitted.

## Fixed Diagnostic

- One 128-thread CTA per output column and row, four-value payload tasks.
- Read each gate/up u32 payload word and its original u16 scale bits at
  task/8, matching the qualified kernel's source-level weight addressing.
- Sum payload word plus scale bits modulo 2^32, separately for gate/up.
  Publish two 128-entry u32 arrays, synchronize once, and reduce using
  shared partials plus warp shuffles. Store two u32 checksums per output.
- Read every plane bit; do not use dummy output, constant input, inline PTX
  cache overrides or a compiler setting that can remove the reads.
- No activation load, dequantization, activation multiplication or SiLU.
  Each row deliberately rereads the same weights, as the control does.
- Own guarded reusable checksum output before timing; enqueue performs no
  allocation/free, copy or stream synchronization. Validate dimensions,
  alignment and nonoverlap before launch. Preserve explicit producer waits.

Verify exact CPU checksums derived independently from original GGUF Q8_0
blocks, output guards, repeated reuse, immutable planes and cleanup. Cover
wrapping arithmetic, signed payload byte extremes, zero bits, tails/odd
output widths, rows 1/4 and invalid arguments. Inspect generated sm_52 loads,
registers, shared memory and spills, plus unchanged control instruction words.

## Measurement And Interpretation

Warm the real FFN for at least 1000 accumulated GPU ms and the diagnostic
for at least 100 ms with a finite existing warmup budget. Run five serial
alternating pairs, ten repeated enqueues per interval. Report the standalone
read/checksum interval, real gate/up/SwiGLU, isolated down, complete real FFN,
launch/dependency counters, payload/scratch bytes and GPU conditions.
Collect no hardware counter estimates in place of unavailable CUPTI counters.

The checksum kernel has different arithmetic, output size and resource use.
Its time is neither pure DRAM time nor a rigorous lower bound on inference.
Any payload/time rate is logical bytes per measured interval, not measured
memory bandwidth. A difference from gate/up timing identifies a surrogate
cost gap, not proven removable arithmetic or an end-to-end speedup. Never
feed checksums to down projection or call their sum a complete FFN timing.

If checksum time is close to control, seek independent evidence before
another arithmetic-only probe. If it is much lower, inspect the qualified
compiled arithmetic/dependency chain before proposing one bounded candidate.
Neither outcome authorizes geometry sweeps or runtime integration. Keep
53.06 HTTP tok/s and 1.30x fresh Ollama as the unchanged exact goals.

## Atomic Delivery

1. `docs(research): bound FFN read diagnostic`: this contract, before code.
2. `feat(bench): measure gate-up weight reads`: CUDA checksum primitive,
   existing shared benchmark integration and behavioral validation. Depends
   on the contract; has no generation selector or inference consumer.
3. `docs(perf): record FFN read costs`: release artifacts, independent audit,
   limitations, status and next evidence-backed action. Depends on the
   verified diagnostic. Do not claim an inference optimization.

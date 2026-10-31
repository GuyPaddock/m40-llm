# Lossless Cross-Column Warp Packets

## Source-Backed Hypothesis

The [prepared read diagnostic](2026-10-26-maxwell-prepared-read-screen.md)
retains about 89% of real one-row down/vocabulary cost. It is not DRAM
timing or a lower bound. Compiled control has four separate 64-bit payload
and four U16 scale loads per eight-value task. Test one lossless packet
layout that reduces load/address issue work without changing logical bytes,
thread ownership, activation precision, FP32 association or reduction.

This is not the closed C10 scalar-input-interleaved consumer. Do not combine
the rejected packed-activation or warp-finish changes, or rescue a loss with
new geometry, unrolling, cache policy or register caps.

## Fixed Layout And Consumer

- Support only K divisible by 256 and N divisible by four, covering the
  actual Qwen down (11008,2048) and vocabulary (2048,151936) shapes. Reject
  unsupported shapes explicitly, before allocation/launch.
- Copy every original scale bit and signed payload byte. Use typed host and
  device ownership with a semantic source identity and explicit format
  version, separate from GGUF and existing execution representations.
- Payload order is
  `values[output_tile][warp_task_tile][half][lane][column][byte]`, with two
  halves, 32 lanes, four columns and four bytes per column. Each half is
  512 contiguous bytes. Scale order is `scales[output_tile][qblock][column]`,
  four original u16 values per record. No padding or quantization is added.
- Keep the original four-column CTA and eight-value task: `task=tid`,
  stride blockDim.x. Load eight scalar FP32 activations as the control does.
  Read one packed scale record and two uint4 payload records. Accumulate
  each column's first four values, then its last four, preserving its
  original FP32 scale/multiply/FMA order. Retain column guards and call the
  original shared reduction tree. No warp-finish substitution.
- Down uses 256 threads at one row and 128 at four; vocabulary uses 128.
  Preserve original stream dependencies. Enqueue does no allocation/free,
  transfer or synchronization. Typed RAII owners retain all plane data.
- For full aligned warps, old four 8-byte-per-lane payload instructions and
  new two 16-byte-per-lane instructions cover the same 1024 bytes and 32
  source-address buckets of 32 bytes summed across instructions. Scale
  instructions can decrease from four to one, buckets from four to two.
  These counts are not measured cache/DRAM transactions or a promised gain.

## Validation And Measurements

Before timing, require exact CPU inverse roundtrip for all original bytes,
including arbitrary scale encodings and signed/zero payloads. Test identity,
byte counts, unsupported shapes, partial-upload cleanup, guarded device
buffers, distinct rows, dependent producer streams, repeated reuse and
invalid input/output spans. Synthetic GPU output must match the original
prepared control bits as well as existing CPU tolerances.

Extend the existing `q8_execution_bench` prepared-projection screen, not a
new benchmark framework. Use the frozen M02 model/input/report identities,
actual down layers 0/17/35 and vocabulary with the existing distinct
layer-0 normalized-input surrogates, rows 1/4. Both arms retain original
prepared controls and the candidate packet. Report packing/upload times,
extra host/device bytes, original cache identities, scratch/counters,
actual dispatch and compiler loads/registers/spills/barriers. Bound extra
packet plus gate/up pair at 128 MiB per down case; vocabulary packet at
512 MiB. Those are setup budgets, not per-token work.

Warm real control for at least 1000 GPU ms and candidate for 100 ms with
finite existing budgets. Run five alternating pairs, ten enqueues per
stage. Measure gate/up, isolated down and complete FFN separately, vocabulary
as its own complete projection. Use existing paired bootstrap and gates:
**>=5% complete one-row FFN at all three down layers with positive lower
bounds**; vocabulary separately needs >=5%/positive interval. Four rows are
not the decode gate. No pooling or geometry rescue; collect fresh prescribed
confirmation only for genuine borderline/variation cases.

Even passing micro gates do not authorize promotion alone. Require enough
independently qualified profile scope, unchanged actual 512 IDs, frozen
dense-valid quality/logit checks, and >=max(3%,twice variation) repeated HTTP
gain before integration is retained. Preserve original 53.06 HTTP tok/s
and 1.30x fresh controlled Ollama requirements. Keep defaults unchanged.
Do not substitute Q8_1, sparse/approximate inference or batching.

## Atomic Delivery

1. `docs(research): bound warp packet experiment`: this contract before code.
2. `feat(cuda): probe lossless Q8 warp packets`: typed host/device packer,
   fixed consumer, shared benchmark integration and behavioral tests;
   depends on the contract. No generation selector or cache rewrite.
3. `docs(perf): record warp packet screening`: clean release pairs, source/
   compiled evidence, independent audit and gate decision; depends on the
   tested implementation. Preserve losing code before any later removal.
4. Only if every applicable gate passes, separately plan conditional runtime
   selection and actual-model/HTTP qualification. Otherwise close the
   experiment without configuration rescue or combining rejected probes.

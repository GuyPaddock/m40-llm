# Library Segmented-Read Reference

## Question

The existing gate/up checksum preserves each output column but also reuses
the qualified kernel's 128-thread CTA mapping. Its roughly 94% real-stage
time is not a DRAM lower bound. Independently compare CUDA's bundled CUB
segmented reduction on the same logical per-column modular checksums.
This is a library reference, not a new FFN geometry or inference candidate.

Local CUDA 12.8 headers contain CUB 2.7.0. Its segmented reduction selects
the SM35 policy on sm_52: one CTA per segment, warp-based block reduction,
library-selected thread/items policy and no requested dynamic shared memory.
Use public `DeviceSegmentedReduce::Reduce`, not copied implementation or
custom policy tuning. Inspect actual launches/resources before attribution.
The compound transform iterator may inhibit ordinary vectorization; a win
is not assumed. Library dispatch/host costs remain included in intervals.

## Fixed Diagnostic

- Use original split gate/up FP16-scale bits and u32 payload words. A
  transform iterator yields a pair of u32 sums per original four-value
  task: gate/up payload word plus original u16 scale bits at task/8.
- Segment boundaries are original output columns; reduce each pair modulo
  2^32 independently. Output remains rows*N interleaved gate/up checksums.
  This is associative integer arithmetic, not FP32 inference or SiLU.
- Flatten rows/columns into segments, rereading the same weights per row,
  matching the existing diagnostic. Support rows 1/4, dynamic complete Q8
  blocks, odd N and checked signed segment/offset bounds.
- Use one caller-owned bounded workspace queried before timing, RAII
  cleanup, original owners and decode stream. No timed allocation/free,
  transfer, explicit sync or inter-stream wait. Include library dispatch and
  all its kernel launches; do not assume one launch from source alone.
- Keep qualified real gate/up, prepared down/vocabulary, original checksum
  and all runtime/default choices unchanged. No source/weight repacking,
  inference selector, alternate CUB policy, geometry/cache/register/unroll
  rescue or combination of closed candidates.

## Checks And Decision

Test exact original-source modular checksums, signed/extreme payload bits,
arbitrary/zero scale bits, dynamic short/long K, odd widths, distinct row
outputs, guards, immutable inputs, ordered producers, scratch reuse, cleanup
and rejected bad dimensions/overlap/alignment/overflow/workspace spans.
Extend existing shared captured-input benchmark and CUDA weight-read tests.
Run applicable cuBLAS on/off, projection/MLP/attention/forward regressions,
non-CUDA warning checks, Clippy and formatting.

Commit tested code before release observation. Use frozen M02 real layers
0/17/35, genuine rows1/4, five alternating pairs, ten repeated enqueues per
stage and existing finite warmups. Compare original and library checksum;
keep real gate/up, isolated down and complete FFN as separate controls.
Record CPU/guard results, counters, resource/dispatch evidence, memory,
clock/process isolation and raw timing, without claiming DRAM or occupancy.

Require at least 10% one-row checksum throughput improvement with positive
paired lower bounds at all three layers to justify further investigation.
A win establishes only diagnostic cost scope; an inference design still
needs its own hypothesis, original >=5% complete-FFN gate, actual512/frozen
quality and HTTP qualification. A losing fixed library reference closes
without policies, new geometries or an inference consumer.

## Atomic Delivery

1. `docs(research): bound CUB read reference`: this contract.
2. `feat(cuda): add segmented read reference`: bounded public-library
   adapter, shared benchmark and behavioral checks, dependent on 1.
3. `docs(perf): record segmented read costs`: clean release/raw/resource
   evidence, independent audit and measured decision, dependent on 2.

Preserve failed code before separate removal. Qualified exact runtime stays
49.146 HTTP tok/s; original 53.06 and 56.037 HTTP tok/s targets are unmet.
No Q8_1, approximation, sparsity, scheduling or default change is authorized.

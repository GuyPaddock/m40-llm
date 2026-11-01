# Qualified Gate/Up Reader Work Audit

## Finding

The current split gate/up reader has **no duplicated logical payload read**
within a one-row layer. Every packed weight word is consumed once at its
original output-column address. It already shares each float4 activation
load between gate and up, uses aligned contiguous payload words, and
finishes with one publication barrier plus warp reductions.

There are repeated scale operands and activation reads across output
columns. They are concrete source work, **not measured DRAM transactions**.
Experiments targeting those repetitions are already closed below their
complete-FFN gates. This audit does not identify a new, independently
supported implementation. Preserve the qualified **49.146 HTTP tok/s**
configuration; do not launch another arithmetic/layout variation merely
because source-level repetition exists.

This is not evidence of a hardware ceiling. Cache misses, physical request
amplification, achieved occupancy and warp stalls remain unmeasured. Nor is
it proof that a different lossless representation cannot work. A future
proposal must identify the actual work it avoids and establish enough scope
under the [admission checklist](../maxwell_q8_execution.md#admission-for-further-optimization).

## Address And Byte Accounting

Inspected clean source `fad8adcadb02609eadbc6b6246dc9cec7aefa25a`:

- [Separate-scale/value packing](../../src/infer/execution_weights/mod.rs#L423)
  copies the original FP16 scale bits and signed payload bytes without
  requantization or padding.
- [Qualified consumer](../../cuda/q8_ffn.cuh#L157) uses
  `task = tid + iteration * 128`, `value = column * (K/4) + task`,
  `scale = column * (K/32) + floor(task/8)`.
- [Launch](../../cuda/q8_ffn.cuh#L835) has 128 threads, N output-column CTAs
  per row and 1024 dynamic shared bytes. Normal generation does not request
  optional gate/up intermediate outputs.

For Qwen K=2048, N=11008, each CTA has 512 tasks, four per thread. Enumerating
the task map covers `[0,512)` exactly once; disjoint column bases preserve
that property across the matrix. Each projection has 64 distinct scales,
each used by eight tasks. A full warp's scale instruction addresses four
distinct scales, not 32 unrelated scale locations. Therefore eight source
uses cannot be relabeled as eight physical memory transactions.

| One gate/up layer, one activation row | Unique bytes | Source operand bytes |
| --- | ---: | ---: |
| Signed payload, both projections | 45,088,768 | 45,088,768 |
| FP16 scales, both projections | 2,818,048 | 22,544,384 |
| FP32 activation vector | 8,192 | 90,177,536 |

Unique gate/up weights total **47,906,816 bytes**. The activation vector is
reused across columns and might be served by caches; this audit measures
no cache behavior. Per layer, normal outputs are 44,032 bytes, shared
partial publication is 11,272,192 logical bytes, and 11,008 CTAs execute
one block barrier each. Shared bytes are not global/DRAM bytes.

The [ledger](2026-10-26-maxwell-reader-work-ledger.json) binds dimensions and
six previously validated layer0/17/35 gate/up tensor identities to the
frozen model/capture. It enumerates the task map and records formulas, not
a new inference or GPU benchmark. Source logical counts are not dynamic
SASS instruction counts; the compiler can peel/unroll the loop.

The existing [control SASS](2026-10-26-maxwell-ffn-read-control.sass.gz)
shows read-only `LDG.E.CI` payload/scale accesses, `LDG.E.CI.128`
activations, half conversion, signed integer conversion, one `BAR.SYNC`
site and shuffle reductions. Reported resources remain 32 registers,
1024 dynamic shared bytes and no stack/local storage. These establish
compiled operations, not achieved occupancy or memory bandwidth.

## Which Avoided-Work Ideas Have Evidence?

| Work targeted | Existing result | Decision |
| --- | --- | --- |
| Separate gate/up addresses and load issues | [Joint packets](2026-10-25-maxwell-joint-ffn-screen.md): +4.66-4.74% complete FFN after fresh twenty-pair confirmation | Below 5%; no packet/geometry rescue |
| Multiple payload instructions for thread-owned tasks | [Task-pair packets](2026-10-26-maxwell-ffn-task-pair-screen.md): +0.23-0.34% complete FFN | Closed |
| Repeated FP16 scale loads/conversions | [Shared scales](2026-10-26-maxwell-shared-scales-ffn-screen.md): +1.99-2.19% complete FFN | Closed |
| Activation reads and CTA setup across output columns | [Two columns](2026-10-25-maxwell-split-cols2-screen.md): -5.85 to -6.01% complete FFN | Closed |
| Payload storage size | [Indexed byte coding](2026-10-26-maxwell-ffn-entropy-study.md): only 2.37-2.69% modeled stored-byte savings | No decoder; excludes predictive formats from its conclusion |
| Temporal reuse across different layer weights | [Grouped/alternating order](2026-10-26-maxwell-layer-reuse-screen.md): changes within +/-0.45% intervals | No material tested order bias |

These are historical experiments, not newly pooled measurements. Their
controls, scopes and limits remain in the linked reports. The checksum
reader's roughly 94% of gate/up latency supports caution about arithmetic
substitutions, but is **not a lower bound** or pure-memory timing.

Fully pre-expanding gate/up into FP32 would remove per-weight conversion
and scale multiplication, but increase the pair from 47,906,816 to
**180,355,072 bytes, 3.765x**. Across 36 layers, FP32 pairs alone occupy
6,492,782,592 bytes. Replacing current split copies adds 4,768,137,216
resident bytes; keeping both adds the entire FP32 allocation. Original
GGUF source is still needed by the qualified prefix path and cannot be
quietly subtracted from the accounting. No claim about actual DRAM traffic
or predicted latency follows, but conversion removal alone is insufficient
justification for that expansion. FP16 pre-expansion is not generally a
lossless replacement for the original FP32 scale/weight product.

The [qualified middle-token profile](2026-10-26-maxwell-packed-stage-audit.json)
still places gate/up at **8.045 ms / 41.11%** of captured GPU stages and down
at **4.771 ms / 24.37%**. Those are overlapping instrumented scope evidence,
not permission to equate weight-byte savings with HTTP speedup. This audit
changes no profiler or runtime code and performs no new timed run.

## Validation And Next Admission

Task coverage, scale multiplicity, tensor dimensions and byte formulas were
checked from the frozen records and independently recounted with integer
arithmetic. Current source hashes are in the ledger. Existing
SASS/resource and CUDA parity reports are reused as historical evidence;
no new hardware test is claimed for this documentation-only audit.

Next require either demonstrably avoidable logical work outside these closed
probes, or independent evidence that a proposed organization reduces costly
physical amplification or improves reuse. Include all new resident/peak
bytes, metadata, setup and per-token work. Keep complete-FFN and actual512/
quality/HTTP gates intact. There is no newly admitted kernel in this audit.
Both original **53.06** and **56.037 HTTP tok/s** targets remain unmet.

[Report and ledger checksums](2026-10-26-maxwell-reader-work.sha256).

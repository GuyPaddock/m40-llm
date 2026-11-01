# Thread-Owned Gate/Up Task Pairs

## Independent Hypothesis

The [cross-column packet](2026-10-26-maxwell-warp-packet-screen.md) realizes
wider loads with unchanged registers and improves isolated projections, but
fails complete-FFN scope. Do not retune or integrate that packet. Inspect
the dominant native gate/up mapping independently: it reads a float4,
two u32 payload words and two u16 scales per task, iterating `task += 128`.
Group two tasks already owned by a thread into one uint4 payload load.
Keep its original two separate scale planes, activation reads, arithmetic,
output ownership, reduction and precise SiLU.

This is not the adjacent-K eight-value mapping, cross-column packet,
paired-scale joint layout, virtual/physical cohort, persistent CTA, or
fixed-K specialization. No closed configuration is combined or rescued.

## Fixed Representation And Consumer

- One output/128-thread CTA, unchanged rows 1/4 decode-stream scheduling.
  Support dynamic K divisible by 1024 and any positive output width.
  Reject unsupported shapes explicitly; no K=2048 template or padding.
- Retain original gate and up scale planes, byte-for-byte. Payload records
  are `[column][bundle][tid][G0,U0,G1,U1]`, four original u32 words.
  Bundle covers 256 four-value tasks. Record half 0 uses
  `bundle*256+tid`; half 1 uses `bundle*256+tid+128`.
- Read one uint4 record per two-task bundle. Process its two original
  tasks in original order, each with the original float4 activation and
  two original scales at `task/8`. Keep each four-value scale/FMA sequence,
  original shared publication/reduction and precise SiLU expression.
- Source-level loads per two tasks decrease from ten to seven: two float4
  activations, four u16 scales, one uint4 payload instead of four u32
  payloads. Payload bytes stay 512 per full warp; summed 32-byte address
  buckets stay sixteen. Scale addressing is unchanged. These are not
  measured transactions, occupancy or a performance forecast.
- Versioned typed identity includes both original model/tensor/offset/
  dtype/shape sources. Keep all three planes RAII-owned; validate output
  disjointness, input/plane alignment and ordered producer streams. Enqueue
  performs no allocation/free, copy or synchronization.

## Evidence And Gates

Require CPU inverse roundtrip of all original bits, including arbitrary
scale encodings, signed/zero payloads, supported multiple-K and odd-N cases.
Test source identity, incompatible pairs/shapes, partial upload cleanup,
GPU control-bit and existing CPU parity, capture/ordinary equality, guarded
buffers, distinct rows, stream dependencies, scratch reuse and invalid spans.

Extend the shared captured-input `q8_execution_bench`, not another framework.
Use frozen M02 actual layers 0/17/35, rows 1/4, qualified split-f32x4-128
control, immutable prepared down/vocabulary. Both arms retain original
and candidate gate/up planes; combined extra preparation <=128 MiB per
case. Report packing/upload/inverse costs and host/device bytes separately
from timing. Inspect actual sm_52 loads, registers/spills/shared/barriers
and unchanged control/down instruction words.

Warm control for 1000 GPU ms and candidate for 100 ms with finite existing
budgets; five alternating pairs, ten repeated enqueues per stage. Gate/up,
isolated down and complete FFN remain separate. Existing **>=5% complete
one-row FFN with positive paired lower bound at all three layers** remains
required. Four rows or isolated gains cannot bypass it; do not expand
geometry to rescue losses. Use fresh prescribed confirmation only for
genuinely borderline results, without pooling.

Any passing screen still requires sufficient independent profile scope,
unchanged actual 512 IDs/frozen dense-valid quality/logits and repeated
HTTP gates before retention. Original 53.06 HTTP tok/s and 1.30x fresh
Ollama remain required. No Q8_1/approximation/scheduling substitution,
runtime selector or default change is authorized by the microbenchmark.

## Atomic Delivery

1. `docs(research): bound FFN task pair study`: this contract before code.
2. `feat(cuda): probe thread-owned FFN packets`: typed three-plane owners,
   fixed consumer, shared benchmark integration and behavioral tests;
   depends on the contract. No generation dispatch/cache change.
3. `docs(perf): record FFN task pair screen`: clean release evidence,
   independent audit, compiled resources and unchanged gate decision;
   depends on the tested implementation. Preserve code before removal.
4. Only if every applicable gate passes, separately plan runtime and actual-
   model/HTTP qualification. Otherwise close the fixed configuration.

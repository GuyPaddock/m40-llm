# Qualified FFN Layer-Reuse Sensitivity

## Question

Repeated single-layer FFN intervals revisit the same weights, unlike a
36-layer decode. Existing full-token events average about 0.2235 ms/layer
for gate/up and 0.1325 ms/layer for down, close to repeated-layer controls
around 0.2213/0.1314 ms. That is indirect evidence against a large reuse
bias, not a controlled cache measurement. Test operation order directly.

## Fixed Diagnostic

- Extend the existing frozen-M02 FFN runner; no CUDA, inference, sampling,
  model, backend or scheduling change. Use qualified split-f32x4-128 and
  the original prepared down/vocabulary cache.
- One row only. Reject four-row inputs explicitly because their cross-stream
  dependencies would confound a single-stream ordering comparison.
- Use actual pairs (0,17), (17,35), (35,0). Retain both original gate/up
  preparations and separate scratch in both arms. Combined preparation is
  95,813,632 bytes, below 128 MiB. No alternative weight representation.
- Compare grouped A^N then B^N with alternating (AB)^N. Every interval has
  exactly N calls per layer and identical stage work, inputs, pointers,
  allocation ownership and kernel configurations. All work uses decode.
- Measure gate/up, isolated down and complete FFN separately. Record total
  interval GPU time and normalize by 2*N actual layer operations. Include
  host enqueue gaps and original dependencies; exclude setup and readbacks.
- Warm each order's complete FFN for at least 1000 GPU ms using existing
  finite budgets. Five alternating paired trials, ten calls per layer per
  interval. Do not overlap GPU work, builds/tests or MemPalace with timing.
- Require bit-identical stage outputs between orders, unchanged CPU bounds,
  capture parity, inputs/guards, cache identities, scratch accounting and
  cleanup. No timed allocation/free, copy, sync, cuBLAS or stream wait.
- Preserve original instruction words. No new kernels, cache flushes,
  geometry/policy/register changes or integration of rejected candidates.

## Interpretation

Report signed alternating/grouped latency change and paired confidence
intervals. If a stage differs by at least 3% with an interval excluding zero,
flag temporal reuse/order sensitivity for future benchmark design. If all
intervals lie within +/-3%, close a material bias for these tested pairs.
Otherwise report uncertainty and apply existing fresh-trial rules, without
pooling or changing the diagnostic threshold after observation.

This is independent frozen-input layer work, not a true 36-layer forward:
there is no hidden-state dependency between layers, QKV/attention, logits,
KV or generated tokens. A difference is not pure cache time, measured
traffic or an optimization. It cannot reopen closed candidates, bypass the
original complete-FFN/quality/HTTP gates or satisfy either exact speed goal.

## Atomic Delivery

1. `docs(research): bound FFN reuse sensitivity`: this contract.
2. `test(bench): compare FFN reuse distance`: shared runner, explicit
   one-row selector, work-order/numerical/ownership tests; depends on 1.
3. `docs(perf): record FFN reuse sensitivity`: clean release raw pairs,
   independent audit, limits, status and diary; depends on tested 2.

Qualified exact runtime stays 49.146 HTTP tok/s versus 43.106 official
Ollama. Both 53.06 and 56.037 HTTP tok/s requirements remain unmet.
Runtime defaults, compressed-KV hierarchy and exact model semantics stay fixed.

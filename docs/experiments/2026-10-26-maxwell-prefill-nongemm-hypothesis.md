# Bound Remaining Packed-Prefix Operations

## Question

The existing 64-token prefix profile captures conversion, SGEMM and QKV
bias, leaving about 140 ms of its packed-forward wall scope unattributed.
The recent activity calibration cannot supply engine-stage deltas. Use the
existing session-owned event profiler to measure the remaining M40 groups
before proposing a kernel or rewriting per-row prefix work.

## Fixed Measurement

- Add grouped spans for prefix embeddings, attention norm, Q/K RoPE plus
  exact KV append, packed attention, attention residual, FFN norm, SwiGLU,
  MLP residual, inter-layer hidden copies and final output copies.
- Retain the existing conversion/SGEMM/bias spans. Place events after
  producer-stream waits, so dependency waiting is not relabeled as stage
  computation. Preserve every original CUDA call, order, pointer, position,
  stream, dependency, allocation owner and final drain.
- One span per operation group, never per prefix token. Grow the existing
  event pool only when prefix profiling is requested, bounded to 24 pairs
  per layer plus 16 fixed pairs. No per-token event allocation or extra
  synchronization, external observer, hardware counters or timing subtraction.
- Keep prefix rows distinct from generated-token rows. Include embedding
  host-copy/enqueue time honestly: events bracket a stream interval, which
  can contain host submission gaps, not pure device execution.
- Extend existing profiler/prefix tests. Require unchanged final logits,
  sampled IDs and ordinary enqueue counters, both supported fixture head
  dimensions, correctly labeled grouped spans, ready events after existing
  drains, pool reuse and unchanged explicit profiling-off behavior.
- Use the existing release HTTP comparison with the qualified packed QKV,
  split FFN, prepared down/vocabulary and dequant-SGEMM prefix. The official
  oracle remains untimed. Five serial alternating profiling-off/on pairs;
  canonical 65 prompt IDs, 512 natural generated IDs, fresh dense/context
  1024 KV, immutable 109 prepared identities and isolated GPU boundaries.
- Measure profile perturbation before interpreting stage scope. Over 3%
  total HTTP slowdown, excessive variation, changed actual IDs/counters or
  missing stage coverage blocks quantitative attribution. Keep all raw
  observations; no collector cap/threshold or model/default changes.

## Interpretation And Atomic Delivery

Report each group's GPU interval, host-call duration and counters, plus
union/timeline and remaining wall interval. Do not sum overlapping stages
as latency or call residual wall time CPU-only. Any optimistic E2E scope
uses the ordinary HTTP denominator and is a screening estimate, not a
predicted gain or permission to relax exact-output/quality/HTTP gates.
No new kernel follows unless demonstrated scope supports a bounded test.

1. `docs(research): bound remaining prefill costs`: this contract.
2. `feat(profile): cover remaining prefill groups`: grouped event spans,
   bounded pool and behavior/ownership tests; no kernel changes.
3. `docs(perf): record remaining prefill costs`: completed release pairs,
   independent artifact audit, raw data, limits, status and diary.

The prior goal turn made progress by measuring and rejecting the joint
activity calibration, not by qualifying a faster runtime. Exact status
remains 49.146 HTTP tok/s vs 43.106 official Ollama. Both 53.06 and 56.037
HTTP tok/s requirements remain unmet and active. No closed FFN probe,
approximate/numerical path or scheduler work substitutes for those goals.

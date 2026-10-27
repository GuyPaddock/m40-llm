# Compound Runtime Cost Checkpoint

## Scope

The [qualified compound](2026-10-27-maxwell-compound-runtime.md) reaches
50.910426 buffered HTTP tok/s against official Ollama's 43.194730. Both
original targets remain binding: 53.06 and 56.153149 HTTP tok/s. The last
stage profile used the earlier split FFN, so it cannot establish the
current cost distribution after the compound's 3.205% incremental gain.

Use the existing sampled CUDA-event profiler and comparison runner without
changing inference or profiler code. No new kernel, geometry, graph,
verification, drafter, precision, scheduling or attention experiment is
admitted. The single compound exception and negative verifier stop remain
binding; measurement does not reopen any losing implementation.

## Fixed Measurement

- Restore the qualified quality report's frozen environment. Keep the
  compound, packed-warp-256 QKV/output, prepared down/vocabulary, bounded
  dequant-SGEMM prefix, explicit dense FP16 KV/context1024 and all36 layers.
- Use canonical65 prompt IDs and greedy actual512 tokens, natural length
  stops, the same source model and completed compound quality evidence.
  Official Ollama0.40.11 is an untimed tokenizer oracle only.
- Run the existing five alternating profile-off/on HTTP pairs, after both
  warmups. Sample generated steps0/255/511 only. Do not add timing layers
  or search another token set. Verify outputs, actual-ID replays and actual
  compound dispatch against completed quality.
- Require seven profiled sessions and21 event rows, separating warmup and
  final replay from five primary sessions. Every row must cover all36
  layers,400 ordinary spans and a single vocabulary projection. Preserve
  event-pool reuse and ordinary enqueue counters.
- Record instrumentation perturbation separately. If absolute HTTP latency
  change exceeds3%, do not use this capture for quantitative attribution;
  retain the evidence and stop instead of tuning the collector.
- Serialize all GPU work, request escalation, verify process isolation and
  stable clocks at boundaries. No builds, tests or MemPalace calls overlap
  timing. Keep model construction and joint preparation outside timing.

## Output And Limits

Report first/middle/late stage medians, shares, GPU interval union and token
wall time. Recompute the optimistic request-time reduction needed to meet
each original target from the fresh official checkpoint. Use stage shares
only to bound the scope of future avoided-work proposals, not to claim an
optimization, physical bandwidth, occupancy, stalls or a hardware ceiling.

Do not subtract profiler overhead from throughput, compare separately
instrumented historical stages as paired gains, or promote a kernel from
the profile. A further implementation still needs a distinct specific
avoided-work hypothesis and the existing complete-FFN/quality/HTTP gates.

## Atomic Delivery

1. `docs(research): scope compound cost checkpoint`: freeze this protocol
   before measurement. No executable change; depends on runtime qualification.
2. `docs(perf): record compound stage costs`: publish raw off/on requests,
   event rows, coverage/identity audits, negative controls, derived costs and
   limitations. Update current status/performance history and MemPalace.
   Depends on the completed serial measurement and offline checks.

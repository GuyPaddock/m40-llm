# Exact-Dense Dataflow Feasibility Audit

## Objectives And Controls
Make cross-kernel/dataflow analysis the next task toward **56.153149 HTTP tok/s**. This task measures opportunities and proposes at most one bounded experiment; it does not implement an optimization.

Preserve the immutable **50.910426 HTTP tok/s** reference, joint-only preferred ownership, qualified kernels, numerical contracts and runtime defaults.

Keep the research tracks separate:
- Preserve explicit prompt-copy speculation for the two validated copy workloads. No automatic classification or broader qualification claim.
- Close history lookup, prompt-copy and online backoff as general browser accelerators. Preserve their implementations and negative evidence.
- Assess learned drafting on paper only.
- Defer shared-read server batching; aggregate throughput is a separate objective.

## Dataflow Inventory And Measurement
Use the unchanged Qwen2.5-3B Q8_0 browser workload: 65 prompt tokens, 512 actual greedy output tokens, explicit dense KV, qualified compound FFN, joint-only ownership, prepared down/vocabulary and existing SGEMM prefill.

- Trace embedding through all 36 layers, output normalization, vocabulary, argmax and host token handling. Include prefill, request setup and buffered HTTP completion separately.
- Record intermediate buffers, sizes, producers, consumers, lifetimes, aliases and streams. Distinguish necessary residual retention from avoidable writes, rereads and copies.
- Inventory launches, dependencies, synchronization and allocations. Separate host enqueue time, blocked host time, GPU intervals and uncovered timeline gaps; do not count overlapping intervals twice.
- Inspect residual/norm boundaries, QKV/RoPE/KV boundaries, FFN intermediate handoff, vocabulary/argmax and the Q8 embedding host round trip. These are audit subjects, not authorized fusion candidates.
- Distinguish already-eliminated work from remaining work. Gate/up/SwiGLU and K-RoPE/KV append are already fused; qualified decode already skips cross-stream waits.
- Use existing counters and sampled CUDA events first. Add only missing opt-in host timing or attribution fields, with no new kernels, execution dependencies or synchronization. Preserve existing interfaces and historical reports.
- Sample generated steps **0,255,511**, covering every layer; collect whole-request timing and operation totals across all 512 tokens. Record logical bytes and static repeated-load estimates separately from physical DRAM traffic, which remains unmeasured.

Run five alternating release profiling-off/on HTTP pairs after warmup, with exclusive GPU access and recorded clocks. Use the existing variation-extension and confirmation rules. If instrumentation changes HTTP latency by more than the existing 3% attribution limit, retain the evidence and stop quantitative attribution rather than subtracting overhead.

## Feasibility Decision
Produce an avoided-work ledger for each plausible boundary: measured affected cost, removable operations, retained work, added computation, memory requirements, synchronization constraints and numerical risks.

- Distinguish measured cost from optimistic elimination bounds and projected net savings. Buffer bytes alone do not establish latency savings.
- Explain why proposed fusion would preserve FP32 association, rounding, residual values, causal attention and ownership. Account for cross-block dependencies, duplicated computation, register pressure and spills.
- Evaluate savings against both the unchanged meaningful-improvement gate and the remaining **938.958 ms/request**, approximately **1.834 ms per generated token**.
- Do not treat the historical approximately 0.83 ms/token wall residual as entirely removable CPU overhead.
- Recommend at most one fixed follow-up experiment only if measured evidence supports worthwhile complete-request savings. State whether it could close the full gap or only part of it.
- If no mechanism qualifies, publish a negative feasibility result and stop. Do not reopen rejected FFN layouts, kernel geometries, graphs or scheduling experiments.

No performance promotion occurs during this audit. A later implementation would require separate correctness qualification and controlled HTTP measurement, including a fresh official Ollama comparison before claiming the relative target.

## Learned-Drafter Paper Assessment
After the dataflow checkpoint, assess three mechanism classes: a smaller autoregressive model, auxiliary prediction heads and a causal hidden-state drafter.

- Use primary papers and implementation sources; identify compatible artifacts, vocabulary requirements, training needs and licenses.
- Apply the measured four-row verifier model, including eligibility, conditional acceptance, correction, rollback, initialization, transfers, vocabulary work and contention on the same M40.
- Present acceptance/cost sensitivity rather than treating approximately 92% acceptance at 1 ms as universally sufficient. Mark unavailable costs and acceptance as unmeasured.
- Rank mechanisms and define evidence required before implementation. No training, model downloads, integration or new drafting heuristic is authorized.

## Validation And Atomic Delivery
1. **`docs(research): scope exact dataflow audit`**: freeze workload, controls, measurement boundaries and stop rules; record the research priorities.
2. **`feat(bench): trace qualified token dataflow`**: add only missing opt-in instrumentation and behavioral tests for accounting, interval overlap and invalid captures. Depends on the frozen protocol.
3. **`docs(perf): record exact dataflow findings`**: preserve raw paired results, hashes, buffer/dependency inventory, instrumentation overhead and the bounded go/no-go recommendation. Depends on validated instrumentation.
4. **`docs(research): assess learned draft costs`**: publish the separate paper assessment and measured-model sensitivity without reopening CPU drafting.

Before measurement, verify unchanged compiled CUDA kernels, prepared ownership and actual greedy IDs/counts/stop reasons, existing full-vocabulary checks, valid KV state, frozen task outputs and cleanup. Run relevant warning-free builds, formatting, Clippy and affected CUDA tests with escalation.

Update `docs/research_status.md`, append new evidence to performance history, and write MemPalace diary checkpoints after each completed unit. Preserve historical artifacts unchanged. Stop after the audit and paper assessment for review.

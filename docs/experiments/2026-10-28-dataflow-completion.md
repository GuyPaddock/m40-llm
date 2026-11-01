# Exact-Dense Dataflow Plan Completion Audit

Scope: [the approved plan](2026-10-28-exact-dataflow-plan.md), not attainment
of the still-unmet throughput objective. The bounded measurement can complete
with a negative optimization decision. The paper-only follow-up is required
even though no new speed kernel qualifies.

## Requirement Evidence

| Requirement | Inspected evidence and outcome |
| --- | --- |
| Preserve immutable reference, prepared ownership, kernels and defaults | All 34 manifest checks pass, including original model, five retained binaries and source archive. Diff from `6ca39d6` changes runtime code only in profiler accounting and the H2D timing wrapper. No CUDA, inference, ownership, CLI-default or scheduling edits. All 138 compiled entries match. |
| Separate copy-only, closed CPU methods, learned drafting and batching | Paper assessment and research status preserve the explicit two-copy-task scope, close the three general CPU candidates and defer batching. Direct-verifier and draft-study checksum sets pass, including retained executables. |
| Fixed workload and release configuration | `dataflow-http.jsonl` manifest records clean `30290eb`, release executable hash, exact model hash and frozen environment. Shared runner pins dense KV, compound128/joint-only, prepared down/vocabulary, SGEMM prefix, canonical 65 prompt IDs and 512 actual output IDs with length stop. |
| Trace every layer and output path | Primary event byte ranges select 15 captures at steps 0/255/511, positions 64/319/575, all 36 layers and 400 spans each. The independent audit rejects wrong counts, positions, samples or streams. |
| Buffer sizes, producers, consumers, lifetimes, aliases and streams | Findings ledger covers embeddings, norms, QKV, exact KV, attention partials, projections, residual retention, FFN handoff, ping-pong scratch and logits/argmax. Checked against forward/workspace/session/attention source. Original sources and allocation ownership remain unchanged. |
| Distinguish existing fusion and actual remaining work | Source and counters show gate/up/SwiGLU and K-RoPE/append already fused; no separate decode SwiGLU or cross-stream wait. Findings separate unused allocated workspace slots from actual reads/writes. |
| Non-overlapping host/device accounting | Optional host offsets use a separate clock; interval-union tests and regenerated summary verify overlap handling. Findings explicitly reject summing host blocking time with GPU work or treating the residual as removable CPU cost. |
| Logical traffic versus unavailable hardware counters | Findings label repeated activation loads as static logical estimates. No measured DRAM traffic, achieved occupancy, warp stalls or hardware-ceiling claim. |
| Whole-request setup, prefill, decode and HTTP boundaries | Raw rows retain separate prefill/decode/generation/HTTP timings and whole-request counters. First-token workspace transition is distinguished from steady zero-allocation decode. Independent medians are not forced to add. |
| Five paired trials, warmup, isolation, variation/confirmation | Runner warms both arms and alternates off/on order; all twenty timed boundaries contain one GPU PID at P0/1113/3004. Raw protocol summary reports five pairs, low variation and no extension/confirmation condition. |
| Instrumentation perturbation gate | HTTP latency rises 1.05125%, below unchanged 3%. Ten primary rows and fifteen captures reproduce the committed summary. No overhead subtraction or speed promotion. |
| Avoided-work ledger and one-or-zero experiment decision | Findings assess all named boundaries, measured scope, retained arithmetic, coordination and association risks against 3% and 938.958 ms. No individual supported mechanism qualifies; no kernel is proposed or implemented. |
| Actual IDs, stop/count, logits, valid KV and task outputs | Quality runner compares actual generation with frozen reference; sampled seven full-vocabulary vectors match retained/joint ownership. Off/on replay compares all 512 IDs, those vectors and 10,616,832 valid KV words bitwise. Twelve frozen outputs are unchanged; seven dense-valid, five inconclusive. |
| Ownership, cleanup and unchanged work | Quality reports 109 logical sources, 37 split physical owners, 36 joint pairs and 1,724,645,376-byte saving. Dispatch/counters match both arms. HTTP and quality teardown return tracked bytes to zero. |
| Relevant build and behavioral checks | Four accounting unit tests, three profiler GPU cases, eight runner/protocol tests and 25 affected CUDA tests pass. Non-CUDA warning-as-error tests, CUDA/server all-target Clippy and formatting pass. Seven corrupt evidence cases are rejected. |
| Paper assessment of three learned mechanisms | Separate report cites primary papers, source, licenses and artifact metadata. It identifies tokenizer/precision/training gaps, actual vocabulary storage risk and unmeasured M40 costs. No weights downloaded, training or implementation. |
| Measured verifier model with full overhead and uncertainty | Existing finite-request model regenerated byte-for-byte. Report includes correction, rollback, eligibility, initial setup, feature/vocabulary costs, same-GPU contention and cost/acceptance sensitivity. Projected resident/HTTP scenarios are explicitly not measurements. |
| Rank mechanisms, define future evidence and stop | Paper ranks evidence gathering only, requires compatible causal artifacts and held-out joint acceptance/cost measurements before implementation. No automatic classification, new CPU heuristic or neural integration. |
| Atomic delivery, repository records and memory | Commits below separate scope, instrumentation, runner, compiled identity and measurements. Paper assessment is the final documentation unit. Research status/performance history link results; MemPalace diary records scope, experiment checkpoints and completion. |

## Verification At Close

The dataflow audit and seven corruption controls pass again; regenerated summary
matches the committed JSON byte-for-byte. The preserved SASS reproduces the
138-entry compiled audit byte-for-byte. All eleven dataflow checksums pass,
including the retained measured executable and source archive. The immutable
manifest's 34 checks and all direct-verifier/draft-study checksum entries pass.

The verifier model's five behavioral cases pass, and its regenerated JSON
matches the frozen model. The paper's sensitivity tables are selections from
that JSON, not fitted equations or new measured results. Shape calculations
are explicit multiplication expressions; all neural latency claims remain
unmeasured or labeled analogies.

Archived CUDA results were inspected in `2026-10-28-dataflow-logs.tar.gz`:
three profiler cases, eight runner/protocol cases and the 25 attention,
transfer, embedding, forward, prepared-weight and joint-weight cases all pass
without ignored cases in those runs. No affected CUDA code has changed since
measurement. Four profiler unit cases were rerun at close. Final non-CUDA
warning-as-error tests and CUDA/server Clippy confirm the current source;
formatting and documentation-link checks complete the documentation delivery.
No GPU benchmark was repeated for these documentation-only changes.

Historical measurements and reports remain unchanged. This goal adds no new
Ollama comparison, no claimed speed qualification and no runtime promotion.
The negative dataflow result is limited to the measured boundaries; it does
not rule out a future structural design supported by new evidence.

## Delivery Sequence

1. `b6a8bdf`: freeze dataflow scope, workload and stop rules.
2. `e936e9a`: add opt-in host offsets and H2D timing with behavioral checks.
3. `abee2fd`: add isolated quality-bound profiling-off/on runner.
4. `30290eb`: establish compiled-kernel identity before measurement.
5. `871fa1a`: preserve raw measurements, retained binary/source, independent
   audit, buffer/dependency ledger and negative feasibility decision.
6. Final `docs(research): assess learned draft costs` commit: separate
   primary-source paper assessment, completion audit and status links.

The original plan's instrumentation unit was split into accounting, runner
and compiled-identity commits so measurement dependencies remain inspectable.
All planned deliverables are covered. Stop for review; no next experiment is
authorized by this completion.

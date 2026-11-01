# Direct-Q8 Multi-Row Verification Feasibility

## Summary

Assess whether sharing Q8 weight reads across FP32 activation rows could make exact target verification worthwhile. Deliver a quantitative assessment and, only if justified, recommend **one fixed isolated microbenchmark**. Do not write its CUDA kernel in this task.

Preserve the immutable 50.910426 HTTP tok/s reference, preferred joint-only compound ownership, runtime defaults, exact-output requirements and existing gates. The rejected temporary-FP32/SGEMM verifier remains rejected.

## Evidence And Cost Model

- Audit the original verifier's raw trials, counters and transaction code. Its measured serial cost was 19.311 ms per target row; verification batches cost 171.687/180.576/206.486 ms at widths 2/4/8. Distinguish that historical split-FFN baseline from the current compound runtime.
- Account for seven dequantize-plus-SGEMM projections per layer per batch. Across 36 layers, these cover 2,774,532,096 weights: approximately 2.948 GB of packed Q8 and 11.098 GB of temporary FP32 output per batch. The corresponding FP32 writes and subsequent matrix reads are **logical traffic**, not measured DRAM transactions.
- Inspect every required consumer: QKV, output, gate/up/SwiGLU, down and vocabulary. Identify which existing qualified operations can process rows independently without changing arithmetic, and which need new machinery. Do not assume an existing multi-row kernel preserves qualified numerical association.
- Model widths 2/4/8, including projections, normalization, RoPE, causal attention, KV append, vocabulary, argmax, transfers, stream dependencies, workspace transitions, commit, rollback and correction. Credit no attention or vocabulary reuse unless demonstrated.
- Preserve per-row task assignment, FP32 scale multiplication, accumulation order, reduction tree and SwiGLU arithmetic. Identify compiler-association risks explicitly.
- Quantify registers, scratch and shared memory. The existing compound uses 32 registers/thread and 1,536 shared bytes. A row-sharing design with separate published partials requires:

| Verification rows | Gate/up accumulators per thread | Shared bytes |
| --- | ---: | ---: |
| 2 | 4 | 2,560 |
| 4 | 8 | 4,608 |
| 8 | 16 | 8,704 |

These are design quantities, not compiled register counts or achieved occupancy. Use Maxwell's documented limits and conventional FP32, integer unpacking, vector loads and warp reductions. Assume neither Tensor Cores nor native DP4A. Cite [NVIDIA's Maxwell guide](https://docs.nvidia.com/cuda/archive/12.8.1/maxwell-tuning-guide/index.html).

## Bounded Diagnostic Session

Use the authorized diagnostic allowance because existing evidence does not separately attribute verifier dequantization, SGEMM and transaction costs.

- Add a research-only diagnostic entrypoint to the existing benchmark tool. Use the frozen Qwen model and token trace, explicit dense KV, current qualified compound settings and preferred joint-only ownership. Keep historical loaders and benchmark modes unchanged.
- Run only the original 65-token prefix and widths 2/4/8. Collect five alternating profiling-off/on pairs per width. Separately collect five rejection cycles at accepted prefixes `{0, floor(K/2), K−1}`, deduplicated.
- Reuse existing CUDA-event instrumentation with a sufficiently sized, preallocated diagnostic event pool. Preserve existing profiler defaults; add no per-kernel synchronization. Record complete batch wall time, stage intervals, launch/copy/sync counts, allocation transitions, rollback and correction time.
- Validate actual predictions, causal lengths, retained prefix state, cleared rejected suffixes and teardown. Label batch records as batches, not individual generated tokens.
- Run serially with escalation and GPU isolation. Reject attribution if profiling exceeds the existing 3% instrumentation allowance. Do not retune instrumentation, rerun a broad matrix or treat these observations as performance qualification.

## Break-Even And Recommendation Gate

- Derive progress from the actual cached-next-token and commit protocol. Do not assume `K+1` new tokens per batch. Report full acceptance separately from rejection plus correction.
- For each width, calculate expected cycle cost and committed progress using accepted-prefix distributions. Publish minimum acceptance and allowable drafter cost for break-even, the unchanged variation-aware improvement threshold, and both HTTP targets.
- Include acceptance sensitivity at 0/.25/.5/.75/.9/.95/.99/1 and drafter costs of 0/.25/.5/1/2/5 ms per cycle. Mark these as scenarios, not observed drafter behavior. Distinguish the known first candidate from genuinely drafted tokens.
- Include prefill, HTTP overhead, 511 continuation rows, serial tails and context-dependent attention. Preserve per-trial accounting rather than subtracting unrelated medians. The current targets require approximately 407 ms and 939 ms of request savings.
- Assess only one prospective implementation: **four-row joint gate/up/SwiGLU**, 128 threads, existing joint layout, D=2048 and H=11008. Share each payload load and scale conversion across rows while retaining independent qualified arithmetic and reductions.
- Its proposed comparison has three arms: four qualified one-row calls; the existing four-row consumer without explicit cross-row reuse; and the proposed shared-read consumer. Keep prepared down unchanged and include complete-FFN cost.
- Recommend that experiment only if the accounted model supports meaningful complete-request savings at non-perfect acceptance. Preserve the existing ≥5% complete-FFN gate and HTTP confidence/variation gates for subsequent qualification. A favorable upper bound alone is insufficient.
- Otherwise publish a negative or inconclusive feasibility decision and stop. Even a favorable decision authorizes no kernel, verifier integration or drafter implementation automatically.

## Atomic Delivery And Validation

1. `docs(research): bound direct-Q8 feasibility` records the evidence ledger, formulas, diagnostic limits and plan in repository docs and MemPalace.
2. `test(bench): expose verifier cost anatomy` adds only diagnostic instrumentation and behavioral checks. Verify CUDA gating, ownership, numerical/output behavior, non-CUDA warnings, Clippy and formatting before collecting diagnostic evidence.
3. `docs(research): decide direct-Q8 feasibility` preserves raw diagnostics, reproducible calculations, assumptions, overhead/resource budgets and the single recommendation or stop decision. Update research status and MemPalace.

Do not modify measured artifacts, download models, implement EAGLE, introduce approximations, change defaults or begin a geometry/batch-size search. Any later HTTP qualification must still include a fresh official Ollama comparison.

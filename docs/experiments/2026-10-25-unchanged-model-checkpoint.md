# Unchanged-Model Phase Checkpoint: 2026-10-25

C18 records the exact-engine result and the conditional speculation decision.
It does not declare the overall research plan or either speed objective complete.

## Requirement Audit

| Requirement | Current authoritative evidence | Result |
| --- | --- | --- |
| Same model and prompt | C11 manifest: Qwen2.5-3B Q8_0 SHA `4420ccb0...e0111d`; rendered prompt SHA `3005a73b...648f9a`; 65 oracle-matched actual IDs | Verified |
| Same fixed generated workload | All 12 warmup/screen records: 512 generated tokens, length stop; no EOS suppression or filler | Verified |
| Fresh prompt KV | Every record: 0 reused / 65 evaluated; Ollama native cache diagnostics and zero-common-prefix replacement, m40 reset per request | Verified |
| Serial resident-weight HTTP comparison | Five alternating release pairs on the same M40; completed requests before next inference; no concurrent tests/builds | Verified |
| Trial variation and confidence | m40 MAD/median 0.0981%, Ollama 0.0755%; 10,000 seed-zero paired bootstrap samples; interval retained | Verified |
| Actual output behavior | C11 companion matches measured m40 output and prior qualified 512 IDs; separate engine continuations retained | Verified, not cross-engine ID equality |
| First bounded kernel/layout round | C07 promotion; independent C09/C10 reports; C10A combined HTTP/parity gate; failed MLP/lane kernels preserved before removal | Complete, no prepared promotion |
| Offline n-gram feasibility | C11A actual trace plus five stable fixed regressions; position/visited-round replay; 2/4/8/16 drafts and cost sensitivity | Complete, negative gate |
| Speculative verifier/decoder | C11A best optimistic browser gain 4.949% < required 10% | C12-C17 postponed by the approved gate |
| At least 53.06 HTTP E2E tok/s | Fresh selected m40 median 31.850227 | Not achieved |
| At least 1.30x fresh Ollama | Ollama 43.031982; required m40 55.941577 | Not achieved |
| Separate model research | C19-C27 still require their own data, implementations, measurements, and gates | Remains to be done |

This audit inspects the committed raw records, not only the summary prose.
The complete hashes, flags, actual IDs, clocks, cache lines, and output remain
in the linked artifacts; abbreviated hashes above are labels only.

## Qualified Engine

Keep native Q8 with C07 parallel head128 attention, partitions=8/threads=256,
decode-stream Q8, native fused QKV and gate/up/SwiGLU, device greedy selection,
and native shared-activation vocabulary argmax. Preparation is off. This short
comparison explicitly uses dense FP16 KV/context1024 and full-context attention.
The project's general compressed-KV default is unchanged.

Fresh HTTP medians are 31.850 tok/s m40 versus 43.032 Ollama 0.40.11, with
m40-to-Ollama throughput difference -25.985% and 95% interval
[-26.118%, -25.305%]. Neither timing-extension rule triggers. C07's original
+33.98% matched m40-to-m40 attention result remains its promotion evidence;
the fresh Ollama comparison does not create a different promotion claim.

The two removed implementations remain available in preservation commits:
prepared fused MLP in `e2dbc61`, removed in `9044eea`; coalesced-lane Q8 in
`bb4d30e`, removed in `f5cda0d`. Remaining split/vector/cache work stays opt-in
research. The down/vocabulary-only timing lead is not qualified because its
continuation diverges. No default, approximation, or benchmark workload was
changed to close the speed gap.

## Speculation Decision and Regression Scope

The lookup and offline simulator are retained as tested research tools, not
integrated into generation. No transactional KV, multi-token verifier,
runtime speculative decoder, or adaptive draft sizing is built under the
failed gate. C12-C17 are deliberately postponed, not claimed implemented.

Browser candidate8/16 commits the reference in 485 rounds, with 43 eligible
rounds and 28 accepted draft tokens. The best zero-extra-overhead 1.0x-verifier
estimate is +4.949% E2E throughput, not the required 10%. A 1.5x verifier gives
less than 1%; 2x and 4x lose throughput. These are assumptions, not actual
verification measurements or a universal speculation limit.

The fixed non-browser checkpoint captures arithmetic/OK/config plus explanation
and Java-code prompts twice each. Actual IDs are stable; smoke outputs are
`4`, `OK`, `37`; explanation/code each reach 128 tokens. All 24 case/draft
studies replay the exact reference. Regression generation was an untimed
in-process capture, not a cross-runtime HTTP performance matrix or C19's
calibration/held-out quality corpus. Secondary repetitions do not override the
failed browser gate.

## Next: Separate Model Research

C19 is now the next strict task, as authorized by the negative C11A branch.
Freeze disjoint calibration and held-out inputs before changing thresholds or
layers. Exclude the browser benchmark from calibration. Add teacher-forced
loss/perplexity, task-specific scores, deterministic scoring tests, and explicit
model-variant quality/performance records, then collect dense-reference results.

Only after that foundation should C20's layer masks and C21's fixed 16-of-32
sparse-Q8 kernel experiments proceed in their dependency order. Pruned weights,
activation thresholds, skipped layers, or reduced attention context are model
or semantics changes. Their throughput must remain separate from the unchanged
Qwen/Ollama target, with memory and quality costs reported explicitly. A faster
modified model will not retroactively satisfy either exact-engine objective.

## Evidence

- [C11 keep/remove decisions, timings, and raw HTTP records](2026-10-25-exact-kernel-checkpoint.md),
  committed in `fa58a83`.
- [C11A acceptance, cost assumptions, and raw traces](2026-10-25-ngram-feasibility.md),
  committed in `f799737`; lookup `fe392cd`, tool `739c195`.
- [Baseline inventory, stage costs, and counter limitations](2026-10-24-qwen-baseline-checkpoint.md).
- [Study commands and test contracts](../ngram_feasibility.md).

This checkpoint adds no new timing observations. CUDA/regression capture,
CPU simulator tests, warning-as-error non-CUDA checks, both Clippy lanes, and
formatting were verified in the preceding units. Benchmark and capture
processes have exited; the last device query reports 0 MiB / 0% utilization.

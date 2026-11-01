# Real-model quantized-KV quality

The complete frozen grid records 1,149 generated tokens across 36 generations.
Both quantized arms preserve the seven cases that the dense reference passes.
The other five cases fail in dense mode and remain inconclusive for approximation.
This is a bounded result on six tasks at two contexts, not a model-wide quality
guarantee or a reason to promote another runtime default.

## Complete-generation results

Counts below include emitted stop tokens. The arm order is dense/top64/top8.
All rows stop naturally except the three long arithmetic generations, which
reach their unchanged 16-token limit. Pass/fail is the original full rubric.

| Task/context | Token counts | Dense/top64/top8 | Quantized conclusion |
| --- | ---: | --- | --- |
| Chat, short | 19/19/19 | pass/pass/pass | Preserved |
| Chat, long | 19/19/19 | pass/pass/pass | Preserved |
| Fact, short | 10/10/10 | pass/pass/pass | Preserved |
| Fact, long | 10/10/10 | pass/pass/pass | Preserved |
| Multiple facts, short | 39/39/39 | pass/pass/pass | Preserved |
| Multiple facts, long | 35/35/35 | pass/pass/pass | Preserved |
| Configuration, short | 3/3/3 | pass/pass/pass | Preserved |
| Configuration, long | 11/11/11 | fail/fail/fail | Inconclusive |
| Arithmetic, short | 3/3/3 | fail/fail/fail | Inconclusive |
| Arithmetic, long | 16/16/16 | fail/fail/fail | Inconclusive |
| Garden explanation, short | 82/82/82 | fail/fail/fail | Inconclusive |
| Garden explanation, long | 123/135/150 | fail/fail/fail | Inconclusive |

There are no dense-pass/candidate-fail regressions in this grid. Eleven of
twelve cases have identical generated IDs in all arms, including every short
case. The long garden explanation first differs at generated index 98 for
top64 and index 14 for top8, both zero-based. Top8 omits the drip-irrigation
concept retained by dense/top64. All three still fail the required two-paragraph
format, so this observable content difference is reported without converting
the dense-failed case into a qualified accuracy comparison.

The long configuration answer contains the correct value inside a sentence
instead of returning the integer alone. Short arithmetic returns 53 instead
of 47; long arithmetic runs out of its fixed output budget before answering.
The short garden answer includes all four concepts in one paragraph. None
of these reference failures was repaired by changing prompts or output limits.

## Matched-prefix observations

There are 52 distinct dense-prefix positions, giving 104 candidate comparisons.
Every reference target ranks first in its recorded full vocabulary logits.
These observations are sparse samples of the dense trajectory, not acceptance
rates for the candidates' divergent continuations.

| Context/arm | Positions | Maximum logit difference | Maximum KL | Minimum top10 overlap |
| --- | ---: | ---: | ---: | ---: |
| Short, top64 | 25 | 0.0062883 | 4.610e-7 | 10 |
| Short, top8 | 25 | 0.0062883 | 4.610e-7 | 10 |
| Long, top64 | 27 | 1.97113 | 0.038640 | 9 |
| Long, top8 | 27 | 9.98412 | 0.188754 | 6 |

The no-old controls have identical generated IDs despite small logit differences.
Long-context differences are materially larger, especially for top8, even
where the simple task rubric passes. Do not equate task-score preservation
with numerical equality or general robustness.

## Model and comparison

The [protocol](2026-10-31-kv-quality-plan.md) was committed before inputs or
inference. Exact prompts and token IDs were frozen at `558203f`; source runner
`421df83e085212b78ca0d2e57b91a20ebb879e1e` ran from a clean worktree. The retained
release executable SHA-256 is
`2cd0408f1019df978a45a791a3a750fbf5bad1a6a135fba743104b10f3aafea0`.
The model is the existing Qwen2.5-3B Q8_0 file, SHA-256
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
No input, scoring rule, policy or model setting changes after generation starts.

Each of six original held-out tasks has a short canonical ChatML prompt and a
controlled long prompt with the unchanged task before irrelevant archive notes.
Long prompts contain 2048–2066 tokens, with the task entirely outside the recent
1024-token window. This artificial archive is a controlled probe, not a natural
long-document benchmark or new training data. Original output limits and natural
stops remain in force. Dense failures must remain inconclusive for approximation.

All three arms use identical original weights, native Q8 FFN/attention
projections, prepared split down/vocabulary, explicit packed dequant-SGEMM
prefill, native dense head128 attention and graphs off. Cache capacity is 3,072,
block size 32. The arms are explicit dense FP16, FP16-K/Q4-V direct with top64,
and the same mixed representation with supported plain top8 selection. At most
38 old blocks can be consumed under these prompt/output bounds, so top64 covers
all old blocks. Top8 omits some. Prefill itself uses full causal attention;
the approximation is consumed during one-row decode.

Top64 versus dense includes quantization, mixed-kernel reduction differences
and downstream effects. Top8 versus top64 additionally includes block selection
and its interactions. Different
attention reductions can also produce a no-old discrepancy; short-case logit
differences must not be called quantization error.

## Evidence boundaries

The runner records all generated text/IDs, counts/stops, rubric scores,
repetition diagnostics, actual attention launches, packed-prefill cuBLAS calls,
per-layer cache lengths and old-block counts. It drains and replaces caches
between arms. Matched-prefix observations use only the dense continuation up
to the prediction position, never the future target. Full vocabulary logits
are compared at fixed sparse positions; reported statistics are not corpus
perplexity or a general model-quality guarantee.

The reported cache allocation is 113,246,208 bytes for dense and 116,499,600
bytes for mixed at capacity 3,072. The mixed implementation allocates recent,
old and summary storage separately; this configuration is not a VRAM saving.
These allocation figures are separate from tracked buffer teardown accounting.
The experiment is not timed for throughput: incidental elapsed fields do not
qualify a speed claim, and no runtime default promotion follows.

## Validation before inference

The [runner validation](2026-10-31-kv-quality-runner.md) records 220 passing
non-CUDA tests, seven CPU example tests and all ten release CUDA example tests,
including the synthetic two-layer prefix/coverage/causality/ownership smoke.
CUDA/server Clippy and formatting pass. These checks qualify the runner's
preflight, not a substitute for the real-model result.

The run exited successfully with all 12 case-completion rows, 36 generation
rows and 24 paired-comparison rows. Every layer's cache length and old-block
count passed, all requested attention and prefill counters matched, and tracked
allocations returned from zero to zero. The post-run GPU snapshot is empty.

The [independent audit](2026-10-31-kv-quality-audit.jq) reconstructs task scores,
repetition statistics, token counts/stops, first output differences, cache
coverage, dispatch counts, observation positions and recorded top10 overlap.
Its [result](2026-10-31-kv-quality-audit.json) passes. All twelve corruptions
in the [negative controls](2026-10-31-kv-quality-audit-checks.sh) are rejected.
Eight known scoring/repetition examples also pass. Full logits are not retained
for an independent recomputation of KL; the Rust numerical implementation has
behavioral tests and the report preserves its metrics.

The local jq 1.8.1 executable exhibits an array-slice equality/uniqueness anomaly
in the retained minimal diagnostic. The audit materializes slice elements before
comparison, and the repetition examples plus all 36 reconstructed diagnostics
pass. This is an observed local tool behavior, not a claim about other versions.

See [raw complete results](2026-10-31-kv-quality-results.jsonl), the
[artifact manifest](2026-10-31-kv-quality-artifacts.sha256), and the retained
`local-artifacts/qualified-references/kv-quality/run-01/` directory in the
original repository for source archive, executable and original execution logs.
No new inference run, policy tuning or default change follows from this result.

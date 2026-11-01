# Real-model quantized-KV quality follow-up

## Scope

This is the next bounded investigation in the October 31 runtime objective.
Base is feature commit `b9333e7`; branch is `research/1031-kv-quality`.
The selector and graph follow-ups remain on their independent local branches.
Do not reopen V3 source acquisition, training, or learned-drafter evaluation.
No production default or approximate-attention contract changes in this study.

The original quantized-KV study used captured K/V, synthetic queries and zero
model-generated tokens. It established implementation correctness and operator
costs, not model quality. This experiment uses complete real-model generations
and identical-prefix logit comparisons, keeping approximation effects visible.

## Frozen comparison

Use the existing Qwen2.5-3B Q8_0 file with SHA-256
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
Use all six held-out tasks from `data/research_quality/v1.json`, whose SHA-256 is
`2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38`.
Retain their scoring rules and original output limits, including natural stops.
This is inference evaluation only; no fitting, calibration or training follows.

Prepare two versions of each task before any inference:

- Short: the original task in the existing canonical Qwen ChatML template.
- Long: place the unchanged task first inside `<task>` tags, followed by an
  explicitly irrelevant `<archive>` of deterministic numbered filing notes,
  then request an answer to the task using its requested format. Append whole
  notes until the rendered prompt first reaches 2048 tokens; require fewer than
  2112 tokens and record every prompt byte and token ID. The task must lie wholly
  before the recent-window boundary, checked using its encoded end position.

The archive grammar is fixed in the preparation code, with no task answer,
model output or scoring result as input. Commit its exact manifest and hashes
before inference. Preserve the original corpus unchanged. The wrapper and
artificial archive define a controlled long-context probe, not a general
natural-text benchmark. A dense failure is inconclusive for approximation.

Allocate capacity 3072 and run these arms in fixed order for every case:

1. Explicit dense reference (`KvCompressionConfig::dense_reference()`).
2. FP16-K/Q4-V direct, recent 1024, block 32, plain top-k, top64, no
   representatives/anchors/fallback. Reject any input whose maximum consumed
   length can exceed 3072. At most 64 old blocks can exist, so this arm retains
   all old blocks and isolates quantization plus its downstream effects.
3. The same direct backing and selection with top8, matching the supported
   default policy. Differences from top64 include selection and its downstream
   interactions; do not describe them as an additive isolated operator error.

All arms use identical source weights, native Q8 FFN and attention projections,
prepared split down/vocabulary, native head128 dense attention, graphs off,
and explicit packed-prefix admission with dequantize-SGEMM Q8 prefill. Prefill
uses the existing full-context causal prompt path; compression applies when
consuming the cache for one-row decode. Record the actual path and reject a
silent fallback. Run one live model at a time; replace/reset the owned cache
between arms and drain producers before reading state or freeing it.

## Evidence and decisions

Record full output text, generated IDs/counts, stop reason, rubric result,
prompt IDs, model/tokenizer/configuration/source identities, executed attention
counters, cache allocation/backing, and old-block counts. Confirm that long
cases actually use old data, top64 covers every old block, and top8 excludes
some old blocks. Short cases must stay within the recent window; compare all
three output sequences as a no-old diagnostic and report any mismatch.

For every case, force the dense-generated continuation through each arm using
identical prefixes. Inspect full logits at continuation positions 0, 1, 8, 32,
64, 127 and the final emitted position, deduplicated and bounded by the actual
output. Record max/mean absolute differences, KL(reference || candidate),
top-10 overlap, reference-token rank and NLL. These are sparse observations
on a dense trajectory, not whole-corpus perplexity or recursive sparse quality.
Do not feed a future target into its own prediction. Track token repetition
with a fixed diagnostic: longest consecutive repeated suffix pattern of width
1..16, and repeated 4-gram fraction; preserve raw outputs for interpretation.

Report each long/short task separately. Dense-pass/candidate-fail is a quality
regression; dense-fail is inconclusive even when the candidate passes. Different
IDs are not automatically a task failure. No model-wide quality guarantee or
performance promotion follows from this six-task probe. No policy tuning or
new tasks after observing results. If a correctness or dispatch gate fails,
preserve the partial evidence and diagnose it before interpreting quality.

## Dependency-ordered commits

1. `docs(research): plan real-model KV quality`: this protocol and status.
2. `feat(research): freeze KV quality inputs`: CPU-only manifest preparation,
   behavioral validation and exact manifest before inference.
3. `feat(research): evaluate real-model KV quality`: separate explicit research
   runner using existing generation/scoring APIs, paired-prefix observations,
   synthetic hardware smoke and relevant checks. Keep generic dense evaluation
   contracts intact. Commit verified implementation before the real-model run.
4. `docs(research): record real-model KV quality`: raw artifacts, independent
   accounting audit, limitations and carry-forward decision.

Use release CUDA builds on the idle M40, serial workloads, escalation for tests
and GPU access, and disk-backed targets. Run relevant behavioral CPU tests,
synthetic dense/mixed cache/session checks, warning-free non-CUDA checks, CUDA
Clippy, formatting and link/whitespace checks. These quality runs are untimed
inference experiments; incidental wall time is not a throughput benchmark.

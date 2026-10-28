# Bounded Causal Draft Feasibility Study

Preserve the fully direct verifier, immutable 50.910426 HTTP tok/s reference,
joint-only ownership, numerical/statistical rules and all runtime defaults.
No new CUDA kernel, model download, neural drafter or SGEMM verifier is in scope.

## Fixed Candidates And Data Boundary

1. Historical longest-suffix lookup, unchanged `decode::ngram::propose`.
   Re-evaluate only as a baseline with the four-row verifier's known seed;
   this does not reverse the 2026-10-25 rejection or authorize lookup tuning.
2. Prompt-copy lookup: longest 4-16 token suffix, most recent matching prompt
   occurrence with three known following tokens. Its mechanism is copying
   input text, not predicting novel prose. No history-derived external corpus.
3. Online backoff counts: orders 1-4, most frequent observed next token,
   stable smallest-ID tie break, recursively propose three tokens. Counts
   include only prompt and committed past; the known seed is available as
   context, not a training transition. No confidence-threshold search.

All candidates receive only their owned prompt/past state and one already
computed target token. Future reference IDs are isolated in the scorer.
History updates occur only after simulated target commit/correction, never
after rejected guesses. Each trace starts a fresh state. Full four-row batches
require three guesses; incomplete proposals use ordinary serial decoding.
Stops and final output tails are not padded or manufactured.

Use existing browser512 and twelve frozen tasks as disclosed development data.
Freeze eight new held-out prompts before generating any target output: prose,
code, verbatim-copy and structured extraction, at nominal 128/384 prompt
tokens each. Deterministic neutral record filler reaches the target length;
actual lengths are recorded. Capture each actual greedy continuation twice
with the qualified compound/joint-only/dense serial path, at most 128 tokens,
and require stable actual IDs/count/stop. These are held-out acceptance data,
not a new general model-quality benchmark. Never fit candidates or dispatch
rules on their outputs. Hash source, prompt definitions, executable and traces.

## Measurements And Model

Freeze source/candidates before study. Warm each CPU source once; measure five
complete causal replays, recording construction, every proposal including
misses, candidate assembly, committed-history maintenance and memory estimates.
Exclude only offline future scoring from proposal costs. Report trial variation
and position latency distributions without calling CPU microtiming a paired
GPU/E2E qualification. Use release builds. Do not run CPU timing beside GPU work.

Report position-wise and actually visited-round eligibility, conditional guess
acceptance 1/2/3, full-batch acceptance, rejection index, proposal cost and context
bin jointly. Preserve actual proposed IDs and provenance so causal audits can
reconstruct each decision. A known seed is not an accepted guess. Separate
initialization/history maintenance from proposal-only time; charge both.

Replay actual rejection/eligibility paths through the completed measured
verifier model: measured batch latencies, ordinary correction, rollback, finite
tails and all-in CPU cost. Do not replace the joint observed distribution with
one IID acceptance average. Use all five measured target pairs. Clearly mark
context extrapolation, unmeasured late rejection, and historical HTTP overhead.
HTTP projections are not measured HTTP performance. Include +1/+5 ms target
overhead scenarios and zero-draft-cost ceilings; do not relax original gates.

Rank these measured CPU sources alongside explicitly unmeasured smaller-model
and learned-head alternatives. Unavailable artifacts, unmeasured training or
acceptance, and guessed neural latency cannot qualify a source.

## Conditional Implementation And Stop

Any candidate needs a credible complete-request gain above the unchanged
variation-aware 3% floor, causal/transaction correctness and non-regression on
its declared held-out workload before a research-only live drafter is justified.
A broad candidate must also improve the browser workload; copy-only success
does not establish a general prose accelerator. Never select per-prompt
thresholds from held-out outcomes. If a fixed candidate is promising, implement
only that source behind explicit research selection and use the existing direct
verifier, retaining exact greedy outputs. No default/CLI/server promotion.
Any live performance qualification still needs the established paired GPU
protocol; offline projections alone cannot qualify runtime throughput.

If no source has sufficient acceptance/coverage, preserve code and raw negative
evidence and stop with the ranked report. Do not rescue it with another
parameter search, future leakage or an unmeasured neural drafter.

## Atomic Sequence

1. `docs(research): bound causal draft study` freezes scope/data/gates and diary.
2. `feat(research): add causal draft cost study` adds isolated sources, trace
   capture, release replay/model and behavioral future-isolation tests.
3. `docs(research): record causal draft feasibility` records clean-source capture,
   CPU costs, joint acceptance, independent audit, projections and ranking.
4. Only if the fixed evidence supports it, a separate `feat(research)` commit
   implements the selected research drafter, followed by independent correctness
   and paired evidence commits. Otherwise stop after the negative assessment.

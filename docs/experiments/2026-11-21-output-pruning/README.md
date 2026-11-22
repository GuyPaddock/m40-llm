# Terminal-layer work during packed prefix prefill

## Frozen question

Can an operation that needs only future KV avoid enough terminal-layer work
to justify a Maxwell prototype? The baseline is Phase 4
48c63332462e009964e891be1cabaf139e3b50f8. The historical Qwen2.5-3B Q8_0
M40 cohort has a 64-token packed prefix, a separately decoded last prompt
token, and 512 generated tokens. It includes per-layer event timelines and
a matching profiler-off request control. Use the five primary sessions
(numbered 2 through 6), excluding warmup and token-ID replay.

Trace every caller that requests a hidden vector, continuation, or KV-only
prefix. In the historical profile extract both the final layer's time window
through the final copy and the portion from its attention onward. The wider
window intentionally credits even required work and gaps. Compare with each
profiled prefix and the median unprofiled complete HTTP wall. Report the
individual observations and distinguish opportunity estimates from measured
candidate speed. This is an admission screen, not a mathematical upper bound
on an unprofiled run.

Prototype when a supported operation has dead work and the screen indicates
at least 5% potential time reduction in the prefix or the whole request.
A smaller short-context window closes that cohort. Reopen for another
measured shape/backend if it crosses the threshold. Never omit intermediate
layer state required by later layers. Any prototype must compare all prefix
KV bytes, full subsequent logits and greedy IDs with identical arithmetic,
then measure complete requests with profiling disabled.

Sources: [Phase 4 forward](https://github.com/GuyPaddock/m40-llm/blob/48c63332462e009964e891be1cabaf139e3b50f8/src/infer/forward.rs),
[historical profile](../2026-10-26-maxwell-prefill-nongemm.md),
[upstream KV-only MTP](https://github.com/ggml-org/llama.cpp/pull/31750),
[upstream prefill dependencies](https://github.com/ggml-org/llama.cpp/pull/31356).

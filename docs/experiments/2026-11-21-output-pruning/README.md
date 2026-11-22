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

## Disposition

**PASS for the dependency finding. FAIL for prototype admission on the
65-input/512-output Qwen workload.** The median window from final-layer
attention normalization through output copy was 9.646 ms. It occupied
2.693% of profiled prefix wall and 0.0920% of the 10,482.350 ms median
unprofiled request. Individual fractions were 2.682–2.760% and
0.0906–0.0936%, respectively. The narrower span from attention dispatch
through output copy was 6.398 ms. Neither window represents a measured
speedup. This profile used the historical prepared SGEMM backend. Longer
contexts and other backends have no timing decision from this analysis.
Keep the general-purpose hidden-output API. Reopen only with a relevant
workload and profile that meets the frozen threshold.

| Operation | Input, output and state dependency at Phase 4 |
|---|---|
| Session packed prefix | In `src/decode_session.rs`, `prefill_packed_prefix_ids` passes all but the last prompt ID and a writable output. It advances `processed_len`; normal forward overwrites the hidden output when it processes the held-out token. Every active layer's prefix K/V remains live. |
| Batched HTTP prefix | In `src/server.rs`, `prefill_sequence` likewise omits the last token. `finish_prefill` advances the session without reading the output. The next decode tick computes its own hidden vector before `finish_batch_token` consumes logits. Each request owns a sequence slot. |
| Packed then compress | In `src/decode_session.rs`, the prefix populates temporary dense KV. Import into the compressed cache consumes the prefix K/V. The held-out token supplies the next requested logits. |
| General packed forward | `src/infer/forward.rs::forward_causal_rows` embeds rows, propagates residuals and appends KV in each active layer. Later layers require the preceding layer's attention and FFN output on every prefix row. Only in the terminal active layer of a KV-only call is everything downstream of cache append dead. Normalization, K/V projections, biases and RoPE-adjusted K append remain live. Direct callers can also explicitly request the final hidden vector. |
| Decode and verification | Ordinary target decode needs the final hidden row for logits or top-1. `src/verification.rs::verify` consumes per-row predictions, including the next prediction after an accepted prefix. Its packed and direct backends require hidden outputs for these rows. Dense rollback truncates valid lengths without another layer evaluation. |
| Specialized MTP | Phase 4 has no qualified learned MTP checkpoint or KV-only MTP graph. A separate checkpoint-specific state graph has no validated workload here. |

Remote main e1403e0dbe2037cd8fc2b0af32addf97a4df3488 also has packed
session/HTTP prefix callers and a general hidden-output forward. Its source
must be traced again for any future port. No kernel, feature flag or
production dispatch is proposed for the measured case.

### Replay

The five per-session observations are in [screen.json](screen.json).
Run from this worktree. The first comparison replays the October input,
execution and profiler audit before extracting the final layer.

```bash
set -eo pipefail
base=docs/experiments/2026-10-26-maxwell-prefill-nongemm
out=docs/experiments/2026-11-21-output-pruning
bash "$base-audit.sh" | diff - "$base-audit.json"
gzip -dc "$base-events.jsonl.gz" |
  jq -s --slurpfile audit "$base-audit.json" -f "$out/screen.jq" |
  diff - "$out/screen.json"
```

SHA-256 of the original compressed events is
`06366256fb347fa0c01067fc080501ce5f6fd1ff50119b369058f13025113864`.
The HTTP JSONL is
`06547c93f56219cc1c2975ed09e4a7b525599e99a52c04ea45f8077ea2469599`.

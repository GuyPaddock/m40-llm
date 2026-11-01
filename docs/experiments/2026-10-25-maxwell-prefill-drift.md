# Sequential Prefill Ordering Diagnostic

The large packed-versus-sequential prompt difference is not explained by
FP16 cache rounding. Explicitly completing sequential prefix forwards changes
the sequential result to the packed path's first token. The original
no-head prefix loop has a buffer-reuse ordering defect; preserve its historical
timings, but requalify against a corrected control before recommendations.

## Reproduction

Release `a26748f`, executable SHA-256
`52acba53830a46f87cbd0a7a7665ab8f7e3122319d97ca8024677b39b56978f9`.
Run `model_quality --q8-prefill-drift` on the original Qwen2.5-3B Q8_0 blob
with context 1024 and the frozen M02 environment. Native FFN, prepared
down/vocabulary, dense KV and the canonical 65 ChatML IDs remain fixed.
The diagnostic is untimed and executes only the prompt. The isolated Ollama
server has already stopped and the GPU starts empty.

| Prompt path | Actual first token | Hidden max/mean diff vs session sequential | Full-logit max/mean diff | Top-10 overlap |
| --- | ---: | --- | --- | ---: |
| Session sequential | 4498 | 0 / 0 | 0 / 0 | 10 |
| Manual sequential, no drain | 4498 | 0 / 0 | 0 / 0 | 10 |
| Sequential, drain before each embedding | 39814 | 54.50098 / 1.76412 | 9.95758 / 1.07233 | 6 |
| Session, greedy head at every prefix | 39814 | 54.50098 / 1.76412 | 9.95758 / 1.07233 | 6 |
| Native packed prefix | 39814 | 54.46373 / 1.76412 | 9.95755 / 1.07224 | 6 |
| Session sequential repeat | 4498 | 0 / 0 | 0 / 0 | 10 |
| Dequant SGEMM packed prefix | 39814 | 54.46945 / 1.76412 | 9.95761 / 1.07226 | 6 |

Manual no-drain reproduces every sequential hidden/logit bit. Drained
sequential and per-prefix greedy-head runs match every final hidden/logit
bit with each other. Thus skipping prefix logits is not semantically neutral
under the existing schedule. A repeat returns the old result, excluding a
one-time warmup difference. Packed paths still have small numerical differences
from drained sequential; exact full-continuation parity remains unproven.

## KV and Attention Evidence

Every sampled physical slot reports device length 65. The report compares
256 K and V values at positions 0, 1, 32, 63 and 64 in every layer, and
records a bounded set of raw dimensions. In layer 0, drained sequential and
the old session agree at positions 0 and 64, but all 256 K/V values differ
at positions 1, 32 and 63. Layer-0 K maximum differences reach 5.521484;
V reaches 2.917969. This is not a small softmax reduction-rounding effect.

For each run, the last-layer query and its physical FP16 cache are replayed
through packed attention after widening to FP32, then compared with an
independent FP64 CPU softmax. Cached decode and packed replay each agree with
CPU within 2.51e-6 max absolute error throughout these probes. This does not
prove every earlier layer correct, but rules out a large generic attention
math error on the captured final query/cache. Different paths have different
cache contents; comparing their final logits alone had obscured that fact.

The no-head loop uploads the next embedding into reused `d_x` while the
previous full-layer forward is only enqueued. The upload's default-stream
completion guarantee does not wait for earlier users on nonblocking decode
or prefill streams. Draining before the next upload or consuming the prior
logits provides that missing completion boundary. Correct this ownership
boundary before investigating residual packed precision differences.

## Evidence and Next Action

- [Raw per-path, per-layer KV and attention comparisons](2026-10-25-maxwell-prefill-drift.jsonl)
- [Diagnostic output](2026-10-25-maxwell-prefill-drift.log.gz)
- [Checksums](2026-10-25-maxwell-prefill-drift.sha256)

All tracked allocations return from zero to zero. The diagnostic's three
focused tests pass, including a real head128 fixture with biases and scratch
cleanup; the evaluator suite has 30 passing tests with only the unrelated
explicit-artifact compact-model test excluded. Non-CUDA evaluator tests and
CUDA/server evaluator Clippy pass. These fixture tests did not reproduce the
real-model timing hazard by themselves.

Add an explicit completion boundary for no-logits prefix forwards, with a
queued-work regression test. Repeat this diagnostic and full-model parity
before re-timing or combining packed prefill with an FFN candidate. This is
a correctness repair, not permission to weaken the original output gate or
to claim the old unsafe continuation as the model's intended output.

# Fixed cheap channel signals fail the predictability screen

None of the tested causal policies passes the predeclared screen at 10% or
20% retained channels. All tested target layers fail. Stop these fixed
policies without a selector kernel, projection search or training round.
The supplied-index Q4 execution result remains valid; this study finds no
usable selector among the tested signals on this six-prompt diagnostic set.

## Observations and fixed decision rule

Clean release analyzer source `f2114c2` consumes the
[complete capture](2026-10-31-selector-capture-result.md) from `d6f6ae9`.
The original Qwen2.5-3B Q8_0 model and six existing calibration prompts are
unchanged. All six observer-off/on generations match. There are 128 emitted
IDs and 122 consumed-token rows at each of five layers; target layers are
0/17/35, with predecessor features from 16/34.

The [frozen protocol](2026-10-31-selector-predictability-protocol.md) requires
at least 90% of eligible sparse rows in every applicable target layer to
achieve both 0.80 top-k recall and 0.95 of the oracle's down-norm-weighted
contribution mass. Neither threshold is a model-quality guarantee.
The oracle ranks absolute completed SwiGLU values, with channel-ID ties;
weighted mass is a separate diagnostic at those selections.

There are **zero passing causal rows out of 5,382 policy-row comparisons**.
These are repeated policy evaluations of the same 122 input positions, not
5,382 independent tokens. Even the weighted-mass criterion alone has zero
passing causal rows. Dense refresh successes are excluded from these counts.

## Descriptive recall results

The best mean recall among the fixed causal policies is listed below.
Listing the best observed mean does not select or tune a new candidate.
Each listed causal policy uses refresh interval eight and has 103 eligible
sparse rows. The previous-token oracle diagnostic has 116 eligible rows;
the means therefore do not form a matched-row causal-versus-oracle comparison.

| Layer | Retained | Best causal policy by mean recall | Mean recall | Previous-token oracle mean recall |
| ---: | ---: | --- | ---: | ---: |
| 0 | 10% | Current-input sketch, refresh 8 | 14.77% | 15.99% |
| 0 | 20% | Current-input sketch, refresh 8 | 24.21% | 24.94% |
| 17 | 10% | Current-input sketch, refresh 8 | 21.71% | 28.29% |
| 17 | 20% | Refresh-history frequency, refresh 8 | 29.83% | 35.60% |
| 35 | 10% | Refresh-history frequency, refresh 8 | 49.52% | 48.96% |
| 35 | 20% | Refresh-history frequency, refresh 8 | 47.66% | 48.28% |

Random selection yields approximately 10%/20% recall at the corresponding
retention levels. Layer 35 shows more persistence than early layers, but
still misses the screen by a wide margin. The highest previous-oracle recall
in any row is 70.07%, below 80%; that diagnostic also cannot meet the gate.

Best mean weighted-oracle mass across causal policies is 31.08%/39.57%
at layer 0, 42.44%/48.85% at layer 17 and 74.16%/74.00% at layer 35,
for 10%/20% retention respectively. The best single causal row reaches
94.17%, still below the fixed 95% mass threshold. These nonnegative mass
proxies do not measure output cancellation, logits or generation accuracy.

All per-task summaries remain in the artifacts. The two-token configuration
answer has only one observed input, so causal policies spend it on refresh
and have no eligible sparse row. It is not counted as a successful policy.
The arithmetic and long-photo generations end earlier than the historical
M02 run; no length extension or task exclusion was used.

## Cost interpretation and stop

Refresh-eight policies use 19 observed dense refreshes plus six unobserved
prompt-terminal bootstrap steps, leaving 103 sparse steps out of 128.
Refresh-sixteen uses 12 refreshes plus six bootstrap steps, leaving 110.
Using rounded historical Q8 and selected-Q4 medians gives an optimistic
remaining selector budget of 147.13–151.12 microseconds at 10% retention
and 53.14–57.26 microseconds at 20%, across layers and the two intervals.
These are point estimates; actual selected-index patterns, feature kernels,
top-k, transfer and cache handling were not timed.

The accuracy proxy fails before implementation-cost admission. Following
the frozen stop rule, no masked-output follow-up, CUDA selector, recursive
sparse generation, quality comparison or whole-model speed measurement runs.
Do not label this a measured sparse-model accuracy failure or a hardware
ceiling. It bounds previous-label reuse, fixed history frequency and these
16-dimensional refresh-bank sketches on six prompts. Other signals and
training-time changes remain separate questions, not automatic continuations.

## Validation and retained artifacts

The analyzer checks complete positions and paired generations, all model and
corpus identities, the complete original weight-region digest and all fifteen
captured tensor digests. Nine CPU behavioral tests cover its arithmetic and
causal access. An end-to-end skipped-label mutation changes the diagnostic
previous oracle while leaving causal predictions unchanged.

The independent [streaming audit](2026-10-31-selector-audit.jq) replays all
8,296 row records and 476 summaries. It verifies sorted unique selections,
recomputes recall from selected indices, checks dense-refresh schedules and
prior label identity, and recounts summary gates. Its
[result](2026-10-31-selector-audit.json) passes. A corrupted-recall control
fails specifically at the index-versus-recall comparison. This audit does not
independently recompute weighted norms/mass; those have behavioral test coverage.

- [Source identities and final decisions](2026-10-31-selector-decision.jsonl)
- [Per-row metrics and source rows](2026-10-31-selector-row-metrics.jsonl.gz)
- [All task/layer/policy summaries](2026-10-31-selector-summaries.jsonl.gz)
- [Checksums](2026-10-31-selector-result.sha256), including build/run logs and negative control

Full selected-channel traces are retained outside Git:

```text
/mnt/array-fastest/home/guyep/playground/2025-12-20-m40-llm/m40-llm/local-artifacts/qualified-references/selector-followup/analysis-01.jsonl
72,043,335 bytes
SHA-256 1a445be5b367e7ee9e2a9ccb688661ed28b841952cc2957c231510d7869e64e2
```

Reproduce from clean analyzer source `f2114c2`, source `scripts/dev-env.sh`,
build `cargo build --release --no-default-features --locked --example channel_predictability`,
then provide the original model, captured inputs and report with `--model`,
`--inputs`, `--capture-report` and a fresh `--output` path. Run
`jq -n -f docs/experiments/2026-10-31-selector-audit.jq ANALYSIS.jsonl`
from a checkout containing this audit. CPU analysis took 23.20 seconds wall
time with peak RSS 126,136 KiB; those are offline analysis costs, not inference.

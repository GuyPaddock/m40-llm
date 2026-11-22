# Mixed KV on 48 Qwen prompt/context cases

**PASS for top-64 answer preservation when all historical blocks fit.
FAIL for top-8 answer preservation.** A third verdict, for model-wide
quality at longer contexts and across other weights, is **INCONCLUSIVE**.
The [generation review](continuations.md) gives the full denominators,
the three losses and two instruction-following failures missed by the
automatic scorer. No speed inference follows from this quality run.

The M40 completed 144 generations, 96 arm comparisons and 3,346 sampled
tokens. The independent [generation audit](generation-audit.json) checked
the frozen scores, stop reasons, repetition, each physical layer length,
packed prefill dispatch, attention launch counts and teardown. Tracked
device bytes returned to their initial zero. The raw report SHA-256 is
`5ae8a7edba03e51543f43d454738cb169dd038b4557190e7c854fa79df222791`.

| Context and historical budget | Identical complete ID traces | Matched-prefix greedy IDs | Median KL | P95 KL | Median / minimum top-10 overlap | Maximum absolute logit error |
|---|---:|---:|---:|---:|---:|---:|
| Short, top-64 | 24/24 | 82/82 | 1.02e−10 | 2.27e−7 | 10 / 9 | 0.01074 |
| Short, top-8 | 24/24 | 82/82 | 1.02e−10 | 2.27e−7 | 10 / 9 | 0.01074 |
| Long, top-64 | 16/24 | 89/92 | 0.000543 | 0.04941 | 10 / 8 | 2.54272 |
| Long, top-8 | 11/24 | 87/92 | 0.009980 | 0.69942 | 9 / 5 | 11.28994 |

Each matched-prefix observation compares all 151,936 vocabulary scores
on the same dense-generated history. The P95 is the nearest rank among
the observed prefixes. These fixed positions and tasks are a diagnostic
sample. They cannot estimate a population failure rate. The higher maximum
KL for top-8 (2.4953 on the first rewrite token) coincided with a correct
answer from that arm on a dense-failing prompt. Numerical drift and task
correctness require separate measurements.

The [comparison data](comparisons.json) include first token divergence,
per-vector errors, block ranges and case IDs. They summarize the measured
run's numerical values. Independent recomputation from the retained FP32
arrays is a separate audit. Replay the summary from this worktree:

```sh
dir=docs/experiments/2026-11-21-kv-quality-result
gzip -dc "$dir/report.jsonl.gz" |
  jq -s -f docs/experiments/2026-11-21-kv-runs/quality-summary.jq |
  cmp - "$dir/comparisons.json"
```

The [raw report](report.jsonl.gz), [execution log](run.log.gz),
[preflight](preflight.txt.gz) and [host load](host-load.log.gz) retain the
source, sm_52 build, executable, model and input hashes, flags, hardware,
generation, cache and dispatch measurements. The model was Qwen2.5-3B
Q8_0, capacity 3,072, recent window 1,024, block size 32. The [input
freeze](../2026-11-21-kv-inputs/README.md) retains prompt bytes and IDs.

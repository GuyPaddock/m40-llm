# Calibration-Ranked Layer Removal

## Decision

C24 is complete as a model-changing research study. Keep every mask opt-in.
The one-layer mask `[8]` preserves all seven dense-valid task passes across
the two frozen splits, but increases held-out perplexity by 3.46%. Removing
two or three layers gives more matched-step speed at larger quality cost.
No mask changes generation/server defaults or satisfies an exact-engine goal.

The next experiment is C25 compact export of the explicitly evaluated `[8]`
variant. This least-structural-change candidate was recorded before held-out
evaluation, not selected by trying more masks against held-out answers.
Compact export must agree with the masked source, not with the unchanged
36-layer model. C24 masks reclaim neither uploaded weights nor KV capacity.

## Provenance and Isolation

- Clean measured revision `68f1b15be403bd10ad267cc3266b860de169c969`.
- Release executable SHA-256
  `b7204ebb9a6dc01f3c247b3c4d016060b002f6f14ef02f4fd036ec2a9feacfd5`.
- Qwen2.5-3B-Instruct Q8_0 source SHA-256
  `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
  The local Ollama blob has that `sha256-` filename; no source bytes change.
- Frozen C19 corpus SHA-256
  `2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38`.
  Six tasks and two loss passages per split; 102 calibration and 101 held-out
  scored tokens. No browser or held-out input fits a mask.
- Base execution SHA-256
  `faf6939dfd4c3447858236bd4f76dfe9b12857f6bc0678f4708881ee2c31b890`.
  Native Q8_0, explicit dense KV, context bound 1024, head128 parallel
  attention split8/256, graph off, no C23 activation transform.
- Rust 1.95.0, NVCC 12.8.93, g++ 14.3.1, CUDA `-O3 -std=c++17`, sm_52 SASS
  and compute_52 PTX, cuBLAS enabled, no `--use_fast_math`.

Calibration and all four quality runs execute serially on Tesla M40 24GB.
No other inference, tests, builds or MemPalace embedding work overlaps timed
GPU operations. GPU checks show no compute application and 0 MiB before
calibration and after the quality matrix; temperature changes from 24 to
43 C. Clocks are not locked. Every completed run returns tracked device bytes
to zero. Exact flags, prompts, token IDs, stop reasons and identities are in
the raw manifests rather than inferred from variant labels.

## Frozen Calibration

Read-only capture measures `1 - cosine(input, output)` on actual FP32 hidden
vectors, accumulating in host FP64. All 36 original layers have 102 scalar
observations from only the two declared teacher-forced calibration passages.
The final unforwarded target is excluded. First/last layers 0/35 are recorded
but cannot be removed.

Iterative loss ranking evaluates every remaining interior layer, not a BI
shortlist. Each trial resets session/KV/workspace state and scores identical
observed prefixes. Select minimum token-weighted calibration mean NLL, with
BI then original ID as deterministic tie-breakers. All 34+33+32 = 99 trials
validate before freezing the artifact and accessing held-out results.

| Stage | Trials | Removed Original ID | Frozen Mask | Calibration PPL | Next-Best NLL Margin |
| --- | ---: | ---: | --- | ---: | ---: |
| 1 | 34 | 8 | `[8]` | 31.458426 | 0.001336 |
| 2 | 33 | 24 | `[8,24]` | 31.439754 | 0.037127 |
| 3 | 32 | 25 | `[8,24,25]` | 33.360452 | 0.049165 |

The dense calibration control exactly reproduces C19's NLL sum
352.0826424885246 and PPL 31.556847839056097. Every applied mask later
reproduces its winning calibration loss exactly.

BI alone would not pick the first loss winner. Layer 8 is 22nd in ascending
dense BI, while the lowest-BI layer 21 gives single-removal calibration PPL
33.106407. The first-stage best losses are layers 8/30/24/19/25, with PPL
31.458426/31.500485/31.916302/32.227826/33.073606. The narrow first-stage margin
is calibration evidence, not proof that layer 8 generalizes optimally.
No alternate mask is refitted after seeing held-out results.

The frozen [ranking artifact](2026-10-25-layer-ranking.json) SHA-256 is
`a9f0f2cc8c952eacaf96d6fe2576e30a35ef225a5f85f4574630545af3ab4d3c`.
Its hash and masks were recorded in MemPalace before variant evaluation.
Capture retains 29,376 bytes of FP64 observations and two 8,192-byte hidden
vectors, 45,760 logical float bytes under a 64 MiB host budget. This is not
peak host memory. The 99 trials total 307.122 ms setup and 279,375.359 ms
diagnostic loss wall time; loss includes full-logit D2H and CPU scoring, so
neither value is a decode throughput measurement.

## Quality and Logit Drift

| Mask | Calibration Tasks | Held-Out Tasks | Calibration/Held-Out Dense-Valid Regressions | Calibration PPL | Held-Out PPL | Held-Out PPL Change |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `[]` | 3/6 | 4/6 | 0 / 0 | 31.556848 | 39.612159 | reference |
| `[8]` | 4/6 | 5/6 | 0 / 0 | 31.458426 | 40.984621 | +3.46% |
| `[8,24]` | 4/6 | 3/6 | 0 / 1 | 31.439754 | 46.184633 | +16.59% |
| `[8,24,25]` | 2/6 | 4/6 | 1 / 1 | 33.360452 | 49.373639 | +24.64% |

The three calibration and two held-out dense-failed tasks remain inconclusive
for model-change conclusions, even if a candidate passes. Aggregate 4/6 on
the three-layer held-out variant hides a dense-valid regression and an extra
pass on a dense-failed task. It does not mean quality is preserved.

| Mask | Calibration Logit Max/Mean Absolute Difference | Held-Out Logit Max/Mean Absolute Difference | Held-Out Mean Top-10 Overlap |
| --- | --- | --- | ---: |
| `[]` | 0 / 0 | 0 / 0 | 10.0000 |
| `[8]` | 7.544676 / 0.544497 | 8.317498 / 0.446589 | 8.1782 |
| `[8,24]` | 10.238578 / 0.777033 | 11.658968 / 0.632990 | 7.7723 |
| `[8,24,25]` | 11.894620 / 0.916003 | 12.038500 / 0.800287 | 7.4059 |

Within the current empty-mask control, full-vocabulary logits are identical
for all 203 teacher-forced positions. All four reference runs also match
C19's twelve generation traces and recorded per-token loss statistics.
C19 does not archive every full logit vector; do not claim a full-vector
comparison against historical C19 data.

The task regressions are exact-format/stop failures with facts still visible:

- `[8,24]`, held-out workshop: all three requested facts appear in JSON,
  but the model opens a Markdown JSON fence and omits its closing fence.
  Reference/candidate generate 39/38 tokens, both stop naturally, with first
  generated divergence at step 36. The strict frozen JSON grader reports
  format failure, not proof that facts disappeared.
- `[8,24,25]`, calibration depot: the correct `room: 18` is inside an
  unclosed JSON fence. Reference/candidate generate 11/14 tokens.
- `[8,24,25]`, held-out tour: correct Saturday/West Entrance facts appear
  inside another unclosed JSON fence. Reference/candidate generate 19/22
  tokens. Both regressions stop naturally and diverge from the reference
  at generated step 0.

Do not relax the grader or choose a different mask to erase these failures.
The one-layer held-out PPL is much better than C20's fixed `[17]` probe,
47.656541, but that is a same-corpus quality comparison, not a paired
historical speed claim.

## Matched-Prefix Timing

Five alternating serial pairs per split time 42 calibration or 43 held-out
steps after eight untimed warm-prefix inputs. Each prediction consumes the
same observed prefix in both variants; sampled predictions are never fed
back. Timings include forward, vocabulary projection, greedy readback and
final completion of both GPU streams. KV reset, session allocation and warmup
are outside the interval. All per-model predictions are stable across trials.

| Mask | Calibration Reference/Candidate ms | Calibration Paired Rate Gain | Held-Out Reference/Candidate ms | Held-Out Paired Rate Gain |
| --- | --- | ---: | --- | ---: |
| `[]` | 1163.600 / 1162.769 | +0.066%, noise | 1192.404 / 1192.287 | +0.017%, noise |
| `[8]` | 1166.204 / 1137.062 | +2.580% | 1196.050 / 1167.428 | +2.484% |
| `[8,24]` | 1166.791 / 1109.319 | +5.166% | 1197.232 / 1140.051 | +5.116% |
| `[8,24,25]` | 1165.724 / 1080.913 | +7.848% | 1195.592 / 1107.760 | +7.937% |

Times are independent medians; gains are medians of paired throughput ratios.
Across the eight reference/candidate groups, MAD/median ranges from 0.0054%
to 0.0642%. Five pairs are a bounded research checkpoint, not broad confidence
or a promotion decision. The one-layer rate gain is below the exact-engine
3% screen, which is not relaxed. These are matched-prefix model-compute
measurements, not generated/HTTP tokens per second or unchanged-model wins.

Free-generation rows retain prefill/decode/total wall times, actual counts
and stop reasons. For example, the 110-token workshop prompt gives:

| Mask | Pass | Generated Tokens | Prefill ms | Decode ms | Generation Total ms |
| --- | --- | ---: | ---: | ---: | ---: |
| `[]` | yes | 39 | 2752.938 | 1056.874 | 3809.825 |
| `[8]` | yes | 35 | 2686.195 | 923.467 | 3609.674 |
| `[8,24]` | no | 38 | 2614.336 | 980.780 | 3595.129 |
| `[8,24,25]` | yes | 39 | 2538.385 | 980.537 | 3518.935 |

All four stop naturally. Different output lengths/styles make these unequal
work; do not derive an engine speedup from this single free-generation row.
Caller admission-to-return times are retained separately in JSONL and include
additional shared-runner work.

## Allocation and Preparation Costs

Every variant still uploads 3,279,519,744 weight bytes and allocates 37,748,736
dense KV bytes per model. Active layers are 36/35/34/33, but physical source
capacity remains 36 layers. There is no prepared-weight cache, FP32
materialization or activation-sparsity allocation. Head128 attention adds
66,560 reusable bytes per model after execution.

The reported combined tracked total, 6,559,588,368 bytes after execution,
includes both resident models and workspaces. It is neither a standalone
model footprint nor measured peak VRAM. Raw per-model fields distinguish
uploaded weights, KV and workspace. Terminal tracked bytes are zero.

Reference/candidate cold-load intervals for counts 0/1/2/3 are
5913.372/6165.483, 5855.257/5951.478, 5954.174/6108.600 and
6114.283/6246.045 ms. They include parsing/read/upload/dense KV/validation,
not fresh-process cold I/O, preflight artifact hashing or later identity
hashing. No separate complete artifact-validation/cold-first-token timer is
available, so do not report zero preparation cost.

## Reproduction and Validation

Use [the ranking-runner commands](../layer_ranking.md#frozen-ranking-runner).
The exact environment is under `configuration.environment` in every quality
manifest. Freeze/hash the completed calibration artifact before running the
empty control and `--remove-layer-count 1/2/3` with `--split all` and
`--step-timing-trials 5`. Run all GPU operations serially.

Post-matrix validation passes 47 hardware tests with cuBLAS on: influence,
mask, teacher-forced scoring, attention grid, projection/MLP wrappers and
full-layer forward. The cuBLAS-off run passes 18 hardware checks and all ten
shared-runner tests. The implementation also passes both warning-as-error
CI lanes, CUDA/server and non-CUDA all-targets Clippy, and fmt. GPU-less CI
does not establish GPU correctness. The explicitly requested observer test
with `CUDA_VISIBLE_DEVICES=''` fails with `m40llm_create_context returned null`
and Cargo exit 101, rather than skipping successfully. After validation the
GPU again has no compute application and 0 MiB allocated.

## Raw Records

- [Frozen artifact](2026-10-25-layer-ranking.json),
  [107-record calibration study](2026-10-25-layer-ranking-study.jsonl),
  [calibration log](2026-10-25-layer-ranking-study-log.txt).
- [Empty-mask quality/timing](2026-10-25-ranked-layer-empty.jsonl),
  [log](2026-10-25-ranked-layer-empty-log.txt).
- [One-layer quality/timing](2026-10-25-ranked-layer-one.jsonl),
  [log](2026-10-25-ranked-layer-one-log.txt).
- [Two-layer quality/timing](2026-10-25-ranked-layer-two.jsonl),
  [log](2026-10-25-ranked-layer-two-log.txt).
- [Three-layer quality/timing](2026-10-25-ranked-layer-three.jsonl),
  [log](2026-10-25-ranked-layer-three-log.txt).
- [Independent audit and derived summary](2026-10-25-ranked-layer-summary.jsonl).
- [cuBLAS-on validation](2026-10-25-ranked-layer-validation-cublas-on.txt),
  [cuBLAS-off validation](2026-10-25-ranked-layer-validation-cublas-off.txt),
  [expected no-device failure](2026-10-25-ranked-layer-validation-no-device.txt).

Each quality file has 36 records, including a terminal completion record,
twelve task rows, four passage rows and ten paired timing trials. Artifact
and source identities are bound into every positive variant manifest. The
summary is derived evidence; raw rows remain authoritative. No mask,
threshold, attention policy or original model file is promoted or changed.

# C19 Dense Model-Quality Reference: 2026-10-25

C19 is complete. This checkpoint freezes the quality measurement foundation
before any layer, sparse-weight, or activation tuning. It does not change
runtime defaults, modify the model, or satisfy either exact-engine speed goal.

## Inputs and Scope

The [raw JSONL](2026-10-25-model-quality-dense-reference.jsonl) contains 24
records: manifest, split identities, separate warmups, 12 task comparisons,
four passage comparisons, two summaries, and successful cleanup. The run used
runner commit `f974162` with a clean worktree and release Rust build.

- Model/reference are independent loads of Qwen2.5-3B Q8_0, SHA-256
  `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
  Architecture `qwen2`, 36 layers, hidden 2048/FFN 11008, 16 query/2 KV heads,
  head128. No layers or weights are changed.
- [Corpus V1](../../data/research_quality/README.md) was committed in `d1294fc`
  before observing results. Its SHA-256 remains
  `2a1270959ef105a3ec19e0be0ab888048fb619880fd4f014ce216acfb8710c38`.
  Both disjoint splits are evaluated without changing any input or rubric.
- The selected [C11 engine flags](2026-10-25-exact-kernel-checkpoint.md) remain
  native Q8, fused QKV/MLP, decode-stream execution, and parallel exact head128
  attention with eight partitions/256 threads. Preparation/materialization and
  graphs are off. **Dense reference KV** is explicit, capacity 1024; the global
  compressed-KV generation default is unchanged.
- Canonical Qwen system+user ChatML is used for tasks. Passages are plain text
  without chat formatting or EOS; this tokenizer has no BOS. Full tokenizer
  metadata and actual encoded inputs match. No truncation occurs.
- Tesla M40 24GB, driver 580.159.04, NVCC 12.8.93, Rust 1.95.0, GCC 14.3.1.
  CUDA uses existing `-O3`/C++17/sm_52+compute_52 flags, no `--use_fast_math`.
  Full configuration, executable hashes, toolchains and stop IDs are in JSONL.
- One isolated serial quality run, reference then candidate for each task.
  No other compute process, build, or test overlaps capture. This is neither
  an Ollama comparison nor a paired throughput-promotion experiment.

## Task Results

Reference and candidate generate identical token IDs on **all 12 tasks**.
There are no generated divergence steps or dense-valid candidate regressions.
The fixed rubric passes 3/6 calibration and 4/6 held-out tasks. The other five
are explicitly **inconclusive for pruning conclusions**, not variant failures.

| Family | Calibration | Held-Out |
| --- | --- | --- |
| Chat | Pass, Wednesday/10:00 | Pass, Saturday/west entrance |
| Factual retrieval | Pass, room 18 | Pass, deck 7 |
| Multi-fact | Fail, obsolete Aster pier 2; score 2/3 | Pass, Indigo hall 12/Monday, Saffron hall 16 |
| Code/config | Pass, `6` | Pass, `14` |
| Arithmetic | Fail, explanation reaches 16-token limit | Fail, explanation reaches 16-token limit |
| Long-form | Fail, one paragraph and lexical coverage 3/4 | Fail, one paragraph despite coverage 4/4 |

The arithmetic failures demonstrate instruction/budget limitations under the
frozen setting; they do not establish inability to compute 31/47 with a different
prompt or larger budget. Photo preservation includes the word `labeled`, not
one of the fixed description aliases; this is a surface-rubric miss, not proof
that the semantic concept is absent. Garden output covers every fact but ignores
the two-paragraph requirement. Preserve these outputs and limits; do not widen
aliases, budgets, or formatting rules after observing this checkpoint. A revised
corpus would need a new version and independent recorded reference.

## Teacher-Forced Loss and Drift

Each scored token is predicted from the observed prefix only. Full vocabulary
logits are compared on the same prefix, independently of greedy continuations.

| Passage/Split | Scored Tokens | Mean NLL | Perplexity |
| --- | ---: | ---: | ---: |
| Calibration clock | 50 | 3.648643 | 38.422507 |
| Calibration checklist | 52 | 3.262509 | 26.114980 |
| **Calibration aggregate** | **102** | **3.451791** | **31.556848** |
| Held-out market | 51 | 3.659499 | 38.841884 |
| Held-out recovery | 50 | 3.699166 | 40.413579 |
| **Held-out aggregate** | **101** | **3.679136** | **39.612159** |

Aggregates use total NLL/scored tokens, not averaged passage perplexities.
Reference/candidate NLLs match exactly; max/mean absolute full-logit drift is
zero and every token's top-ten overlap is 10. All 203 scored tokens are retained
with target ID, rank/logit, NLL, and top-ten IDs. This self-comparison verifies
reproducibility/reset behavior; it does not measure a pruned model's quality.
The authored passages are not a standard corpus perplexity benchmark.

## Resident Generation Timing

These are single reference observations in seconds, not benchmark medians.
Prefill means generation admission through the first sampled token; decode
means first through last sampled token. Wall means `generate_text` admission
to return, including detokenization. Loading, hashing, HTTP and queueing are
excluded. Prompt/generated counts and stop reasons are actual, not requested
counts; `stop` includes a sampled stop token omitted from output text.

| Task | Prompt/Generated | Stop | Prefill s | Decode s | Wall s |
| --- | ---: | --- | ---: | ---: | ---: |
| cal-chat-reminder | 86/21 | stop | 2.171 | 0.555 | 2.770 |
| cal-fact-depot | 73/11 | stop | 1.873 | 0.277 | 2.197 |
| cal-multi-delivery | 101/34 | stop | 2.532 | 0.916 | 3.494 |
| cal-config-retry | 80/2 | stop | 2.027 | 0.028 | 2.100 |
| cal-arithmetic-invoice | 61/16 | length | 1.570 | 0.416 | 2.032 |
| cal-long-photo | 69/79 | stop | 1.761 | 2.165 | 3.972 |
| held-chat-tour | 87/19 | stop | 2.199 | 0.501 | 2.749 |
| held-fact-observatory | 69/10 | stop | 1.770 | 0.250 | 2.066 |
| held-multi-workshop | 110/39 | stop | 2.752 | 1.057 | 3.854 |
| held-config-cache | 74/3 | stop | 1.885 | 0.056 | 1.991 |
| held-arithmetic-crates | 58/16 | length | 1.499 | 0.417 | 1.960 |
| held-long-garden | 71/80 | stop | 1.812 | 2.197 | 4.056 |

Candidate measurements and exact internal totals remain in JSONL. No timing
percentage is promoted from one reference/candidate observation. Loss timing
includes interleaved serial evaluation of both models, full-logit D2H and CPU
scoring; it is not ordinary decode throughput. Recorded load setup is 6.595 /
6.278 s reference/candidate, excluding later hashing and not cold disk I/O.

## Memory and Cleanup

Each model uploads 3,279,519,744 weight-payload bytes and reserves 37,748,736
dense-KV bytes, exactly 36 MiB. After forward use each also reports 66,560 bytes
of head128 attention workspace. Prepared/materialized weights are zero.

The Rust allocation tracker reports 6,559,588,368 combined bytes after warmup
for both models/workspaces. This is not standalone peak VRAM and must not be
summed with individual fields indiscriminately. Native allocations such as KV
are accounted separately and are not fully covered by that tracker. These
resident-capacity fields are not active attention traffic or measured DRAM.

The completion record returns Rust-tracked bytes 0 -> 0. A separate post-run
`nvidia-smi` query confirms no compute processes and 0 MiB / 0% GPU use; initial
GPU use was also 0 MiB / 0%. No model or benchmark process is left running.

## Validation and Next Step

CPU loss/scoring/runner tests, full warning-as-error non-CUDA tests, both
all-target Clippy lanes, release build, formatting and diff checks pass.
CUDA/cuBLAS-on validates 43 evaluator/grid/projection/MLP/forward tests;
cuBLAS-off validates 15 evaluator/grid/projection/MLP tests. Four runner tests
also pass, including real tiny-model generation and report preservation.
Raw-artifact validation checks all 12 generated traces and all 203 scored token
comparisons, both summaries, frozen corpus identity, and completion/cleanup.

Next is C20's diagnostic layer mask, followed by C21's separately gated fixed
16-of-32 sparse-Q8 work. Use calibration for choices and retain held-out
quality/loss evidence; failed dense tasks cannot establish pruning quality.
C12-C17 remain postponed by C11A. Exact HTTP targets 53.06 and 55.9416 tok/s
remain unmet; a faster modified model cannot satisfy them retroactively.

See [runner commands and contracts](../model_quality.md). Raw JSONL SHA-256:
`64aad89c204549dd0015aac85ecf5f83643d70b5281264a180b4ead33336e8ad`.

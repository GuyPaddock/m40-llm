# Frozen Drafter Data And Pilot Admission

## Review Decision Requested

Review candidate dated 2026-10-29, based on source `caf4ee1`. This is a plan,
not authorization to collect teacher features, train, switch GPUs or integrate
speculation. The accompanying [data specification](2026-10-29-drafter-data-spec.json)
pins preparation rules and source metadata. It is NOT a completed row-level
data manifest. Dataset records have not been downloaded or selected.

Approve the specification first. Then permit CPU-only source acquisition,
grouping and exclusion checks, and review the realized row manifest before any
teacher generation. Missing document provenance, IDs, hashes or split audits
must block that second approval; an incomplete manifest cannot authorize GPU
work. Review the small pilot's outcome separately before any larger allocation.

The user accepts the [rank256 cost probe](2026-10-29-compact-draft-cost.md):
1.538085 ms median and 1.555598 ms p95. Preserve implementation `c9c1853`, report
`caf4ee1`, resource evidence and executable. This establishes proposal cost,
not learned acceptance or runtime speedup.

The [ILP4 stop](2026-10-29-ilp4-decision.md) remains closed. The original FP32
control failed 101 synthetic dynamic-range SwiGLU tolerance checks; ILP4 did
not fail that fixture. No ILP4 performance or quality regression was
demonstrated. Its admission contract stopped measurement, and this task does
not change that contract, revive its code or propose a rescue.

Preserve compound-runtime-63240c5-v1 at 50.910426 HTTP tok/s, joint-only
ownership, all exact-output requirements and runtime defaults. Keep the
separate exact portfolio out of this study. Use the last official Ollama
43.194730 result; targets remain 53.06 and 56.153149 HTTP tok/s.

## Fixed Target And Feature Contract

Use the original local Qwen2.5-3B-Instruct Q8_0 GGUF with SHA256
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
No alternate quantization, teacher, vocabulary, output precision or model
download. Use the qualified compound consumer, joint-only prepared ownership,
original prepared down/vocabulary, explicit dense reference KV and existing
qualified prefill settings. No approximate mode, packet portfolio or compressed
attention. Record executable, source, environment, tokenizer and tensor hashes
in every realized teacher manifest.

Capture the width2048 FP32 residual after transformer block35, BEFORE output
RMSNorm. The current forward output is `DecodeSession::d_out`; output norm
uses separate scratch in [logits.rs](../../src/infer/logits.rs). Do not substitute
normalized hidden states, embeddings or a later token's output. The fixed
cost-probed head has three independent 2048->256 affine/SiLU maps and one
shared 256->151936 vocabulary projection. It does not consume the seed as an
additional embedding or condition head2/head3 on generated guesses.

For zero-based target tokens x and a prefix ending at absolute position t:

| Field | Exact meaning |
| --- | --- |
| Feature | h_t after processing x_0 through x_t |
| Committed KV length | t+1, every layer, sequence0 |
| Known seed | x_{t+1}, greedy argmax of the ordinary target logits at h_t |
| Guess1 label | x_{t+2} |
| Guess2 label | x_{t+3} |
| Guess3 label | x_{t+4} |

For a prompt of P tokens, the first feature is h_{P-1}, the seed is generated
token y_0, and labels are y_1/y_2/y_3. A feature after consuming y_0 is already
one step too late for this decision. All teacher trajectories use serial
target greedy continuation, not draft-conditioned prefixes or dataset answers.
Absolute positions and causal masks stay unchanged.

Snapshot on the producing stream after forward completion and before scratch
reuse. The exported FP32 row must own its bytes, not retain a mutable session
pointer. Store feature bits, t, committed length, seed, three label IDs and
validity bits, prompt/trajectory identity, stop IDs and stop reason. The seed
and labels never enter the learned feature input. Dataset reference answers
are neither labels nor features.

Include the first target stop ID as a supervised label, then mask all later
labels. Never fill with repeated EOS or invent post-stop continuation. A
terminal known seed creates no proposal. Length-limit tails are censored,
not EOS. Retain zero-valid-label positions in the trajectory ledger but exclude
them from the loss. Do not filter examples by target answer quality or by how
well the drafter guesses them.

Before bulk export, behavioral tests must establish the alignment with distinct
token IDs at each offset, prompt-end capture, each stop position, truncation,
and rejection/rollback boundaries. Changing only future labels must leave
features/proposals unchanged. Compare ordinary versus instrumented target
IDs, full logits and valid KV under the existing contracts; copied features
must be bit-identical to the completed forward buffer. Test interleaved
sessions, producer completion and teardown. These tests are planned, not run.

## Sources And License Review

Only primary metadata/license text was fetched for this plan. Pinned revisions
and SHA256 hashes of those texts are in the data specification. Payload hashes
and row IDs remain unresolved until CPU-only preparation is approved.

| Source | Proposed use | Provenance and restriction |
| --- | --- | --- |
| Databricks Dolly15k | Natural instructions and supplied context | Databricks human-authored data; some Wikipedia context; CC BY-SA3.0 |
| Google Research MBPP, full configuration | Basic Python problem prompts | Google Research/Austin et al.; CC BY4.0; preserve official source splits |
| Original Qwen target | Frozen teacher outputs/features | Qwen RESEARCH LICENSE AGREEMENT, not the runtime repository's code license |

The pinned [Dolly card](https://huggingface.co/datasets/databricks/databricks-dolly-15k/raw/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/README.md)
and [CC BY-SA3.0 terms](https://creativecommons.org/licenses/by-sa/3.0/legalcode)
require retaining attribution, terms and modification records. Resolve the
source document for Wikipedia-derived context; a paragraph hash alone does
not establish document-disjoint splits. If provenance cannot be resolved,
quarantine the record rather than claim isolation.

The [MBPP card](https://huggingface.co/datasets/google-research-datasets/mbpp/raw/4bb6404fdc6cacfda99d4ac4205087b89d32030c/README.md)
and [Google source description](https://github.com/google-research/google-research/tree/master/mbpp)
identify its provenance and official partitions. Use only problem text as
input. Reference code/tests may inform overlap and template grouping but are
never executed, supplied to the teacher or used as labels.

The pinned [Qwen license](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE)
limits its standard grant to non-commercial research/evaluation and specifies
obligations for distributed derivatives and output-trained models. This study
is local non-commercial research only. No commercial-use or redistribution
clearance is assumed. Preserve Qwen, Databricks, Wikipedia and Google notices
and have the combined feature/checkpoint use reviewed before admission. This
is a license checklist, not a legal conclusion that trained weights inherit
one particular data license. The pinned HF target revision identifies license
metadata; it does not establish the GGUF conversion's source revision.

## Frozen Selection And Splits

Proposed pilot size, measured in distinct prompts rather than correlated
feature rows:

| Stratum | Train | Calibration | Locked held-out |
| --- | ---: | ---: | ---: |
| Dolly prose/open QA | 256 | 48 | 96 |
| Dolly grounded QA | 256 | 48 | 96 |
| Dolly extraction/classification | 256 | 48 | 96 |
| MBPP code | 256 | 48 | 96 |
| Total | 1024 | 192 | 384 |

For Dolly, use `instruction` and nonempty `context` only. Exclude brainstorming
and summarization from this first pilot to keep categories fixed. The literal
user content is instruction, followed by `\n\nContext:\n` and context if present.
Prose/open QA uses `creative_writing` and `open_qa`; grounded QA uses `closed_qa`
with nonempty context; extraction/classification uses `information_extraction`
and `classification`. Retain the source category in the realized manifest.
For MBPP it is `text` alone. Use the runtime's QwenChat user/assistant wrapper,
without a system message or an extra BOS; freeze final UTF-8 prompt bytes and
token IDs. Reject embedded chat-control tokens rather than interpret them as
new roles. Do not pad, paraphrase, truncate facts or manufacture filler.

Prompt lengths including ChatML must be 65-447 tokens. Report buckets65-127,
128-255 and256-447. Require at least32 held-out prompts in each bucket across
the corpus and at least16 independent source/template groups per stratum.
Do not require MBPP to supply long prompts it does not naturally contain.
Generate at most128 target tokens, stopping naturally. This stays within the
serial cost model's measured65-575 positions. Allow four-row proposal replay
only at measured batch positions65-569; later rows use serial tail costs.
Report context coverage gaps rather than extrapolate beyond these ranges.

Group BEFORE selecting rows. Use canonical upstream document IDs, substantive
task-template IDs and near-duplicate connected components. Use NFKC, lowercase
and collapsed whitespace for overlap checks without changing emitted prompt bytes.
Connect exact duplicates, shared contiguous32-token spans, or normalized
five-word-shingle Jaccard >=0.5; manually review paraphrases, repeated task
skeletons and unresolved provenance. For code, also group equivalent problem
templates and normalized solution ASTs. Never execute source code. Shared
ChatML and the `Context:` transport wrapper are exempt from template grouping;
shared substantive task templates are not.

Assign whole connected groups to only one split. Stable ordering uses SHA256
of UTF-8 `20261029:` plus canonical group ID, with deterministic quota filling and
an explicit rejection ledger. Keep MBPP source train IDs601-974 in train,
validation511-600 in calibration, and test11-510 in held-out. Exclude few-shot
IDs1-10. Quarantine groups crossing those official partitions. Dolly groups
may fill only one of the three splits. Unresolved document identity or an
unmet quota stops manifest admission; do not silently substitute data.

The realized manifest must list source file hash/revision, row/task ID,
document/template/component IDs, stratum/split, prompt bytes/hash/token IDs,
tokenizer identity, attribution, exclusion decisions and manual reviewer.
Freeze data, tokenizer, split, exclusion-corpus and preparation-program hashes
before teacher generation. Keep held-out features/labels inaccessible to the
trainer and checkpoint selection. New held-out means unused by this project;
we cannot establish that Qwen never saw public MBPP or Dolly during pretraining.

## Benchmark Exclusions

Exclude all existing benchmark prompts, answers, paraphrases and substantive
templates from ALL pilot partitions, not only training. This includes the
browser512 trace, twelve frozen tasks/passages, eight prior drafting held-out
tasks, both copy tasks, KV needle/realistic fixtures and their generated outputs.
The old corpus's `calibration` label grants no permission to use it for training.
Do not use captured FFN inputs, verifier future traces or synthetic probe
coefficients as teacher examples.

The data specification pins known exclusion roots. CPU preparation must expand
them into a reviewed exclusion-content manifest covering tracked fixture
modules and retained experiment JSON/JSONL plus referenced local artifacts.
Reject exact/near overlap using the same group rules, reserved markers such as
`ZXQ-NEEDLE-41729` and `M40_BATCH_LIMIT`, and manually reviewed topic/template
overlap, including browser page loading. Missing referenced fixtures block
the audit. Hash the complete exclusion manifest, not just the seed list.

Repeat overlap checks against generated teacher text before training. Unexpected
verbatim benchmark answers quarantine the whole source group. If this breaks
quotas, stop for a revised manifest review instead of selecting replacement
rows after viewing teacher quality. Preserve quarantined hashes and counts.

Source sufficiency is not yet established. Dolly's document provenance and
MBPP's often-short problem text may leave too few eligible independent groups.
The CPU census must report these losses before R1. Do not pad short problems
or weaken document/template isolation to force the proposed quotas to pass.

## Teacher Throughput And Storage

Upper bound1600 prompts x128 feature positions =204800 rows. Each FP32 feature
is8192 bytes, giving1,677,721,600 bytes or1.5625 GiB before IDs, labels and
metadata. The train upper bound is131072 rows. No full-vocabulary logits are
stored per row; save full logits only for the frozen export correctness sample.
Reserve8 GiB free disk for source records, temporary/final features, manifests,
audit samples and two optimizer checkpoints. Write atomic hashed shards and
do not keep unbounded intermediate copies.

These are decode-only scenarios, NOT measured feature throughput:

| Exported rows/s | 204800-row decode time |
| --- | ---: |
| 50 | 1.138 hours |
| 25 | 2.276 hours |
| 10 | 5.689 hours |

HTTP50.910426 is not a teacher-export rate. Prefill, target load, feature copies,
disk serialization and ownership checks add cost. After manifest approval, a
fixed16-prompt training-only scout, four per stratum, runs for at most0.25 GPU
hours and counts inside a2-hour pilot teacher limit. Report cold setup,
prefill/decode/export times, rows/s, tokens/s, bytes/s and host/device peaks.
Use the slowest observed per-stratum rate plus measured prompt/setup costs and
25% headroom to price the remaining corpus. If it cannot fit the remaining
teacher pilot time and disk budget, stop before bulk export. Do not consume
the20-hour parent allowance to finish this pilot. Reuse the scout's valid rows.

All GPU process time, including setup, failed attempts and validation, counts.
CPU manifest preparation is logged separately, not billed as GPU time. Bounds
are abort limits, not permission to run until exhausted.

## One Fixed Training Pilot

Train only the cost-probed rank256 head, with the target absent from the
training device. Input features remain raw FP32. Use full-vocabulary masked
cross entropy, equal weight per valid head label, no distillation-temperature
search, vocabulary restriction, target updates or additional seed embedding.

Freeze the following schedule before execution:

- Seed20261029; Xavier-uniform matrices and zero hidden biases, no vocabulary bias.
- FP32 parameters/gradients/Adam states; no AMP, TF32 or reduced-precision GEMM.
- AdamW, learning rate0.0003, betas0.9/0.999, epsilon1e-8; weight decay0.01 on matrices and0 on biases.
- Batch128 feature rows; include the last short batch; deterministic epoch shuffle.
- Gradient global-norm clipping1.0, no dropout, no gradient accumulation, constant learning rate.
- Four complete epochs, at most4096 optimizer steps at the row cap; stop at2 GPU hours if incomplete.
- Fixed final-epoch checkpoint; save an initial checkpoint and final resumable state, no best-held-out selection.
- Calibration loss/acceptance after each epoch is diagnostic only; one locked held-out evaluation after epoch4.

Pin the training code, dependencies, device, deterministic settings and tensor
layout in the run manifest before execution. No package/version or optimizer
substitution after seeing results. Check serialization against the existing
FP32 proposal implementation and full vocabulary. If the available framework
cannot run on M40, stop and request an A4000 switch; do not attempt both GPUs.
Every switch, including return to M40, requires user confirmation.

Weights, gradients and two Adam moments occupy647,507,968 bytes. Batch128
three-head FP32 logits occupy233,373,696 bytes before logits gradients,
activations, allocator and library workspaces. These are lower bounds, not a
claimed training peak. Measure peak memory and first32 optimizer steps inside
the2-hour training limit, including initialization. Stop if projected completion
with25% headroom exceeds that limit, outputs/gradients become nonfinite, feature
identities change or deterministic replay fails. No architecture/rank/batch
rescue or extra epochs in this pilot.

## Acceptance And Cost Evaluation

Report training/calibration separately from locked held-out. Evaluate the final
checkpoint once, with no answer-aware routing. Each proposal receives only the
causal feature and permitted past metadata. The evaluator may inspect future
teacher labels only after the proposal is fixed. No live speculative runtime
integration is needed for this offline acceptance study.

Use the preserved M40 proposal consumer with serialized trained parameters and
recorded causal features for acceptance, not an unverified training-framework
argmax substitute. Check parameter layout and sampled full-vocabulary agreement
against the existing independent numerical oracle before held-out evaluation.
This requires a separately confirmed return if training used A4000. It counts
inside the1-hour evaluation cap and adds no target-model generation path.

Replay actual visited decision positions using the existing verifier's progress
rules, including committed correction tokens and terminal stops. Do not count
every overlapping teacher position as an independent generated decision. Report
all-position head accuracy separately, never as realized acceptance.

For each stratum, context bucket and aggregate report:

- Guess1 acceptance and numerator/denominator over eligible valid first labels.
- Guess2 acceptance conditional on guess1 matching, with a valid second label.
- Guess3 acceptance conditional on both earlier guesses matching, with a valid third label.
- Full three-guess acceptance over all eligible decisions AND over uncensored three-label decisions.
- Eligible decision coverage, tokens progressed, rejected-guess histogram, accepted stops and censored tails.
- Proposal cost, setup, coverage/cost jointly, and per-prompt resident/HTTP projections with uncertainty.

Use null for missing denominators. Short terminal sequences are not three-label
failures or successes. Eligibility depends only on known seed/stop state,
remaining requested length, context capacity, feature availability and the
measured-position restriction. Preserve the existing conservative tail rule
`max_tokens - 1 - generated_row >= 4`. Never gate on future EOS, expected
answers or observed acceptance. Report the reasons for ineligibility.

Use the [measured recurrence](2026-10-29-compact-draft-model.pl) and original
[verifier results](2026-10-28-direct-verifier-measurement.jsonl) unchanged as
inputs. A separate versioned evaluator must reproduce the five existing
perfect-draft totals before ingesting learned predictions. Preserve serial
costs, four-row target costs, rejection-specific rollback/correction, setup,
finite tails and stop handling. Do not substitute57.820ms/4 for token latency.
For explanatory probability tables, use unequal conditionals p1/p2/p3 and
weights `1-p1`, `p1*(1-p2)`, `p1*p2*(1-p3)`, `p1*p2*p3`. Primary estimates
come from observed visited trajectories, not independent Bernoulli guesses.

Charge1.5380845ms per attempted learned proposal, and repeat with1.5555975ms
on every attempt. Charge any feature/decision overhead absent from that probe;
do not double-charge its8192-byte D2D snapshot. Ineligible checks are not free:
measure their CPU overhead or conservatively charge a full proposal cost.
Include first-request checkpoint loading separately from warm costs. Repeat
the existing unmeasured+1ms/+5ms per-verification sensitivity. Trained-weight
M40 proposal costs and live cache interference remain unmeasured until later
approval; A4000 training speed is not an M40 inference measurement.

The historical554.537660ms HTTP fixed overhead is a browser-workload scenario,
not each new prompt's measured overhead. Report actual teacher prefill/setup
separately and both prompt-specific and historical512-token projections.
Applying pilot acceptance to a512-token request is an explicit stationarity
assumption, not measured browser acceptance or measured HTTP throughput.
Carry all five paired target-cost samples, not only a favorable median.

As context, equal acceptance at all guesses would need92.5948% at full coverage
or96.0774% at50% coverage for56.153149 HTTP tok/s using median proposal cost.
The full-coverage p95-cost threshold is92.6131%. These are inherited model
scenarios, not acceptance gates independently applied to each head.

## Stops And Review Gates

No existing exact-output or runtime statistical gate is relaxed. These new
pilot criteria decide whether to request further research, not whether to
promote a runtime mode.

1. R0: approve this specification and bounded schedule. Only CPU manifest work follows.
2. R1: review actual source/row/split hashes, provenance, license checklist, exclusions, pinned training toolchain and export test plan. Only then authorize the bounded scout, exporter and pilot. Missing items mean stop.
3. Stop on any causal leak, overlap, invalid ownership, changed target outputs/KV, nonfinite result, cost/resource failure or budget overrun. Preserve partial evidence; no silent retries or new data.
4. An incomplete four-epoch schedule is inconclusive, not a passing trained model. Underpowered acceptance is also inconclusive: require at least256 reached third-guess decisions overall and32 per stratum in held-out, with independent group counts reported.
5. For a positive pilot recommendation, require the final held-out projection's lower95% paired gain bound to exceed the existing `max(3%, twice larger timing variation)` screen on the balanced non-copy aggregate, using p95 proposal cost. Require no negative point gain in any stratum, and a historical512-token median-cost projection reaching56.153149 in all five target-cost scenarios. Otherwise stop with a negative or inconclusive feasibility report.
6. R2: review measured acceptance, uncertainty, leakage audit, GPU ledger, model artifacts and projections. Even a positive result does not authorize the remaining training budget, GPU switching or speculative integration.

Use10000 bootstrap resamples with seed20261029 at independent document/template
group level, preserving within-group tokens and stratifying task families.
Report micro and equal-stratum macro estimates. Carry target-cost pair variation
separately; five historical pairs do not establish new HTTP confidence. The
pilot bound is conditional on that cost model, not a replacement for later
paired release HTTP measurements. No held-out threshold fitting or extra trials
selected after seeing results. Borderline evidence returns for review.

## GPU Ledger And Atomic Delivery

| Phase | Parent cap, GPU hours | Already used | Proposed additional pilot cap |
| --- | ---: | ---: | ---: |
| Proposal cost | 4 | 0.005923 | 0 |
| Teacher/export and export validation | 20 | 0 | 2, including0.25 scout |
| Training and training diagnostics | 40 | 0 | 2 |
| Final acceptance/cost evaluation | 8 | 0 | 1 |
| Aggregate | 72 | 0.005923 | 5 |

Phase caps are non-transferable. Record start/end/device/phase for all GPU
work; replays and failures count. A successful pilot would leave the unused
66.994077 aggregate hours locked, not scheduled. A4000 is optional head-only
training capacity after a confirmed switch; release M40 ownership first.
No GPU work or switching occurs while this plan awaits review.

Future dependency-ordered commits, only after their approval boundary:

| Commit | Purpose and checks | Depends on |
| --- | --- | --- |
| `docs(research): freeze drafter pilot data` | Realized CPU manifest, license/exclusion/group audits and deterministic selection tool with behavioral tests | R0 |
| `feat(research): export causal draft features` | Owned raw-hidden export and alignment/stop/session parity tests; no generation selector | R1 and frozen manifest |
| `docs(research): record teacher pilot gate` | Scout, measured storage/throughput, admitted bulk feature hashes or documented stop | Exporter checks |
| `feat(research): train fixed compact head` | Fixed trainer, serialization/masking/reproducibility tests, no target edits | Teacher gate |
| `feat(research): score causal draft acceptance` | Visited-trajectory evaluator and existing-recurrence reproduction tests, stop/censor/coverage behavior | Frozen schema; before held-out access |
| `docs(research): record drafter pilot outcome` | Final weights/hashes, observed acceptance, projected costs, limits and R2 recommendation | Fixed training/evaluation |

Keep implementation and measured-evidence commits traceable. Retain failed
code before any separately authorized removal. Record checkpoints in MemPalace
and research status. No new performance-baseline measurement entry is warranted
for this planning-only change; original reports and immutable artifacts remain
untouched.

# Learned-Drafter Paper Feasibility

## Decision

**No learned drafter is ready for implementation on the evidence reviewed.**
This is an artifact and cost/acceptance evidence gap, not a proof that neural
drafting cannot work on Maxwell. No model weights were downloaded, no training
ran, and no inference or runtime default changed during this assessment.

The [dataflow audit](2026-10-28-dataflow-findings.md) is complete and recommends
no new speed kernel. The immutable exact-dense reference remains
**50.910426 HTTP tok/s**; both 53.06 and 56.153149 remain objectives, not achieved
results. Joint-only preferred ownership and the fully direct four-row verifier
are preserved. Prompt-copy stays an explicitly selected copy-only experiment;
history lookup, prompt-copy and online backoff remain closed as general browser
accelerators. Shared-read server batching remains a separate aggregate-throughput
research direction, not a single-request speed claim.

## Evidence And Accounting

The [measured verifier model](2026-10-28-direct-verifier-feasibility.md) uses a
57.820 ms complete four-row target, not an isolated FFN estimate. It includes
causal attention, all projections, vocabulary, target transaction costs and
ordinary correction after rejection. One input is a known target seed; only
three tokens are guesses. The remaining 511 input rows include finite serial
tails. Early rejection/correction cycles cost about 75.8-76.0 ms. Late-context
rejection cost is modeled, not measured for every wrong prefix.

The following values are reproduced from the existing
[model JSON](2026-10-28-direct-verifier-model.json), without changing its
recurrence or acceptance gates. `p` is equal conditional acceptance at each
guess position, not an observed average token match rate. Coverage is 100%
except in the explicit eligibility table. All-in proposal cost `D` is charged
on every drafting decision, including an ineligible lookup.

| Assumed p | Assumed D ms | Projected median resident tok/s | Minimum resident gain across five inputs | Full-batch acceptance |
| --- | ---: | ---: | ---: | ---: |
| 0.90 | 1 | 55.714 | 7.53% | 72.90% |
| 0.90 | 5 | 52.572 | 1.48% | 72.90% |
| 0.95 | 1 | 59.913 | 15.60% | 85.74% |
| 0.95 | 5 | 56.421 | 8.88% | 85.74% |

These are hypothetical resident scenarios, **not measured neural or HTTP
throughput**. A resident result above 56 tok/s does not establish the HTTP goal.
The HTTP scenario reserves 554.538 ms historical fixed overhead plus 0.370 ms
extra session setup. It is not a contemporary HTTP measurement. The new
dataflow profiling pairs do not replace those inputs opportunistically.

| All-in D ms | Minimum modeled p for 56.153149 HTTP scenario |
| --- | ---: |
| 0.25 | 91.24% |
| 1 | 92.03% |
| 2 | 93.08% |
| 5 | 96.12% |

At p=0.90 even a free drafter misses the stronger HTTP target. At p=0.95 its
all-in budget is 3.884 ms/decision; at p=0.99 it is 7.964 ms. The perfect-draft
8.999 ms budget is an oracle ceiling, not a real candidate budget. Unknown
additional work of 1/5 ms per eligible attempt raises the D=0.25 ms HTTP
acceptance requirement to 92.29%/96.36%.

| Eligibility at p=0.95, D=1 ms | Projected median resident tok/s | Minimum resident gain |
| --- | ---: | ---: |
| 25% | 54.705 | 5.62% |
| 50% | 57.340 | 10.67% |
| 100% | 59.913 | 15.60% |

These equal-p/constant-coverage scenarios ignore correlations. A real study
must jointly measure conditional acceptance by position, full acceptance,
eligibility, rejection location, proposal cost and context/prompt class.
Do not multiply independently favorable acceptance and timing averages.
Draft prefill, feature capture, cache repair, transfers, vocabulary, allocation
and synchronization are additional costs unless explicitly included in D or
charged once per request. Draft initialization must not disappear into warmup.
Target rejection costs are already charged and must not be charged twice.

## Mechanisms And Compatibility

Primary sources below were inspected on 2026-10-28. Repository `main` links
are discovery references, not pinned implementations qualified for this runtime.
No borrowed implementation is included in this report.

License scope also includes the target, not just the drafter's code. The
[Qwen2.5-3B-Instruct license](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/main/LICENSE)
is the Qwen Research License, not the smaller model's Apache-2.0 license. It
limits the grant to non-commercial purposes, requests a separate commercial
license and contains requirements for distributed models trained from its
outputs. Any later distillation or redistribution proposal needs a review of
the actual target, draft checkpoint and dataset terms; this assessment grants
no new permission to train or distribute them.

### Smaller Autoregressive Model

An independent small model proposes a causal continuation; target verification
and correction determine the emitted result. The original speculative-sampling
paper provides the distribution-preserving algorithm, but our present prototype
is qualified only for exact greedy decoding. Stochastic sampling would require
its own algorithm and validation, not reuse of the greedy claim.
[Leviathan et al., ICML 2023](https://arxiv.org/abs/2211.17192v2).

The official [Qwen2.5-0.5B-Instruct model](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
is a concrete existing small-model candidate, with an Apache-2.0 model card.
Its [configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json)
declares 24 layers, hidden896, FFN4864, 14 query/2 KV heads, head64,
vocabulary151936, tied embeddings and BF16 storage. This is not the qualified
3B head128 shape. Conversion to supported storage, prepared weights and all
operator paths would need separate validation. No compatible local draft
artifact or M40 proposal timing was established here.

Inference: three autoregressive guesses require three draft forward steps,
including vocabulary work, plus prompt initialization and repair after rejected
guesses. Sharing a GPU does not make that weight traffic free or concurrent
with a dependent target pass. Equal vocabulary size is insufficient: token-ID
mapping, added/special tokens, ChatML/BOS/EOS behavior and the exact tokenizer
revision must match or have a proven translation. No mapping equality is
claimed. Off-the-shelf drafting need not train a new model, but its agreement
with this Q8_0 target is entirely unmeasured. A same-family name does not imply
the roughly 92% conditional acceptance needed at D=1 ms.

### Auxiliary Prediction Heads

Medusa adds future-token heads and uses tree verification. Medusa-1 freezes the
target; Medusa-2 also trains the backbone and therefore cannot preserve this
immutable model. Typical acceptance or published tree speedups are not evidence
for our strict greedy, one-seed/three-guess linear verifier.
[Medusa paper](https://arxiv.org/abs/2401.10774v3).

The [reference head implementation](https://raw.githubusercontent.com/FasterDecoding/Medusa/main/medusa/model/medusa_model.py)
uses residual hidden transforms and a separate full-vocabulary projection per
head. Applying that structure to three Qwen-sized heads gives
`3 * 2048 * 151936 = 933494784` projection parameters: **1,866,989,568 FP16
bytes**, before hidden transforms and scratch. Ideal Q8_0 payload would be
991,838,208 bytes, but quantizing a trained head needs its own acceptance test.
These are shape calculations, not measured allocations or required dimensions
for every possible auxiliary-head design.

Our existing target vocabulary stage is 1.564480 ms at the middle sample.
Three equally costly heads would consume 4.693440 ms before hidden transforms,
argmax or integration. This is only a local cost analogy: different precision,
weights, reuse or dispatch can change it. It already exceeds the p=0.95 HTTP
budget, so an auxiliary head is not automatically a sub-millisecond drafter.
The mechanism still avoids running a separate full transformer and is worth
reconsidering if a suitable compact trained artifact and measured costs exist.

The [published artifact list](https://raw.githubusercontent.com/FasterDecoding/Medusa/main/README.md)
contains Vicuna Medusa-1 and Vicuna/Zephyr Medusa-2 checkpoints; it does not
establish a frozen Qwen2.5-3B head compatible with our target. A new head would
need training on separate target continuations or distillation data, followed
by held-out evaluation. Training on the final benchmark traces is excluded.
The [code license](https://raw.githubusercontent.com/FasterDecoding/Medusa/main/LICENSE)
is Apache-2.0; checkpoint and training-data terms must be checked separately.

### Causal Hidden-State Drafter

EAGLE-3 combines target features from multiple layers and predicts tokens
autoregressively, with training that exercises generated intermediate states.
This differs from earlier feature-extrapolation versions.
[EAGLE-3 paper](https://arxiv.org/abs/2503.01840v3).
The [maintainers' source and artifact list](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/README.md)
uses trained target-specific heads and tree setups. Its listed official
checkpoints do not establish compatibility with our Qwen2.5-3B Q8_0 target.
Older Qwen2 code support is not a suitable 3B trained artifact; the README also
warns about FP16 overflow for its Qwen2 path. BF16 assumptions must not be
silently carried into this Maxwell FP32-compute runtime. The
[repository license](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/LICENSE)
is Apache-2.0, with separate checkpoint/data provenance still required.

Inference: target features could avoid a second full transformer, but feature
capture must preserve the exact target execution and originate only from the
committed causal prefix. Three 2048-wide FP32 feature rows would be 24,576 bytes
per captured token. This is a payload estimate, not the entire drafter state or
a measured transfer cost. Projection/attention/vocabulary work, draft KV,
feature retention through target rollback, and post-rejection repair remain.
Acceptance of a head trained against another target precision is unknown.
Tree acceptance cannot be assigned to our linear three-guess proposal.

A bounded exact-model artifact search found no verified compatible Medusa or
EAGLE head. Search-name matches alone are insufficient: for example,
[qwen-2.5-3b-instruct-eagle-ft](https://huggingface.co/jeqcho/qwen-2.5-3b-instruct-eagle-ft)
is presented as a PEFT fine-tune, not a verified speculative head. This is a
bounded search result, not a claim that no compatible artifact can exist.

## Ranking And Reopening Criteria

This ranks future evidence gathering, not implementations authorized now.

| Rank | Mechanism | Credible avoided work | Missing evidence / current decision |
| --- | --- | --- | --- |
| 1 | Compact causal hidden-state drafter | Avoid a separate full transformer by reusing committed target features | Target-specific artifact, legal/data provenance, linear acceptance, precision compatibility and all-in M40 cost unknown; defer |
| 2 | Frozen-target auxiliary heads | Parallel proposals without autoregressive transformer layers | No verified compatible head; full-vocabulary storage/cost is substantial; defer |
| 3 | Independent Qwen0.5B | Smaller per-guess model with an existing public artifact | Three serial forwards, extra KV/prefill/repair and tokenizer compatibility unmeasured; defer |

1. Before an implementation proposal, identify an exact artifact/revision,
   license, target/tokenizer mapping, trained-data provenance, dtype, memory
   budget and a causal interface to the unchanged four-row verifier. Keep the
   original target and prepared down/vocabulary ownership unchanged.
2. Seek proposal-only measurements under separate authorization. Precommit
   calibration and held-out prompts across prose, code, retrieval and copying,
   plus context-length bands. Capture conditional acceptance, coverage and cost
   jointly. No future-token access, benchmark fitting or automatic copy routing.
3. Charge draft initialization, missed proposals, synchronization, source
   maintenance, draft rollback, target correction and final tails. Re-evaluate
   the measured finite-request model; late-context penalties need measurement.
   Stop if credible all-in costs and acceptance cannot pass the original screen.
4. Only a separately authorized live prototype can qualify speed. Require the
   existing full-logit tolerances, exact actual greedy IDs/counts/stops, valid
   KV state, all rejection positions, cleanup and session isolation, then the
   unchanged paired release variation/confidence and HTTP/Ollama rules.

No measured neural candidate clears those prerequisites today. Preserve the
specialized copy capability and negative CPU evidence; stop for review rather
than training, downloading or integrating a speculative runtime.

## Reproduction

The existing `2026-10-28-direct-verifier-model.pl` regenerates the unchanged
JSON used above; its five behavioral cases cover finite tails, correction,
ineligible work and extra costs. Select `curves` with `extra_per_attempt_ms=0`,
`conditional_guess_acceptance=0.9/0.95` and `drafter_ms=1/5` for the resident
table. `minimum_acceptance`, `drafter_budgets` and `eligibility_sensitivity`
provide the remaining rows. No new performance measurements or model equations
are introduced by this paper assessment.

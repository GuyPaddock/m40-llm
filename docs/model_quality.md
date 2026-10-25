# Model-Variant Quality Evaluation

C19 supplies the quality foundation for later layer/weight/activation changes.
It does not tune a variant, change generation defaults, or replace the unmet
unchanged-Qwen HTTP throughput objectives.

## Frozen Inputs

[Corpus V1](../data/research_quality/README.md) contains disjoint calibration
and held-out inputs, with six task families and fixed loss passages in each.
Its data and rubrics were committed before any dense/model-variant result was
observed. Calibration may guide model choices; held-out results report quality
changes. New data or grading rules require a new identified corpus version,
not edits to make a failing baseline pass.

Exact/JSON/concept scores expose partial credit and format failures. They are
deterministic surface rubrics, not a general semantic judge. Retain raw output,
and mark pruning conclusions inconclusive wherever the dense reference fails.
The loss and logit measurements complement these bounded task checks.

## Teacher-Forced Loss

`evaluation::loss::compare_teacher_forced` predicts token i using exactly
ids[..i]. It never samples a token or conditions on the target/future. The
caller supplies the complete tokenization, including any BOS. No implicit EOS,
truncation, sliding window, or position renumbering occurs. Passages must fit
the allocated/model context or fail explicitly.

Token NLL uses FP64 stable log-sum-exp over full FP32 logits. It subtracts the
maximum before exponentiation and computes log(sum_exp_shifted) + max - target
without losing the log partition at huge equal logits. NaN/infinite rows,
empty vocabularies, invalid targets, and changing vocabulary sizes are errors.

Aggregate NLL is weighted by scored tokens, never by averaging passage PPLs.
Perplexity is exp(total_nll / scored_tokens); overflow is explicitly labeled
with perplexity_overflow=true rather than an unexplained null. These definitions
follow the standard [causal-language-model perplexity formulation](https://huggingface.co/docs/transformers/perplexity).
Tokenization/BOS/context choices affect PPL and must match across variants.
The small synthetic corpus is not a WikiText or general benchmark claim.

Each token comparison records the actual target, one-based rank/logit, NLL,
top-ten IDs/overlap, and **full-vocabulary** max/mean absolute logit differences.
Ties use the first vocabulary index, including signed zero. Target-conditioned
drift is measured on the same forced prefix, not after two sampled outputs
have branched. Raw logit shifts can differ while probabilities/loss agree;
retain both kinds of measurement.

## CUDA Ownership

`evaluation::teacher_forced::compare_loaded_models` exclusively borrows two
loaded models, keeping session model pointers live and preventing concurrent
high-level workspace use during evaluation. The models must not share mutable
KV storage. Compressed caches are rejected; evaluation requires explicit dense
reference allocation, not an implicit default or fallback.

Each passage resets both caches and creates one reusable decode session per
model. Existing full-layer forward/logits APIs process the forced prefix
incrementally. Source weights remain immutable. No new CUDA kernel or sampling
path is introduced, and session scratch is released on success or error.
These timings include full logits readback and CPU scoring; they are not
ordinary generated-token latency measurements.

## Validation and Remaining Work

CPU checks cover analytical/uniform/shifted/extreme logits, weighted loss,
overflow, strict-prefix history despite conflicting top predictions, tail-logit
drift, ties, and invalid input. CUDA tests use two independent tiny models,
repeat passages after different histories, reject context/token/aliased-cache
errors, and verify subsequent usability and allocation cleanup.

```bash
source scripts/dev-env.sh
cargo test --no-default-features --test research_quality --test teacher_forced_loss
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test teacher_forced_cuda -- --test-threads=1
```

The next C19 unit is the release evaluation CLI and dense-reference run over
both frozen splits. It must identify source/reference/variant hashes, actual
tokenization and transforms, corpus/split/config hashes, output/task/loss/logit
results, resident memory scope, and prefill/decode/total timing boundaries.
In-process generation E2E must not be relabeled HTTP E2E; single quality
observations cannot establish a throughput promotion. No layer/sparsity tuning
before the complete C19 checkpoint.

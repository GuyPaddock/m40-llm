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

## Release Runner

`examples/model_quality` compares two explicitly named model files with
independent dense caches. It requires Maxwell, native execution weights,
`large-model`, materialization disabled, and graphs off. Supported chat
formatting is Qwen2 or Llama3; unsupported tokenizers fail rather than using a
byte fallback. Full tokenizer metadata hashes, special IDs, vocabulary sizes,
and actual encoded inputs must agree. A different file hash identifies an
external variant but does not prove its pruning provenance.

For the unchanged-Qwen C19 reference, use the selected C11 environment:

```bash
source scripts/dev-env.sh
checkpoint=docs/experiments/2026-10-25-qwen-exact-checkpoint.jsonl
while IFS=$'\t' read -r key value; do
  export "$key=$value"
done < <(head -n 1 "$checkpoint" | jq -r \
  '.environment | to_entries[] | [.key, .value] | @tsv')
model="$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
cargo build --offline --release --features cuda --example model_quality
# Inspect GPU processes first; run no other inference, tests, or builds during capture.
target/release/examples/model_quality \
  --reference-model "$model" --model "$model" \
  --variant-id unmodified-qwen2.5-3b-q8_0 \
  --split all --context-tokens 1024 --output /tmp/model-quality.jsonl
```

The output must not exist. Each completed JSONL row is flushed; a partial run
has no `model_quality_complete` record and must not be reported as complete.
Use `--split calibration` for tuning and reserve `--split held-out` for
identified quality checkpoints. Do not modify V1 inputs/rubrics after results.
Unset `M40LLM_KV_*` diagnostics before evaluation; the runner rejects them.

For the separate C20 model-changing diagnostic, `--skip-layers 17,18` applies
only to the candidate before KV allocation. The reference stays unchanged.
Source hashes, active original IDs and explicit in-process transforms identify
the variant; source weights and physical KV capacity are not compacted.
Optional `--step-timing-trials 5` adds alternating matched-prefix timing pairs,
with eight warm prefix tokens by default. These are forced-prefix greedy
prediction steps, not free-generated or HTTP TPS. See [layer masks](layer_masks.md)
for bounds, supported paths and the frozen one/two/three-layer probe choices.

The manifest records model/corpus/config/executable hashes, source revision,
worktree, toolchains, build flags, device, and selected splits. Split records
hash the serialized selected data. Task rows retain raw text, rendered prompt,
actual prompt/generated IDs, stop reason, fixed score, dense-reference status,
prefill/decode/total and admission-to-return wall time. The first-use warmup is
separate; every task/passage resets KV. Passage rows retain forced IDs and
per-token full-logit/loss comparisons. Split summaries weight loss by tokens.

All generation timings are single quality observations, not paired performance
trials or HTTP E2E. They include request setup and detokenization but exclude
model loading, HTTP, and queueing. Loss timing includes both models' full-logit
readback and CPU scoring, so it is not normal decode throughput. Generated
counts include sampled stop tokens omitted from output. Prefill is admission
through first sampled token; decode is first through last sampled token.

Memory fields identify per-model uploaded weight payload, resident KV,
head128 workspace, prepared and materialized buffers. Tracked device total
includes **both** resident models and shared workspaces; native allocations
such as KV are reported separately and are not fully covered by the tracker.
It is not standalone peak VRAM. Final device accounting must return to its
pre-run value. File load timings include parsing/read/upload/KV/validation and
exclude later hash reads; they do not establish uncached cold-disk latency.

## Validation

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
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --example model_quality -- --test-threads=1
```

Runner tests check reference-dependent conclusions, report preservation,
tokenizer/context preflight, and actual tiny-model dense generation records.
The [real release reference over both splits](experiments/2026-10-25-model-quality-dense-reference.md)
completes C19: identical generated traces/full logits, 3/6 calibration and 4/6
held-out fixed-rubric passes, with dense failures retained as inconclusive for
pruning. No model tuning was used to establish this foundation.

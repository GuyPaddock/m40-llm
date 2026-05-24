# Experimental KV Compression

The compressed KV-cache work is experimental. It exists to measure whether an
M40 can support longer effective contexts by combining exact recent KV with
older compressed summaries or selected exact blocks. The project default is now
compressed KV when the runtime/model supports the preferred exact-old backend.
Dense `off` remains available as an explicit reference/compatibility mode.

This work is inspired by DeepSeek-V4, but it does not reproduce that
architecture exactly:

```bibtex
@misc{deepseekai2026deepseekv4,
  title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
  author={DeepSeek-AI},
  year={2026},
}
```

## Modes

CLI generation accepts:

```bash
--kv-compress-mode off|dense-recent-only|block-select-exact|recent-only|block-summary|block-select-lossy
--kv-recent-window 1024
--kv-compress-block 32
--kv-compress-top-blocks 8
--kv-compress-representatives 0
--kv-compress-representative-policy last|stride
--kv-exact-old-backing dense|q8|fp16-k-q4-v
--kv-exact-old-attention staged|q8-direct|fp16-k-q4-v-direct
```

- default: `block-select-exact` with direct FP16-K/q4-V exact-old backing,
  plain score-ranked top-k, and `top_blocks=8`.
- `off`: dense exact KV reference/compatibility mode.
- `dense-recent-only`: diagnostic dense-KV sliding-window attention over the
  same absolute recent-token range used by the compressed sidecar. It preserves
  absolute RoPE positions and does not renumber the window.
- `block-select-exact`: keeps old exact KV and uses block summaries only as an
  index for selecting old blocks.
- `recent-only`: diagnostic mode that attends only to exact recent KV while
  using the compressed sidecar state.
- `block-summary`: attends to exact recent KV plus old block mean K/V summaries.
- `block-select-lossy`: scores old blocks cheaply, attends to selected lossy
  summaries and optional representatives.

The preferred experimental runtime path is the default compressed-KV path:

```bash
m40-llm generate model.gguf "..." \
  --kv-compress-mode block-select-exact
```

With no KV flags, `generate` and `run` select `block-select-exact`,
`fp16-k-q4-v`, `fp16-k-q4-v-direct`, and `top_blocks=8` when supported. Use
`--kv-compress-mode off` for dense reference mode. Use
`--kv-compress-top-blocks 4` only for efficiency/YMMV runs where minimizing
active KV and decode cost matters more than retrieval robustness. Top16 remains
diagnostic/escalation-only rather than a blanket robustness default. This path
preserves recent K/V as FP16, stores old K as FP16, stores old V as packed
signed q4 with scales, and dequantizes old V directly inside selected-block
attention instead of staging selected old V into FP16.
Compressed modes fail early when the model context/head dimension cannot support
the selected backend, for example when `--kv-recent-window` exceeds the model
context or a head_dim=128 model requests a non-FP16-K/q4-V direct backend.

In the realistic code/config lookup validation, top4 selected the archived
value `M40_BATCH_LIMIT=73`, while default top8 recovered the active value
`M40_BATCH_LIMIT=37`.

Long generations should keep `--max-tokens` within the model context remaining
after prompt tokenization. The CLI rejects requests whose prompt plus requested
generation length would exceed the model context instead of relying on a later
KV-cache failure. For long default-compressed runs, use
`M40LLM_LONG_DECODE_LOG=1` or `M40LLM_LONG_DECODE_LOG=N` to print low-volume
progress and KV configuration state while preserving the normal decode path.
Avoid running multiple large model generations on the same M40 unless you are
explicitly testing contention. The CLI and server emit a best-effort warning
when `nvidia-smi` reports other compute processes on the selected GPU; set
`M40LLM_GPU_BUSY_WARN=0` only when that warning is expected.

Representative storage is opt-in. For `block-summary` and
`block-select-lossy`, `--kv-compress-representatives N` stores up to `N` exact
old-token K/V representatives per compressed block. `last` keeps the last `N`
old tokens per block; `stride` keeps approximately even representatives.

## Quality Harness

The long-context retrieval harness requires an explicit model path:

```bash
M40LLM_LONG_CONTEXT_RETRIEVAL_MODEL=/path/to/model.gguf \
M40LLM_ENABLE_CUBLAS=1 \
cargo test --features cuda --test kv_compression_long_context -- \
  --nocapture --test-threads=1
```

By default it runs bounded 64-token and 512-token old/recent retrieval smoke
cases. Set `M40LLM_KV_QUALITY_TARGETS=512,1024` for an explicit target list or
`M40LLM_KV_QUALITY_FULL=1` for the optional 64/512/1K/2K/4K sweep when the
model context permits.
Set `M40LLM_KV_QUALITY_MAX_TOKENS=<n>` if the retrieval answer needs more than
the default generated-token budget.

Use `M40LLM_KV_QUALITY_REPORT=path/to/report.jsonl` to write one JSONL record
per case. Records include mode, prompt tokens, generated tokens, pass/fail,
output text, prefill/decode/total timing, token rates, KV byte accounting, and
compression ratio. CUDA rows also include
`materialized_f32_cache_entries` / `materialized_f32_cache_bytes`, before/after
prompt cache totals, and prompt/total cache-added deltas so Qwen-like
cross-model runs can distinguish cold FP32 weight materialization from steady
decode/prefill behavior. `attention_compression_elapsed_ms` is currently `null`
unless a per-attention counter is available.

Set `M40LLM_KV_QUALITY_MINIMAL_TELEMETRY=1` for cross-model smoke runs where
diagnostic overhead would dominate. This leaves pass/fail, output, timing, and
KV accounting enabled but does not force selection telemetry or logit traces.

Set `M40LLM_KV_QUALITY_WARMUP_MATERIALIZATION=1` to run one unreported dense
warmup generation before top-k multitask quality rows. JSONL rows then include
`materialization_warmup_elapsed_ms` and `materialization_warmup_prompt_tokens`.
This is diagnostic-only: on Qwen2.5-3B the full-prompt warmup exposed that cold
materialization and dense packed-prefix timing still need to be separated more
carefully before using the first dense row as a steady-state latency number.
Set `M40LLM_KV_QUALITY_WARM_ROWS=1` for the same warmup behavior when the
intent is to label measured rows by `materialized_f32_warm_row`; a row is warm
when it adds no materialized FP32 cache bytes.

## Prefill Experiments

- `M40LLM_PREFILL_CHUNK_SIZE=<n>` enables experimental packed-prefix prefill
  for dense `off` runs when the prompt length is within the bound.
- `M40LLM_KV_COMPRESSED_PREFILL_CHUNK_SIZE=<n>` enables compressed-aware
  chunked prefill while preserving sequential sidecar updates.
- `M40LLM_KV_PACKED_THEN_COMPRESS_PREFILL=1` runs dense packed prefill into a
  temporary dense KV cache, then builds the compressed sidecar from that cache
  for `block-summary` and `block-select-lossy`.

Packed-then-compress reports temporary dense KV allocation separately from the
final compressed allocation.

## Diagnostic Sweeps

`M40LLM_KV_QUALITY_LOSSY_PACKED_SWEEP=1` runs dense `off`,
`block-summary`, and `block-select-lossy` at bounded long-context targets. It
skips `block-select-exact` by default because that mode is diagnostic and
dense-backed.

`M40LLM_KV_QUALITY_REPRESENTATIVES=0,1,2,4` controls representative counts for
lossy packed sweeps. `M40LLM_KV_QUALITY_REP_POLICIES=last,stride` controls
representative policies.

`M40LLM_KV_QUALITY_EXACT_SELECTION_SWEEP=1` includes dense `off`, exact sparse
`block-select-exact`, and lossy modes while recording block-selection telemetry:

- `needle_block_index`
- `selected_block_indices`
- `needle_block_selected`
- `needle_block_rank`
- `total_old_blocks`
- `top_blocks`

`M40LLM_KV_QUALITY_TOP_BLOCKS=4,16` overrides the top-block counts used for
exact/lossy selection.

`M40LLM_KV_EXACT_BLOCK_RETRIEVAL_SWEEP=1` runs the focused summary-indexed
exact-block diagnostic. It defaults to 2048-token old/recent retrieval cases,
compares dense `off` with `block-select-exact`, and defaults
`top_blocks=1,2,4,8,16`. JSONL rows include the active attended KV working set:

- `active_attended_kv_tokens`
- `active_attended_kv_bytes`
- `active_attended_kv_bytes_all_layers`
- `active_attended_old_block_tokens`
- `active_attended_recent_tokens`

`M40LLM_KV_EXACT_BLOCK_STAGING=1` keeps `block-select-exact` semantics but
first gathers selected exact old K/V plus recent exact K/V into compact device
buffers, then attends over that staged working set. `DecodeSession` allocates a
reusable staging workspace for this mode; low-level callers without a session
can still fall back to the older per-call temporary allocation path. This is a
diagnostic bridge to a future q8 exact-old backing store. JSONL rows include:

- `exact_block_staging_enabled`
- `staged_workspace_reused`
- `staged_workspace_bytes`
- `staged_workspace_capacity_tokens`
- `staged_workspace_allocations`
- `staged_kv_tokens`
- `staged_kv_bytes`
- `staged_old_tokens`
- `staged_recent_tokens`
- `staged_position_min`
- `staged_position_max`

`M40LLM_KV_EXACT_OLD_BACKING=q8` switches staged `block-select-exact` to a
hybrid compressed/exact layout. The runtime stores the recent window as exact
FP16, keeps summary/index metadata for block scoring, incrementally quantizes
tokens that age out of the recent ring into q8 old K/V, and dequantizes selected
old blocks into the reusable staging workspace. Dense `off` remains a dense FP16
reference path and does not allocate q8 backing just because the env var is set.
JSONL rows include:

- `exact_old_backing`
- `exact_old_attention_backend`
- `q8_old_backing_bytes`
- `q8_old_backing_scale_bytes`
- `old_k_fp16_bytes`
- `q4_old_v_payload_bytes`
- `q4_old_v_scale_bytes`
- `recent_fp16_bytes`
- `summary_index_bytes`
- `final_kv_allocated_bytes`
- `dense_equivalent_kv_bytes`

`M40LLM_KV_EXACT_OLD_BACKING=fp16-k-q4-v` switches staged
`block-select-exact` to the deployable mixed exact-old layout. The runtime keeps
the exact recent ring in FP16, stores old K in FP16, stores old V as packed
signed q4 with FP32 per-token/per-head scales, and keeps the same summary/index
metadata for selecting old blocks. This mode does not allocate q8 old K/V and
does not allocate the diagnostic dense FP16 shadow. It currently dequantizes
selected q4 V into the reusable FP16 staging workspace before attention.

Set `M40LLM_KV_EXACT_OLD_ATTENTION=fp16-k-q4-v-direct` with
`M40LLM_KV_EXACT_OLD_BACKING=fp16-k-q4-v` to use the experimental direct mixed
attention backend. This keeps old K in FP16, reads old V from the packed q4
backing, and unpacks q4 V inside the attention value accumulation instead of
materializing selected old V into FP16 staging. The direct mixed path preserves
the staged/FP16 selected-context pass/fail pattern at 2048 and the known 4096
rows, and is now the recommended experimental mixed attention backend. It
supports `head_dim=64` and `head_dim=128` models; the older staged/q8 exact-old
and summary/lossy CUDA paths remain `head_dim=64` only. It remains opt-in while
top-block robustness is still under investigation.

Current long-context quality evidence makes direct FP16-K/q4-V the preferred
experimental backend and plain score-ranked top-k the preferred selection
policy. `top_blocks=8` is the recommended compressed-KV default because it is
the safer retrieval/quality setting and passed dense-valid Llama and Qwen
validation. `top_blocks=4` remains the efficiency/YMMV setting with the
smallest active attended KV. `top_blocks=16` is not a safe robustness default
because quality is non-monotonic. Dense `off` remains useful for reference
tests, and compression conclusions should be drawn only from rows where dense
also passes.

Score-cluster evidence is promising but not enough to promote it. The Llama
4096 multi-needle row is useful for selection anatomy, but it is formally
inconclusive because dense `off` fails. The top16 regression appears to be
cumulative support-set distribution shift rather than a single toxic block:
top8 passes, top8 plus individual top16-extra blocks passes, and the cumulative
tail first fails at `[88,57,90]`. In that checkpoint, score-cluster-adaptive
with `min_k=8` avoided the failing top16 tail without regressing dense-valid
rows. A Qwen2.5 2048 follow-up is dense-valid across single-needle,
multi-needle, distractor-needle, and early-fact QA; score-cluster min8/max12
and min8/max16 passed all rows. A later telemetry confirmation fixed Qwen
head_dim=128 selected-block reporting and showed score-cluster min8/max12 uses
top8-sized support on the checked Qwen rows, not top4-sized support. Keep
score-cluster-adaptive opt-in until it is validated across more dense-valid
4096+ prompts and model families.

`M40LLM_KV_SCORE_CLUSTER_DIFF_VALIDATE=1` runs a narrower targeted
score-cluster differentiation suite. It constructs clustered distractor,
boundary distractor, and weak-support multi-needle prompts, then compares top4,
top8, top16, score-cluster min8/max12, and score-cluster min8/max16. JSONL rows
include `selected_support_size`, `selected_support_bucket`,
`active_attended_kv_bytes_all_layers`, `decode_elapsed_ms`, selected block
indices/scores, and `selection_elapsed_ms` where the debug-selection path can
time block scoring. The first Llama/Qwen runs did not find a dense-valid
top9-top12 score-cluster case: all dense-valid targeted rows collapsed to
top8-sized support and passed, so score-cluster remains opt-in rather than a
preferred default.

`M40LLM_KV_REALISTIC_PROMPT_VALIDATE=1` runs a small realistic-prompt suite for
the preferred backend/policy instead of synthetic selection tuning. It compares
dense `off`, top4, and top8 over long chat history QA, document QA over
early/middle/late facts, multi-fact extraction with distractors, and
code/config lookup. Top16 is skipped unless top4/top8 disagree, or unless
`M40LLM_KV_REALISTIC_INCLUDE_TOP16=1` is set. Dense `off` remains the
reference; compressed rows are inconclusive for compression-policy conclusions
when dense fails the same prompt. Use `M40LLM_KV_MULTITASK_TASKS=...` to run a
subset of realistic task names:

- `real-chat-history-qa`
- `real-doc-qa-early`
- `real-doc-qa-middle`
- `real-doc-qa-late`
- `real-multifact-distractor-extract`
- `real-code-config-lookup`

Top-block selection diagnostics are also opt-in:

- `M40LLM_KV_BLOCK_SELECT_POLICY=topk|neighbors|threshold|anchor|anchor-neighbors|explicit|explicit-score-order`
  controls how selected old blocks are promoted after score ranking for direct
  exact-old attention diagnostics. The default `topk` preserves existing
  behavior.
- `neighbors` adds +/-1 old-block neighbors around score-ranked hits.
- `threshold` keeps blocks within `M40LLM_KV_BLOCK_SCORE_DELTA` of the best
  score, with optional `M40LLM_KV_BLOCK_MIN_BLOCKS` and
  `M40LLM_KV_BLOCK_MAX_BLOCKS` caps.
- `anchor` and `anchor-neighbors` always include anchor blocks from
  `M40LLM_KV_ANCHOR_BLOCKS`, defaulting to block 0 when unset.
- `explicit` uses `M40LLM_KV_FORCE_INCLUDE_BLOCKS` /
  `M40LLM_KV_FORCE_EXCLUDE_BLOCKS` for selected-set ablations.
- `explicit-score-order` selects only forced-included blocks while preserving
  score order before optional `M40LLM_KV_SELECTED_BLOCK_ORDER` canonicalization.
- `M40LLM_KV_SELECTED_BLOCK_ORDER=score|chronological|descending` controls the
  physical/materialized selected-block order for order-invariance diagnostics.

JSONL rows include `block_select_policy`, `base_selected_block_indices`,
`policy_added_block_indices`, `final_selected_block_indices`, and the threshold
or cap values when set. These knobs are diagnostic-only; do not use them as a
default policy until the 4096 top-block instability is understood.

`M40LLM_KV_POLICY_DIAG=1` runs the current focused top-block policy diagnostic:
4096-token recent-needle retrieval with dense `off` plus direct
FP16-K/q4-V `block-select-exact`. It evaluates fixed policy cases for `topk`,
small score thresholds, anchor block inclusion, and anchor-plus-neighbor
promotion. Rows include `block_policy_case`.

`M40LLM_KV_ANCHOR_NEIGHBOR_VALIDATE=1` runs a broader 4096 validation matrix
for dense `off` plus direct FP16-K/q4-V `block-select-exact`. It compares
`topk` with `anchor-neighbors` for old/top4 and recent/top4/top8/top16. Use
this before treating anchor-neighbor promotion as a candidate default.

`M40LLM_KV_FALLBACK_DIAG=1` runs the bounded fallback diagnostic for direct
FP16-K/q4-V exact-block retrieval. It keeps top-k as the primary policy and
tests whether answer-agnostic retry gates can selectively expand fragile rows to
top16 without increasing active KV for already-stable rows. The matrix is
limited to old/top4 and recent/top4/top8/top16 at 4096 tokens.

Fallback rows include:

- `fallback_case`: `topk`, `oracle-eot-top16`, `eot-anomaly-top16`,
  `low-margin-top16`, or `score-spread-top16`.
- `fallback_triggered`
- `fallback_trigger_reason`
- `fallback_policy_used`
- `initial_output`
- `initial_active_kv_bytes_all_layers`
- `fallback_active_kv_bytes_all_layers`
- `initial_decode_elapsed_ms`
- `fallback_decode_elapsed_ms`
- `total_decode_elapsed_ms_with_retry`
- `initial_eot_rank_min`
- `initial_eot_logit_margin_min`
- `initial_top_margin_min`
- `score_cutoff_margin`

`oracle-eot-top16` is answer-aware and exists only as a diagnostic baseline.
Runtime policy work should prefer answer-agnostic gates such as EOT confidence
anomalies, low top-token margin, score-spread near the block cutoff, or a fixed
conservative top-block cap when quality is more important than active-KV size.

`M40LLM_KV_FALLBACK_MULTITASK_DIAG=1` runs a bounded multi-task validation
suite for the answer-agnostic fallback gates. It uses direct FP16-K/q4-V
`block-select-exact` retrieval, compares dense `off`, top-k only, and top16
fallback rows for EOT-anomaly, low-margin, score-spread, and a combined gate.
The default target is 1024 tokens so the suite is routine enough to run; set
`M40LLM_KV_QUALITY_TARGETS=4096` for the heavier long-context version. The suite
includes single-needle, multi-needle, distractor-code retrieval, early-fact QA,
early-fact summary, and a normal long-chat smoke. JSONL rows include
task-specific score, fallback trigger metadata, active KV before/after, retry
decode time, final KV allocation, and whether a fallback regressed a row that
top-k already passed.

Use `M40LLM_KV_MULTITASK_TASKS=single-needle,distractor-needle,...` and
`M40LLM_KV_MULTITASK_FALLBACK_CASES=topk,score-spread-top16,combined-top16` to
bound expensive 4096-token runs. The multi-task case names are
`single-needle`, `multi-needle`, `distractor-needle`, `early-fact-qa`,
`early-fact-summary`, and `long-chat-smoke`. Some diagnostics also enable
additional prompt shapes:

- `boundary-single-needle`: places the answer near a block boundary.
- `far-apart-multi-needle`: places two required answers far apart in the
  prompt.

`M40LLM_KV_TOPK_MULTITASK_DIAG=1` runs the same multi-task prompt suite without
fallback. It compares dense `off` with direct FP16-K/q4-V `block-select-exact`
for top-k selected old blocks. Use `M40LLM_KV_MULTITASK_TOP_BLOCKS=4,8,16` or
the shared `M40LLM_KV_QUALITY_TOP_BLOCKS=4,8,16` alias to choose the tested
`top_blocks` values.

`M40LLM_KV_TOPK_SENSITIVITY_DIAG=1` runs the focused 4096 multi-needle
selection-set diagnostic. It forces `M40LLM_KV_MULTITASK_TASKS=multi-needle`
and `M40LLM_KV_MULTITASK_TOP_BLOCKS=4,8,16`, then emits richer JSONL telemetry
for comparing selected block scores, selection records, selected-block
attention masses, top attended entries, and dense-vs-compressed per-token logit
traces. Use this before adding more policy complexity; it is meant to explain
why top4/top16 fail while top8 passes, not to add fallback/retry behavior.

`M40LLM_KV_TOPK_ABLATION_DIAG=1` runs the 4096 multi-needle selected-set
ablation for direct FP16-K/q4-V exact-old retrieval. It tests top4 plus each
single top8-delta block, top8 minus each selected block, top8 plus each
top16-extra block, selected pair/quartet support-shape cases, and score-cluster
policies. Use `M40LLM_KV_ABLATION_CASES=case-a,case-b` to restrict expensive
runs to named cases. The diagnostic uses:

- `M40LLM_KV_BLOCK_SELECT_POLICY=explicit`
- `M40LLM_KV_FORCE_INCLUDE_BLOCKS=...`
- `M40LLM_KV_FORCE_EXCLUDE_BLOCKS=...`
- `M40LLM_KV_BLOCK_SELECT_POLICY=score-cluster`
- `M40LLM_KV_BLOCK_SELECT_POLICY=score-cluster-adaptive`

These are diagnostic-only selection controls. `score-cluster` starts from the
base top-k set and adds score-near candidates up to `M40LLM_KV_BLOCK_MAX_BLOCKS`
using `M40LLM_KV_BLOCK_SCORE_DELTA` relative to the top-k cutoff score.
`score-cluster-adaptive` uses the same cutoff rule but also honors
`M40LLM_KV_BLOCK_MIN_BLOCKS`, so it can enforce a minimum support-set size.

`M40LLM_KV_ORDER_EQUIV_DIAG=1` runs a focused 4096 multi-needle
order-equivalence diagnostic. It compares baseline top8, baseline top16,
explicit same-set top8 in score order, ascending order, and descending order,
plus score-cluster-adaptive min8/max12 and min8/max16 candidate rows. JSONL rows
include `attention_step_trace` with per-generated-step top8-core mass,
top16-tail mass, selected-old entropy, recent mass, old exact mass, and selected
block materialization order. Current evidence shows same-set top8 passes under
all tested orderings; top16 failure is therefore better explained as cumulative
tail distribution shift than simple candidate order.

Tail-prefix ablation cases are available through
`M40LLM_KV_TOPK_ABLATION_DIAG=1` and `M40LLM_KV_ABLATION_CASES`:
`top8-plus-tail1` through `top8-plus-tail8` cumulatively add the top16 tail
blocks `[88,57,90,71,46,53,73,93]` to the passing top8 core. JSONL
`generated_logit_trace` rows include `dense_reference_token_margin_compressed`,
defined as compressed top logit minus the dense-reference token logit. The
4096 multi-needle diagnostic shows `top8-plus-tail1` and `top8-plus-tail2`
pass, while `top8-plus-tail3` and all larger prefixes fail; block `90` is the
first transition point in this prompt.

`M40LLM_KV_SCORE_CLUSTER_VALIDATE=1` runs the bounded score-cluster candidate
validation suite after the anatomy diagnostics. It compares dense `off`, plain
top4/top8/top16, and:

- `score-cluster-adaptive` with `min_k=8,max_k=12`
- `score-cluster-adaptive` with `min_k=8,max_k=16`

Use `M40LLM_KV_SCORE_CLUSTER_CASES=top4,top8,score-cluster-adaptive-min8-max12`
to restrict expensive validation runs to named cases. Score-cluster JSONL rows
include the selected block set, scored selection records, score cutoff margin,
policy score delta, min/max block caps, active KV bytes, and
`selected_support_size` / `selected_support_bucket` (`top4-sized`,
`top8-sized`, `top16-sized`, or `other`).

The default targets are 2048 and 4096 when the model context permits. Set
`M40LLM_KV_QUALITY_TARGETS=2048` or `4096` to bound the run, and use
`M40LLM_KV_MULTITASK_TASKS=...` to resume individual expensive prompt shapes.
The 2048 suite includes single-needle, multi-needle, distractor-needle,
early-fact QA, boundary-single-needle, and far-apart-multi-needle. The 4096
suite omits early-fact QA and focuses on single-needle, multi-needle,
distractor-needle, boundary-single-needle, and far-apart-multi-needle. Dense
`off` remains the reference: compressed rows are marked inconclusive for
compression-policy conclusions when dense fails the same prompt.

`M40LLM_KV_CAPTURE_GENERATED_STEP=<n>` sets
`M40LLM_KV_ATTENTION_CAPTURE=token:<prompt_last_token + n>` when no explicit
attention capture selector is already set. This is useful for capturing the
actual divergence step instead of only the first generated answer token.

The quality harness supports row filtering for expensive exact-block sweeps:
`M40LLM_KV_QUALITY_MODES=block-select-exact` limits the mode matrix, and
`M40LLM_KV_EXACT_BACKEND_VARIANTS=fp16-k-q4-v-direct` limits exact-block backend
cases.

By default, q8 exact-old attention still dequantizes selected old blocks into
the reusable FP16 staging workspace before attention. Set
`M40LLM_KV_EXACT_OLD_ATTENTION=q8-direct` to use the experimental direct q8
attention backend instead. The direct backend skips the FP16 staging buffers for
old selected blocks and dequantizes q8 old K/V inside the attention kernel while
preserving the staged path's FP16 rounding semantics. Keep this opt-in until the
2048 and larger quality sweeps are characterized across more prompts.

`M40LLM_KV_Q8_DRIFT_DIAG=1` runs a narrow exact-block backend comparison in the
quality harness. It defaults to 4096-token prompts and limits the mode matrix to
dense `off` plus `block-select-exact`. For `block-select-exact`, it runs:

- `fp16-exact`: dense/FP16 exact old backing.
- `staged-q8`: q8 old backing dequantized into FP16 staging.
- `direct-q8`: q8 old backing dequantized inside the attention kernel.

The diagnostic focuses on the known failing rows by default: old needle with
`top_blocks=4` and recent needle with `top_blocks=8`. JSONL rows include
`exact_block_backend_variant` to distinguish these cases.

Attention telemetry records first-captured probability mass by group:
recent exact tokens, selected exact old tokens, old summaries, representatives,
and other entries. It also reports top attended entries, needle-block mass when
applicable, and pre-softmax logit stats for recent, summary, and representative
groups.

Set `M40LLM_KV_ATTENTION_CAPTURE=first|all|layer:<n>|token:<n>|layer:<n>,token:<n>`
to choose which decode attention calls are retained. The default is `first`.
Layer/token capture is diagnostic-only and can be expensive when set to `all`.

In exact-selection diagnostics, `block-select-exact` may use
`M40LLM_PREFILL_CHUNK_SIZE` for packed-prefix prefill.

`M40LLM_KV_LOGIT_COMPARE=1` makes the harness retain prompt logits and first
decode-step logits for dense `off`, `dense-recent-only`, `block-select-exact`,
`recent-only`, `block-summary`, and `block-select-lossy`. JSONL rows then
include dense-vs-mode max/mean logit differences, top-10 overlap, top token
IDs, and the expected first answer token's rank/logit when it can be derived
from the tokenizer. Rows after `dense-recent-only` also include dense-window-vs
mode prompt/first-decode differences and dense-window expected-token rank/logit.
For compressed modes, the same JSONL records include dense-window candidate
absolute positions, compressed recent candidate absolute positions, and physical
ring slots so recent-ring ordering can be compared directly. The stderr table
prints concise first/last/count summaries for these arrays; use JSONL for the
full position list.

`M40LLM_KV_LOGIT_TRACE=1` records per-generated-token logits. In those trace
rows, `expected_answer_token` is only first-answer-token metadata. Use
`dense_reference_token` and its rank/logit fields as the per-step dense
reference. Trace rows also report EOT token `128009` rank/logit so premature
end-of-turn drift can be diagnosed.

`M40LLM_KV_RECENT_EQUIV_SEQUENTIAL=1` disables packed-then-compress for
compressed modes during exact-selection quality diagnostics. Use it when
comparing `dense-recent-only` and compressed `recent-only` under matching
sequential prefill semantics. In this mode the harness limits the mode matrix
to dense `off`, `dense-recent-only`, compressed `recent-only`, and
`block-select-exact`.

JSONL rows also include absolute-position diagnostics:

- `recent_ring_absolute_start`
- `recent_ring_absolute_end`
- `needle_token_absolute_positions`
- `question_token_absolute_positions`
- `needle_tokens_in_recent_ring`
- `question_tokens_in_recent_ring`

## Current Interpretation

The current quality evidence separates three concerns:

- Dense `off` is the reference capability check.
- `block-select-exact` tests whether summary scoring can find the relevant old
  block while still attending exact K/V.
- `dense-recent-only` distinguishes "the old context is genuinely needed" from
  compressed recent-ring construction or indexing bugs.
- `block-summary` and `block-select-lossy` test whether lossy summaries and
  representatives preserve enough information for retrieval.

If dense fails, the case is inconclusive for compression. If `block-select-exact`
passes but lossy modes fail, summary scoring is not the primary bottleneck; the
representation or mixing strategy needs work.

The current architectural direction is summary-indexed exact-block retrieval:
summaries should select old blocks, while attention consumes exact K/V for the
selected old blocks plus the exact recent window. The Phase 1 prototype remains
dense-backed on GPU through `block-select-exact`. A later memory-saving variant
should replace old dense backing KV with q8 exact old KV, stage selected blocks
into a small FP16/FP32 working buffer, and report recent exact KV, summary index
KV, exact old backing KV, selected working-set KV, and temporary dense prefill KV
separately.

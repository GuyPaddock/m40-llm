# Experimental KV Compression

The compressed KV-cache work is experimental. It exists to measure whether an
M40 can support longer effective contexts by combining exact recent KV with
older compressed summaries or selected exact blocks.

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
--kv-compress-mode off|block-select-exact|recent-only|block-summary|block-select-lossy
--kv-recent-window 1024
--kv-compress-block 32
--kv-compress-top-blocks 16
--kv-compress-representatives 0
--kv-compress-representative-policy last|stride
```

- `off`: dense exact KV.
- `block-select-exact`: keeps old exact KV and uses block summaries only as an
  index for selecting old blocks.
- `recent-only`: diagnostic mode that attends only to exact recent KV while
  using the compressed sidecar state.
- `block-summary`: attends to exact recent KV plus old block mean K/V summaries.
- `block-select-lossy`: scores old blocks cheaply, attends to selected lossy
  summaries and optional representatives.

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
compression ratio. `attention_compression_elapsed_ms` is currently `null`
unless a per-attention counter is available.

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
decode-step logits for dense `off`, `block-select-exact`, `recent-only`,
`block-summary`, and `block-select-lossy`. JSONL rows then include dense-vs-mode
max/mean logit differences, top-10 overlap, top token IDs, and the expected
first answer token's rank/logit when it can be derived from the tokenizer.

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
- `block-summary` and `block-select-lossy` test whether lossy summaries and
  representatives preserve enough information for retrieval.

If dense fails, the case is inconclusive for compression. If `block-select-exact`
passes but lossy modes fail, summary scoring is not the primary bottleneck; the
representation or mixing strategy needs work.

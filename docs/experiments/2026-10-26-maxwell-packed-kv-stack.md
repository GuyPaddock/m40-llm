# Optimized Q8 Runtime Stacked With Exact-Old KV

## Result

The recent Q8 optimizations work with the DeepSeek-inspired summary-indexed
exact-old KV backend, but the combination is **slower on the short browser
workload**. Five measured rounds after warmup give:

| KV configuration | Resident tok/s | Prefill ms | Decode ms | Total ms | Final KV MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense off | 49.104 | 508.962 | 9,865.397 | 10,426.840 | 36.000 |
| Compressed defaults, window1024/top8 | 19.694 | 514.489 | 25,429.156 | 25,997.741 | 58.887 |
| Compressed window128/top8 | 20.217 | 507.503 | 24,774.414 | 25,325.056 | 30.238 |
| Compressed window128/top4 | 22.765 | 508.295 | 21,933.267 | 22,490.879 | 30.238 |

These are **resident-generation wall times**, excluding model loading and
KV allocation, **not buffered HTTP or a new Ollama comparison**. All twenty
measured requests naturally generate 512 tokens. Prefill is effectively
unchanged; the large difference appears in decode. This is an exploratory
stack comparison, not an exact-output promotion or a defaults change.

The 1024-token recent window contains the entire workload: 65 prompt tokens,
512 sampled tokens, and 576 KV tokens consumed by the final forward. There
are no old tokens to select in that row. Its compressed cache still reserves
old backing capacity, making resident KV **larger** at this short allocation
bound. The window128 variants reduce KV allocation by about 16%, but are
still much slower than this runtime's optimized dense attention.

This does not locate a long-context crossover or invalidate earlier
long-context memory/quality measurements. It shows that compressed KV is
not an additive short-context speedup to these projection optimizations.
Keep dense off for the exact short-workload target; leave project defaults
and the preferred compressed backend/policy hierarchy unchanged.

## Output And Semantics

Dense actual IDs match the previously qualified 512-token continuation.
Compressed defaults preserve those IDs in every round. The smaller windows
change output consistently at zero-based generated steps **323 for top8**
and **196 for top4**. All per-case outputs are stable across rounds.

The recorded top4 continuation becomes repetitive; top8 also introduces
repetition and inaccurate parsing/rendering descriptions. These are
qualitative observations of this explanation prompt, not a formal retrieval
quality score. Fluent text alone would not establish quality preservation.
The sparse rows do not satisfy the original exact-output requirement.
No forced continuation, retry, fallback, representative tuning or selection
policy change was used.

At the final forward, the logical attended-token upper bounds per query
head are 576 dense/default, 384 window128/top8, and 256 window128/top4.
The sparse figures include 128 recent tokens plus at most 8/4 blocks of 32
old tokens. They are **upper bounds**, not captured selected block counts or
physical allocation sizes. This run does not collect per-attention timing;
the penalty cannot be attributed numerically to scoring, metadata or the
direct kernel from these wall times alone.

## Fixed Runtime And Memory

Clean source `0ea619bb4113359136eecb2169632fedcac3ae13`, release CUDA test
executable SHA-256
`ab22e0fe9a55f8b306e6000145702c9226ea4dd19e01075be7e29813492420af`.
Same Qwen2.5-3B Q8_0 GGUF hash `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`,
canonical ChatML IDs, greedy sampling, natural stopping and context 1024.

Every case uses packed-warp-256 QKV/output, split-f32x4-128 FFN, prepared
down/vocabulary and the 64-token dequant-SGEMM prefix. Actual dispatch checks
require 18,432 launches of each packed projection, 18,432 split-FFN,
18,944 prepared down/vocabulary and 252 conversions/SGEMMs. Compressed rows
add 18,432 direct FP16-K/q4-V attention calls; no old-V staging path is used.
Dense uses the qualified parallel head128 kernel. Numerical stage attribution
would require a separate profile; dispatch counts are not GPU timings.

All 109 prepared identities match the qualified runtime: 3,279,519,744
source plus 2,917,580,800 prepared bytes, totaling 6,197,100,544 bytes
before KV/workspace. One resident model is reused and its preparation is
checked unchanged afterward. Temporary SGEMM capacity is 90,177,536 bytes.

| Compressed allocation component | Default window1024 | Window128, top4/top8 |
| --- | ---: | ---: |
| Recent FP16 K/V | 37,748,736 B | 4,718,592 B |
| Old FP16 K | 18,874,368 B | 18,874,368 B |
| Old q4 V payload | 4,718,592 B | 4,718,592 B |
| Old q4 V scales | 294,912 B | 294,912 B |
| Summary/index/metadata | 110,880 B | 3,100,752 B |
| Final allocated KV | 61,747,488 B | 31,707,216 B |

Dense allocation is 37,748,736 bytes. Capacities are requested resident
payloads, not peak VRAM or DRAM traffic. KV allocation medians of roughly
3.1-4.9 ms are reported separately and excluded from generation timing.

## Reproduction And Validation

Build the release `qwen_decode_throughput` test with CUDA/cuBLAS enabled.
Export the frozen environment from the
[quality reproduction](2026-10-25-maxwell-packed-projection-runtime-quality.md#configuration-and-reproduction),
then override the generic loader selectors explicitly:

```bash
export M40LLM_Q8_EXECUTION_WEIGHTS=split
export M40LLM_Q8_EXECUTION_FAMILIES=gate-up,down,vocabulary
export M40LLM_Q8_FFN_KERNEL=split-f32x4-128
export M40LLM_Q8_PREFILL_BACKEND=dequant-sgemm
export M40LLM_Q8_ATTN_PROJECTIONS=packed-warp-256
export M40LLM_KV_BLOCK_SELECT_POLICY=topk
export M40LLM_QWEN_THROUGHPUT_MODEL="$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
export M40LLM_QWEN_THROUGHPUT_REPORT=/tmp/new-packed-kv-stack.jsonl
cargo test --release --features cuda --test qwen_decode_throughput \
  q8_kv_stack::test_preferred_q8_runtime_with_compressed_kv \
  -- --include-ignored --nocapture --test-threads=1
```

Source `scripts/dev-env.sh` first. Unrelated `M40LLM_KV_*` diagnostic settings
are rejected. The output must not exist. Each case warms once, then five
rounds alternate forward/reverse case order. Every request starts with new
KV; allocation occurs outside timing. Warmups are excluded from all medians.

All forty measured GPU boundaries show P0, 1113 MHz SM / 3004 MHz memory,
with only benchmark PID 2351467. The eight warmup boundaries include one
949 MHz SM observation, retained separately. No inference, build, test or
MemPalace work overlaps the benchmark. Session 30959 exits zero after
534.51 seconds; tracked bytes return zero to zero and GPU compute is empty.
The report-behavior unit test, CUDA/server Clippy, formatting and independent
artifact audit pass. Missing rounds, wrong dispatch, changed dense IDs and
incorrect memory accounting must be rejected.

- [Raw JSONL and complete outputs](2026-10-26-maxwell-packed-kv-stack.jsonl)
- [Audit](2026-10-26-maxwell-packed-kv-stack-audit.jq) and [derived medians](2026-10-26-maxwell-packed-kv-stack-audit.json)
- [Run log](2026-10-26-maxwell-packed-kv-stack.log.gz) and [checksums](2026-10-26-maxwell-packed-kv-stack.sha256)

This side experiment leaves the exact track at **49.146 HTTP tok/s** versus
**43.106 official Ollama**, below the unchanged **53.06** and **56.037**
requirements. Its next step remains profiling the newly qualified dense
runtime; any longer-context compressed comparison needs its own matched
workload and dense-valid quality checks.

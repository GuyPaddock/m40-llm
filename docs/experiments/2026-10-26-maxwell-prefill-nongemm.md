# Remaining Packed-Prefix Costs

## Decision

Do not implement another prefix kernel from this profile. The newly captured
groups total **106.794 ms** across the 64-token prefix. Even eliminating all
of them gives only **1.029% optimistic HTTP throughput scope**, below the
unchanged 3% E2E floor. Including the existing QKV bias spans increases this
estimate to only **1.446%**. These estimates are feasibility screens, not
causal bounds or predicted gains.

Preserve the qualified **49.146 HTTP tok/s** configuration, split-f32x4-128
FFN, packed-warp-256 QKV/output, prepared down/vocabulary and dequant-SGEMM
prefix. The official comparison remains **43.106 HTTP tok/s**. Both original
requirements, **53.06** and **56.037 HTTP tok/s**, remain unmet. No runtime,
model, attention semantics or default changes follow.

This does **not** establish Maxwell's hardware ceiling. DRAM transactions,
achieved occupancy and warp stalls remain unmeasured. Further optimization
needs a credible account of expensive work avoided, not another arithmetic
substitution or a rescue of a closed probe. See the
[implementation admission requirements](../maxwell_q8_execution.md#admission-for-further-optimization).

## Measurement And Coverage

Five serial alternating release profiling-off/on pairs preserve all actual
512 generated IDs, frozen completed-quality output, ordinary enqueue counts
and 109 prepared tensor identities. Profiling changes HTTP throughput from
**48.844007 to 47.980453 tok/s**, a **1.768% reduction**, or **1.800% latency
increase**. MAD/median is 0.339% off and 0.279% on. This passes the fixed 3%
perturbation screen, not an optimization or fresh Ollama qualification.

The oracle verifies tokenization/backend only; its inference is untimed and
does not overlap M40 requests. No external observer or hardware collector
is loaded. Original CUDA calls, streams, dependencies and drains are kept.

Seven profiled sessions contain one warmup, five primary requests and one
actual-ID replay. Each prefix has **829 spans** in the bounded **880-pair**
event pool, versus 540 spans previously. Decode still samples generated
steps 0/255/511, 400 spans each, reusing the pool. New groups are per-layer,
not per-token event allocations. Warmup/replay are excluded from medians.

| Prefix scope, all layers | Median event ms | Summed host-call ms |
| --- | ---: | ---: |
| Embeddings | 1.356 | 1.353 |
| Attention norm | 0.530 | 0.208 |
| RoPE and KV append | 47.560 | 42.830 |
| Packed attention | 54.584 | 1.225 |
| Attention residual | 0.336 | 0.162 |
| FFN norm | 0.524 | 0.143 |
| SwiGLU | 1.339 | 0.153 |
| MLP residual | 0.302 | 0.145 |
| Inter-layer hidden copies | 0.218 | 0.260 |
| Final output copy | 0.004 | 0.008 |
| Existing Q8 conversion | 88.679 | 0.900 |
| Existing SGEMM | 91.496 | 1.906 |
| Existing QKV bias | 42.730 | 264.808 |

Packed-forward wall median is **355.603 ms**, captured event union
**329.498 ms**, and remaining wall **25.719 ms**. Independent medians need
not sum. The prior roughly 140 ms unattributed scope is substantially
explained, but these are separate instrumented runs, not subtractable
engine-speed measurements.

Events bracket stream intervals and can include host submission gaps.
Host-call durations overlap GPU work and are not CPU-only latency. In
particular, **264.808 ms bias host time is not removable exclusive cost**.
That group enqueues 192 residual-add launches per layer after the original
producer wait; previous GPU work or enqueue blocking can appear inside the
host interval. This profile does not identify their causal contributions.
Do not count both host and event time as separate savings.

Per-sample sums, then medians, give 106.794 ms for new groups and 149.399 ms
including bias. The ordinary HTTP median is 10,482.350 ms. Calculating
`T / (T - scope) - 1` gives the optimistic estimates above. The attention
and RoPE groups alone cannot provide a qualifying standalone E2E gain.
This does not authorize combining unrelated closed probes to clear a gate.

## Identity, Memory And Isolation

Clean measurement source `cf411b63c042c5b3b5146f8b3c56508c754d6f72`,
release comparison executable SHA-256
`0998e7e9940cac90d99319d080f111659b47314f515bd7473aba0d376daa12aa`.
CUDA 12.8.93 flags remain `-O3`, sm_52 SASS plus compute_52 PTX, no fast math.
Qwen2.5-3B Q8_0 model SHA-256 is
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
The frozen 65 ChatML IDs, greedy sampling, natural 512-token length stop,
fresh dense KV/context 1024 and completed quality evidence are unchanged.

All 109 prepared identities retain 2,917,580,800 prepared bytes plus
3,279,519,744 source bytes. Dense KV is 37,748,736 bytes and the reusable
prefill temporary is 90,177,536 bytes. Event-pool setup is around 0.85 ms;
these capacities are not peak-VRAM measurements.

Every prefix retains 14,292 launches, 252 cuBLAS calls, 70 stream syncs,
324 waits, 21 allocations/20 frees, 68 H2D and 64 D2H copies. Every full
request retains 237,013 launches, 252 cuBLAS calls, 1097 syncs, 324 waits
and 42 allocations/frees. Grouped spans add no ordinary CUDA work.

All twenty primary GPU boundaries show isolated P0, 1113 MHz SM / 3004 MHz
memory. Benchmark PID 3753060 and the idle owned oracle runner 3754140 are
the only compute processes. No builds, tests or MemPalace work overlap
timing. Benchmark session 85848 exits zero after actual-ID replay; owned
oracle session 22529 exits zero after cleanup. A subsequent GPU process
query is empty.

An initial nested-shell environment export was rejected **before model
loading or timing**, with `runtime comparison requires qualified control
M40LLM_ENABLE_NVCC=1`. Its output is empty and no events file exists. A
startup script fixes the quoting and uses new `-screen-` paths. No failed
trial is pooled, restarted on observation timeout or called a kernel failure.

## Validation And Reproduction

The extended profiler test covers head64 and Qwen-shaped head128, exact
raw KV contents, logits, sampled IDs, counters, new group coverage/streams,
pool capacity/reuse and explicit profiling-off behavior. The hardware run
passes 22 CUDA checks across attention parity, profiler, MLP, projection and
prefill suites. CUDA/server all-target Clippy, formatting and 217
warning-as-error non-CUDA checks pass. Additional cuBLAS-disabled conversion
and backend-rejection checks pass, five total. Twelve corruption controls
reject changed IDs/work, missing groups, wrong streams/layers, allocations,
pool/union errors, unrelated GPU processes and optimization claims.

Use the [previous profile reproduction](2026-10-26-maxwell-prefill-attribution.md#validation-and-reproduction)
with this source, the same quality report and new output paths. The
profiling flags are unchanged. The compressed events artifact can be audited
without hardware:

```bash
bash docs/experiments/2026-10-26-maxwell-prefill-nongemm-audit.sh
bash docs/experiments/2026-10-26-maxwell-prefill-nongemm-audit.sh --negative-controls
```

- [Raw HTTP requests and IDs](2026-10-26-maxwell-prefill-nongemm-http.jsonl)
- [Raw prefix/decode events](2026-10-26-maxwell-prefill-nongemm-events.jsonl.gz)
- [Audit](2026-10-26-maxwell-prefill-nongemm-audit.jq), [derived results](2026-10-26-maxwell-prefill-nongemm-audit.json)
- [Run log](2026-10-26-maxwell-prefill-nongemm-run.log.gz), [oracle log](2026-10-26-maxwell-prefill-nongemm-oracle.log.gz), [rejected preflight](2026-10-26-maxwell-prefill-nongemm-preflight-rejected.log.gz)
- [Validation archive](2026-10-26-maxwell-prefill-nongemm-validation.tar.gz), [checksums](2026-10-26-maxwell-prefill-nongemm.sha256)

No next kernel is nominated by this result. The next candidate must name
the dominant reader work it removes, bound all new storage/conversion work
and pass the existing complete-FFN and E2E gates before promotion.

# Qwen Q8_0 Buffered Stage Profile, 2026-10-24

## Configuration

Use the [controlled comparison command](../controlled_benchmark.md), unchanged
Q8 backend flags, release Rust, CUDA 12.8 `-O3`/`sm_52`, dense FP16 KV, context
1024, the shared 65-ID browser prompt, and 512 greedy generated IDs. Add
`--compare-stage-profile`, `M40LLM_STAGE_PROFILE_TOKENS=0,255,511`, and
`M40LLM_STAGE_PROFILE_REPORT=/tmp/m40-llm-c04-stage-events.jsonl`.

The runtime base is `663c414` plus the C04 worktree recorded in the manifest;
the benchmark executable hash identifies the measured binary. No projection
or attention kernel was optimized in this checkpoint. The two models remain
resident, but only m40-llm generates during this experiment. Oracle verification
finishes before timing. No hardware tests overlap the timed requests.

## Perturbation Check

| Pair | Profile Off HTTP Tok/s | Profile On HTTP Tok/s |
| --- | ---: | ---: |
| 0 | 23.8394 | 23.7061 |
| 1 | 23.7798 | 23.7008 |
| 2 | 23.7910 | 23.6673 |
| 3 | 23.7722 | 23.6642 |
| 4 | 23.7569 | 23.6510 |
| Median | 23.7798 | 23.6673 |

Profile-on throughput is 0.473% lower. Median client wall is 21.531 s off and
21.633 s on. MAD/median variation is 0.0468% off and 0.0689% on. This is a
bounded instrumentation screen, not evidence for an inference optimization.
The companion runs match all 512 generated IDs, not just decoded text.

The profiler records session 1 for the enabled warmup, sessions 2-6 for the
five enabled measured requests, and session 7 for the enabled untimed trace.
Only sessions 2-6 contribute to the stage medians below.

## GPU Event Spans

| Stage | Step 0 | Step 255 | Step 511 |
| --- | ---: | ---: | ---: |
| Embedding | 0.001 ms | 0.023 ms | 0.024 ms |
| Attention norm | 0.393 ms | 0.386 ms | 0.387 ms |
| QKV | 2.492 ms | 2.490 ms | 2.495 ms |
| Q RoPE | 0.262 ms | 0.255 ms | 0.257 ms |
| K RoPE / KV append | 0.262 ms | 0.262 ms | 0.259 ms |
| Attention | 2.509 ms | 11.421 ms | 20.387 ms |
| Attention output projection | 1.715 ms | 1.713 ms | 1.720 ms |
| Attention residual | 0.158 ms | 0.157 ms | 0.158 ms |
| FFN norm | 0.405 ms | 0.404 ms | 0.405 ms |
| Gate/up/SwiGLU | 11.852 ms | 11.857 ms | 11.858 ms |
| Down projection | 6.315 ms | 6.315 ms | 6.318 ms |
| MLP residual | 0.160 ms | 0.159 ms | 0.159 ms |
| Output norm | 0.010 ms | 0.011 ms | 0.010 ms |
| Vocabulary / greedy argmax | 3.438 ms | 3.432 ms | 3.434 ms |
| Union of all stage intervals | 29.987 ms | 38.889 ms | 47.898 ms |

Layer stages aggregate 36 layers. Steps are generated-prediction indices;
their absolute input positions are 64, 319, and 575. Separate medians need not
sum exactly. This configuration uses the decode stream throughout, but the
collector also records producer-stream offsets and interval unions for
multi-stream work; it never treats overlapping durations as additive.

These are event-delimited spans, not isolated kernel service times. Host enqueue
gaps or synchronization already inside a wrapper can affect them. There are
no added per-operation synchronizations. Event readiness is queried only at
the existing token completion boundary.

The first capture of each fresh event pool has 97.55 ms median token wall but
29.99 ms GPU interval union. Its CPU call spans are also unusually large.
This gap is not explained by the current counters, and first-use profiling
costs have not been isolated. Do not use it as ordinary CPU-cost evidence.
Reused middle/late captures measure 39.72/48.71 ms wall and 38.89/47.90 ms GPU
union. Host completion waits overlap GPU execution and cannot be added to it.

## Evidence and Next Work

- [Raw HTTP records and exact token traces](2026-10-24-qwen-profile-comparison.jsonl)
  retain every pair, output, build/environment manifest, and GPU observation.
- [Raw stage JSONL, gzip](2026-10-24-qwen-stage-events.jsonl.gz) retains all
  warmup/measured/trace sessions with per-layer GPU/CPU/counter records.
- Uncompressed stage JSONL SHA-256:
  `8da090e0b171670c81fce7a0e8dfe5bd9e301da8fe72ae54811bcfdbcf98dd4a`.
- Gzip SHA-256:
  `6add55fe7ca4bdfea47c69f5a73379c728c7c1f0ee99972d7b0be83999409110`.

The middle-token FFN cost is about 18.17 ms and attention is 11.42 ms.
Attention rises to 20.39 ms late while the fixed projection costs remain
stable. This justifies the bounded exact head128 attention comparison and
Q8 layout experiments, with particular attention to gate/up as well as down.
It does not justify changing model semantics. C05 will establish which legacy
Maxwell traffic/occupancy counters are actually available; none are measured
or inferred here.

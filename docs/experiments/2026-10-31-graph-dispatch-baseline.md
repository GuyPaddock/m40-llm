# Graph dispatch negative controls

Base `4ac6902`; CUDA kernels remain identical to `b9333e7`. Test-scoped
injection and full-forward cases come from `a45cfcf`; direct attention checks
are new. These ignored research checks intentionally fail before the candidate.

The native full-forward replay case reproduces 93 unequal FP16 cache comparisons
in 32 tokens, maximum cache absolute difference 6.1035156e-5 and maximum logit
difference 3.7718564e-8. Tracked allocations return to baseline. Its unchanged
exact-cache assertion fails, exit 101.

The short direct-attention case compares 144 outputs across six kernel/cache
choices, two physical slots, changing lengths/queries, and invalid lengths.
It finishes all cache/input/guard/allocation checks but fails exact FP32 output
parity, exit 101. The final raw log records the unequal-element total.

An initial uncommitted fixture attempt incorrectly treated the append API's
position argument as independent of its rotary position. Its cache-length
assertion failed after the first choice. That run is retained as a fixture
failure and is not dispatch evidence. The fixture now uses ordinary unrotated
FP32 append and verifies every initialized FP16 cache value against its input.

Commands, after sourcing `scripts/dev-env.sh`, selecting the M40 UUID
`GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852`, and confirming no compute processes:

```bash
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --lib \
  decode_session::graph_tests::test_native_q8_graph_sequences \
  -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test attention_graph_dispatch test_graph_attention_short_dispatch \
  -- --include-ignored --nocapture --test-threads=1
```

Logs are `2026-10-31-graph-dispatch-baseline.log`,
`2026-10-31-graph-dispatch-kernel-baseline.log` and
`2026-10-31-graph-dispatch-fixture-failure.log` in this directory.
These are correctness checks; their wall times are not performance evidence.

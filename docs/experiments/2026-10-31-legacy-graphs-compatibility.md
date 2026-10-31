# Legacy graph compatibility checkpoint

The four existing graph tests pass on the Tesla M40 with CUDA 12.8 and
cuBLAS enabled, from clean source `2f4a93e`. This establishes current graph
API and tiny-fixture compatibility. It does not establish native-Q8 sequence
correctness, failure recovery or a performance gain.

The explicit hardware run passes one-layer forward capture/replay, one-layer
session replay, multilayer session replay and graph diagnostic synchronization.
All four tests run with ignored tests included and one test thread; none skips
an unavailable device. Session tests assert graph launches, and the diagnostic
test also asserts synchronization and the hidden-to-logits stream dependency.
These fixtures use FP16 weights and the capturable cuBLAS path. The diagnostic
durations in the log come from a debug build and are not performance baselines.

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/legacy-graphs/target
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test forward_with_layer_smoke graph \
  -- --include-ignored --nocapture --test-threads=1
```

The [raw log](2026-10-31-legacy-graphs-existing-tests.log.gz) records four
passed tests and twenty filtered tests. The filtered tests are not passes.
The GPU was idle before this serial run.

## Remaining native correctness gate

The source audit's next gate remains open. Reuse the nonzero two-layer Qwen
Q8 fixture in `tests/common/q8_model.rs`, with head128 GQA and explicit dense
KV. Compare graph-off and graph-on logits and greedy IDs across growing
prefixes. Inspect actual FP16 K/V tokens and per-layer/sequence lengths using
the existing debug APIs. Include two separate sequence slots, repeated session
construction and teardown, changed positions and allocation cleanup.

Exercise capture failure after recording operations and launch rejection before
submission through test-only failure injection. Execute the actual session
fallback against real CUDA/model/KV state. Assert that the warmed result or
normal fallback leaves the same logits, KV values and lengths as graph-off,
and that subsequent tokens continue normally. A pre-submission rejection test
does not establish recovery from an asynchronous device fault; keep that
distinction explicit.

`DecodeSession::forward_with_optional_graph` currently warms ordinary forward
before capture. Capture failure returns that warmed result and disables
graphs. Launch failure calls ordinary forward again. Tests must validate
observable state in these paths rather than merely count a fallback branch.
The current native-only FFN/prefill restrictions and compressed-KV exclusion
remain unchanged.

Only after this correctness gate should the branch measure the same Qwen
model's native graph-off/on complete token cost. Separate warm forward,
capture/instantiation, first replay, steady replay, CPU submission and complete
token time as required by the [source audit](2026-10-31-legacy-graphs-audit.md).
Compound support remains conditional on a native complete-operation win.

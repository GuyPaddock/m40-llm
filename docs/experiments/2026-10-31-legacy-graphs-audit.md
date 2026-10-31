# Legacy graph source and history audit

The runtime already has opt-in decode graph capture. Revisit its native path
under controlled conditions before extending graph support to the qualified
compound FFN. Earlier local investigations recorded slow replay, so lower CPU
submission time alone is insufficient evidence of improvement.

## Upstream reference

[PR 30727](https://github.com/ggml-org/llama.cpp/pull/30727) remained open and
unmerged on October 31. The inspected head was
`0114179ee5fee313f7c4b5b69e4f281b4708e488`, against base
`c0fd78626405c6d9b934efef0ec7291fcc20e3d6`.

Its graph change removes the pre-Volta architecture disable from graph
enablement. The broader patch also changes cuBLAS precision/algorithm
selection and workspaces. The contributor's preliminary Kepler performance
report does not isolate those changes or establish Maxwell speed. This is a
reason to test the existing local graph implementation, not to import the
whole patch.

## Local restrictions and retained evidence

At checkpoint `359dee7`, [`DecodeSession`](../../src/decode_session.rs)
allocates device position/sequence-length parameters, warms normal forward
before capture, and replays through the decode stream. Logits consumption
orders against graph output. Capture failure returns the warmed result and
disables graphs; launch failure falls back to normal forward. These paths need
behavioral verification of token and KV state before performance conclusions.

Session construction rejects graph use with non-native Q8 FFN or prefill.
[`forward.rs`](../../src/infer/forward.rs) separately restricts graph-parameter
forward to native FFN. Compressed KV bypasses graph replay. Use explicit dense
KV for this experiment and retain all those production restrictions initially.

The [performance history](../perf_baselines.md) contains May 12 graph replay
and diagnostic-sync investigations, plus a later Qwen fused-path probe. Those
entries distinguish fast host enqueue from slow GPU completion and retain
output concerns. They are historical context, not current matched benchmarks.

Existing tests in [`forward_with_layer_smoke.rs`](../../tests/forward_with_layer_smoke.rs)
exercise dense graph replay and diagnostic synchronization. Primitive graph
tests also exist in `cuda_elementwise.rs` and `q8_ffn_joint.rs`. A captured
primitive does not prove whole-session compound support.

## Next bounded experiment

First run the native dense session's correctness checks at changing positions,
with separate sequences and repeated sessions. Verify capture/launch failure
behavior before enabling any new dispatch. Then compare native graph off/on
with identical Qwen inputs and explicit dense KV, separating warm forward,
capture/instantiation, first replay, steady GPU replay, host submission and
complete token time. Use the controlled protocol's five alternating pairs and
variation rules; actual output traces must match.

Only a native complete-operation win can justify a narrow compound capture
extension. A negative native gate closes that dependent work for this study.
No new measurement, failure reproduction or graph default change has occurred
in this audit.

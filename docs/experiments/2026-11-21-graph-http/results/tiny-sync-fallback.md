# TinyLlama F16 graph admission

**FAIL for capture with the synchronous large-model F16 projection.**
Both 128-ID companion traces matched and both ran every layer. Graph-on
reported capture failure at the first Q projection, shape 1×2048×2048,
and fell back to ordinary execution. The benchmark stopped before timing
equivalent fallback requests as a graph comparison. All evidence is in
the [original run](tiny-sync-fallback.tar.gz).

The C wrapper returns -3 when synchronizing the prefill stream fails.
The Rust runtime already offers a dedicated asynchronous one-row F16
projection on the decode stream. The separate `tiny-decode-stream`
cohort selects `M40LLM_F16_DECODE_KERNEL=1` and
`M40LLM_F16_DECODE_STREAM=decode` for both arms. Keep the same model,
request, budget, timing and parity gates. This comparison tests the
existing capture-compatible hypothesis without adding a new runtime
path or attributing a synchronous fallback's wall time to graph replay.

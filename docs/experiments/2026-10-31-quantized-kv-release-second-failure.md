# Standalone quantized KV selection fails again

Clean `2cbbc28` passes three real cases, then fails at layer 0, context 2,048,
Q8/head64, sequence 5, query head 9. It selects old blocks
`[6,17,12,11,18,7,9,13]`; the independent reference selects
`[6,17,19,12,11,18,7,9]`. Missing block 19 has FP64 mean-dot score
`-10.221068283590888`, while substituted block 13 scores
`-10.425311266934152`. This difference is not a near-tie ordering issue.
No timing ran and there is no full-grid completion.

The [raw records](2026-10-31-quantized-kv-release-second-failure.jsonl.gz),
[run log](2026-10-31-quantized-kv-release-second-failure.log.gz),
[build log](2026-10-31-quantized-kv-release-second-failure-build.log.gz) and
[script](2026-10-31-quantized-kv-release-second-failure.sh) preserve the
standalone replay. The earlier passing test-executable diagnostics do not
establish standalone correctness.

Inspection finds a possible race in both existing Q8 quantizers: after
reducing K maxima into shared `reduce[0]`, every thread reads that maximum,
then immediately reuses the shared array for V without a block barrier.
Warp zero can overwrite the maximum before another warp reads it. Both
sites exist unchanged in base `0eee578`; the head128 staged specialization
did not introduce them. Verify this diagnosis and preserve the exact selection
and reconstruction gates when correcting synchronization. Do not change the
quantizer format or selector to hide this result.

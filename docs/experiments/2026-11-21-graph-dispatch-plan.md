# B1 graph qualification

Phase 4 baseline 48c6333. Reuse tests b5ea002 followed by dbceae1 from the October graph study. They branched before the feature acquired selected-Q4 and Q8 barriers. Verify the three-way source differences and preserve both changes. Compare ordinary and graph decode on the repaired binary. Retain a separate unfixed hardware observation using the original controls. Historical failure is four full-forward cases with 93 different FP16 cache comparisons. The repaired October grid passed 228 comparisons. Do not infer that this makes capture profitable.

## Correctness

Run release hardware tests serially on the M40 with cuBLAS enabled and disabled. Preserve the 8192/8193 specialized/fallback boundary, head64, head128 and generic paths, full logits/greedy IDs/valid KV bits and the October unsupported parallel attention case. Check independent sessions, cancellation before reuse and teardown, repeated replay and cache length updates. Fail on invalid ownership or stale continuation state. Count executed tests and retain skip reasons. Inspect graph node construction to distinguish conditional early returns from duplicate useful attention work.

## Performance

Use A2 at baac624 on the consolidation branch for model and measurement provenance. Freeze ordinary/graph candidates to the same supported model, kernel, precision, projection backend, dense KV, request, generation limit, concurrency one and diagnostics disabled. Audit that graph replay actually ran. Run Qwen Q8_0 and an eligible local F16 model if supported. Screen restricted configurations explicitly. For head128 parallel rejection, use the common ordinary/graph-compatible fallback on both sides, then report opportunity cost against best ordinary serving separately. Do not force unsupported modes.

Collect first-capture, instantiation and replay timings separately, warm alternating pairs, complete HTTP time and per-token boundaries. Record device and host resource costs, including peak VRAM and graph lifecycle. Derive break-even tokens from setup divided by positive per-token savings; when replay loses, report no positive break-even. A trace showing redundant nodes can admit at most one bounded change with a fresh control. Apply A2 reproducibility and five/ten/twenty rules. A correctness fix has its own verdict. For speed require at least 5% complete request gain above variation and positive paired lower bound. Retain negative and inconclusive measurements. Keep graph dispatch opt-in.

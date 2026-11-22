# Frozen three-layer sparse-Q4 quality gate

The [held-out operator screen](operator-result/README.md) admits
packed and zigzag on layers 0, 17 and 35. Compare four variants
with the unchanged Qwen2.5-3B Q4_0 model in separate M40 processes.
Each process runs its own native reference with matching input IDs,
explicit dense FP16 KV, sequential prefill and greedy seed zero.

| Arm | Repacked layer weights | Selection |
|---|---|---|
| Packed full | Lossless channel-major | All channels |
| Packed thresholded | Same lossless weights | Per-layer median cutoff |
| Zigzag full | Requantized output-tiled | All channels |
| Zigzag thresholded | Same requantized weights | Per-layer 75th percentile cutoff |

Use the six held-out generation tasks and likelihood passages in
`data/research_quality/v1.json`. Record each dense reference failure
separately. Stop an arm on the first lost dense pass, new repetitive
generation, nonfinite logits, or per-passage perplexity more than
3% above the native reference. The repetition diagnostic checks any
16 identical consecutive generated IDs or four adjacent copies of a
nonconstant token sequence of length 2–8, present only in the
candidate. Completed answers and matched-prefix teacher-forced logits
are independent observations. Require at least one dense-passing
case to advance the initial gate. A surviving variant still needs
the larger November quality suite before any speed qualification.

Validate the original GPU projection geometry with both layouts,
unmasked and thresholded. Check that an exact cutoff tie is excluded,
an all-zero row leaves a zero result, allocations remain stable
and surrounding output guards are preserved. Before scoring an arm,
verify the complete checkpoint, frozen cutoffs, original source
weights and each down-tensor identity. Charge weight allocation,
repacking, and workspace in each process's memory record. The timed
scope in these reports is quality diagnostics with two models
resident. Any later throughput claim needs its own warmed,
interleaved complete-request comparison.

From a clean worktree under the shared M40 lock, use
`bash docs/experiments/2026-11-22-sparse-q4-live/check-quality.sh MODEL BUILD_REPORT`
to build and run the same guards with cuBLAS on and off. It also
recompares all generated weight planes against the original operator
inputs. Then invoke
`bash docs/experiments/2026-11-22-sparse-q4-live/run-quality.sh MODEL NEW_RAW_DIR MODEL_QUALITY`.
Retain all early stops and separate hardware guard logs.

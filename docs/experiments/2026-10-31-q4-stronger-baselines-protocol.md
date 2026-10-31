# Stronger references for Q4 channel selection

The [first Q4 screen](2026-10-31-q4-channel-screen.md) passes its matched
native-Q4 gate, but full-retention repacking is already faster and the
qualified Q8 FFN is a stronger runtime reference. Freeze the follow-up before
measuring it.

Use the same model, source hashes, capture, encoding and layers 0, 17 and 35.
Restrict this follow-up to batch one and exact-Q4-hidden-magnitude selections
at 10% and 20% retention. Keep the original numerical tolerance.

Pair each selected case against two references:

1. Full-retention Q4 through the same repacked selected-channel kernels.
2. The existing qualified Q8 joint/shared-scale compound gate/up/SwiGLU and
   prepared warp-packet down projection.

Prepare all three Q4 owners and the Q8 owner before timing a layer. Retain
the same allocations in every arm. Q8 preparation retains split and joint
diagnostic owners as required by the existing benchmark helper; it does not
change runtime ownership defaults. Native Q8 weights and qualified down/vocabulary
preparation remain resident throughout. This is a cross-representation
timing comparison, not an equal-quality comparison.

Verify Q8 with its existing captured-input/FP64 checks. Verify Q4 against
independent masked FP64 references, and report output drift from full Q8
separately. Check guards, unchanged inputs, immutable owners, repeatability
and per-layer allocation cleanup after the pairs.

Warm each of the four arms for at least one second of GPU work. Reuse the
first screen's five-pair, ten-pair variation extension and fixed twenty-pair
borderline confirmation rules, with ten complete-FFN repetitions per sample.
All twelve comparisons include host submission/completion and GPU-event
measurements. No predictor, recovery, transfer or allocation enters the timed
FFN. Never pool these pairs with the original grid.

For a passing Q8 comparison, estimate the remaining overhead budget as
`Q8 median / 1.05 - selected Q4 median`, in milliseconds per FFN. This is a
point-estimate budget to retain the 5% complete-FFN gate, not a confidence
bound or a prediction of eventual selector performance. Use the tightest
layer budget at each retention level when choosing the next screening probe.
Any integrated method still needs direct complete-pipeline measurements,
sequence-quality checks and whole-model evaluation.

Use the prior reproduction command with `--ffn-q4-channels-baselines --rows 1`
and omit `--ffn-q4-channels-correctness-only`. The new option rejects competing
modes and any other row grid. Require a clean release build, the exact capture
environment, a fresh output path and an otherwise idle M40.

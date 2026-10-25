# Calibration-Based Layer Ranking

C24 is research-only. Layer masks change the model; they do not improve the
unchanged-engine benchmark or reclaim resident weights/KV. Normal generation
still executes every layer.

## Evidence Contract

Collect `BI = 1 - cosine(block_input, block_output)` for each original layer
and teacher-forced input position in the frozen C19 calibration passages.
Inputs include the normal metadata BOS, preserve absolute positions and omit
the final unforwarded target. Compute cosine on the host in FP64 from the
actual FP32 vectors. Reject nonfinite or zero-norm vectors rather than invent
a redundancy score. This follows [ShortGPT's block-influence measure](https://arxiv.org/html/2403.03853v1#S2.SS2).

The plan binds model/corpus/execution SHA-256, original layer/hidden dimensions,
and exact calibration passage IDs/token arrays. Capture must visit every
declared position once per original layer. First/last layer scores are recorded
but cannot become removal candidates. Scalar observations have a checked
memory budget; retaining all hidden vectors is unnecessary.

BI alone does not prove a layer is safe to remove. Evaluate every remaining
interior layer's removal on the same calibration targets after each selected
removal. Minimize token-weighted mean NLL; exact ties use ascending BI then
original ID. Repeat for one/two/three removals and freeze those masks before
held-out evaluation. This uses [SLEB's iterative loss-verification idea](https://arxiv.org/html/2402.09025v2#S3.SS2),
not its source code or a claim to reproduce its datasets/results.

Held-out loss/task results characterize the frozen masks, not the choice of
layers. Do not select another mask after inspecting held-out failures. All
original tensor/KV IDs remain intact through the existing
[diagnostic layer-mask path](layer_masks.md).

## Atomic Implementation

1. `feat(eval): define layer influence rankings`: numerical/identity contract,
   bounded capture, iterative calibration-loss ranking and CPU tests.
2. `feat(infer): capture calibration block influence`: model-local read-only
   observations of ordinary sequential forward, with GPU inertness and routing
   tests. Depends on the contract.
3. `test(quality): rank calibration layer removals`: shared quality-runner
   calibration capture, iterative loss study, frozen mask application and
   artifact lineage. Depends on capture and existing C19/C20 evaluation.
4. `docs(perf): record ranked layer tradeoffs`: actual calibration artifacts,
   held-out quality, matched-prefix/free-generation timing and allocations.
   Depends on complete serial release measurements.

No runtime or threshold is promoted by this study. Compact export is C25 and
requires an explicitly quality-evaluated candidate. C22/C27 sparse-weight
research remains subject to its separate negative kernel gate.

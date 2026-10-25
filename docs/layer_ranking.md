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
3. `feat(eval): score single-model token loss`: the same full-logit NLL and
   strict prefixes as paired scoring, without recomputing an unchanged control.
   Verify CPU and CUDA agreement and reset behavior.
4. `test(quality): rank calibration layer removals`: shared quality-runner
   calibration capture, iterative loss study, frozen mask application and
   artifact lineage. Depends on capture and existing C19/C20 evaluation.
5. `docs(perf): record ranked layer tradeoffs`: actual calibration artifacts,
   held-out quality, matched-prefix/free-generation timing and allocations.
   Depends on complete serial release measurements.

No runtime or threshold is promoted by this study. Compact export is C25 and
requires an explicitly quality-evaluated candidate. C22/C27 sparse-weight
research remains subject to its separate negative kernel gate.

## Read-Only Model Capture

`LoadedModel::begin_layer_influence_capture(plan, host_budget_bytes)` configures
model-local host state before KV/workspace/session creation, without a layer
mask or activation transform. Begin/finish each frozen passage explicitly,
then export with `finish_layer_influence_capture`. Incomplete finish is an
error and retains capture state. The host budget includes two reusable FP32
hidden vectors and the per-token FP64 scalar observations. No GPU buffer is
allocated by the observer.

Ordinary full-layer sequential forward drains both producer streams before
reading the layer's unchanged input/output buffers. This is diagnostic capture,
not a performance path. Without capture, there are no readbacks or new waits.
Only dense KV, logical `sequence_id=0` and non-graph sequential execution are
supported; packed/batched/graph paths fail before launches. Finish capture
before applying a diagnostic mask. This state is never process-global.

```bash
source scripts/dev-env.sh
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test layer_influence_cuda \
  -- --include-ignored --test-threads=1
```

## Frozen Ranking Runner

The existing release `model_quality` example accepts `--rank-layers new.json`
only with `--split calibration`, no activation/mask variants, and no timing
trials. The initial runner requires native Q8_0 projection coverage. It uses
the same model loader, passage tokenizer and full-logit session scorer as C19.
Single-model scoring uses exactly the paired evaluator's NLL calculation.

It first records every layer's influence on the two frozen calibration
passages, then re-evaluates every remaining interior removal at each stage.
For 36 layers and three removals this means 34+33+32 = 99 trials, not a BI
shortlist. Source weights stay resident. Between trials the exclusively owned
model has no scoring session, both streams drain, and old KV/workspace owners
are dropped before installing the next checked mask. Normal runtime setup
guards are not relaxed. Setup/loss elapsed times are diagnostic, not decode TPS.

The artifact is created only after all stages validate and device owners are
released. The report is append-only with a terminal `layer_ranking_complete`
record; an interrupted run is not a completed ranking. Capture binds model,
corpus, base execution and actual calibration token arrays. No held-out task,
loss or answer is used to choose a layer. First/last layers remain protected.

Use the exact native-Q8 environment from the
[C19 runner command](model_quality.md#release-runner), then run serially:

```bash
cargo build --release --features cuda --example model_quality
target/release/examples/model_quality --reference-model "$model" --model "$model" \
  --variant-id calibrated-layer-ranking --split calibration \
  --rank-layers /tmp/qwen-layer-ranking.json \
  --output /tmp/qwen-layer-ranking.jsonl

# Freeze/hash the completed artifact before any held-out variant evaluation.
for count in 1 2 3; do
  target/release/examples/model_quality --reference-model "$model" --model "$model" \
    --variant-id "ranked-removal-$count" --split all \
    --layer-ranking /tmp/qwen-layer-ranking.json --remove-layer-count "$count" \
    --step-timing-trials 5 --output "/tmp/qwen-ranked-$count.jsonl"
done
```

`--layer-ranking` application verifies model/corpus/base-execution hashes,
dimensions and calibration passage tokenization, then uses only the requested
already-frozen mask. It conflicts with ad-hoc `--skip-layers` and activation
transforms. Manifests retain artifact SHA and original skipped IDs. The ordinary
reference stays all-layers. Also run an empty-mask control, retaining the usual
dense-failed inconclusive classifications and full-logit/task/timing records.

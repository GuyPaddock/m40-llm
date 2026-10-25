# SwiGLU Input-Sparsity Research

C23 thresholds the completed SwiGLU vector before the down projection. Gate/up
weights and computation stay unchanged. This changes model semantics and is
not an unchanged-model engine optimization. Application requires the explicit
model-local research API or quality runner; generation defaults are unchanged.

## Selection Contract

`infer::activation_sparsity::ActivationThreshold` accepts finite nonnegative
magnitudes. Values with `abs(value) <= threshold` become zero; nonfinite values
are retained so selection cannot hide upstream numerical failures. Retained
values keep their exact FP32 bits and increasing original input indices.
Zero threshold omits zeros only. An empty retained set is valid and projects
to zero; an empty source vector is invalid.

Quantile calibration consumes caller-supplied finite observations only. It
does not discover models or mix calibration and held-out data. Tied magnitudes
may exceed the target sparsity, so reports must measure the retained count.
Per-layer model calibration follows a useful device-consumer screen, not
assumed savings from zero-valued activations alone.

## Bounded Implementation Sequence

1. `feat(infer): define sparse activation selection`: CPU threshold/index
   contract and behavioral tests. Depends on the frozen C19 evaluation split.
2. `perf(cuda): skip Q8 down-projection inputs`: reusable RAII compaction
   workspace, deterministic GPU selection, and a decode-only consumer of the
   retained C10 input-tiled layout. Compare against the same zeroed dense
   inputs and preserve zero-threshold parity. Depends on the selection contract.
3. `test(bench): screen sparse activation projections`: release real-down-tensor
   probe, including selection/indexing time, resident duplicate layout bytes,
   achieved sparsity and requested weight loads. Depends on CUDA parity.
4. Record measurements before any removal. Only proceed to per-layer
   calibration and model-wide opt-in evaluation if the complete selection plus
   projection path reduces measured latency. No browser/held-out tuning.

This uses dense Q8 weights with sparse inputs, not C21's rejected 16-of-32
sparse-weight representation. The initial device screen is single-row decode,
tile widths8/16 and threads128/256, four configurations total. Test requested
activation sparsity 0/25/40/50/60%; do not reopen dense-layout tuning. Actual
DRAM traffic is not implied by a retained-value count. Full latency and later
quality must justify any continuing integration cost.

## Device Probe Contract

`DeviceSparseInputWorkspace` owns reusable K-entry u32 indices, K-entry FP32
values and one u32 count, exactly `8*K+4` requested bytes. A single 256-thread
CTA compacts contiguous input ranges with deterministic prefix sums. The next
kernel reads the device count and only loads weights for retained indices;
there is no host count read, allocation or stream synchronization in enqueue.
Both operations use decode-stream ordering. Diagnostic readback drains decode.

The Q8 consumer shares the retained dense-tiled value loads and FP32 reduction.
It accepts only tile widths8/16, threads128/256, one input row and matching
typed layout/dimensions. Source weights are unchanged. Workspace, weights and
outputs must outlive stream completion, including failure paths. This is a
standalone research API, not a generation flag or a trained threshold artifact.

```bash
source scripts/dev-env.sh
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test sparse_activation_cuda \
  --test q8_tiled_projection -- --include-ignored --test-threads=1
```

Hardware checks compare GPU-selected indices/value bits with the CPU contract,
guard output tails, reject invalid calls without launches, and compare every
output against native Q8 with the same zeroed input. Independent FP64 samples
cover real `K=11008,N=2048` down dimensions. No speed or model-quality conclusion
follows from these numerical tests.

## Release Screen

The existing execution-weight benchmark has an explicit `--sparse-inputs`
mode. It requires release, `--layout input-tiled`, `--rows 1` and explicitly
named FFN down tensors. Other flags retain their previous lossless meaning.
All source bytes, hashing, preparation, isolation checks and JSONL IO are
shared with the existing runner, not another model-loading implementation.

```bash
source scripts/dev-env.sh
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo build --release --features cuda --example q8_execution_bench
target/release/examples/q8_execution_bench --sparse-inputs \
  --layout input-tiled --rows 1 --tensors blk.0.ffn_down.weight \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --trials 5 --iterations 100 --output /tmp/sparse-input-screen.jsonl
```

Each of twenty cells has five alternating native/candidate pairs after warmup.
Native consumes the same CPU-thresholded input as the selected-input reference;
the candidate includes GPU selection and projection. Selection-only and
projection-only timings are separate diagnostics, not additive E2E predictions.
The deterministic synthetic inputs screen the layout prerequisite; their fitted
quantile is not per-layer model calibration or a held-out quality result.
Index/value equality, full native output parity and FP64 samples are checked
before timing. Preparation cost, source plus duplicate tiled weights, workspace
capacity and achieved sparsity are explicit. Source-level requested load bytes
are not DRAM transactions; repeated scale/index loads may hit caches. Tracked
device payloads are released before the terminal completion record; primary
CUDA context overhead can remain until process exit.

## Current Checkpoint

The [clean release screen](experiments/2026-10-25-activation-input-sparsity.md)
passes the layout prerequisite at roughly 40-60% synthetic input sparsity,
including GPU selection cost. The subsequent
[frozen real-model evaluation](experiments/2026-10-25-activation-quality.md)
completes C23 without a promotion: target 25% is slower, target 40% regresses a
dense-valid task, and targets 50/60% improve held-out matched-prefix rate about
4.3/6.7% while increasing PPL 8.35/17.38%. All positive targets add about
822.46 MiB of weights/scratch/counters. Zero-threshold full-model execution
retains native projection and exactly matches the reference. These are
approximate-model tradeoffs, not HTTP/generated throughput claims. Both
unchanged-model speed objectives remain unmet; runtime defaults are unchanged.

## Calibration and Application Sequence

The calibration implementation follows these atomic units without promoting
the experiment:

1. `feat(infer): define activation calibration artifacts`: bounded exact
   observations, frozen calibration-task eligibility, generated-input absolute
   positions, model/corpus/execution identity and per-layer thresholds. Reject
   incomplete/held-out/nonfinite populations; include CPU behavioral tests.
2. `feat(infer): apply calibrated sparse activations`: model-local opt-in
   capture/application, immutable prepared down weights, reusable scratch and
   explicit unsupported-path rejection. Depends on the checked artifact.
   Verify zero-threshold full-model parity and observation boundaries on M40.
3. `test(quality): evaluate sparse activation variants`: calibration-only
   capture tooling plus explicit artifact/target inputs to the frozen quality
   runner. Depends on application. Record outputs, actual IDs, losses, logit
   drift, setup/memory and matched-prefix timing with selection included.
4. `docs(perf): record activation quality tradeoffs`: freeze artifacts before
   comparing targets0/25/40/50/60 on both C19 splits. Do not choose thresholds
   from held-out/browser results or change defaults.

The initial capture cap is 32 generated-token input rows per calibration task,
declared before capture. Prompt rows and the final sampled token that is never
forwarded are not observations. Equal ranges are required at every layer.
The cap bounds host payload and biases the sample toward early generation;
report that limitation rather than implying model-wide calibration coverage.

`LoadedModel::begin_activation_calibration` and
`apply_activation_calibration` configure model-local state before KV/workspace
creation. Capture begins/ends explicitly per eligible task and exports only
after all layers observed the same generated ranges. Application checks the
uploaded weight digest and exact down tensor names/offsets/shapes; the runner
must also check the original file, corpus and execution hashes. Preparation
reserves duplicate down weights only, one reusable selection workspace and
two u64 counters per layer. Packing/upload proceeds one tensor at a time.

This path requires sm_52, Q8_0 down weights, decode-stream Q8 scheduling,
explicit dense KV and sequential forward/prefill. Graphs, packed/batched work,
layer masks and other prepared-weight policies are rejected. It is not wired
into ordinary CLI/server defaults. Device counters accumulate retained/input
values in the existing selector CTA, on its ordered stream. Reading/resetting
them is a report/setup operation; ordinary application adds no host readback.

Zero-threshold layers keep native projection and allocate no tiled copy.
Their inputs are unchanged, so compaction has no semantic benefit and can add
reduction-order drift that crosses FP16 KV rounding boundaries. The full-model
zero control requires exact native logits; independent sparse-kernel parity
still covers threshold zero numerically. Statistics list native bypass layers
explicitly rather than inventing selected-input counts for that route.

The existing `model_quality` runner accepts `--capture-activations new.json`
only with `--split calibration`, no candidate transforms and no timing trials.
It uses the same task formatting, greedy generation, token accounting and
model loader as evaluation. Capture JSONL retains actual prompt/generated IDs,
outputs and fixed scores. Diagnostic readback timing is not decode performance.
The artifact is created once after all eligible tasks finish; keep its hash.

Evaluation uses `--activation-calibration new.json --activation-target 0.4`
for the candidate only. Model file, uploaded source, corpus, base execution,
task IDs and layer identities must match. All 0/25/40/50/60 targets are explicit;
there is no implicit activation default. `--activation-budget-mb` counts extra
device payload, while `--activation-observation-budget-mb` bounds capture host
payload. Per-task counters cover all forwarded prompt/generated inputs since
reset, not just decode, and never report a selected fraction for native bypass.

Use the unchanged native-Q8 environment from the C11 checkpoint, with no
prefill chunking or KV diagnostics. Keep that exact environment/context for
capture and every evaluation; execution mismatch is an error, not a refit.

```bash
source scripts/dev-env.sh
while IFS=$'\t' read -r key value; do
  export "$key=$value"
done < <(head -n 1 docs/experiments/2026-10-25-qwen-exact-checkpoint.jsonl | \
  jq -r '.environment | to_entries[] | [.key,.value] | @tsv')
model="$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
cargo build --release --features cuda --example model_quality
target/release/examples/model_quality --reference-model "$model" --model "$model" \
  --variant-id native-activation-calibration --split calibration \
  --capture-activations /tmp/qwen-activation-calibration.json \
  --output /tmp/qwen-activation-capture.jsonl

# Freeze the artifact before evaluating any candidate or held-out model output.
for target in 0 0.25 0.4 0.5 0.6; do
  target/release/examples/model_quality --reference-model "$model" --model "$model" \
    --variant-id "activation-target-$target" --split all \
    --activation-calibration /tmp/qwen-activation-calibration.json \
    --activation-target "$target" --step-timing-trials 5 \
    --output "/tmp/qwen-activation-$target.jsonl"
done
```

All GPU runs are serial, without overlapping tests/builds or other inference.
Both task and loss reports preserve dense-failed cases as inconclusive. The
paired timing is identical forced-prefix work including selection and final
completion, not HTTP/generated TPS. Inspect complete generation timings too;
different stop lengths cannot justify an engine speed claim.

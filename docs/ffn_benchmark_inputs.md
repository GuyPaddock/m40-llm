# Paired FFN Benchmark Inputs

M02 in the [Maxwell Q8 investigation](maxwell_q8_execution.md) needs actual
normalized gate/up inputs and completed SwiGLU inputs to down from the same
forward pass. Activation-sparsity calibration stores only the latter and
forbids the qualified prepared down/vocabulary control. Do not enable a
sparsity transform to obtain these observations.

## Artifact Contract

`evaluation::ffn_capture` provides a CPU-only collector and checked JSON
artifact. It has no inference policy, threshold fitting or CUDA dependency.
The shared benchmark consumer remains a separate M02 step.

- A plan binds model, corpus and execution SHA-256 identities, original layer
  count, selected original IDs, dimensions and ordered calibration task IDs.
  The real Qwen checkpoint selects layers 0, 17 and 35, with dimensions 2048
  and 11008. Tests may use smaller complete Q8_0 blocks.
- Gate, up and down sources retain GGUF names, relative tensor byte offsets,
  Q8_0 type, logical shapes and hashes of the native tensor bytes. Aliased
  source ranges, mismatched shapes and renamed original layer IDs are invalid.
- Each row pairs normalized FP32 input to gate/up with completed FP32 SwiGLU
  input to down, plus task ID and absolute token position. The vectors are
  row-major and unmodified. They are not repeated synthetic rows.
- Capture excludes prompt inputs. If the prompt has `P` tokens and generation
  actually emits `G` tokens, eligible generated-input positions start at `P`
  and contain `min(G - 1, row_cap)` rows. The last emitted token has not yet
  been processed. The row cap is 1 to 32 per task.
- Every selected layer must supply the same contiguous positions for every
  frozen calibration task. Incomplete, duplicate, out-of-order and nonfinite
  pairs fail. A rejected pair appends neither vector; a failed task completion
  leaves the active task available for correction.
- Reserve all FP32 observation storage before capture under an explicit byte
  budget. Report used and reserved FP32 bytes separately. These counts exclude
  metadata, serialization, allocator overhead and model-observer read scratch.
  They are not process RSS or GPU memory measurements.
- Export requires all tasks complete and at least one observed row. JSON is
  bounded to 64 MiB and validates positions, sources, finite values and memory
  counts. Prompt and actual generated token IDs remain in the artifact.

The parser establishes internal consistency, not trusted provenance. Consumers
must call `plan.validate_corpus` with the actual frozen corpus/hash, check the
original model and tensor hashes and execution identity, and verify the row
selection against their benchmark configuration. No held-out or browser input
may supply these calibration observations.

## Model Observer

`LoadedModel::begin_ffn_capture` installs model-owned host observations before
KV or forward workspace creation. It permits prepared down/vocabulary weights,
but forbids concurrent activation transforms, layer influence and layer masks.
It requires sm_52, decode-stream Q8 projections, sequence zero, explicit dense
KV and ordinary sequential forward. Packed prefill, batched forward and graph
entrypoints reject capture before device work. Configure this only for untimed
diagnostics, never an authoritative throughput trial.

Call `begin_ffn_capture_task` with the next frozen task ID and encoded prompt
IDs. Immediately before down projection, the observer drains the decode stream
once and reads normalized input plus completed SwiGLU for selected original
layers at eligible positions. Inactive capture and unselected layers add no
GPU copies, launches or synchronization. `finish_ffn_capture_task` supplies the
actual generated IDs; `finish_ffn_capture` validates all tasks before removing
the observer and exporting the artifact. Incomplete export preserves the state.

Setup reserves two reusable host read buffers in addition to the observation
vectors, under the same host budget. Their capacity is
`4 * (d_model + hidden_dim)` bytes. The artifact's observation counts exclude
these temporary read buffers, which are freed on export/model drop. Capture
owns no device allocation and never writes model buffers.

## Current Validation

Eight CPU tests cover paired values and distinct rows, original IDs and source
shapes, signed-zero/subnormal JSON round-trip, calibration-only eligibility,
row bounds, invalid-pair atomicity, incomplete capture, short generation,
budget/overflow/alias failures, external artifact corruption and corpus binding.
Two explicit M40 observer tests cover fused/unfused gate/up, all/selected-layer
capture, prepared down/vocabulary, exact logits and KV-byte parity, no added
kernel/allocation activity, the documented D2H/sync deltas, paired-input replay
and an independent CPU projection/SwiGLU reference. They also cover invalid
setup, incomplete export, unsupported paths and device-allocation cleanup.

The warning-as-error non-CUDA test lane and CUDA/server all-targets Clippy
pass. M40 attention/projection/MLP parity passes with cuBLAS enabled/disabled;
the cuBLAS-enabled lane also passes activation calibration, layer influence,
full forward/prefill and allocation ownership regressions. Explicitly invoking
the observer hardware test with `CUDA_VISIBLE_DEVICES=''` fails with a context
creation error, as required; GPU-less CI does not establish observer parity.

Real-model capture evidence and complete FFN timing still remain. No performance
improvement follows from these contracts or correctness tests.

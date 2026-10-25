# Paired FFN Benchmark Inputs

M02 in the [Maxwell Q8 investigation](maxwell_q8_execution.md) needs actual
normalized gate/up inputs and completed SwiGLU inputs to down from the same
forward pass. Activation-sparsity calibration stores only the latter and
forbids the qualified prepared down/vocabulary control. Do not enable a
sparsity transform to obtain these observations.

## Artifact Contract

`evaluation::ffn_capture` provides a CPU-only collector and checked JSON
artifact. It has no inference policy, threshold fitting or CUDA dependency.
The model observer and shared benchmark consumer are separate M02 steps.

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

## Current Validation

Eight CPU tests cover paired values and distinct rows, original IDs and source
shapes, signed-zero/subnormal JSON round-trip, calibration-only eligibility,
row bounds, invalid-pair atomicity, incomplete capture, short generation,
budget/overflow/alias failures, external artifact corruption and corpus binding.
This contract alone is not GPU observer parity, real-model capture evidence,
complete FFN timing or a performance improvement.

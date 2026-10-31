# Rotated activation producer checkpoint

The isolated staged and fused producers now pass the numerical and ownership
tests on the M40 with cuBLAS enabled and disabled. This establishes correctness
for the fixed experiment. Timing and a compatible rotated-model consumer
remain pending.

## Implementation

`DeviceRotatedActivation` owns immutable signs and guarded signed-input,
canonical FP32 and Q8_1 buffers. Construction fixes the shape, validates signs
and completes uploads. Every enqueue recomputes both outputs. The caller
orders producer/consumer streams and retains the owner through completion.
Diagnostics synchronize both streams before reading outputs and canaries.

The staged schedule launches sign multiplication, a normalized Walsh128
rotation, and the existing Q8_1 quantizer. The fused schedule applies signs on
load, performs the same butterfly and normalization, writes canonical FP32,
then encodes each warp directly into Q8_1. Both retain identical allocations.
The original quantizer and fused producer share the same per-warp encoder,
including FP32 max/sum reductions, ties-away rounding and FP16 scale/sum
storage. Its upstream copyright and MIT notice remain in place.

The C ABI and Rust wrapper accept rows 1, 2, 4 and 8 and widths divisible by
128. Flattening complete blocks preserves the existing Q8_1 row contract.
The unchanged gate/up consumer processes one row at a time. The entry is
research-only; the runtime does not select this producer.

## Validation

Three new ignored hardware tests run explicitly, serially, and pass with
cuBLAS on and off. The main test covers all twelve combinations of widths
128, 2048 and 11008 with rows 1, 2, 4 and 8. It includes zeros, tied values,
distinct rows, large signed values and overwritten inputs at the same address.

- Every canonical output is checked against independent dense FP64 Walsh
  coefficients, with `1e-5 + 2e-6*abs(reference)` tolerance.
- Every encoded byte matches the existing CPU Q8_1 reference applied to the
  validated canonical values.
- Staged and fused canonical values, encoded bytes and downstream output
  bits match exactly, on both streams with explicit upstream and downstream
  dependencies.
- Independent owners with different shapes can be enqueued on separate
  streams. Reusing one preserves the other's outputs.
- Guard bytes, immutable inputs/signs/weights and tracked allocation cleanup
  pass. Each producer call records exactly three or one launches, with no
  allocations, transfers, cuBLAS calls, synchronization or stream waits.
- Invalid shapes, signs, input alignment, null/overflowing spans and aliases
  fail before GPU work or allocation.

Existing Q8_1 tests pass with both builds: five activation tests, including
three CPU contract tests and two hardware tests, and two consumer hardware
tests. The cuBLAS-enabled compound forward/ownership suite passes three tests,
and the attention parity grid passes eleven. The Q8 FFN runtime suite passes
ten tests. CUDA/server all-target Clippy, the warning-as-error non-CUDA suite
and formatting pass. Ignored or feature-excluded hardware tests in the
non-CUDA run are not counted as hardware passes.

## Initial failures and reference correction

The first test build failed on a missing FP32 type annotation and missing
stream-wait labels. Those test setup errors were corrected in this unit.

An extra consumer check initially reconstructed each Q8_1 activation to FP32
before a scalar projection and used only a result-relative output tolerance.
A large synthetic input produced `-2110.2441` versus `-2110.2946164380774`,
exceeding that tolerance. The test failure also poisoned the process-local
CUDA test lock, so the subsequent rejection test was not a valid result.

The corrected independent reference operates on the encoded integer values
and block scales, matching the consumer's representation. Its numerical
bound accounts for FP32 product/reduction roundoff using `gamma_n` times the
sum of absolute terms, then propagates the two projection bounds through
SwiGLU. This protects cancellation-heavy synthetic cases without assigning a
new model-quality tolerance. The frozen producer tolerance and all exact
byte/output-bit requirements are unchanged. The existing Q8_1 regression
tests retain their original tolerances. Both failed logs are preserved.

Logs beside this report use the prefix `2026-10-31-rotated-activation-` and
suffix `.log.gz`. The `on`, `off`, `regression`, `runtime`, `clippy` and `noncuda` logs
record the passing checks above. `initial-build` and `initial-consumer` record
the failures described here.

## Reproduction and next unit

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/activation-preparation/target
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test rotated_activation --test q8_activation --test q8_ffn_q81 \
  -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=0 cargo test --features cuda \
  --test rotated_activation --test q8_activation --test q8_ffn_q81 \
  -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --test q8_compound_runtime \
  -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --test attention_parity_cuda_grid \
  -- --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --test q8_ffn_runtime \
  -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo clippy --features cuda,server --all-targets \
  -j 4 -- -D warnings
RUSTFLAGS='-D warnings' cargo test --no-default-features --locked -j 4
cargo fmt --all -- --check
```

The next unit implements the source-bound producer-only and producer-plus-
consumer release comparison from the
[fixed protocol](2026-10-31-rotated-preparation-protocol.md), with retained
inputs and matched 256-channel rotated weight fixtures. No performance,
whole-model accuracy or runtime integration claim follows from these tests.

## Source-bound benchmark unit

`q8_execution_bench --ffn-rotated-preparation` is exclusive with other modes,
requires the retained capture and exact environment, and checks clean release
source before loading the model. It fixes rows `1,2,4,8`, five pairs and
100 operations per sample. The higher iteration count is fixed before timing
to reduce event-timing noise for short producer operations.

Each of 24 layer/width/row cases compares preparation alone and preparation
plus the same 256-channel Q8_1 consumer, for 48 comparisons. Both arms use the
decode stream, with the consumer enqueued after preparation on that stream.
Warm each arm and boundary for at least one GPU second. Warmup uses batches
of 256 operations and a maximum of 4096 batches so short operations can reach
the target without an unbounded loop. Warmup samples never enter primary pairs.

The timing callback reuses the established alternating-order and bootstrap
protocol, with this topic's 3% complete-operation floor. Extend five pairs to
ten only when throughput MAD/median exceeds 5%; continued instability fails.
Collect one fresh twenty-pair confirmation near the gain gate or when the
point estimate clears it but the lower confidence bound does not. Report both
phases. Preparation-only results are diagnostic.

The benchmark fits matched rotated Q8_0 weight slices from the original Q8
source, recording source rows, hashes, fit error, preparation and upload costs.
It checks all eight retained inputs are distinct at each width. Before and
after timing, both paths must pass dense FP64 Walsh checks, CPU Q8_1 encoding,
the encoded-consumer numerical check, and exact canonical/encoded/output bit
comparisons. Input and scratch guards, immutable signs/weights and allocation
cleanup are checked separately. All owners stay resident in both arms.

Logical traffic counts describe the signed scratch write/read and canonical
quantizer read removed by fusion, 12 bytes per activation element. The canonical
and encoded output writes remain. No measured DRAM or sector claim follows
from that count. All model, kernel and consumer qualifications remain unchanged.

The benchmark unit passes 66 CPU example tests, CUDA/server all-target Clippy
and formatting. Twenty existing ignored hardware tests are not counted as
passes. Logs are `2026-10-31-rotated-benchmark-tests.log.gz` and
`2026-10-31-rotated-benchmark-clippy.log.gz`. Release measurements are pending.

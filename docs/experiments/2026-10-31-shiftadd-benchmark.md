# ShiftAdd projection benchmark implementation

The existing `q8_execution_bench` now accepts `--ffn-shiftadd`, with
`--rows 1,2,4,8 --trials 5 --iterations 100`. The
[frozen protocol](2026-10-31-shiftadd-projection-protocol.md) defines the
comparison and admission gates. No measured performance result is recorded
in this implementation checkpoint.

The benchmark checks the retained capture, model and environment identities,
requires a clean release source tree, and refuses to overwrite output. Each
of nine full projection matrices is fitted once, then tested at four batch
sizes. It holds all four arms' weights resident throughout the comparison.
Native Q4/Q8 use one decode launch per row; SA4 and prepared Q8 use one batch
launch. Prepared Q8 uses the existing packed-input 4-column/128-thread path.
This is an explicitly selected projection control, not a claim that it is
the best complete FFN implementation.

Independent FP64 decoded-weight dots and FP32 operation-count bounds check
GPU arithmetic. The ShiftAdd bound uses absolute plane contributions before
sign cancellation. Weight and captured-projection fit errors versus original
Q8 are reported separately. Before/after checks cover finite results, input
and output guards, immutable input and encoded/prepared weights. Each timing
sample verifies the expected operation and launch count, no allocation or
free, and unchanged tracked residency. Per-case and final cleanup are checked.

The paired timing code reuses the FFN fusion sampler and existing bootstrap,
with the frozen 3% projection floor. No new statistical procedure is added.
Reusable CUDA events bracket complete projections; CPU reference work,
fitting, uploads, allocations and warmup are outside primary samples and
reported separately. Logical payload sectors are a cache-independent lower
bound on the backing footprint, not memory transaction measurements.

The CPU example suite passes 65 tests, including six new checks for signed
Q8 decoding, distinct rows, corrupt-output rejection, plane cancellation,
fit-error calculation, pair ordering/failure, decisions and CLI constraints.
Twenty unrelated hardware tests remain ignored and are not counted as
passes. CUDA/server all-target Clippy passes. The warning-as-error non-CUDA
suite passes 230 tests. Formatting and whitespace checks pass. Raw logs have
prefix `2026-10-31-shiftadd-benchmark-`; hardware behavior of the candidate
consumer is documented in the [CUDA qualification](2026-10-31-shiftadd-cuda.md).

Next build this committed source in release, inspect its SA4 kernel, restore
the capture environment and run one serial M40 screen. The benchmark itself
checks all four projection arms before collecting each case's timings.
No generated tokens, full-model quality or whole-model throughput is inferred.

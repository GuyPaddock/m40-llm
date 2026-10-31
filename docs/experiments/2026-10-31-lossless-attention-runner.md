# Lossless attention fixture admission and runner checkpoint

The CPU fixture tool admits the complete eight-sequence capture and all twelve
layer/context cases. It rejects partial, misidentified or corrupted inputs,
preserves original FP16 bits when converting token-major capture views to head
streams, and reports all encoded bytes. The runner's fourteen tests pass,
including hardware append/history checks, with cuBLAS enabled and disabled.
Real-input release qualification and measurements remain pending.

## CPU fixture unit

`examples/lossless_attention_bench/capture.rs` and `capture_tests.rs` reuse the
frozen reader from quantized-KV commit `5362671` without changes. The files'
SHA-256 digests are respectively
`89eb4151fe22719db61c669207ed69f436a83b241cbb713f3b23cae89cd21ff9` and
`850628c36a5e33bcc357c3295a0bb8b003b87e2a7bd3ed7677132385eb650f77`.
Both branches use the same capture identity, record and stream checks.

`fixture.rs` builds original head128 K/V streams and an independent FP64
full-context GQA reference. Its tests use hand-computable means to check
sequence/head mapping and require an appended captured token to restore the
original prefix bits. Signed zero survives the layout conversion. Invalid
lengths or values fail before mutating append state. CPU encoding includes
format validation and exact round-trip checks.

Four inherited reader tests cover the complete binary hash and stream bounds,
metadata and model identity, duplicate or incomplete records, per-stream
digest binding, non-finite inputs and distinct synthetic queries. The new
fixture tests add two cases. These tests require no model or GPU.

The standalone committed entry point inspects twelve layer/context cases
using layers 0/17/35 and contexts 1K/2K/4K/8K. It requires a complete capture;
the [full capture](2026-10-31-lossless-capture-results.md) now passes. Reports include input
hashes, encoded stream statistics and the extra device pointer directories.
Queries remain deterministic synthetic vectors, not captured model queries.

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/lossless-kv/target
RUSTFLAGS="-D warnings" cargo test --no-default-features --locked \
  --example lossless_attention_bench
cargo run --release --no-default-features --example lossless_attention_bench -- \
  --capture-report COMPLETE_CAPTURE.jsonl --kv-inputs COMPLETE_CAPTURE.bin \
  --output NEW_INSPECTION.jsonl
```

## GPU runner

`pipeline.rs`, `benchmark.rs`, `append.rs` and the GPU CLI additions implement
the [frozen comparison](2026-10-31-lossless-kv-consumer.md). They are committed
separately after the consumer. Kernel and append hardware tests execute and
pass; CUDA/server all-target Clippy also passes. The
[validation record](2026-10-31-lossless-consumer-validation.md) distinguishes
supported regression passes from command failures and optional skipped diagnostics.

Each case retains eight distinct captured sequences and shared guarded query,
output, staging and partial buffers. The device codec supports decoding a
prefix of its stream directory, so a batch-one staged operation reconstructs
only batch-one K/V. Its hardware test now checks that inactive streams remain
unchanged and invalid prefix counts fail.

Before timing, every active batch checks exact reconstructed K/V, FP64
attention with absolute tolerance `1e-3`, matched prototype output bits,
repeated output stability, original input bits and guard regions. The native
head128 control retains its separate numerical tolerance and reduction order.
An observed numerical failure is recorded and stops the case before timing.

Five attention comparisons run per batch:

- Matched dense versus separate reconstruction plus attention.
- Staged versus fused reconstruction inside attention.
- Matched dense versus fused attention.
- Existing native serial attention versus batched fused attention.
- Serial versus batched execution of the same fused kernel.

The last comparison separates dispatch savings from codec savings. Native
attention uses the existing eight-partition, 256-thread path and full token
coverage. None of these operations invokes model inference or changes defaults.

Every timed output must match its own untimed output, with readback outside
the timing interval. Attention-only intervals reject device allocation/free
or host copies. Three warm observations precede five alternating pairs; the
existing variation extension and fresh-confirmation rules apply. Reports
preserve both completed host wall time and CUDA-event elapsed time. The
complete grid contains 12 cases, 48 active batches, 240 attention comparison
groups and 48 append-plus-attention groups.

Append comparisons recreate a length-minus-one prefix before each observation.
The native arm truncates its actual cache and verifies its shorter length.
Its timed interval appends the resident final GPU token and runs native
attention. The CPU-compressed arm retains an original-FP16 host history and
an encoded device prefix. Its timed interval reads the new GPU token back,
extends that host history, re-encodes and validates every stream, allocates
and uploads replacement storage, retires old device storage, and runs fused
attention. Recreating pre-existing prefixes is outside the interval and
reported separately. New-token readback, replacement allocation and retirement
are charged. The host-history assumption and its bytes are explicit.

Append output checks compare with each arm's already validated full-prefix
output. New compressed storage must also decode to the captured K/V words.
Native valid lengths and tracked cleanup are checked after every observation.
Append CUDA-event elapsed time includes host preparation gaps; it is not GPU
active kernel time. No incremental encoder or unmeasured amortization is assumed.

The runner requires a clean release build, a full hash-verified capture and
the selected M40. It rejects attention/logging overrides and other GPU compute
processes before context creation and between comparisons. Correctness-only
mode executes all numerical and append checks without collecting paired
performance observations. Capture and initial hardware qualification have completed.

```bash
# Reproduction only; require an idle GPU before each command:
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --test lossless_kv_cuda \
  --example lossless_attention_bench -- --include-ignored --test-threads=1
# Run affected regressions from the consumer checkpoint, then commit the
# verified CUDA consumer before the dependent GPU runner.
M40LLM_ENABLE_CUBLAS=1 cargo build --release --features cuda \
  --example lossless_attention_bench --locked -j 4
target/release/examples/lossless_attention_bench \
  --capture-report COMPLETE_CAPTURE.jsonl --kv-inputs COMPLETE_CAPTURE.bin \
  --benchmark --correctness-only --output NEW_CORRECTNESS.jsonl
target/release/examples/lossless_attention_bench \
  --capture-report COMPLETE_CAPTURE.jsonl --kv-inputs COMPLETE_CAPTURE.bin \
  --benchmark --iterations 10 --output NEW_TIMINGS.jsonl
```

Full capture audit and initial serial hardware tests pass. Real-input release
qualification, controlled timings, result analysis and the branch decision
remain required. The
[validation checksum manifest](2026-10-31-lossless-attention-runner.sha256)
retains CPU and compile-only logs. The prior 225-test non-CUDA suite remains
recorded in the consumer checkpoint; it was not relabeled as GPU coverage.

## Queued qualification

This section preserves the original queue contract. Capture `95030` and
qualification `68704` are now terminal with exit code zero. The
[completed logs](2026-10-31-lossless-initial-qualification.sha256) establish
passing execution. No GPU slot remains reserved by either session.

Session `68704` reserved the next GPU slot after capture. Its
[retained queue script](2026-10-31-lossless-qualification-queue.sh) waits for the
existing capture PID to retire. It then checks the frozen source hashes, all
96 streams, 384 compression rows, 24 readbacks, eight sequences, completion,
tracked cleanup and the 201,326,592-byte artifact's digest. It requires an idle
M40 before CPU fixture admission and before each hardware test.

The CPU fixture inspection reconstructs all twelve real layer/context cases.
The job then runs `lossless_kv_cuda` followed by the `lossless_attention_bench`
example tests with `--include-ignored --test-threads=1`. Every failure stops the
queue. It performs no real-grid performance measurement and makes no commits.
Review its logs before the affected regression suites, code commits and clean
release comparisons.

The executed copy and outputs are under
`/tmp/lossless-qualification-20261031-01/`: `queue.log`, `capture-summary.json`,
`fixture-inspection.jsonl`, `fixture-inspection.log`, `kernel-tests.log` and
`runner-tests.log`. All phases completed and the logs are now retained in the
repository. Do not restart the capture or qualification merely to refresh them.

The [source manifest](2026-10-31-lossless-qualification-source.sha256) pins the
pending implementation and tests. The
[queue checksum manifest](2026-10-31-lossless-qualification-queue.sha256)
preserves the exact launch script and capture-summary program. These records
document queued work, not passing hardware evidence.

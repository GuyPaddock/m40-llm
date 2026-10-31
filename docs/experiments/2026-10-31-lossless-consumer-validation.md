# Lossless consumer hardware validation

The opt-in CUDA decoder and matched dense/fused attention consumer pass their
M40 hardware tests. Both decoder/attention tests execute with cuBLAS enabled
and disabled. The runner's fourteen tests also pass in both configurations,
including its hardware append/history/output check. These qualify the bounded
implementation for the real-input release study; no timing gain is established.

The source bytes match the twelve-file
[qualification manifest](2026-10-31-lossless-qualification-source.sha256).
The [initial logs](2026-10-31-lossless-initial-qualification.sha256) retain
cuBLAS-enabled decoder and runner execution. The
[regression manifest](2026-10-31-lossless-regressions.sha256) retains the exact
three invocation scripts, all outputs, failures and final idle-device checks.

## Executed coverage

With cuBLAS enabled, affected correctness coverage passes for batched attention,
the eleven-case attention grid, all five stream/transfer ownership tests,
minimal and toy forward parity, layer forward cases, mixed and fallback GEMM,
head128 parallel parity/workspace reuse, projection/MLP wrappers and tracked
weight replacement/cleanup. There are 75 successful executed correctness
tests in these regression suites, excluding the two optional model diagnostics
that returned without running and the release-only benchmark described below.
The initial two lossless and fourteen runner tests are recorded separately.

With cuBLAS disabled, 46 supported tests execute successfully across minimal
and toy forward parity, fallback GEMM, 22 mixed GEMM cases, both lossless tests,
projection/MLP wrappers and the fourteen runner tests. Decoder reconstruction
covers all 65,536 FP16 bit patterns. Attention checks cover dimensions 64/128,
batches 1/2/4/8, five lengths, GQA, two allocation lifetimes, guards, invalid
aliases and explicit producer/consumer dependencies.

Previous non-CUDA warning-as-error, CUDA/server Clippy and formatting results
remain recorded in the [consumer protocol](2026-10-31-lossless-kv-consumer.md).
No source change occurred between those checks and this hardware execution.

## Retained failures and omitted diagnostics

The first broad command used `--include-ignored` on the head128 suite. That
selected `benchmark_head128_geometries`, which rejected the debug executable
before timing. This command exited 101. The two head128 correctness tests
still executed successfully. The unused geometry search is not rerun: the
lossless comparison already fixes eight partitions and a native 256-thread
control, and will run its own complete-operation release benchmarks.

The broader cuBLAS-disabled command also exited 101 at
`materialized_gguf_f32_sgemm_matches_cpu_rectangular_2x3x4`. The existing FP32
SGEMM wrapper returns `-5` without `M40LLM_HAVE_CUBLAS`. Its implementation and
test are unchanged from branch base `0eee578`; the CUDA translation unit's
only change is including the new lossless header. The same test passes with
cuBLAS enabled. This unsupported configuration is not relabeled as passing
cuBLAS-disabled parity. The supported remaining suites ran in a subsequent
serial command and passed.

The optional real-Qwen mixed-prefill diagnostic and optional head128 generation
smokes did not execute because their model inputs were not supplied. They
provide no model-quality evidence. The new lossless tests and append test
did execute; none skipped for unavailable hardware. The source manifest
matches after testing and the final external M40 check reports zero allocation
and no compute process.

## Next qualification

Commit the dependent GPU runner, build it in release mode and run the entire
captured input grid with `--correctness-only`. Only then run the paired
attention and append comparisons. Preserve full-context coverage, exact KV
reconstruction, matched output bits and separate native numerical tolerance.
Runtime dispatch and defaults remain unchanged.

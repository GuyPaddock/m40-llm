# Quantized KV reference preserves integer zero semantics

The [initial staged-reference failure](2026-10-31-quantized-kv-initial-failure.md)
is a CPU oracle error. The focused replay identifies head64, physical slot 3,
length 9, query head 0, token 1, channel 3. Input `-0.0084991455` reconstructs
to GPU FP16 bits `0x0000`; the test expected `0x8000`.

The [diagnostic log](2026-10-31-quantized-kv-zero-diagnostic.log.gz) retains
both bit patterns. CUDA's existing Q4 quantizer converts the rounded value
to an integer code. Integer zero has no sign, and multiplication by the
positive scale reconstructs positive zero. The test had retained a rounded
floating-point negative zero, which incorrectly survived FP16 conversion.
The benchmark's separate CPU reference already converts codes to integers.

The correction changes only the staged hardware test's reconstruction oracle:
convert the clamped rounded code to `i32`, then reconstruct it as FP32 before
FP16 rounding. No CUDA arithmetic, encoded representation, runtime backend,
selection policy, numerical tolerance or exact-byte assertion changes.
A focused test checks that negative inputs quantized to integer zero produce
positive zero, while recent unquantized FP16 history preserves negative zero.
Failure diagnostics now identify slot, width, length, token, channel and bits.

The failing staged implementation remains in `ebc5d5f`. The correction is
separate because failed experiments are retained and history rewriting is
not authorized. Full serial hardware qualification must pass before committing
the dependent runner or admitting a clean real-input timing grid.

## Hardware qualification

The corrected oracle passes the staged hardware test at head64 and head128,
and its focused zero-semantics test passes. All seventeen benchmark-runner
tests pass, including real M40 pipeline/ownership execution over the synthetic
fixture grid. Both test targets also pass with cuBLAS disabled. The additional
cuBLAS-disabled projection, MLP and minimal/toy forward checks pass.

The affected cuBLAS-enabled regression run reports 49 cases. One optional
real-model diagnostic has no model input and does not execute; the remaining
48 checks execute and pass. They cover attention, transfer ordering,
projection/MLP, forwards including the included graph checks, and allocation
ownership. This establishes no model-quality result. Source hashes match
before and after the run. The external final M40 query reports zero MiB
and no compute process.

The [qualification script](2026-10-31-quantized-kv-qualified.sh),
[initial corrected tests](2026-10-31-quantized-kv-qualified-initial.log.gz),
[affected regressions](2026-10-31-quantized-kv-qualified-affected.log.gz),
[cuBLAS-disabled checks](2026-10-31-quantized-kv-qualified-cublas-off.log.gz)
and [post-process log](2026-10-31-quantized-kv-qualified-queue.log.gz) retain
these results. Full real-input release qualification and timing remain pending.

Fresh CUDA/server all-target Clippy and formatting pass, as do all nine
warning-as-error CPU fixture/reference tests. Their
[Clippy log](2026-10-31-quantized-kv-qualified-clippy.log.gz) and
[CPU log](2026-10-31-quantized-kv-qualified-cpu.log.gz) are included in the
[qualification checksums](2026-10-31-quantized-kv-qualified.sha256).

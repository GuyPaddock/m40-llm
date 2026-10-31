# Synchronize shared Q8 quantizer reductions

Both Q8 quantizers reused their shared reduction array immediately after
reading the K maximum. Without a barrier, warp zero could store its V value
into `reduce[0]` before another warp loaded the K maximum. Those later threads
could then encode K with a different scale from the scale stored by thread
zero. Both sites predate this research branch and exist in base `0eee578`.

The correction adds a block barrier after reading the K maximum, before the
array is reused for V. It applies to streamed recent-token eviction and the
dense-to-Q8 builder. Quantization arithmetic, representation, selected positions,
attention semantics and exact validation requirements remain unchanged.

The original standalone runs fail at different positions, including an
omitted block whose independent score is substantially better than the selected
replacement. Passing isolated test replays did not reproduce those failures.
The preserved evidence supports investigating the cross-warp race; it does
not justify changing the FP64 rank reference or weakening exact reconstruction.
CUDA race-checking tools are not installed in the inspected toolkit paths,
so no dynamic race-detector result is claimed.

A new hardware regression uses K values much larger than V, two warps per
head, all old tokens selected, eight sequences and four allocation lifetimes.
It checks reconstructed bytes and independent attention, with guarded outputs
and tracked cleanup. The real first-layer replay preserves original case
ordering. Both complement the clean standalone full-grid qualification,
which remains required before timings.

## Qualification before standalone release

Both staged tests and all nineteen runner tests pass with cuBLAS enabled
and disabled. This includes the distinct-scale regression and all twelve
real first-layer cases in their original order. The affected runtime run
again executes 48 checks successfully; its one optional model diagnostic
has no model input and does not run. CUDA/server all-target Clippy and
formatting pass. Source hashes remain unchanged during qualification.
The external final M40 query reports zero MiB and no compute process.

The [executed script](2026-10-31-quantized-kv-barrier-qualification.sh),
[cuBLAS-enabled tests](2026-10-31-quantized-kv-barrier-tests.log.gz),
[affected regressions](2026-10-31-quantized-kv-barrier-affected.log.gz),
[cuBLAS-disabled tests](2026-10-31-quantized-kv-barrier-off.log.gz),
[Clippy log](2026-10-31-quantized-kv-barrier-clippy.log.gz) and
[post-process log](2026-10-31-quantized-kv-barrier-qualification.log.gz) are
retained with [checksums](2026-10-31-quantized-kv-barrier.sha256).

Hardware coverage exercises streamed eviction. The dense-to-Q8 builder
receives the identical synchronization correction by inspection and compiles,
but has no separate end-to-end hardware invocation in this qualification.
The full standalone real-input grid remains outstanding; prior test-executable
passes were insufficient to establish its correctness. No timing is admitted
by this checkpoint alone.

# First main-based graph hardware run

**PASS for short attention parity in both repaired builds.**
Each fixed build completed 96 exact comparisons, including guards and
cache ownership. The unrepaired short fixture reported FP32
mismatches. The [exit codes](exits.tsv) record two passes and seven
failed invocations. The [raw archive](measurements.tar.gz) contains
the nine logs, exact binary/model preflight and GPU snapshots.

**INCONCLUSIVE for the long boundary and TinyLlama.** The long grid
discovered a short cache immediately after the test's asynchronous
seeding. In main's dense append, the next host length read uses a
blocking CUDA copy while the preceding length write was queued on the
nonblocking decode stream. The fixture must wait for each write before
calling append again. The repaired fixture reached a stored length
of 8,175 for an intended 8,193 in both builds. The original build
showed attention differences before failing the same length check.

All three TinyLlama controls stopped at full-layer validation, which
ran before the fixture allocated any layer slots. They executed no
graph replay. Move the validation after allocation. Keep the initial
failure logs and run the corrected fixture against both runtime images.
Verify the archive with `sha256sum -c archive.sha256`.

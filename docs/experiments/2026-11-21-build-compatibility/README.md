# Research build compatibility

**INCONCLUSIVE for source-specific validation with the shared Cargo target.**
All eleven initial runs exited zero. In two consolidation checkouts differing
only in the Q4 admission example and benchmark files, the number of passing
non-CUDA tests changed from 226 to 223. The second execution omitted
`metrics_round_trip_without_sample_offsets`, `sampled_offsets_include_the_stop_token`
and `test_november_kv_answers_cover_all_families`. All three still existed.
The graph-node checkout had just used the same target directory. Both
invocations ran an identically named library test binary from that directory.

The [index](shared-target/index.tsv) records observed counts, source at
invocation and archive hashes. Each tar.gz holds the complete commands,
stdout/stderr, timestamps, toolchain, build environment and exit codes.
For example, compare `tar -xOf shared-target/consolidation.tar.gz ./noncuda.log`
with the same member of `consolidation-02.tar.gz`. These retained counts
cannot qualify the individual source trees. Formatting ran directly on
each checkout. GPU timing binaries in the separate study directories
are outside this compatibility-target check.

The [revised runner](../2026-11-21-protocol/check-build.sh) gives each
source a new target directory, rejects reuse, and verifies the expected
package source path in both build logs. Each run hides the GPU. The
qualification comprises formatting, warning-as-error non-CUDA tests, and
NVCC/cuBLAS CUDA/server all-target Clippy. Hardware tests are recorded in
their individual studies.

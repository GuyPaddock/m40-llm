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

## First source-isolated results

**PASS for formatting, the invoked non-CUDA tests, and CUDA/server
compilation on the six revisions below.** Each started with an empty
target directory named for its complete source commit. The test binaries
ran from that directory. Both build logs identify the expected worktree.

| Research branch | Source | Reported non-CUDA passes |
|---|---|---:|
| consolidation | f2b8802 | 226 |
| graph-node-cost | 21aca1c | 223 |
| graph-dispatch | 66f21e3 | 223 |
| q8-projections | 110f8c7 | 223 |
| quantized-kv | 44f15e9 | 224 |
| small-n-verification | 5faa6ad | 223 |

All six suites reported zero failures. Consolidation again executed
`metrics_round_trip_without_sample_offsets`, `sampled_offsets_include_the_stop_token`
and `test_november_kv_answers_cover_all_families`. This resolves the
three missing checks in its prior shared-target run. The [new index](isolated/index.tsv)
identifies each complete archive, including command lines and all logs.
The [packaging check](pack-isolated.sh) verifies the compile paths, run
paths, exits, source revision and uncompressed log bytes. Separate
device tests remain necessary for any CUDA correctness or timing claim.

## Real-activation Q4 study

At 5dde384 on research/1121-sparse-q4-live, formatting, 224 non-CUDA
tests and CUDA/server all-target Clippy passed with the GPU hidden.
The [source-isolated index](isolated/index.tsv) includes the archive.
The independent capture, threshold and paired-timing tests ran separately.
This compile and CPU regression evidence precedes the capture and charged
operator measurements on the M40.

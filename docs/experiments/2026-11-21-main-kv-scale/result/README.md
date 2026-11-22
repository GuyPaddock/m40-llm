# Q8 history-scale checks on pinned main

**PASS for the repaired incremental path. INCONCLUSIVE for reproducing
the race on main.** The unfixed binary also passed its one executed
eviction test. The guarded fixture compared both reconstructed planes,
selected positions and attention across four sequence slots and four
reset/reuse lifetimes. Both fixed builds passed it and all ten applicable
attention parity tests. No tests were ignored in the measured invocations.

| Source | Build | Eviction fixture | Attention grid |
|---|---|---:|---:|
| a898ac2 | cuBLAS on | 1/1 | — |
| 75ed22f | cuBLAS on | 1/1 | 10/10 |
| 75ed22f | cuBLAS off | 1/1 | 10/10 |

The original failure occurred in the Phase 4 lineage. On main, the
same two CUDA source locations still use shared scratch for distinct K
and V maxima, and the port places one barrier before each reuse. The
measured comparison establishes the fixed behavior on main. The unfixed
run supplies no new failure witness there. The attention grid exercises
dense and mixed formats. It does not exercise main's separate bulk
dense-to-Q8 builder, whose required storage combination lacks an admitted
constructor. The bulk barrier is compile-covered. This study timed no
production workload.

Run d56ca894a974ac5c80508689933fcb2c7fe5f81a's
[serial driver](../run.sh) on the M40 with the five exact binaries listed
in [binary hashes](../binaries.sha256). The [raw archive](measurements.tar.gz)
contains the release binary hash checks, preflight, outcomes and before/
after device snapshots. Every checked boundary had zero resident compute
processes. Verify the archive with `sha256sum -c archive.sha256`.

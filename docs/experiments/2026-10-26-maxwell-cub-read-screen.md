# Public CUB Segmented Read Is Slower Here

## Decision

Close the fixed library reference. One-row checksum throughput regresses
**22.87-23.01%**, with negative paired intervals at all three real layers;
four rows regress **29.28-29.31%**. This fails the prespecified 10% read
improvement gate. No custom policy, geometry/register/cache rescue or
inference consumer follows. Preserve implementation `4c1680b` and isolation
fix `dfb6451` before removing the unprofitable adapter from the active build.

The reference computes identical source-derived modular checksums, but
has different scheduling, integer reduction, transform-iterator and host
dispatch costs. This does not identify a stall category or show that the
original kernel saturates DRAM. Library reputation or static instruction
counts cannot substitute for this measured result.

## Five-Pair Release Results

GPU milliseconds per call, five alternating pairs and ten repetitions per
interval. Throughput gain is original/library latency minus one. Paired 95%
bootstrap intervals use 10,000 resamples and seed zero, without pooling.

| Layer | Rows | Original checksum ms | CUB checksum ms | Throughput gain | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.207926 | 0.269578 | -22.87% | -23.06% to -22.69% |
| 17 | 1 | 0.207920 | 0.270000 | -22.99% | -23.27% to -22.47% |
| 35 | 1 | 0.207856 | 0.269984 | -23.01% | -23.18% to -22.82% |
| 0 | 4 | 0.816179 | 1.154262 | -29.29% | -29.34% to -29.19% |
| 17 | 4 | 0.816166 | 1.154534 | -29.31% | -29.36% to -28.74% |
| 35 | 4 | 0.816925 | 1.155152 | -29.28% | -29.44% to -29.18% |

The separate qualified real controls remain about **0.2211-0.2213 ms
gate/up**, **0.1314-0.1316 ms down** and **0.3519-0.3524 ms complete FFN**
at one row. They are not replaced by either checksum result. Four-row
controls use genuine captured activation rows; checksum deliberately rereads
the same weight pair for every row, as the original diagnostic does.

## Provenance, Compiler And Isolation

Clean source `dfb64513f9a48e56db32c5838e492537ceaf5637`; release executable
SHA-256 `5e6f775b1f993225d1e7a4b68fe667a60453bd0af01aebdcbdd12fc02a557fcd`.
Frozen model, fixture, capture and environment match M02. CUDA 12.8.93,
sm_52 SASS plus compute_52 PTX, `-O3`, no fast math, cuBLAS enabled.
The public-library header reports **CUB 2.7.0 / 200700**; SDK header hashes
are retained. No CUB implementation source is copied or custom policy used.
Retain the [upstream BSD notice](../third_party/cub-LICENSE.txt).

The first integration exposed C99/POSIX declarations globally. An untimed
code check found changes to six RoPE entries, so that build was never used
for the screen. `dfb6451` isolates CUB in its own CUDA translation unit and
restores original inference flags. All **100,400 original words across
135 entries** match the prior loaded native image, aggregate SHA-256
`7a954317700bf84cdc40440471d00b2dfb2b5b126cfb562e88c5566b8449970a`.
The two new entries are CUB's reduction and its unused empty metadata kernel.

CUB reduction resources are **62 registers, 80 static shared bytes,
zero stack/local bytes**, versus original checksum's **32 registers** and
1024 requested dynamic shared bytes. CUB requests zero dynamic shared bytes.
Its 2904 encoded words hash to
`f4f076d1f6bacd28c0fdc4aee8ec70c65e5ecb3c8e0abc7c42bff49dc8f71218`.
These are compiler resources, not measured achieved occupancy or a causal
explanation of the regression.

An independent untimed release parity test records **24 CUB kernel launches**,
48 matched driver/runtime records, 256 threads and zero requested dynamic
shared bytes. Grids equal rows*N; no empty metadata kernel is launched.
There are no dropped/failed/unknown launches and unsubscribe succeeds.
Observer perturbation is excluded from the untraced primary intervals.

## Memory, Checks And GPU State

Both readers retain the same original gate/up pair, **47,906,816 bytes**,
plus source and the unchanged 37-key prepared down/vocabulary cache,
**4,472,455,168 bytes**. There is no new weight representation. CUB queries
one byte of caller-owned scratch, below the 1 MiB cap, and reuses it.
Guarded checksums occupy 88,128/352,320 bytes at rows1/4; real scratch is
201,152/803,264 bytes. Tracked case totals are **4,520,651,265** and
**4,521,517,569 bytes**. These are requested payloads, not peak VRAM/traffic.

CUDA checksum/bounds tests pass **9 with cuBLAS on and 9 off**, including
odd N, dynamic K through 8192, arbitrary scale bits, wrapping, guards,
ordered producers, reuse, alias/alignment/offset/workspace rejection and
cleanup. Shared benchmark CPU tests pass **36**, with 16 hardware checks
ignored, not counted as passes; the cross-stream pipeline test also passes.
Attention **11**, forward **20** with four optional checks ignored,
projection **2**, MLP **1**, warning-as-error non-CUDA **217**, Clippy and
formatting pass. The independent audit rejects twelve corrupted reports.

Session **91782**, PID **3559220**, terminates successfully. All sixty measured
GPU boundaries show only that PID, P0, 1113 MHz SM and 3004 MHz memory.
Six cases provide 30 pairs and 150 serial primary intervals. Real FFN warms
for at least 1000 GPU ms; each reader warms for at least 100 ms, under the
existing finite budgets and outside primary data. No inference, build,
test or MemPalace work overlaps timing. Enqueue counters show no allocation,
free, copy or explicit synchronization. Tracked allocations return zero;
after process exit, a separate hardware check finds no GPU compute process.

## Reproduction And Artifacts

At `dfb6451`, source `scripts/dev-env.sh` and restore the frozen environment
as shown in the [original read experiment](2026-10-26-maxwell-ffn-read-screen.md#reproduction-and-evidence).
Build release `q8_execution_bench`, then add `--ffn-weight-read-cub` to its
`--ffn-weight-read --rows 1,4 --trials 5 --iterations 10` command, with a new
output path. Request hardware access and check GPU exclusivity first.

- [Contract](2026-10-26-maxwell-cub-read-hypothesis.md), [raw report](2026-10-26-maxwell-cub-read-screen.jsonl), [audit](2026-10-26-maxwell-cub-read-audit.sh) and [derived costs](2026-10-26-maxwell-cub-read-audit.json)
- [Compiled comparison](2026-10-26-maxwell-cub-read-code-comparison.json), [rejected global-flags comparison](2026-10-26-maxwell-cub-read-unisolated-code-comparison.json), [resources](2026-10-26-maxwell-cub-read-resources.txt.gz) and [CUB assembly](2026-10-26-maxwell-cub-read-candidate.sass.gz)
- [Untimed launch trace](2026-10-26-maxwell-cub-read-launch-trace.jsonl.gz), [release parity log](2026-10-26-maxwell-cub-read-launch-test.log.gz), [SDK identities](2026-10-26-maxwell-cub-read-sdk-headers.sha256) and [checksums](2026-10-26-maxwell-cub-read.sha256)
- [Timing log](2026-10-26-maxwell-cub-read-screen.log.gz), [build](2026-10-26-maxwell-cub-read-build.log.gz), [CUDA on](2026-10-26-maxwell-cub-read-cuda-on.log.gz), [CUDA off](2026-10-26-maxwell-cub-read-cuda-off.log.gz), [benchmark tests](2026-10-26-maxwell-cub-read-example-tests.log.gz), [regressions](2026-10-26-maxwell-cub-read-regressions.log.gz), [non-CUDA](2026-10-26-maxwell-cub-read-noncuda.log.gz) and [Clippy](2026-10-26-maxwell-cub-read-clippy.log.gz)

No actual512 generation, frozen quality or HTTP qualification follows a
losing diagnostic. Qualified exact runtime remains **49.146 HTTP tok/s**
versus **43.106 official Ollama**; **53.06** and **56.037 HTTP tok/s** remain
unmet. Preserve defaults and pursue independently supported exact FFN work,
not further tuning of this library reference or closed candidates.

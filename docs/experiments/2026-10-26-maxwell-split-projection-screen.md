# Split QKV/Output Storage Does Not Qualify

## Decision

The [fixed split-storage experiment](2026-10-26-maxwell-split-projection-hypothesis.md)
preserves qualified output bits but passes neither independent stage gate.
One-request QKV throughput regresses **0.78-2.12%**. Output's median gain
is **2.02-2.49%**, below the unchanged 5% gate, and every one-request output
interval spans zero. These five-pair observations do not prove that the
true output gain is positive or exclude 5% in every interval. They do not
qualify the candidate for runtime integration.

Output's optimistic standalone profile-weighted scope is only
**0.102-0.125% E2E**, not a measured E2E gain. Failed stage gates cannot
be combined to bypass the rule. Keep primitives `0b96c46` and benchmark
`34a2427` as reference, with no generation selector, geometry/cache/register
search, runtime promotion or default change. Both original targets remain
unmet; qualified status stays **49.146 HTTP tok/s versus 43.106 official
Ollama**, below **53.06** and **56.037 tok/s**.

## Independent Results

Five alternating pairs per shape, ten repeated projections per arm, follow
at least 1000/100 GPU milliseconds of untimed control/candidate warmup.
Latencies cover all requests in a case. Gains are throughput changes,
`control_ms/candidate_ms - 1`, not latency reduction percentages.

| Stage | Layer | Requests | Packed control ms | Split candidate ms | Throughput change | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| QKV | 0 | 1 | 0.036627 | 0.036960 | -0.900% | [-3.346%, -0.147%] |
| QKV | 17 | 1 | 0.036698 | 0.036986 | -0.779% | [-4.076%, +1.220%] |
| QKV | 35 | 1 | 0.036726 | 0.037523 | -2.123% | [-6.595%, +0.078%] |
| QKV | 0 | 4 | 0.146192 | 0.147901 | -1.155% | [-2.514%, -0.711%] |
| QKV | 17 | 4 | 0.146147 | 0.147878 | -1.171% | [-2.408%, -0.941%] |
| QKV | 35 | 4 | 0.146259 | 0.148509 | -1.515% | [-2.503%, -0.945%] |
| Output | 0 | 1 | 0.026698 | 0.026170 | +2.018% | [-2.186%, +5.822%] |
| Output | 17 | 1 | 0.026634 | 0.026042 | +2.273% | [-0.859%, +3.401%] |
| Output | 35 | 1 | 0.026768 | 0.026118 | +2.487% | [-0.436%, +5.375%] |
| Output | 0 | 4 | 0.107024 | 0.104541 | +2.375% | [+1.033%, +4.486%] |
| Output | 17 | 4 | 0.106714 | 0.104608 | +2.013% | [+0.969%, +2.897%] |
| Output | 35 | 4 | 0.106458 | 0.104742 | +1.638% | [-0.129%, +2.838%] |

No additional confirmation is used to promote these below-gate point
estimates. Four-request rows are distinct independent M=1 requests with
one launch each, not a batched GEMM. Inputs are frozen actual normalized
FFN vectors used as projection surrogates, not captured attention-norm or
attention-output vectors. No actual-512 generation or HTTP qualification
is claimed for a candidate that fails its prerequisite screen.

The latest middle-token profile supplies QKV/output fractions
**0.070945564817451 / 0.05158930127407999**, bound to profile audit SHA-256
`3ff95bf79269368145bf3044bb4d351ec826f11444b87cadafbfe7868e29177e`.
The standalone calculation omits every other cost and is not runtime
evidence. No combined feasibility value is admitted when a stage fails.

## Fixed Computation And Compiled Evidence

Only storage/addressing changes. The existing typed split layout holds
the original signed Q8 bytes and FP16 scale bits in aligned separate
planes. A single `uint2` replaces four U16 payload reads per eight weights.
Both arms retain two `float4` activation reads, signed conversions,
per-weight FP32 scaling/FMA association, 256-thread task assignment,
original-tree warp finish and optional QKV bias addition. QKV keeps one
column per block, output four columns. No activation quantization or new
lossy representation is used. No upstream code is copied.

Release sm_52 instructions confirm `LDG.E.CI.64` payload reads in place
of four U16 sites. Register use stays **32 QKV / 40 output**, with zero
reported stack/local storage. Dynamic shared capacity stays **1024 / 4096
bytes**; the resource dump's zero static shared bytes does not describe
that launch-time allocation. Source load changes and equal registers do
not establish occupancy, measured DRAM traffic or a bandwidth bottleneck.

Extracted control/candidate SASS text hashes are:

```text
454b49344db42bca364831a51571d1e0708f9d77844f02fad6cab502fba90bdd
25d644ae9b398003b79b1a0f316e804dc0c6675e5d98f3793a719f370428f4df
```

The control disassembly's formatting/header text differs from the older
qualified artifact, but every encoded instruction/control word is unchanged.
Extracting `0x` followed by 16 hexadecimal digits from both files yields
the same SHA-256 **1440814f985d0c8d264b299b00ee0945050993eaaafb9452a8ce71f5b1430aef**.
The warmup-only rebuild also preserves both candidate and control code.
Do not describe the full disassembly text as byte-identical.

All twelve real-input validation cases preserve control bits, guarded
outputs and immutable input/raw/prepared/bias spans. Maximum scalar FP64
reference errors are **3.8485e-6 QKV / 1.1852e-6 output**, not drift from
the bit-identical GPU control. Synthetic tests cover signed extremes,
zero scales, unequal widths/tails, biases, four distinct rows, both producer
streams, aliases, rejected inputs and RAII cleanup.

## Warmup Correction And Identity

The first output process stopped before primary pairs because the old
128-iteration/256-batch warmup cannot accumulate 1000 ms for a control
faster than 30.518 us. The [preserved failure](2026-10-26-maxwell-split-projection-warmup-failure.md)
has manifest plus numerical validation only, not a completed performance
result. Its initial QKV counterpart remains separate and is not pooled
with the fresh common-source screens. The initial QKV gate also fails.

Correction `5719ffc` increases only untimed iterations per batch to 256,
keeps the 256-batch limit and minimum GPU durations, and rejects invalid
samples. It does not change primary ten-iteration pairs, kernels, launch
geometry or gates. CPU tests simulate the failing fast-control condition;
the hardware adapter verifies the sampler. Final warmups all reach their
requested durations within the limit.

Both final screens use clean source
`5719ffcbfbcf57c91acad67c691120ffc95ecc01`, release benchmark SHA-256
`517829832eea508f4c71f10e442a817306ef3886e90b8657c7c36f48970f43eb`.
Frozen M02 model, fixture, capture and environment identities are unchanged.
CUDA uses `-O3`, sm_52/compute_52, no fast math and toolkit 12.8.93.
Disassembly uses cuobjdump 12.8.90. These are not debug-build timings.

## Memory, Isolation And Validation

Both processes retain 3,279,519,744 source bytes plus the qualified
37-tensor down/vocabulary cache of 1,192,935,424 bytes. This is not the
full runtime's 109 prepared identities. The extra candidate planes are
resident in *both* timing arms, outside the timed allocation boundary:

| Stage | Extra prepared bytes | Scratch, requests 1 / 4 | Tracked total, requests 1 / 4 |
| --- | ---: | ---: | ---: |
| QKV | 5,570,560 | 18,688 / 73,984 | 4,478,044,416 / 4,478,099,712 |
| Output | 4,456,448 | 16,512 / 65,664 | 4,476,928,128 / 4,476,977,280 |

Raw records report packing/upload time, semantic tensor identities, host
preparation bytes, plane sizes and tracked deltas. A full-model combination
would add an estimated 360,972,288 device bytes if original GGUF storage
stays resident; no runtime integration or that allocation is performed.
Every primary enqueue has zero allocations/frees, copies, cuBLAS calls,
stream waits or synchronizations. Completion uses an event wait outside
those counters. Both final processes return tracked payload zero to zero.

Sixty pairs comprise 120 serial arm intervals and 120 GPU boundary samples.
All boundaries show P0, 1113 MHz SM and 3004 MHz memory, with only benchmark
PID **2743769 QKV / 2745566 output**. No build, test, inference or MemPalace
work overlaps either timing process. Sessions **53005 / 85857** exit zero,
and the post-process GPU query is empty. The initial output failure is
terminal before the harness correction, not a timeout-triggered duplicate.

Primitive parity/rejection tests pass with cuBLAS enabled and disabled.
Qualified packed QKV/output checks, attention grid, projection, MLP and
forward fixture regressions pass; four optional forward tests remain
ignored. The benchmark CPU suite passes 15 tests with 11 hardware checks
ignored in that invocation; the expanded adapter is explicitly run and
passes separately. All 211 warning-as-error non-CUDA tests pass at final
source. CUDA/server all-target Clippy, formatting and diff checks pass.

The independent audit recomputes raw pair medians, gains and gate decisions,
checks possible resampling bounds for the benchmark's seeded intervals,
and verifies coverage, dispatch, typed preparation, memory, clocks and
non-overlap. It does not claim to reproduce the random bootstrap stream.
Six damaged-report checks reject missing pairs, changed operation, altered
timing, broken parity, wrong preparation and overlapping measurements.

## Reproduction And Artifacts

Source `scripts/dev-env.sh`, export the first record's frozen environment
from `2026-10-25-maxwell-ffn-capture.jsonl`, and build the release CUDA
`q8_execution_bench`. Run these independently and serially, with new output
paths and the original model:

```text
--qkv-f32-split
--output-f32-split
```

Common arguments are `--ffn-inputs target/research-fixtures/m02-ffn-inputs.json`,
`--ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl`,
`--rows 1,4 --trials 5 --iterations 10`, `--model` and `--output`.
Audit with `jq -n` and the six `--slurpfile` arguments listed in the
[audit program](2026-10-26-maxwell-split-projection-audit.jq).

- Raw [QKV](2026-10-26-maxwell-split-projection-qkv.jsonl) and [output](2026-10-26-maxwell-split-projection-output.jsonl), [derived audit](2026-10-26-maxwell-split-projection-audit.json)
- [QKV log](2026-10-26-maxwell-split-projection-qkv.log.gz) and [output log](2026-10-26-maxwell-split-projection-output.log.gz)
- [Control SASS](2026-10-26-maxwell-split-projection-controls.sass.gz), [candidate SASS](2026-10-26-maxwell-split-projection-candidates.sass.gz), [control words](2026-10-26-maxwell-split-projection-control.words.gz), [resources](2026-10-26-maxwell-split-projection-resources.txt.gz)
- [Validation logs](2026-10-26-maxwell-split-projection-validation.log.gz), [negative audit checks](2026-10-26-maxwell-split-projection-negative-audit.log.gz), [checksums](2026-10-26-maxwell-split-projection-screen.sha256)

Preserve these implementations before any separate removal. Next work
needs an independent measured/source-backed hypothesis; it must not rescue
this layout with another geometry or combine closed FFN/projection probes.

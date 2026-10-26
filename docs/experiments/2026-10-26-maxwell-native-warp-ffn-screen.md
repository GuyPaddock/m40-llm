# Native Warp FP32 FFN Regresses

## Decision

Keep `dd78a1a` benchmark-only. The fixed native 32-partial mapping regresses
complete one-row FFN throughput **4.72-4.77%**, with negative paired intervals
at all three layers. Four-row point regressions are **3.22-4.40%**. It fails
every unchanged 5% complete-FFN gate; no runtime, actual512/frozen quality or
HTTP qualification follows. Do not rescue it with another warp/thread count,
unroll, register cap, cache policy or rejected layout.

This [prespecified association experiment](2026-10-26-maxwell-native-warp-ffn-hypothesis.md)
is distinct from the closed 128-partial emulation. It avoids shared publication
and its block-wide barrier with fewer registers than emulation, but still
loses to the qualified 128-thread control. Neither fewer barriers nor fewer
source accumulators establishes a useful performance gain.

The qualified exact runtime remains **49.146 HTTP tok/s** versus **43.106
official Ollama**, not remeasured here. Both original goals, **53.06** and
**1.30x fresh official Ollama**, currently **56.037 HTTP tok/s**, remain
unmet. Runtime defaults, KV hierarchy, SGEMM prefix and prepared down/head
control are unchanged.

## Complete FFN Results

Five alternating pairs per shape use ten repetitions each for gate/up/SwiGLU,
isolated prepared down and complete FFN. Values are GPU-event median
milliseconds. Gain is control/candidate latency minus one; paired 95%
bootstrap intervals use 10,000 resamples and seed zero.

| Layer | Rows | Split control ms | Native-warp ms | Gain | Paired 95% gain interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.351917 | 0.369424 | -4.74% | -4.81% to -4.59% |
| 17 | 1 | 0.352042 | 0.369693 | -4.77% | -4.95% to -4.76% |
| 35 | 1 | 0.352109 | 0.369571 | -4.73% | -4.90% to -4.70% |
| 0 | 4 | 1.541882 | 1.593181 | -3.22% | -4.58% to 0.04% |
| 17 | 4 | 1.541194 | 1.610365 | -4.30% | -6.06% to -3.79% |
| 35 | 4 | 1.591565 | 1.664771 | -4.40% | -7.76% to -3.90% |

One-row gate/up/SwiGLU regresses from **0.2218-0.2220 ms** to
**0.2386-0.2388 ms**. Unchanged isolated down remains approximately
**0.1311-0.1316 ms**. Different stage histories mean those medians need not
sum to complete FFN. Four-row noise does not reopen the negative one-row
gate. There is no generated-token or HTTP speedup claim.

## Arithmetic And Correctness

Each warp owns one output column, with four output warps per 128-thread CTA.
Lane l visits four-value tasks l, l+32, etc., keeping one gate/up accumulator
pair. A native five-step warp reduction combines 32 partials. The original
split FP16 scale/signed-byte planes, float4 activation loads, per-weight
FP32 scaling/FMA and precise SiLU expression remain unchanged.

This changes summation ownership/association from the control's 128 partials.
It is not the four-virtual-partial-per-lane emulation, and does not quantize
activations or modify model weights. No source/output rounding equivalence
is assumed. Existing strict-control variants retain their old bit checks.

All six real captured-input cases pass full FP64 CPU references with the
original finite `1e-3 + 2e-5 * abs(reference)` tolerance. Candidate capture
and ordinary SwiGLU bits match. Drift is reported, not hidden or called
bit equality:

| Output | Max absolute error vs CPU | Max absolute drift vs split control |
| --- | ---: | ---: |
| Gate | 2.30e-6 | 1.91e-6 |
| Up | 2.45e-6 | 3.81e-6 |
| SwiGLU | 5.92e-5 | 7.63e-5 |
| Complete down | 3.96e-5 | 9.16e-5 |

These are layer-local checks, not final model logits or actual generation.
Unchanged actual512 output/frozen dense-valid quality remains mandatory
for any future exact-output candidate; the failed micro gate does not
authorize testing this kernel into a new default.

## Compiler And Memory

Clean source `dd78a1a2f6b37f7c78f9b727bef40c099d85b87e`, release executable
SHA-256 `c769e6507f368e5b1b8117a417bd71d3e1086fbf16c73978b9e4410a4baf68ec`.
NVCC 12.8.93 uses `-O3`, sm_52 SASS and compute_52 PTX, no fast math.

| Kernel | Registers | Dynamic shared bytes | Block-wide barriers |
| --- | ---: | ---: | ---: |
| Qualified split control | 32 | 1024 | 1 |
| New native 32-partial warp | 40 | 0 | 0 |
| Closed 128-partial emulation | 56 | 0 | 0 |

All report zero stack/local storage. The candidate realizes LDG128 activation
loads, u32 payload/U16 scale loads and ten shuffle sites, with no shared
publication. Dependency barriers in its compiled loop are distinct from a
block-wide `BAR.SYNC`; this is not a claim of barrier-free execution.
Fewer accumulators than emulation did not restore control's 32-register usage.
These observations do not measure achieved occupancy, cache hits, stalls or
DRAM traffic. Legacy CUPTI counters remain unavailable.

Original control's 560 instruction words remain unchanged, SHA-256
`31cd38f6ca0d4a0bebd03206f484bf8045834d8af592a645cdb168cfaf02e321`.
Prepared down/head's 552 words remain unchanged, SHA-256
`fafef036cf36582ad7c230dd9b2502575a53356ada1396aaeac8e9baf0872397`.
Closed emulation words also match their archived binary, SHA-256
`b9531d39be132a05ea0c2bd06c10eed24f00bc07072b1aa4ca1a27124f413ba8`.
It was neither modified nor retimed as a candidate here.

Both arms retain the same original model plus qualified 37 down/vocabulary
keys, **4,472,455,168 bytes**, and the same original split gate/up pair,
**47,906,816 bytes**. No new weight plane, activation quantization or KV
cache is allocated. Guarded scratch stays **201,152 / 803,264 bytes** for
rows1/4; tracked totals stay **4,520,563,136 / 4,521,165,248 bytes** and
return zero to zero at final cleanup.

## Validation And Isolation

CUDA numerical/ownership tests pass **2/2 with cuBLAS on and off**, including
all earlier strict-control variants plus the new CPU-bounded association.
Coverage includes signed payload extremes, positive/negative-zero scales,
dynamic K=32/96/2048/4096, tiny/odd output tails, distinct rows, ordinary/
capture parity, immutable inputs/weights, async producers, reuse and cleanup.
The new API rejects full/partial span aliases, alignment/address overflow,
bad dimensions and invalid capture buffers before GPU work. The initial
test compile exposed a missing test-only `c_void` import; it was corrected
inside the implementation before the successful runs and commit.

Benchmark CPU tests pass **26**; 14 hardware tests are ignored in that
invocation and are not counted as passes. Regressions pass attention **11**,
forward **20**, projection **2** and MLP **1**; four optional forward tests
are ignored. Non-CUDA warning-as-error tests pass **217**. CUDA/server
all-target Clippy, formatting and diff checks pass.

Session **12599**, PID **3192070**, exits zero. Thirty paired records contain
180 serial primary GPU intervals. All sixty GPU boundary samples show only
this process, P0, 1113 MHz SM / 3004 MHz memory. No other inference, build,
test or MemPalace operation overlaps timing; GPU compute is empty after
exit. Bounded warmup is 1000 GPU ms control / 100 ms candidate, excluded
from primary timings.

Ten repeated complete FFNs issue twenty launches, isolated stages ten each.
The native-warp dispatch records 2752 blocks per row, four output warps,
32 partials/output and no shared storage. Enqueue introduces no allocation,
free, transfer, cuBLAS call or stream synchronization. Existing four-row
dependency waits remain included; one event completion wait follows each
repeated stage, outside enqueue counters.

The audit checks frozen provenance/cache/source identities, real CPU errors
and reported control drift, capture parity/guards, memory/scratch, dispatch,
timed side effects, raw medians/gains, possible bootstrap bounds, serial
coverage, clocks and cleanup. It does not independently regenerate the
seeded bootstrap sequence. Corruption controls must reject missing work,
wrong partial count, excessive drift, timed allocation, altered identities,
false medians, failed capture, competing GPU work and cleanup failure.

## Reproduction And Artifacts

Source `scripts/dev-env.sh`, restore the frozen M02 environment using the
[task-pair reproduction](2026-10-26-maxwell-ffn-task-pair-screen.md#reproduction-and-artifacts),
request hardware access and verify an idle GPU. Use a new output path:

```bash
cargo build --release --features cuda --example q8_execution_bench
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-f32-native-warp --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-native-warp-ffn-screen.jsonl
```

- [Raw pairs](2026-10-26-maxwell-native-warp-ffn-screen.jsonl), [audit](2026-10-26-maxwell-native-warp-ffn-audit.jq), [derived results](2026-10-26-maxwell-native-warp-ffn-audit.json) and [corruption controls](2026-10-26-maxwell-native-warp-ffn-negative-controls.sh)
- [Candidate SASS](2026-10-26-maxwell-native-warp-ffn-native_warp.sass.gz), [closed emulation SASS](2026-10-26-maxwell-native-warp-ffn-virtual_warp.sass.gz), [resources](2026-10-26-maxwell-native-warp-ffn-resources.txt.gz), [control SASS](2026-10-26-maxwell-native-warp-ffn-control.sass.gz) and [control words](2026-10-26-maxwell-native-warp-ffn-control.words.gz)
- [Prepared control SASS](2026-10-26-maxwell-native-warp-ffn-prepared-control.sass.gz) and [words](2026-10-26-maxwell-native-warp-ffn-prepared-control.words.gz)
- [Run log](2026-10-26-maxwell-native-warp-ffn-screen.log.gz), [release build log](2026-10-26-maxwell-native-warp-ffn-build.log.gz) and [checksums](2026-10-26-maxwell-native-warp-ffn.sha256)

Next work needs independent measured scope rather than another warp-count
or resource-limit rescue. Preserve this association's negative result and
keep all original exact-output targets and acceptance rules intact.

# Role-Parallel Gate/Up Regresses

## Decision

Keep `0d99dc8` benchmark-only. Separate physical gate/up roles preserve
control bits but regress complete one-row FFN throughput **14.53-14.66%**.
Every paired 95% interval is negative. Four-row regressions are
**10.88-14.29%**, also with negative intervals. All unchanged 5% gates fail.
No runtime selector, actual512/frozen quality run or HTTP qualification
follows. Do not rescue this mapping with geometry, registers, cache or unroll.

The [fixed contract](2026-10-26-maxwell-role-parallel-ffn-hypothesis.md)
separates gate and up within one output CTA, not two output cohorts or
independent output warps. Each physical thread has one accumulator and
preserves its original per-role task/FMA order. Threads and logical
activation reads per output double. Fewer accumulators did not reduce
compiled registers or improve latency. These observations do not isolate
occupancy or memory traffic as the cause.

Qualified performance remains **49.146 HTTP tok/s** versus **43.106
official Ollama**, not remeasured here. Both original requirements remain
unmet: **53.06 HTTP tok/s** and **1.30x fresh official Ollama**, currently
**56.037 HTTP tok/s**. Runtime/KV defaults and qualified prepared
down/vocabulary, SGEMM prefix and dense attention remain unchanged.

## Timing

Five alternating pairs per shape repeat each stage ten times. These are
CUDA-event GPU milliseconds, not generation throughput. Gain is
control/candidate latency minus one; paired bootstrap intervals use 10,000
resamples and seed zero.

| Layer | Rows | Control ms | Role-parallel ms | Gain | Paired 95% gain interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.351725 | 0.411949 | -14.62% | -14.71% to -14.52% |
| 17 | 1 | 0.352035 | 0.412493 | -14.66% | -14.76% to -14.48% |
| 35 | 1 | 0.352074 | 0.411926 | -14.53% | -14.76% to -14.47% |
| 0 | 4 | 1.587472 | 1.781238 | -10.88% | -18.00% to -10.23% |
| 17 | 4 | 1.597875 | 1.830294 | -12.70% | -14.37% to -12.18% |
| 35 | 4 | 1.528720 | 1.783574 | -14.29% | -15.88% to -13.08% |

One-row gate/up/SwiGLU grows from **0.2216-0.2223 ms** to
**0.2806-0.2809 ms**. Unchanged isolated prepared down remains about
**0.1313-0.1319 ms**. Separate-stage medians need not sum to complete FFN.
Four-row variation does not reopen the decisive negative one-row gate.

## Correctness, Compiler And Memory

Clean source `0d99dc8748cfe0cd36ef2435ab264a620313a141`; release executable
SHA-256 `04347df8f326233e03b172b56d4b3759e1c2373ab3bdb23458ef2658ed37e2ec`.
NVCC 12.8.93 uses `-O3`, sm_52 SASS/compute_52 PTX, no fast math.

The candidate launches one output per 256-thread CTA: gate threads 0-127,
up threads 128-255. Original split FP16 scales, signed bytes, float4 input,
per-weight FP32 scale/FMA, precise SiLU and two 128-partial reduction trees
remain. Every thread publishes one partial; the first warp reduces both
trees after one barrier. Dynamic complete Q8 blocks and odd N are supported.

| Kernel | Registers/thread | Dynamic shared bytes | Block barriers |
| --- | ---: | ---: | ---: |
| Qualified split control | 32 | 1024 | 1 |
| Role-parallel candidate | 40 | 1024 | 1 |

Both report zero stack/local storage. Candidate SASS has 128-bit activation
reads, u32 payload/u16 scale loads, one `BAR.SYNC` and ten shuffle sites.
No measured occupancy, cache-hit, stall or DRAM attribution is available.
Original encoded instruction hashes remain unchanged:

- Gate/up: `31cd38f6ca0d4a0bebd03206f484bf8045834d8af592a645cdb168cfaf02e321`.
- Prepared down/head: `fafef036cf36582ad7c230dd9b2502575a53356ada1396aaeac8e9baf0872397`.

All six real layer/row cases preserve gate/up, SwiGLU and completed-down
bits versus control, including capture/ordinary parity. Independent FP64
CPU references pass the unchanged finite `1e-3 + 2e-5 * abs(reference)`
tolerance. Maximum CPU errors are **2.07e-6 gate**, **3.41e-6 up**,
**1.72e-5 SwiGLU** and **5.01e-5 down**. This is layer-local validation,
not final-model qualification.

Both arms retain the same 37 prepared down/vocabulary identities and source
payload, **4,472,455,168 bytes**, plus original split gate/up pair
**47,906,816 bytes**. No new layout, quantization or KV is allocated.
Guarded scratch remains **201,152 / 803,264 bytes** for rows1/4; tracked
totals are **4,520,563,136 / 4,521,165,248 bytes**. Timed enqueues add no
allocation/free, H2D/D2H copy, cuBLAS call or stream synchronization.
Tracked allocation returns zero to zero.

## Validation And Isolation

CUDA primitive numerical/guard/span tests pass **2/2 with cuBLAS on and
off**. Focused complete-FFN fixture/CLI tests pass **2/2**, including bits,
actual dispatch, scratch reuse, cache immutability and cleanup. Coverage
includes K=32/96/2048/4096, tiny/odd outputs, signed extremes, zero/negative
zero scales, distinct rows and async producers. Full/partial aliases,
alignment and address overflow are rejected before GPU work.

Benchmark CPU tests pass **31**; **15 hardware tests are ignored**, not
counted as passes. The focused hardware test above is run explicitly.
Attention **11**, forward **20**, projection **2** and MLP **1** pass;
four optional forward tests remain ignored. Non-CUDA warning-as-error
tests pass **217**; CUDA/server all-targets Clippy and formatting pass.

Session **67790**, PID **3261330**, terminates zero. Thirty pairs contain
180 nonoverlapping primary intervals and 60 measured GPU boundaries, all
with only that process, P0, 1113 MHz SM and 3004 MHz memory clocks. No other
inference, build, test or MemPalace call overlaps the timed run. GPU compute
is empty afterward. The independent report audit passes; ten corruptions
are rejected: missing pairs, extra allocation, wrong role dispatch, drift,
capture mismatch, wrong weight identity, bad medians, accounting, cleanup
and competing processes.

## Reproduction And Artifacts

Source `scripts/dev-env.sh`, restore frozen M02 environment from the
[task-pair reproduction](2026-10-26-maxwell-ffn-task-pair-screen.md#reproduction-and-artifacts),
request hardware access and verify idle GPU state. Use a new output path:

```bash
cargo build --release --features cuda --example q8_execution_bench
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-f32-role-parallel --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-role-parallel-ffn-screen.jsonl
```

- [Raw pairs](2026-10-26-maxwell-role-parallel-ffn-screen.jsonl), [audit](2026-10-26-maxwell-role-parallel-ffn-audit.jq), [derived results](2026-10-26-maxwell-role-parallel-ffn-audit.json), [corruption controls](2026-10-26-maxwell-role-parallel-ffn-negative-controls.sh) and [checksums](2026-10-26-maxwell-role-parallel-ffn.sha256).
- [Candidate SASS](2026-10-26-maxwell-role-parallel-ffn-candidate.sass.gz), [control SASS](2026-10-26-maxwell-role-parallel-ffn-control.sass.gz), [prepared control SASS](2026-10-26-maxwell-role-parallel-ffn-prepared-control.sass.gz) and [resources](2026-10-26-maxwell-role-parallel-ffn-resources.txt.gz).
- [Run](2026-10-26-maxwell-role-parallel-ffn-screen.log.gz), [build](2026-10-26-maxwell-role-parallel-ffn-build.log.gz), [cuBLAS on](2026-10-26-maxwell-role-parallel-ffn-on.log.gz), [off](2026-10-26-maxwell-role-parallel-ffn-off.log.gz), [pipeline](2026-10-26-maxwell-role-parallel-ffn-pipeline.log.gz), [CPU](2026-10-26-maxwell-role-parallel-ffn-cpu.log.gz), [regressions](2026-10-26-maxwell-role-parallel-ffn-regression.log.gz), [non-CUDA](2026-10-26-maxwell-role-parallel-ffn-noncuda.log.gz) and [Clippy](2026-10-26-maxwell-role-parallel-ffn-clippy.log.gz) logs.

Next work needs independent evidence of sufficient execution-cost scope,
not another tensor-role mapping. The full exact-output objective stays active.

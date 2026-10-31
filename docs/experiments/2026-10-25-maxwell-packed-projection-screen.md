# Packed QKV/Output Projection Screens

## Decision

Both independent stages pass. One-request QKV throughput improves
**46.87-51.97%** and output projection improves **40.02-46.78%**, with
positive paired intervals and bit-identical control outputs. Four-request
results also pass their stage screens. The combined optimistic scope is
**5.23-5.33% E2E** using the current qualified profile, above the 3% floor.
This permits a combined runtime-quality experiment, not promotion or a
measured E2E claim. No below-gate FFN candidate participates.

Next integrate only an explicit model-local packed projection choice,
keeping split-f32x4-128 FFN, SGEMM prefix, all 109 prepared weight identities
and dense reference KV fixed. Then run actual 512-ID/frozen-task quality,
incremental HTTP and fresh official Ollama gates. Do not change defaults
or call these microbenchmark gains generated-token gains.

The qualified runtime remains **47.379 HTTP tok/s** versus **43.097**
official Ollama. Both original targets remain unmet: **53.06** and
**56.026 tok/s** for 1.30x that fresh median.

## Configuration

The [fixed hypothesis](2026-10-25-maxwell-packed-projection-hypothesis.md)
was committed as `9e8ff51`. Both screens use clean revision
`8b7b20c4ae0e9840a9a67a139e66865a0e7a16a7`, release executable SHA-256
`1a82c50a6f74686fb151ac3582d2c4c4ae62337f2a8a10facdcb911c7fdcd251`.
Original Qwen Q8_0, frozen M02 input/capture/environment identities and
layers 0/17/35 stay fixed. Use the existing runner with one of:

```text
--qkv-f32-packed
--output-f32-packed
```

Common arguments are `--ffn-inputs target/research-fixtures/m02-ffn-inputs.json`,
`--ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl`,
the original `--model`, `--rows 1,4 --trials 5 --iterations 10` and a new
`--output` path. The two commands run serially in separate processes.
Each case has bounded warmup followed by five alternating pairs. Event
timing includes all ten repeated projections and required work; setup,
CPU references and readback are excluded. Bootstrap uses 10,000 paired
resamples with seed zero.

Control is the **qualified scalar-load warp kernel**, not the older shared
reduction. Candidate keeps 256 threads, one QKV column or four output
columns, eight-value tasks, raw 34-byte Q8_0 blocks, original signed
conversion, FP32 scale/FMA/reduction association and optional biases.
Only input/payload loads change: two float4 activation loads and four U16
payload loads per eight-value task. Scale loads stay FP16. No repacking,
quantized activations, cache hints or new scratch participates.

Inputs are actual frozen normalized FFN vectors used as projection
surrogates, not captured attention-norm/output inputs. Four requests are
distinct independent M=1 projections with one launch per row, not batched
GEMM. Both limits are explicit in raw manifests and are why actual-model
quality/E2E gates remain required.

## Independent Results

Gains below are throughput, `control_ms/candidate_ms - 1`, not latency
reduction percentages. Latencies cover all requests in each case.

| Stage | Layer | Requests | Control ms | Packed ms | Throughput gain | Paired 95% gain interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QKV | 0 | 1 | 0.055715 | 0.037936 | +46.866% | [+41.530%,+54.682%] |
| QKV | 17 | 1 | 0.055318 | 0.036400 | +51.974% | [+42.051%,+53.556%] |
| QKV | 35 | 1 | 0.055667 | 0.037702 | +47.649% | [+40.552%,+55.061%] |
| QKV | 0 | 4 | 0.218675 | 0.146202 | +49.571% | [+47.856%,+50.375%] |
| QKV | 17 | 4 | 0.219232 | 0.146627 | +49.517% | [+47.154%,+50.451%] |
| QKV | 35 | 4 | 0.218576 | 0.147165 | +48.525% | [+47.395%,+49.859%] |
| Output | 0 | 1 | 0.038669 | 0.026346 | +46.775% | [+26.054%,+48.083%] |
| Output | 17 | 1 | 0.037677 | 0.026909 | +40.017% | [+36.380%,+47.790%] |
| Output | 35 | 1 | 0.038806 | 0.027200 | +42.671% | [+36.452%,+48.428%] |
| Output | 0 | 4 | 0.152256 | 0.106557 | +42.887% | [+40.316%,+45.044%] |
| Output | 17 | 4 | 0.152787 | 0.107264 | +42.440% | [+39.759%,+44.887%] |
| Output | 35 | 4 | 0.152096 | 0.107021 | +42.118% | [+39.969%,+44.508%] |

Every stage/case clears >=5% and has a positive interval well away from
that boundary; no borderline confirmation is required at this stage.
The [current profile](2026-10-25-maxwell-projection-stage-profile.md)
uses middle-token shares **0.09239512146033983 QKV** and
**0.06640904254002505 output**, bound to profile audit SHA-256
`9445906c521ae61700935e0b934d8744d98eb0db247f7abeee667e2cc9e3066d`.
Optimistic combined gains are **5.335% / 5.327% / 5.228%** for layers
0/17/35 one-request cases. This Amdahl-style calculation is not an observed
HTTP gain or confidence interval. Output alone has only 1.93-2.16% scope;
its raw case-level runtime flag is false for that standalone interpretation.
The combined audit admits it only alongside independently passing QKV.

## Compiled Code And Correctness

Actual release candidates emit U16 payload loads and 128-bit activation
loads instead of U8/scalar sequences. QKV keeps **32 registers**, output
keeps **40**, both with no reported stack/local storage. Dynamic shared
capacity stays **1024 / 4096 bytes**. Qualified control SASS is unchanged
against the [earlier compiled-code observation](2026-10-25-maxwell-byte-ffn-projection-followup.sass.gz).
Combined control/candidate SASS hashes are respectively:

```text
b65ff289acc889a08937b483000119b667f28c84ea56925d6cc938176bbee3f6
bce96319a68adf6b9cde83594925c621c60f5311cf66250b26d751bb07cffecf
```

These observations support changed compiled loads, not measured cache-hit
rate, occupancy or DRAM traffic. No upstream kernel code was copied.

All twelve real-model validation cases preserve control output bits,
guards and source/bias/input contents. Maximum absolute CPU-reference
errors are **3.8485e-6 QKV** and **1.1852e-6 output**. Synthetic hardware
checks cover two-byte-only weight bases, zero scales, signed extremes,
width/column tails, missing/mixed/all biases, producer ordering on both
streams, output guards and cleanup. Inputs at offsets 4/8/12 fail before
launch for packed APIs. Existing warp APIs retain four-byte input support;
packed paths require sixteen bytes and never silently fall back.

Each ten-iteration timing dispatches ten or forty kernels for one/four
requests. Actual operation keys distinguish qualified warp from packed
warp; both correctly report warp reduction and explicit candidate roles.
Timed enqueue has no allocations/frees, copies, cuBLAS calls, waits or
synchronizations. Completion uses one event wait outside these counters.

## Memory, Isolation And Validation

Both processes use **3,279,519,744 source bytes**, unchanged **1,192,935,424
prepared bytes** and identical 37 down/vocabulary tensor keys. Additional
prepared weights and resident KV are zero. This is not all 109 full-runtime
preparations. Guarded input/output scratch is reused in both arms and
unchanged throughout timing: **18,688 / 73,984 bytes QKV** and **16,512 /
65,664 output** for one/four requests. Both processes return tracked payload
0->0.

The sixty pairs contain 120 serial arm intervals and 120 GPU boundary
observations. Boundaries show P0, SM 1113 MHz, memory 3004 MHz and only each
benchmark's own process, **1676964 QKV / 1678416 output**. No inference,
build, test or MemPalace overlaps timing. Both processes exit successfully;
the subsequent GPU query is empty.

- QKV **2** and output **2** hardware tests pass with each cuBLAS setting.
- Attention grid **11**, projection **2**, MLP **1** pass in each setting;
  qualified FFN **2** also pass in the cuBLAS-disabled run.
- Benchmark CPU checks **11 pass**, nine hardware checks ignored. The
  expanded adapter hardware check is separately invoked and passes across
  both comparison types, stages and request counts. This does not claim
  that all nine ignored checks ran.
- CUDA/server all-target Clippy, non-CUDA library tests **77 pass** with
  `-D warnings`, formatting and whitespace checks pass. Two redundant test
  borrows found by lint were corrected before the implementation commit.

## Artifacts

Raw [QKV](2026-10-25-maxwell-packed-projection-qkv.jsonl) and
[output](2026-10-25-maxwell-packed-projection-output.jsonl) records,
[QKV log](2026-10-25-maxwell-packed-projection-qkv.log.gz),
[output log](2026-10-25-maxwell-packed-projection-output.log.gz),
[control SASS](2026-10-25-maxwell-packed-projection-controls.sass.gz),
[candidate SASS](2026-10-25-maxwell-packed-projection-candidates.sass.gz),
[candidate PTX](2026-10-25-maxwell-packed-projection-candidates.ptx.gz),
[resources](2026-10-25-maxwell-packed-projection-resources.txt),
[validation logs](2026-10-25-maxwell-packed-projection-validation.log.gz),
[audit program](2026-10-25-maxwell-packed-projection-audit.jq),
[audit result](2026-10-25-maxwell-packed-projection-audit.json) and
[checksums](2026-10-25-maxwell-packed-projection-screen.sha256) bind source,
execution, input/control/cache identities, numerical checks, actual dispatch,
paired medians, independent/combined gates, interval isolation and cleanup.

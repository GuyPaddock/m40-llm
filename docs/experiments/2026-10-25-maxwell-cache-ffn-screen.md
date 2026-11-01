# Fused FFN Weight-Cache Screen

## Decision

Close this candidate without runtime integration. L2-only weight/scale
loads preserve qualified control bits but improve complete one-row FFN
throughput only **0.095-0.184%**, far below the unchanged >=5% screen.
Every one-row interval's upper bound is below 0.37%. No confirmation,
cache-operator/geometry search, actual-512 qualification or HTTP run is
justified. Implementation `0cff8ab` remains benchmark-only.

The qualified runtime remains **47.379 HTTP tok/s** versus fresh official
Ollama **43.097**. Neither original target is met: **53.06** and **56.026
tok/s** for 1.30x that fresh median. Runtime defaults are unchanged.

## Configuration And Measurements

The [hypothesis](2026-10-25-maxwell-cache-ffn-hypothesis.md) was committed
as `acb3a66` before implementation. Clean source is
`0cff8abc9878edbb124394738b907ca47ce8eae3`; release benchmark SHA-256 is
`72d3e2317fc4a2f0dbe1ce58b18388898700d0cbf91ec00f68d9e224ff0c375b`.
Use the frozen M02 environment, source model, layers 0/17/35 and genuine
rows 1/4. The existing runner arguments are:

```text
--model <original Qwen Q8_0 GGUF>
--ffn-inputs target/research-fixtures/m02-ffn-inputs.json
--ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl
--ffn-f32-weight-cg --rows 1,4 --trials 5 --iterations 10
--output <new report path>
```

Both arms retain one-column/four-value/128-thread fused gate/up/SwiGLU,
runtime K, original FP32 association/reduction and reciprocal-multiply
SwiGLU. Only the candidate's weight/scale load hints change to `.cg`;
activations retain `.nc`. Prepared planes, scratch, down/vocabulary,
dependencies and streams stay fixed. No KV, quantized activations,
approximate arithmetic, scheduler or runtime-policy changes participate.

Five alternating pairs follow bounded warmup. Ten iterations per stage use
CUDA events and include all launches/dependencies. CPU references,
preparation/readback and setup remain outside primary timing. Paired
bootstrap uses 10,000 resamples with seed zero.

| Layer | Rows | Control complete ms | L2-only complete ms | Throughput gain | Paired 95% gain interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.351917 | 0.351574 | +0.097% | [+0.040%,+0.290%] |
| 17 | 1 | 0.352707 | 0.352371 | +0.095% | [-0.279%,+0.366%] |
| 35 | 1 | 0.352054 | 0.351408 | +0.184% | [-0.775%,+0.299%] |
| 0 | 4 | 1.574662 | 1.504211 | +4.684% | [-1.267%,+6.776%] |
| 17 | 4 | 1.588784 | 1.566842 | +1.400% | [+1.230%,+2.867%] |
| 35 | 4 | 1.593219 | 1.566275 | +1.720% | [+1.029%,+4.243%] |

Four-row results are independent, more variable and not a qualifying
single-token gain. The layer-0 four-row interval spans zero; it is not a
confirmed improvement. One-row gate/up medians are **0.22192-0.22292 ms
control** versus **0.22039-0.22059 ms candidate**. Isolated down remains
about **0.13149-0.13202 ms** versus **0.13148-0.13175 ms**. Independent
stage medians and complete medians have different cache histories and do
not have to sum. None of these timings is generated-token throughput.

## Emitted Code, Correctness And Memory

The preflight and actual release binary emit distinct weight instructions:
`LDG.E.CG` / `LDG.E.CG.U16` instead of `LDG.E.CI` /
`LDG.E.CI.U16`; activation loads remain `LDG.E.CI.128`. Archived candidate
PTX has `ld.global.cg.u32` / `ld.global.cg.b16` weights and
`ld.global.nc.v4.f32` activations. The qualified control SASS remains
byte-for-byte unchanged, SHA-256
`7c2d3c8e46c37367a7ce14d14d865ea5a6476b4ab5baf4e2578b45adfa182b22`.
Candidate SASS SHA-256 is
`e799bb776a011030238e48c1b0ab2615bffd6509b3a2062aec859cfe591b7c8f`.

The compiler uses **40 registers candidate versus 32 control**, no reported
stack/local storage, and the same 1024 dynamic shared bytes. Scheduling and
loop code also differ despite unchanged source geometry/math. This tests
the compiled policy candidate as a whole; it does not isolate a cache-hit
effect from compiler scheduling or register costs. No bandwidth, cache-hit
rate or occupancy is measured or inferred.

All eighteen validation records pass. Six candidate cases preserve current
control hidden/completed-down bits and captured/ordinary output agreement.
Synthetic hardware parity also compares exact gate/up bits with control.
CPU maximum absolute differences are **2.0643e-6 gate**, **3.4088e-6 up**,
**1.7127e-5 SwiGLU** and **5.0079e-5 down**, inside the existing tolerance.
Historical native/256-thread differences remain separately labeled.

The source is **3,279,519,744 bytes**; the unchanged 37-tensor prepared
down/vocabulary cache is **1,192,935,424 bytes**. One current-layer prepared
gate/up pair adds **47,906,816 bytes** in both arms. Extra candidate weight
or scratch allocation is zero. This microbenchmark does not retain all
109 full-runtime preparations or allocate KV. Cache identities, guards,
source planes and scratch remain unchanged; tracked payload returns 0->0.

Each ten-iteration gate/up timing launches ten fused kernels; complete FFN
launches twenty kernels in both arms. Actual candidate operation counts
are checked, including zero candidate calls in isolated down. Enqueue has
no allocations/frees, H2D/D2H copies, cuBLAS calls or synchronization.
Event completion occurs once after a repeated interval, outside counters.

All sixty primary GPU boundary samples report P0, SM 1113 MHz and memory
3004 MHz, with only PID `1627047`. The 180 primary stage intervals do not
overlap. No build, other inference/test or MemPalace ran during timing.
The runner exited successfully and a subsequent hardware query found no
compute processes. Driver context residency before exit is not a payload
leak.

## Validation And Artifacts

- Release FFN parity: **2 pass** with cuBLAS enabled and **2 pass** disabled.
- Attention grid **11**, projection **2**, MLP **1** pass in each cuBLAS mode.
- Benchmark CPU checks: **11 pass**, eight hardware checks ignored. The two
  affected full-pipeline checks were separately invoked and **both pass**;
  this is not a claim that all eight ignored checks ran.
- CUDA/server all-target Clippy, non-CUDA library tests with `-D warnings`
  **77 pass**, formatting and whitespace checks pass.

Raw [JSONL](2026-10-25-maxwell-cache-ffn-screen.jsonl), compressed
[log](2026-10-25-maxwell-cache-ffn-screen.log.gz),
[control SASS](2026-10-25-maxwell-cache-ffn-control.sass.gz),
[candidate SASS](2026-10-25-maxwell-cache-ffn-candidate.sass.gz),
[candidate PTX](2026-10-25-maxwell-cache-ffn-candidate.ptx.gz),
[resources](2026-10-25-maxwell-cache-ffn-resources.txt),
[audit program](2026-10-25-maxwell-cache-ffn-audit.jq),
[audit result](2026-10-25-maxwell-cache-ffn-audit.json) and
[checksums](2026-10-25-maxwell-cache-ffn-screen.sha256) preserve the evidence.
The [validation log](2026-10-25-maxwell-cache-ffn-validation.log.gz) records
the listed test and lint invocations.
The audit binds source/executable/model/input/environment identities,
37 prepared-cache keys, coverage, arithmetic checks, actual dispatch,
cleanup, GPU boundaries and independently recomputed complete medians.

Next inspect the qualified signed-byte conversion/FP32 dependency chain
for an independent exact-arithmetic hypothesis. Do not expand cache hints,
retune rejected geometries, combine below-gate variants or substitute
numerical/model changes for the original goal.

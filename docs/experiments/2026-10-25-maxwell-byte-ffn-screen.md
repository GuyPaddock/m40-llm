# Exact Byte-Permute FFN Screen

## Decision

Close the conversion experiment without runtime integration. It preserves
qualified control bits, removes the byte-to-float instructions and retains
32 registers, but complete one-row FFN gains are only **0.138-0.295%**.
Every paired 95% interval crosses zero and its upper bound is below 0.40%.
This decisively misses the unchanged >=5% screen. Four-row results are
also near neutral. No confirmation, alternative conversion scheme,
geometry search, actual-512 qualification or HTTP run follows.
Implementation `d5a3350` remains benchmark-only.

The qualified runtime remains **47.379 HTTP tok/s** versus fresh official
Ollama **43.097**. Neither original target is met: **53.06** and **56.026
tok/s** for 1.30x that fresh median. Defaults are unchanged.

## Configuration And Results

The [prespecified hypothesis](2026-10-25-maxwell-byte-ffn-hypothesis.md) was
committed as `1ce4178`. Clean source is
`d5a33505435f102abc0db5c19051c793fec69741`; release executable SHA-256 is
`749ec46e178d5d9d8ad0f214dbbd0785e62eaad16d7a37cee27a74a81c4d1f4c`.
Use the frozen M02 environment, original model and inputs from layers
0/17/35 with genuine rows 1/4. Runner arguments are:

```text
--model <original Qwen Q8_0 GGUF>
--ffn-inputs target/research-fixtures/m02-ffn-inputs.json
--ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl
--ffn-f32-byte-permute --rows 1,4 --trials 5 --iterations 10
--output <new report path>
```

Only signed-byte conversion changes. XOR bias plus byte permutation
constructs `2^23 + signed_byte + 128`; rounded subtraction of `8388736`
recovers the exact original float. All original scaling, FMA association,
task order, reduction and SwiGLU stay fixed. Geometry is one column,
four values, 128 threads and runtime K. Both arms retain the same read-only
loads, split planes, scratch, streams and qualified prepared down/vocabulary.
No KV or activation quantization participates.

Five alternating pairs follow bounded warmup. Each interval includes ten
repetitions with CUDA events, all dispatch/dependencies and one completion
wait. Setup, preparation, CPU references and readback are excluded. Paired
bootstrap uses 10,000 resamples with seed zero.

| Layer | Rows | Control complete ms | Byte-permute complete ms | Throughput gain | Paired 95% gain interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.351930 | 0.351283 | +0.184% | [-0.081%,+0.394%] |
| 17 | 1 | 0.352381 | 0.351344 | +0.295% | [-0.401%,+0.348%] |
| 35 | 1 | 0.351661 | 0.351178 | +0.138% | [-0.184%,+0.341%] |
| 0 | 4 | 1.478054 | 1.470941 | +0.484% | [-0.132%,+2.645%] |
| 17 | 4 | 1.595811 | 1.587450 | +0.527% | [-2.001%,+2.057%] |
| 35 | 4 | 1.585318 | 1.590054 | -0.298% | [-1.599%,+0.072%] |

One-row gate/up medians are **0.22159-0.22175 ms control** versus
**0.22042-0.22067 ms candidate**. Isolated down is **0.13124-0.13167 ms**
versus **0.13109-0.13149 ms**. Independently sampled stage medians have
different cache histories and need not sum to the complete interval.
These are kernel-pipeline timings, not generated-token speedups.

## Instructions, Correctness And Memory

Actual release SASS has **24 static PRMT sites and no I2F** in the
candidate, versus **24 static I2F sites** in the control. The compiler
emits rounded FP32 subtraction before the unchanged scale multiply.
Static sites include peeled/loop bodies; they are not dynamic counts.
Both kernels use **32 registers**, no reported stack/local storage and
1024 dynamic shared bytes. No cache, occupancy, bandwidth or stall
attribution follows. Eliminating these conversions did not yield a useful
complete-FFN gain.

Control SASS remains byte-for-byte unchanged, SHA-256
`7c2d3c8e46c37367a7ce14d14d865ea5a6476b4ab5baf4e2578b45adfa182b22`.
Candidate SASS SHA-256 is
`f88ec38798ba627fa2de0638ea97656ffd6586e2ad6ffbe7d99da63a13507ff3`.

All eighteen validation records pass. Six candidate cases retain current
control hidden and completed-down bits, with captured/ordinary output
agreement. Synthetic parity additionally checks gate/up bit equality and
asserts coverage of **all 256 byte values in each packed position with
nonzero scales**, alongside zero scales, signed extremes, distinct rows,
width/hidden tails, misalignment, aliases, guards and invalid dimensions.
CPU maximum absolute differences remain **2.0643e-6 gate**, **3.4088e-6
up**, **1.7127e-5 SwiGLU**, **5.0079e-5 down**. Historical native/256-thread
drift is reported separately from current-control equality.

Source weights are **3,279,519,744 bytes**; unchanged prepared
down/vocabulary is **1,192,935,424 bytes**, with identical 37 tensor keys.
Both arms retain the same current-layer **47,906,816-byte** prepared
gate/up pair and scratch. No additional candidate weight/scratch is
allocated. This is not all 109 runtime preparations or KV residency.
Tracked payload returns 0->0.

Ten gate/up iterations launch ten fused kernels; complete FFN launches
twenty kernels for both arms. Actual candidate operation counts are
validated, including zero candidate calls in isolated down. Timed enqueue
has zero allocations/frees, H2D/D2H copies, cuBLAS calls or synchronization.
All 180 primary stage intervals are serial. Sixty GPU boundaries report P0,
SM 1113 MHz, memory 3004 MHz and only PID `1647356`. No build, test, other
inference or MemPalace overlaps timing. The runner exits successfully and
the post-exit hardware query shows no compute processes.

## Tests And Evidence

- Release FFN **2**, attention grid **11**, projection **2**, MLP **1** pass
  in each cuBLAS mode.
- Benchmark CPU checks **11 pass**, nine hardware checks ignored; the three
  affected complete-pipeline checks are separately invoked and **all pass**.
  This is not a claim that all nine ignored checks ran.
- CUDA/server all-target Clippy, non-CUDA library tests **77 pass** under
  `-D warnings`, formatting and whitespace checks pass.

Raw [JSONL](2026-10-25-maxwell-byte-ffn-screen.jsonl),
[log](2026-10-25-maxwell-byte-ffn-screen.log.gz),
[control SASS](2026-10-25-maxwell-byte-ffn-control.sass.gz),
[candidate SASS](2026-10-25-maxwell-byte-ffn-candidate.sass.gz),
[candidate PTX](2026-10-25-maxwell-byte-ffn-candidate.ptx.gz),
[resources](2026-10-25-maxwell-byte-ffn-resources.txt),
[validation logs](2026-10-25-maxwell-byte-ffn-validation.log.gz),
[audit program](2026-10-25-maxwell-byte-ffn-audit.jq),
[audit result](2026-10-25-maxwell-byte-ffn-audit.json) and
[checksums](2026-10-25-maxwell-byte-ffn-screen.sha256) preserve provenance,
coverage, control/cache identities, counters, medians, isolation and cleanup.

## Independent Next Source Finding

The qualified QKV/output paths account for **9.24% / 6.64%** of sampled
middle-token GPU work. Source and [actual release SASS](2026-10-25-maxwell-byte-ffn-projection-followup.sass.gz)
still show scalar loads there: eight U8 weight loads per eight-value task,
plus eight scalar FP32 activation loads. QKV's first task has weight loads
at `03a8` through `0498`; output's first column has them at `04a8` through
`0588`. These paths have qualified warp reductions, but not packed loads.

Next bound an independent alignment-safe load-vectorization experiment
there: four U16 weight loads and two float4 activation loads for each
eight-value task, while retaining existing work assignment, 256 threads,
one QKV output/four output-projection columns, original signed conversion,
scale/FMA/reduction association and bias behavior. No repacking is required
for two-byte-aligned Q8_0 payloads. Check emitted code before timing; retain
the qualified FFN/down/vocabulary and all actual-quality/HTTP gates. This
is not permission to combine the failed conversion/cache experiments or
expand their geometry. Source load counts are not measured traffic or a
performance forecast.

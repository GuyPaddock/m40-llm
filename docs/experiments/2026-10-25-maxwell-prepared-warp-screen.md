# Prepared Projection Warp-Finish Screen

## Decision

Keep the implementation **benchmark-only**. The fixed original-tree warp
finish preserves every tested control bit but improves complete one-row FFN
throughput only **2.44-2.60%**, below the unchanged **5%** gate. Isolated
down gains of **7.53-8.12%** do not override that complete-FFN requirement.
One-row vocabulary improves **6.54%**, but its optimistic standalone
profile-weighted E2E opportunity is only **0.474%**, below the **3%** floor.
Do not integrate either candidate or expand geometry to rescue the result.

The qualified runtime remains **47.379 HTTP tok/s** against **43.097**
official Ollama. Neither original goal, **53.06** or **56.026 tok/s** for
1.30x that fresh median, is met. Defaults remain unchanged. No new model
generation, actual-512 qualification or HTTP benchmark is warranted by this
below-gate screen.

## Fixed Experiment

Primitive implementation `8d1c15d`; benchmark implementation and clean run
source `5218f41805467e595b5a5e4bdfa78ae4ece8cdc2`. Release executable SHA-256:
`6aab5b02fcf77ba3efb259589526d23fe3ced73a53480a75eb7a7fbdb4ab9f75`.
The same original Qwen2.5-3B Q8_0 model, frozen calibration inputs and
qualified prepared down/vocabulary cache remain. No KV is allocated.

Only the reduction changes: original FP32 tasks, accumulation association,
eight-value packed loads, scale/value planes, four-column width and launch
threads remain. One-row down uses 256 threads; four-row down and vocabulary
use 128. Candidate publication is followed by an original-tree warp finish.
Gate/up/SwiGLU stays `split-f32x4-128` in **both** complete-FFN arms.
Four rows are genuine distinct matrix rows, not repetitions or four separate
decode launches. Vocabulary uses real model weights and frozen normalized
FFN layer-0 input surrogates, not captured output-normalized hidden states.

Source `scripts/dev-env.sh`, export the frozen M02 environment as in the
[quality reproduction](2026-10-25-maxwell-projection-runtime-quality.md#configuration-and-reproduction),
and run the release `q8_execution_bench` with:

```text
--model <same Qwen Q8_0 GGUF>
--ffn-inputs target/research-fixtures/m02-ffn-inputs.json
--ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl
--prepared-f32-warp --rows 1,4 --trials 5 --iterations 10
--output <new JSONL path>
```

Each case uses five alternating contemporaneous pairs. Warmup precedes
primary timing and is not pooled into results. CUDA events include all
required stream dependencies; setup, CPU references, plane readback and
packing are outside timing. Complete FFN and isolated down are measured
separately. Bootstrap intervals use 10,000 paired resamples, seed zero.

## Results

| Layer | Rows | Complete FFN control/candidate ms | Complete gain | Isolated down control/candidate ms | Down gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.351757 / 0.343200 | +2.493% | 0.132125 / 0.122202 | +8.120% |
| 17 | 1 | 0.351885 / 0.343501 | +2.441% | 0.132051 / 0.122803 | +7.531% |
| 35 | 1 | 0.351907 / 0.343002 | +2.596% | 0.131693 / 0.122339 | +7.646% |
| 0 | 4 | 1.591043 / 1.575021 | +1.017% | 0.588387 / 0.571990 | +2.867% |
| 17 | 4 | 1.526371 / 1.531814 | -0.355% | 0.565098 / 0.559421 | +1.015% |
| 35 | 4 | 1.536349 / 1.521539 | +0.973% | 0.561357 / 0.555811 | +0.998% |

One-row complete-FFN 95% gain intervals are **[1.868%,3.742%]**,
**[2.308%,2.739%]** and **[2.044%,2.742%]**. All upper bounds remain below
5%; this is not a borderline pass requiring twenty fresh pairs. Four-row
complete intervals include zero and supply no independently qualified gain.

| Vocabulary rows | Control ms | Candidate ms | Gain | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.562976 | 1.466970 | +6.545% | [6.480%,6.580%] |
| 4 | 6.212550 | 5.844864 | +6.291% | [6.278%,6.311%] |

Vocabulary clears its separate projection micro gate, not runtime
feasibility. Using its 7.6763% middle-stage share from the
[qualified profile](2026-10-25-maxwell-projection-stage-profile.md), the
optimistic standalone throughput estimate is **0.473753%**. The rejected
down result cannot be pooled in to bypass its own complete-FFN gate. No
DRAM bandwidth, occupancy or E2E gain is inferred from these timings.
The raw isolated-down summaries reuse the generic summarizer's
`passes_micro_screen` name; only `complete_ffn` is the down eligibility gate.

## Instructions, Memory And Correctness

The control's release SASS is byte-for-byte unchanged from the previously
qualified executable: SHA-256
`333013085bd1e550afc35c6bf329183b675be4679e71bea69cdf994ea6673fa1`.
It has two static barrier sites, one inside the reduction loop, yielding
nine dynamic block barriers at 256 threads or eight at 128. Each candidate
has one publication barrier and twenty static shuffle instructions. Control
and both candidates use **40 registers**, with zero reported stack/local
storage. Shared capacity remains 4096 / 2048 bytes for 256 / 128 threads.
Fewer barriers alone do not provide enough complete-FFN improvement here.

The screen retains **3,279,519,744 source bytes** and the full **37-tensor,
1,192,935,424-byte prepared down/vocabulary cache**, totaling
**4,472,455,168 bytes** before scratch. Each FFN case temporarily prepares
the same **47,906,816-byte gate/up pair** for both arms. This microbenchmark
does not keep all 36 prepared gate/up pairs resident, unlike the qualified
runtime's 109-tensor cache. Candidates add **zero** weight/scratch capacity.
All repeated enqueues avoid allocations/frees, copies, cuBLAS and internal
synchronization, with actual selected operation counts verified. Source
planes, guarded buffers, distinct input rows and model cache identity remain
unchanged. Tracked payload returns from zero to zero; driver context storage
remains until process exit and is not reported as tracked KV/model bytes.

Debug and release primitive tests each pass the two control and two new
warp tests. CPU references cover tails, signed extremes and non-binary
inputs; new tests include guards, both streams, dependent consumers, invalid
calls and ownership. CUDA/server all-target Clippy and 77 warning-as-error
non-CUDA library tests pass. With cuBLAS disabled, attention parity has 11
passes, projection/MLP wrappers 2/1, and the new warp tests 2. The forward
suite passes its fixtures; the external real-Qwen diagnostic has no supplied
model and is not new real-model evidence. Benchmark argument/provenance/bit
tests have 11 CPU passes, plus the explicitly run new synthetic full-FFN
hardware test. Other ignored hardware cases are not claimed as run here.

All eighty primary boundaries show P0 and 1113/3004 MHz with only benchmark
PID 1568441 present. No builds, tests, Ollama or MemPalace work overlaps
timing. The benchmark exits successfully and the post-exit GPU process list
is empty.

## Evidence And Next Work

- [Raw paired measurements](2026-10-25-maxwell-prepared-warp-screen.jsonl)
- [Audit program](2026-10-25-maxwell-prepared-warp-audit.jq) and [derived medians/gates](2026-10-25-maxwell-prepared-warp-audit.json)
- [Control SASS](2026-10-25-maxwell-prepared-warp-control.sass.gz), [candidate SASS](2026-10-25-maxwell-prepared-warp-candidate.sass.gz) and [resource usage](2026-10-25-maxwell-prepared-warp-resources.txt)
- [Run log](2026-10-25-maxwell-prepared-warp-screen.log.gz) and [checksums](2026-10-25-maxwell-prepared-warp-screen.sha256)

Run `jq -s` with `--slurpfile capture` pointing to frozen capture,
`--slurpfile profile` pointing to the qualified projection stage audit,
`-f` pointing to this auditor, and the raw screen path. Negative controls
with wrong operation, timed allocation or fabricated gain must be rejected.

Stop this reduction experiment. Preserve `8d1c15d` / `5218f41` and the
measurements, with no generation wiring or geometry expansion. Return to
the qualified gate/up hot loop's instruction/load/conversion dependencies
before proposing another independent bounded exact hypothesis. Do not
revive failed layouts or substitute numerical/model-changing work for the
original throughput goals.

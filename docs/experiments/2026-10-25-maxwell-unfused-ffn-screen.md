# Unfused Prepared FFN Screen

## Decision

Stop this strategy. The fixed unfused gate/up/SwiGLU decomposition preserves
qualified control bits but **reduces complete one-row FFN throughput by
14.14-14.27%**. Four-row throughput falls **12.18-12.76%**. Every paired
95% interval is negative. This decisively misses the unchanged >=5% gain
gate; no additional geometry, runtime integration, actual-512 qualification
or HTTP run follows. Implementation `286cb9a` remains benchmark-only.

Pinned Ollama's unfused Maxwell dispatch was a reason to test decomposition,
not evidence that decomposition alone explains its performance. Ollama also
uses Q8_1 activation quantization and a different dot-product organization.
This result does not reopen that numerical track as a substitute for the
original exact-output goal.

The qualified runtime stays **47.379 HTTP tok/s** versus fresh official
Ollama **43.097**. Both original targets remain unmet: **53.06** and
**56.026 tok/s** for 1.30x that fresh median. Defaults remain unchanged.

## Configuration

Clean source `286cb9ab621662e779ab163927b06e3fd7fe5db4`, release benchmark
SHA-256 `1f6ca5ab9ed206829c368528e5fab6ef57e70eb17badf886b394da1bd16cce20`.
The [prespecified hypothesis](2026-10-25-maxwell-unfused-ffn-hypothesis.md)
is committed first as `e5ecee2`. It records existing compiler load overlap
and the pinned architecture guard, rather than assuming prefetch is absent.

Use the frozen M02 environment, original Qwen2.5-3B Q8_0, layers 0/17/35,
genuine rows 1/4 and the existing release `q8_execution_bench` with:

```text
--model <original Qwen Q8_0 GGUF>
--ffn-inputs target/research-fixtures/m02-ffn-inputs.json
--ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl
--ffn-f32-unfused --rows 1,4 --trials 5 --iterations 10
--output <new report path>
```

Both arms use the same prepared planes, FP32 per-output task/reduction
association and qualified prepared down/vocabulary. The control is the
one-column split-f32x4-128 fused FFN. The candidate uses two fixed
four-column/128-thread projections plus a control-order SwiGLU kernel.
It reuses existing gate/up scratch. No layout, quantization, activation
format, KV, attention, scheduler or runtime policy changes participate.
The general division-based SwiGLU kernel is deliberately not substituted
for the qualified reciprocal-then-multiply expression.

Warmup precedes primary timing. Each case uses five alternating pairs,
ten iterations per stage, CUDA events and all required stream dependencies.
All three candidate launches and intermediate traffic enter gate/up timing;
complete FFN also includes unchanged down. Preparation, CPU references and
readback are excluded. Bootstrap uses 10,000 paired resamples, seed zero.

| Layer | Rows | Control complete ms | Unfused complete ms | Throughput gain | Paired 95% gain interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.351834 | 0.410310 | -14.252% | [-14.413%,-14.201%] |
| 17 | 1 | 0.352269 | 0.410301 | -14.144% | [-14.913%,-13.989%] |
| 35 | 1 | 0.351869 | 0.410458 | -14.274% | [-14.415%,-14.146%] |
| 0 | 4 | 1.513174 | 1.723091 | -12.183% | [-13.773%,-11.515%] |
| 17 | 4 | 1.465763 | 1.678192 | -12.658% | [-14.640%,-11.637%] |
| 35 | 4 | 1.494368 | 1.712886 | -12.757% | [-14.879%,-11.214%] |

One-row gate/up/SwiGLU increases from **0.22185-0.22192 ms** to
**0.27996-0.28005 ms**. Isolated down remains about **0.13124-0.13162 ms**.
Independent stage and complete medians have different cache histories and
need not sum. No kernel-only number is presented as generated-token speed.

## Instructions, Correctness And Ownership

The qualified control SASS is byte-for-byte unchanged, full-dump SHA-256
`7c2d3c8e46c37367a7ce14d14d865ea5a6476b4ab5baf4e2578b45adfa182b22`.
It retains 32 registers, no reported stack/local storage and 1024 dynamic
shared bytes. The new projection uses 48 registers, no reported stack/local
storage and 2048 dynamic shared bytes. Its separate SwiGLU uses 17 registers
and no shared/stack/local storage. These resource differences and extra
launches are observed; their individual contribution to regression is not
measured occupancy or bandwidth attribution.

All eighteen validation records pass. Six candidate cases preserve current
control hidden/completed-down bits; captured gate/up and normal output
agree. CPU maximum absolute differences are **2.0643e-6 gate**, **3.4088e-6
up**, **1.7127e-5 SwiGLU** and **5.0079e-5 down**, within the original
tolerance. Drift versus the older native/256-thread path is reported
separately, not mislabeled as drift from the current split timing control.

The 37-tensor prepared down/vocabulary cache is **1,192,935,424 bytes**;
source weights are **3,279,519,744 bytes**. Each FFN case retains the same
**47,906,816-byte** gate/up pair for both arms. Candidate extra weight and
scratch capacity is zero. Both arms retain the existing guarded gate/up
buffers even though the fused control ordinarily leaves them unused.
There is no full-model gate/up preparation or KV allocation in this micro
screen. Tracked payload returns from zero to zero.

Every ten-iteration candidate gate/up timing records **20 projection + 10
SwiGLU launches**; complete FFN records 40 launches versus the control's 20.
Timed enqueue has no allocations/frees, copies, cuBLAS calls or internal
synchronization. Per-row stream waits remain explicit. Guards, input/source
planes, prepared-cache identity and scratch capacity remain unchanged.

Optimized cuBLAS-enabled primitive tests pass both FFN tests, covering the
new strategy alongside retained variants, intermediate bit equality, signed
extremes, zero scales, hidden tails, distinct rows, guards and invalid calls.
The new full synthetic pipeline test passes rows 1/4, exact outputs, cache
identity, scratch reuse and all four complete launches. The CPU-only
benchmark invocation has 11 passes and seven ignored entries; only the
separately invoked new hardware test is claimed here.
CUDA/server all-target Clippy passes. With cuBLAS disabled, FFN has two
passes, attention grid eleven, projection wrappers two and MLP wrapper one.
All 77 warning-as-error non-CUDA library tests and formatting pass.

All sixty primary GPU boundaries show P0, 1113/3004 MHz and only benchmark
PID 1603095. No builds, tests, Ollama or MemPalace work overlaps timing.
The process exits successfully; a separate post-exit check finds no GPU
compute processes. The in-process final snapshot's residual driver context
is not tracked model/KV memory and disappears on exit.

## Evidence And Next Work

- [Raw paired report](2026-10-25-maxwell-unfused-ffn-screen.jsonl)
- [Audit program](2026-10-25-maxwell-unfused-ffn-audit.jq) and [derived result](2026-10-25-maxwell-unfused-ffn-audit.json)
- [Control SASS](2026-10-25-maxwell-unfused-ffn-control.sass.gz), [candidate SASS](2026-10-25-maxwell-unfused-ffn-candidate.sass.gz) and [resources](2026-10-25-maxwell-unfused-ffn-resources.txt)
- [Qualified control PTX entry](2026-10-25-maxwell-unfused-ffn-control.ptx.gz)
- [Run log](2026-10-25-maxwell-unfused-ffn-screen.log.gz) and [checksums](2026-10-25-maxwell-unfused-ffn-screen.sha256)

Reproduce the audit with `jq -s --slurpfile capture` pointing to frozen
capture, `-f` pointing to this program and the raw report path. It checks
actual pairs, full candidate dispatch, arithmetic/ownership evidence,
medians, negative gate decisions, clocks, processes and serial intervals.
Wrong-control, missing-launch and fabricated-gain negative controls fail.

Keep this decomposition closed without column/thread retuning. The
qualified control's PTX already uses `ld.global.nc` for both FP32 activations
and weight payload/scales, so merely adding `__ldg` would not establish a
different execution path. Next examine whether an explicitly different
weight-cache policy is justified, with emitted-code evidence before timing
and no measured cache-hit claim. Existing fused arithmetic, layouts,
qualified down/head and original micro/quality/HTTP gates remain fixed.

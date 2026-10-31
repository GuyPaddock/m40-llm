# Multi-row Q4 fusion result

The bounded fusion experiment is complete with a negative integration decision.
No case clears the 5% complete-FFN gate against all three controls. The new
multi-row launch improves complete FFN throughput by at most 0.81% against
repeated existing fused gate/up calls using the same down consumer. Batch-one
gate/up fusion already exists. Keep runtime dispatch unchanged.

## Inputs and timing

The clean release executable was built from
`9aa906e33c62b3f2f39d7627252b6c7687aa51cd`; its SHA-256 is
`f6b4014b68d1140792004e91298dd109cf0df59e892c46b97e96fb617ebe033a`.
The run used the Tesla M40, driver 580.159.04, CUDA 12.8, retained Qwen2.5-3B
inputs from layers 0, 17 and 35, and distinct rows at batch sizes 1, 2, 4 and 8.
Gate/up dimensions are K2048/H11008. The source-bound manifest records model,
capture, executable and encoded-weight hashes and the exact environment.

All arms use the same derived Q4 bytes. These fixtures add quantization to the
original Q8 checkpoint; they do not establish the quality of a released Q4
model. Source conversion errors remain in each layer record. Original and
prepared Q8 weights, native Q4 weights and all diagnostic buffers remain
resident in every arm.

Each complete arm warms for at least one GPU second. Five alternating pairs
per comparison, ten operations per sample, produce 300 pairs across 36
complete-FFN and 24 gate/up comparisons. All comparisons are stable; maximum
throughput MAD/median is 1.8972%. None triggers the frozen extension or
confirmation rule. Paired confidence intervals use 10,000 bootstrap resamples
and seed zero. CUDA events include required stream dependencies; raw records
also contain host enqueue and completion times and enqueue counters.

The terminal record confirms all 12 cases and 60 comparisons completed.
A subsequent device inspection found no compute processes and zero MiB used.
The terminal record itself precedes model teardown and therefore still shows
the benchmark process and its model allocation.

## Complete FFN comparisons

Values below are throughput changes, `control time / candidate time - 1`,
against the same-down control. Each timing is the median milliseconds for
the entire row batch. The summary retains all confidence intervals and the
other paired controls.

| Layer | Rows | Existing fused rows then batched down, ms | New fused rows then same down, ms | Gain |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1 | 1.887731 | 1.909437 | -1.137% |
| 17 | 1 | 1.886467 | 1.884698 | 0.094% |
| 35 | 1 | 1.912566 | 1.913747 | -0.062% |
| 0 | 2 | 3.268774 | 3.268368 | 0.012% |
| 17 | 2 | 3.335510 | 3.327005 | 0.256% |
| 35 | 2 | 3.286742 | 3.281239 | 0.168% |
| 0 | 4 | 5.904547 | 5.860301 | 0.755% |
| 17 | 4 | 5.984064 | 5.955405 | 0.481% |
| 35 | 4 | 5.937968 | 5.895878 | 0.714% |
| 0 | 8 | 10.761945 | 10.676678 | 0.799% |
| 17 | 8 | 10.692678 | 10.615613 | 0.726% |
| 35 | 8 | 10.789024 | 10.702441 | 0.809% |

The candidate beats the unfused batched diagnostic by 95.0% to 125.1%.
Existing gate/up fusion explains most of that difference. Against serial
gate/up and serial down, the candidate loses at rows 1, 2 and 4, and gains
13.66% to 14.95% at eight rows. The latter comparison also changes down
execution; the same-down results above isolate the incremental fusion value.
Gate/up-only gains over existing serial fusion range from 0.019% to 1.469%.
No comparison qualifies a whole-model or HTTP speed claim.

## Correctness and memory

All 12 cases pass the four arms' independent FP64 hidden/output comparisons.
The largest absolute error is 0.000635173 within the frozen numerical bounds.
New hidden outputs match repeated existing single-row fusion bit for bit,
and complete outputs match the same-down fused control bit for bit. Repeated
checks after timing, input/buffer guards, immutable-weight hashes and all
case/layer ownership checks pass. The
[implementation checkpoint](2026-10-31-ffn-fusion-implementation.md) records
the earlier cuBLAS on/off, stream, regression, CPU and Clippy checks.

Each layer retains 38,043,648 bytes of native Q4 weight payload. Diagnostic
scratch ranges from 148,800 to 1,188,160 bytes; peak tracked device residency
is 4,511,686,976 bytes with the original and prepared model still resident.
Layer fixture preparation/upload takes 1317.3 to 1332.3 ms outside timing.
The candidate avoids the unfused path's logical gate/up writes and SwiGLU
reads, 176,128 bytes per row. This is not measured DRAM traffic; no hardware
memory-sector or instruction-count claim is made.

## Reproduction and decision

Build and invoke the benchmark following the
[fixed implementation protocol](2026-10-31-ffn-fusion-implementation.md).
Restore exactly the capture manifest's environment before execution, using
the capture report already present in this branch. Runtime arguments were:

```text
--model <Qwen2.5-3B Q8 blob with SHA-256 4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d>
--output target/q4-fusion-timing-01.jsonl
--ffn-inputs <retained m02-ffn-inputs.json>
--ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl
--ffn-q4-fusion --rows 1,2,4,8 --trials 5 --iterations 10
```

The [raw timing records](2026-10-31-q4-fusion-timing.jsonl.gz),
[run log](2026-10-31-q4-fusion-timing.log.gz) and
[release build log](2026-10-31-q4-fusion-release.log.gz) preserve the run.
The timing archive SHA-256 is
`a3a42d9a0cf2f42090b95a7a3611fb1566fa0ab927fbeed4cbc8865e7c25d6a0`.
Regenerate the [summary](2026-10-31-q4-fusion-summary.json) with:

```bash
gzip -dc docs/experiments/2026-10-31-q4-fusion-timing.jsonl.gz |
  jq -s -f docs/experiments/2026-10-31-q4-fusion-summary.jq
```

Stop this candidate before runtime integration or model-quality evaluation.
Retain its verified experimental implementation at `fa09568` and benchmark
at `9aa906e`. The result applies to this fixed multi-row launch and these
captured dense-Qwen inputs. It does not rule out other fusion designs or
establish MoE prefill performance. The independent activation-preparation
branch remains the next planned investigation.

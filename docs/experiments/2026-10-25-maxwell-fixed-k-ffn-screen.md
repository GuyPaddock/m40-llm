# Fixed-Width Split FFN Screen

## Decision

The single K=2048 address/loop specialization preserves control bits but
improves complete one-row FFN throughput only **1.24-1.32%**. All confidence
bounds are far below the unchanged 5% gate. Four-row point gains are
0.73-1.12%, with layer 0's interval crossing zero. Keep implementation
`deef80b` benchmark-only. This is a clear below-gate result, not a borderline
case requiring more trials. No runtime integration, unroll/register/geometry
rescue, actual-512 quality run or HTTP trial follows.

The last qualified split-f32x4-128 plus SGEMM runtime still measures
45.075 HTTP tok/s versus official Ollama 43.119. Both original goals remain
unmet: 53.06 and 56.055 tok/s for 1.30x that fresh median. No default changes.

## Fixed Experiment

Clean release source `deef80ba14cbc9f6c78412dcbef34bdd6790ebf9`, benchmark
SHA-256 `0734046dedda93b5723d54622c46b2ee712190532cce79b6f89c7472a9c7503e`.
The [pre-implementation hypothesis](2026-10-25-maxwell-fixed-k-ffn-hypothesis.md)
identified runtime address products and loop control for a known width.
The candidate exposes K=2048, spells out four ordered tasks per thread and
keeps original separate scale/value planes, one column, 128 threads, per-weight
FP32 math, publication and reduction. It does not use the rejected joint layout.

Use the frozen M02 environment and inputs with the shared runner:

```bash
source scripts/dev-env.sh
while IFS=$'\t' read -r key value; do
  export "$key=$value"
done < <(head -n 1 docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl |
  jq -r '.execution_configuration.environment | to_entries[] | [.key,.value] | @tsv')
target/release/examples/q8_execution_bench \
  --model "$MODEL" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-f32-fixed-k --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-fixed-k-ffn-screen.jsonl
```

Five alternating pairs per original layer/row shape, ten iterations per
stage. Gain is median control latency / median candidate latency minus one.
Paired bootstrap uses 10,000 resamples with seed zero. The timing control
is the qualified one-column split, not older packed/warp128.

| Layer | Rows | Dynamic K control ms | Fixed K ms | Throughput gain | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.352112 | 0.347770 | +1.25% | +1.05% to +1.40% |
| 17 | 1 | 0.352358 | 0.347754 | +1.32% | +1.19% to +1.41% |
| 35 | 1 | 0.352019 | 0.347722 | +1.24% | +0.84% to +1.25% |
| 0 | 4 | 1.594704 | 1.577117 | +1.12% | -2.16% to +3.63% |
| 17 | 4 | 1.535139 | 1.524064 | +0.73% | +0.36% to +1.57% |
| 35 | 4 | 1.533654 | 1.517107 | +1.09% | +0.01% to +2.62% |

One-row gate/up decreases from about 0.2216-0.2220 ms to
0.2167-0.2168 ms. Isolated prepared down stays about 0.1312-0.1316 ms.
Stage sums and complete medians have different cache histories and need
not coincide. Small isolated gains are not an end-to-end speed claim.

## Instructions, Numerics And Ownership

The candidate eliminates all 21 static prologue XMAD sites before its first
global load, and removes the hot loop's backward branch. It retains twenty
static global loads for its four unrolled tasks. The dynamic control has a
peeled task plus a two-task loop, so static site totals are not dynamic
instruction counts. Output-row indexing and the SiLU epilogue still contain
integer arithmetic; this is not an XMAD-free kernel.

Both kernels use 32 registers, no reported stack/local spill storage, 1024
dynamic shared bytes and one block-wide barrier. The qualified control's
extracted SASS remains byte-identical, SHA-256
`84a8cfe7d64c3ac38fc52666e7f509a79893422e50f683db591f7e4c4a1c69fd`.
Removing these address/loop sites produces only a small measured improvement.
It does not establish a DRAM bandwidth or occupancy explanation, and does
not justify a further unroll/geometry sweep.

All eighteen validation records pass. The six candidate shapes preserve
control hidden and completed-down bits. Captured gate/up and ordinary
SwiGLU agree. Largest CPU absolute errors are 2.0644e-6 gate, 3.4088e-6 up,
1.7127e-5 SwiGLU and 5.0080e-5 completed down, within unchanged tolerance.
Synthetic checks cover signed extremes, zero scales, odd hidden tails,
distinct rows, immutable inputs/planes, guards and unsupported arguments.
K!=2048 fails clearly before GPU work, with no fallback to another kernel.

Both arms retain the same 47,906,816-byte prepared pair: 2,818,048 scale
bytes and 45,088,768 payload bytes. There is no joint pair or quantized
activation scratch. Original plus prepared down/vocabulary is
4,472,455,168 bytes; scratch is 201,152 / 803,264 bytes for rows 1/4.
Tracked totals are 4,520,563,136 / 4,521,165,248 bytes. Median per-case
packing/upload is 26.740 / 8.178 ms, outside timing. Tracked payload
returns zero to zero; after process exit the GPU compute-process query
is empty. In-process driver context memory is not included in that claim.

All thirty pairs / 180 primary intervals are nonoverlapping. Every one of
sixty boundaries records sole process 1366156, P0 and 1113/3004 MHz clocks.
No other GPU workload, build, tests or MemPalace embedding runs during timing.
There are zero timed allocations, frees, copies, cuBLAS calls or stream
synchronizations; required row-batched producer waits remain included.

Before timing, cuBLAS-on/off each pass four primitive and twelve benchmark
tests. Non-CUDA warning-as-error tests pass 210 cases; CUDA/server all-target
Clippy, formatting and 48 affected CUDA runtime/attention/forward/wrapper
tests pass. The optional external real-Qwen fixture was unconfigured, not
real-model generation proof. Real model FFN inputs above are not a substitute
for the gated full-model quality run.

## Evidence And Next Direction

- [Raw records](2026-10-25-maxwell-fixed-k-ffn-screen.jsonl), [pair audit](2026-10-25-maxwell-fixed-k-ffn-audit.json) and [audit program](2026-10-25-maxwell-fixed-k-ffn-audit.jq)
- [Memory audit](2026-10-25-maxwell-fixed-k-ffn-memory-audit.json), using the unchanged [split memory auditor](2026-10-25-maxwell-split-ffn-memory-audit.jq)
- [Compiler resources](2026-10-25-maxwell-fixed-k-ffn-resources.txt), [SASS](2026-10-25-maxwell-fixed-k-ffn-sass.txt.gz), [log](2026-10-25-maxwell-fixed-k-ffn-screen.log.gz) and [checksums](2026-10-25-maxwell-fixed-k-ffn.sha256)

The auditor recomputes actual-pair medians and the gate, checks bitwise/
enqueue/clock/process/cleanup evidence, and rejects wrong-control, timed
allocation and fabricated-gain records. Historical artifacts are unchanged.

Stop this specialization here. The current
[qualified stage profile](2026-10-25-maxwell-split-stage-profile.md)
also measures QKV at 2.522 ms / 11.72% and output projection at 1.773 ms /
8.25% of middle-token GPU work. Inspection of the current native fused Q8
split4 QKV kernel shows a publication barrier plus eight shared-memory
reduction barriers at 256 threads. An independent next investigation can
test whether preserving that exact reduction tree with warp shuffles removes
meaningful cost. Do not simply reactivate prior losing prepared QKV/layout
paths. Keep weights, activation precision, prepared down/head, actual-token
quality and full HTTP gates fixed; no new runtime choice is authorized by
this below-gate FFN result.

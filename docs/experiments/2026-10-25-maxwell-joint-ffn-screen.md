# Joint Gate/Up Packing Screen

## Decision

Lossless joint gate/up packing gives a small, repeatable improvement, but
**does not clear the unchanged 5% complete-FFN gate**. Five initial pairs
give 4.68-4.75% one-row gains. One fresh twenty-pair confirmation gives
**4.66-4.74%**, with all three upper confidence bounds below 5%.

Keep implementation `8881763` benchmark-only. Do not integrate it into
generation, lower the gate, tune more layouts/geometries, or time a full
runtime before qualification. This is a below-threshold gain, not a slowdown
or a numerical failure. The additional trials resolve a borderline result;
neither the best trial nor the best run is used for promotion.

The qualified split-f32x4-128/SGEMM runtime remains at 45.075 HTTP tok/s
versus fresh official Ollama 43.119. Both original objectives remain unmet:
53.06 and 56.055 tok/s for 1.30x that Ollama median. There is no new HTTP
speed claim or default change.

## Fixed Setup And Paired Results

Clean source `8881763d1599d51f8c8228d4a36f3a4333379649`, release
`q8_execution_bench` SHA-256
`071a85f99cf318abbff4b13e7fd8b39416118e1ea5bbf130726a88db45a86c59`.
The [pre-implementation audit](2026-10-25-maxwell-joint-ffn-audit.md)
identified separate payload/scale/address work. This experiment pairs gate/up
four-byte payloads and original FP16 scales, retaining one column, 128
threads, four FP32 inputs, original per-weight scaling/FMA and reduction.
No activation quantization or change to prepared down/vocabulary occurs.

Use the frozen M02 environment and original real layer 0/17/35 inputs:

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
  --ffn-f32-joint --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-joint-ffn-screen.jsonl
```

The confirmation uses `--trials 20` and a new output path. Both runs use
the same clean source, executable, shape, input and environment. Arms and
stage order alternate. Each latency below is the median complete-FFN GPU
time; gain is control/candidate minus one. Paired bootstrap uses 10,000
resamples with seed zero and averages the middle observations for even
trial counts.

| Run | Layer | Rows | Split control ms | Joint ms | Throughput gain | Paired 95% interval |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Initial 5 | 0 | 1 | 0.351926 | 0.335965 | +4.75% | +4.66% to +4.81% |
| Initial 5 | 17 | 1 | 0.352016 | 0.336278 | +4.68% | +4.35% to +5.07% |
| Initial 5 | 35 | 1 | 0.351907 | 0.336102 | +4.70% | +4.06% to +4.90% |
| Initial 5 | 0 | 4 | 1.503747 | 1.453626 | +3.45% | +1.86% to +4.82% |
| Initial 5 | 17 | 4 | 1.520634 | 1.455270 | +4.49% | +3.01% to +5.16% |
| Initial 5 | 35 | 4 | 1.585290 | 1.522611 | +4.12% | +1.19% to +5.72% |
| Fresh 20 | 0 | 1 | 0.351869 | 0.336086 | +4.70% | +4.60% to +4.80% |
| Fresh 20 | 17 | 1 | 0.352162 | 0.336232 | +4.74% | +4.65% to +4.86% |
| Fresh 20 | 35 | 1 | 0.351856 | 0.336182 | +4.66% | +4.59% to +4.79% |
| Fresh 20 | 0 | 4 | 1.537363 | 1.485576 | +3.49% | +2.62% to +4.82% |
| Fresh 20 | 17 | 4 | 1.530027 | 1.482688 | +3.19% | +2.60% to +5.18% |
| Fresh 20 | 35 | 4 | 1.528179 | 1.477840 | +3.41% | +2.86% to +4.29% |

Confirmation gate/up improves from about 0.2214-0.2217 ms to 0.2050-0.2051
ms, while isolated down remains about 0.1313-0.1315 ms. The smaller complete
gain includes that unchanged down cost. Four-row timing has more variation;
it is separate evidence, not a substitute for the one-row decode gate.

## Numerical, Instruction And Memory Evidence

Both runs pass every numerical/guard record. All six candidate shapes
preserve control hidden and completed-down bits. Captured gate/up also
matches ordinary SwiGLU arithmetic. Largest CPU absolute differences are
2.0644e-6 gate, 3.4088e-6 up, 1.7127e-5 SwiGLU and 5.0080e-5 completed
down, within the unchanged tolerance. Synthetic tests additionally exercise
signed extremes, zero scales, odd hidden tails, rows 1/4, invalid arguments,
source identity rejection, guard protection, scratch reuse and cleanup.

Release sm_52 SASS confirms three global loads per task instead of five:
one 64-bit paired payload load, one 32-bit paired scale load and one
128-bit activation load. FP16 conversion reads the two halves of the same
scale register. Both kernels use 32 registers, no reported stack/local
storage, 1024 dynamic shared bytes and one block-wide barrier. The control
SASS SHA-256 remains
`84a8cfe7d64c3ac38fc52666e7f509a79893422e50f683db591f7e4c4a1c69fd`.
This establishes removed issued work and unchanged control, not measured
DRAM bandwidth, occupancy or a unique causal explanation for the gain.

One joint pair holds 45,088,768 payload bytes and 2,818,048 scale bytes,
the same 47,906,816 bytes as the control pair. Both layouts are resident
for both arms: **95,813,632 device bytes and the same retained host bytes**.
Do not describe the microbenchmark as a memory-saving runtime. Original
plus prepared down/vocabulary is 4,472,455,168 bytes. With scratch, tracked
totals are 4,568,469,952 at one row and 4,569,072,064 at four rows. No KV
or quantized activation storage is allocated. Confirmation median joint
packing/upload is 34.382 / 7.253 ms, outside primary timing.

There are zero allocations, frees, copies, cuBLAS calls or stream
synchronizations inside primary enqueue intervals. Required four-row
stream waits are included. Both runs return Rust-tracked payload bytes
from zero to zero. Their final in-process GPU samples still include the
driver context; a subsequent process query is empty after each executable
exits. Tracked cleanup is not a claim of zero driver RSS during execution.

## Validation And Raw Evidence

Before timing, cuBLAS-on and cuBLAS-off each pass four primitive and twelve
benchmark tests. Warning-as-error non-CUDA tests pass 210 cases, including
joint byte packing and rejection checks. CUDA/server all-target Clippy,
formatting and 48 affected CUDA runtime/attention/forward/wrapper tests
pass. The optional external real-Qwen fixture was unconfigured and is not
counted as real-model generation evidence. The frozen real-model FFN
measurements above are not a full generation quality gate.

The initial run has 30 pairs / 180 primary intervals / 60 GPU boundary
samples with sole process 1334267. Confirmation has 120 / 720 / 240 with
sole process 1336467. Every primary boundary observes P0, 1113 MHz SM and
3004 MHz memory; all intervals are nonoverlapping. No other GPU workload,
build, tests or MemPalace embedding ran during timing. Both handles are
terminal and the GPU is empty.

- [Initial raw records](2026-10-25-maxwell-joint-ffn-screen.jsonl) and [audit](2026-10-25-maxwell-joint-ffn-screen-audit.json)
- [Fresh confirmation records](2026-10-25-maxwell-joint-ffn-confirm.jsonl) and [audit](2026-10-25-maxwell-joint-ffn-confirm-audit.json)
- [Audit program](2026-10-25-maxwell-joint-ffn-audit.jq), including actual-pair medians, the unchanged gate and combined residency
- [Compiler resources](2026-10-25-maxwell-joint-ffn-resources.txt) and [SASS](2026-10-25-maxwell-joint-ffn-sass.txt.gz)
- [Initial log](2026-10-25-maxwell-joint-ffn-screen.log.gz), [confirmation log](2026-10-25-maxwell-joint-ffn-confirm.log.gz) and [checksums](2026-10-25-maxwell-joint-ffn.sha256)

The auditor accepts both complete raw runs and rejects wrong timing control,
timed allocation and fabricated positive-gain records. Historical auditors
and artifacts are unchanged. Stop this packing experiment here; any next
optimization needs an independent instruction/profile hypothesis and the
same numerical, micro, actual-token quality and HTTP acceptance gates.

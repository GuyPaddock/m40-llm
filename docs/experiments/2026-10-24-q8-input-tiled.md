# Input-Tiled Q8 Execution Weights

## Decision

C10 completes the second independent lossless Q8 layout experiment. Retain
the vector mapping as a per-family candidate, especially for larger row
counts. It does not displace C09 split planes for single-token down/vocabulary
projection. The coalesced-lane mapping is a negative result; preserve its code
and measurements before any later removal. The twelve-configuration budget is
closed. No runtime dispatch, model cache, generation default, or KV policy changes.

Next is C10A: integrate only useful per-family execution layouts, with memory
budgets and model parity, then apply the paired HTTP E2E gate against C07's
selected split8/256 attention configuration. The 31.852 HTTP tok/s checkpoint
has not changed; neither 53.06 nor the current 55.945 tok/s target is reached.

## Format and Kernels

The C08 format orders records by `[input_block][output_tile]`. Each record
contains original FP16 scales, a 16-byte-aligned value offset, and signed
bytes ordered `[quant_index][output_column]`. Packing transposes bytes only;
it never dequantizes or requantizes. It preserves stored K tails, zeros
alignment/padded-column bytes, and leaves GGUF source weights unchanged.

Two conventional Maxwell CUDA consumers use the same storage:

- `vector`: one warp lane per input index, with 4/8/16-byte loads for adjacent
  output columns, separate FP32 accumulators, warp reduction, and a shared
  warp-partial merge. One block-wide barrier.
- `coalesced-lanes`: lanes divide output columns and input-index groups,
  reusing a scale across several input values and accumulating one FP32 sum.
  Grouped warp reduction and the same merge. One block-wide barrier.

Each mapping tests widths 4/8/16 and 128/256 threads, **twelve configurations
total**, not twelve per mapping. The first six vector cases did not beat C09
on single-token down/vocabulary. The remaining six tested whether lower
accumulator count and scale reuse helped. They did not. No further mapping,
unroll, or register-limit sweep was added.

The host packer supports all C08 descriptor widths 1/2/4/8/16/32. The CUDA
experiment deliberately accepts only 4/8/16. Mismatched layout/launch widths,
unsupported choices, invalid dimensions, undersized/aliased buffers, and
wrong layout variants fail before enqueue. Owners and buffers must remain
alive through explicit decode-stream completion.

## Method

Tesla M40 24GB, UUID `GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852`, driver
580.159.04. Qwen2.5-3B Q8_0 model SHA-256
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
Rust release, NVCC 12.8 V12.8.93, unchanged `-O3`, sm_52 SASS and compute_52
PTX, cuBLAS enabled for the measured build. Manifests retain executable hashes,
the `a48d87b` base revision, and dirty working-tree status.

Real layer-0 Q/K/V/output/gate/up/down tensors and tied vocabulary weights;
deterministic, row-distinct FP32 activations. Compare every output against the
current GGUF projection and sixteen samples against FP64 CPU sums. These
are isolated projections, not fused QKV, gate/up/SwiGLU, or lm-head argmax.

Controls use existing GGUF split4/coltile4 decode and existing multi-row
dispatch. Three warmup pairs precede five alternating serial pairs per
configuration; CUDA-event brackets repeat 10/3 calls at rows 1/4. A bounded
rows=64 Q/gate/down check repeats two calls. Setup, readback, and GPU snapshots
are outside timing. Repeated tiny matrices can fit cache, and event spans can
include host submission gaps; these are not measured DRAM-traffic results.

The initial vector binary predates the explicit access field; those records
are vector cases. Its CUDA vector consumer is unchanged in the final binary.
Later records explicitly name `vector` or `coalesced-lanes`. A small split-plane
refresh checks the shared runner still reproduces C09's important rows.

All timing runs completed before regression tests resumed. The executable
rejects other GPU compute PIDs. Across all five datasets, 1,230 pairs / 2,460
measured intervals have **zero overlapping intervals** and no missing absolute
timestamps; boundary process counts are always one. GPU samples are P0,
memory 3004 MHz, SM 949-1113 MHz, 30-39 C. These bursty microbenchmarks do not
share the sustained-clock conditions of the HTTP workload.

## Single-Token Results

Median event milliseconds per projection. K/V use the longer confirmation;
other vector rows use five pairs. All best vector cases are width16/128
except K/V, which prefer width4/256.

| Projection | K x N | GGUF ms | Best Vector ms | Ratio |
| --- | --- | ---: | ---: | ---: |
| Q | 2048 x 2048 | 0.05435 | 0.03641 | 1.493x |
| K | 2048 x 256 | 0.01109 | 0.00951 | 1.166x |
| V | 2048 x 256 | 0.01105 | 0.00948 | 1.166x |
| Attention output | 2048 x 2048 | 0.05401 | 0.03636 | 1.485x |
| Gate | 2048 x 11008 | 0.22884 | 0.14125 | 1.620x |
| Up | 2048 x 11008 | 0.22876 | 0.14087 | 1.624x |
| Down | 11008 x 2048 | 0.19103 | 0.16136 | 1.184x |
| Vocabulary | 2048 x 151936 | 2.58632 | 1.66078 | 1.557x |

The split refresh gives gate **0.13962**, down **0.14215**, and vocabulary
**1.62315 ms**, close to C09. These separate run windows are not paired directly
against each other, and the small gate/vocabulary differences are not a new
promotion claim. Down clearly favors split planes in this screen. The best
coalesced-lane rows are slower: gate 0.20833, down 0.23157, vocabulary 2.50206 ms.
Down is slower than the native GGUF control, despite lossless parity.

Nine initial vector cells and eight coalesced-lane cells exceed 5% MAD/median
variation, all K/V rows. The vector decode K/V confirmation uses ten pairs and
100 calls per bracket; worst baseline/candidate variation falls below
0.95%/1.62%. It confirms width4/256, while several width8/16 variants are
neutral or slower than GGUF. No precise K/V rows=4 or coalesced K/V gain is
claimed from the remaining noisy cells. Width16
at N=256 launches only sixteen blocks for M=1, fewer than M40's 24 SMs;
do not assume the best large-projection tile is good for narrow K/V.

## Multi-Row Results

Best vector cases, all width16/128. Five pairs each.

| Projection | Rows | GGUF ms | Vector ms | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Q | 4 | 0.52404 | 0.10202 | 5.137x |
| Gate | 4 | 2.34459 | 0.52418 | 4.473x |
| Down | 4 | 1.90988 | 0.48193 | 3.963x |
| Vocabulary | 4 | 25.07055 | 6.58309 | 3.808x |
| Q | 64 | 6.73574 | 1.24400 | 5.415x |
| Gate | 64 | 29.77133 | 7.64850 | 3.892x |
| Down | 64 | 20.42986 | 6.47952 | 3.153x |

C09's earlier rows=64 Q/down are 1.517/7.353 ms; tiled vector is a candidate
for these larger row counts, not proof of full-model prefill or verification
speed. Neither consumer reuses weights between rows or competes here against
materialized cuBLAS. Coalesced-lane rows=4 are worse than vector and were not
expanded into a rows=64 tuning sweep.

## Cost and Correctness

At aligned Qwen shapes, width4 has 5.88% payload growth from the padded scale
header; widths8/16 preserve original payload size. Padded output columns can
increase small/ragged tensor size further. Final consumers use 64-512 bytes of
dynamic shared memory, depending on width/threads.

| Kernel Mapping | Registers at Width4/8/16 | Stack / Local Bytes |
| --- | --- | --- |
| Vector | 32 / 32 / 48 | 0 / 0 |
| Coalesced lanes | 38 / 47 / 48 | 0 / 0 |

The lower-register hypothesis failed: fewer accumulators did not produce fewer
registers after compilation. SASS confirms 32/64/128-bit vector value loads
and byte loads for coalesced lanes. Read-only load selection is the compiler's
ordinary optimization, not a newly enabled `__ldg`/texture experiment. Actual
occupancy and memory-traffic counters remain unavailable under C05's stack.

First-run down/gate packing takes roughly 44-57 ms per tensor, versus about
15 ms for split planes. Vocabulary packing takes 1.12-1.22 s versus about
0.21 s for split planes. Uploads remain about 3.4-4.0 ms for FFN and 47-54 ms
for vocabulary. These are single setup observations, not load medians.
The benchmark retains original and execution device payloads: width16
vocabulary requests 661,225,472 bytes combined; width4 requests 680,673,280.
Host peak payloads, execution growth, and source bytes are explicit in each
preparation record. Model-wide caching would be extra resident storage,
not a free replacement of subranges in the original upload.

All numerical checks pass. Across the five datasets, worst max/mean difference
from GGUF is `1.23978e-5 / 1.03632e-6`; worst sampled FP64 error is `8.75443e-7`.
Host reconstruction tests preserve scale bits including signed zero/NaNs and
all signed bytes, including inactive tails and padded-column zeros. The CUDA
test compares all twelve cases to GGUF/CPU with poisoned output guards,
distinct rows, zero/negative scales, and rejected-call output preservation.
Enqueue adds no device allocation or stream synchronization.

Both cuBLAS-on/off attention/projection/MLP regressions, forward/ownership
tests, complete non-CUDA warning-as-error tests, and Clippy checks pass.
No whole-model quality or E2E throughput improvement is claimed for C10.

## Reproduction

```bash
source scripts/dev-env.sh
M40LLM_ENABLE_CUBLAS=1 cargo build --release --features cuda \
  --example q8_execution_bench
MODEL="$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
target/release/examples/q8_execution_bench --layout input-tiled \
  --tiled-access vector --model "$MODEL" --output /tmp/tiled-vector.jsonl
target/release/examples/q8_execution_bench --layout input-tiled \
  --tiled-access coalesced-lanes --model "$MODEL" --output /tmp/tiled-lanes.jsonl
target/release/examples/q8_execution_bench --layout input-tiled \
  --tiled-access vector --model "$MODEL" --output /tmp/tiled64.jsonl --rows 64 \
  --tensors blk.0.attn_q.weight,blk.0.ffn_gate.weight,blk.0.ffn_down.weight
target/release/examples/q8_execution_bench --layout input-tiled \
  --tiled-access vector --model "$MODEL" --output /tmp/tiled-kv.jsonl \
  --rows 1 --trials 10 --iterations 100 \
  --tensors blk.0.attn_k.weight,blk.0.attn_v.weight
target/release/examples/q8_execution_bench --layout separate \
  --model "$MODEL" --output /tmp/split-refresh.jsonl --rows 1 \
  --tensors blk.0.ffn_gate.weight,blk.0.ffn_down.weight,token_embd.weight
```

Use hardware access and run commands serially; output paths must not exist.
The actual first two runs visit FFN tensors before QKV, as recorded in raw rows.

Raw [vector](2026-10-24-q8-tiled-vector.jsonl.gz),
[coalesced lanes](2026-10-24-q8-tiled-lanes.jsonl.gz),
[64-row](2026-10-24-q8-tiled-prefill64.jsonl.gz),
[K/V confirmation](2026-10-24-q8-tiled-kv-confirm.jsonl.gz), and
[split refresh](2026-10-24-q8-tiled-split-refresh.jsonl.gz) include all trials.
The existing [C09 summary program](2026-10-24-q8-separate-summary.jq) derives
each corresponding `*-summary.json` file independently; do not merge different
access mappings before grouping. The [validation program](2026-10-24-q8-tiled-validation.jq)
slurps all five raw files to reproduce [serial/configuration checks](2026-10-24-q8-tiled-validation.json).
[Compiler resources](2026-10-24-q8-tiled-resources.txt),
[vector SASS](2026-10-24-q8-tiled-vector-sass.txt.gz), and
[lane SASS](2026-10-24-q8-tiled-lanes-sass.txt.gz) retain the code-generation evidence.

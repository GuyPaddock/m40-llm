# Prepared Reads Retain Most Projection Cost

## Finding

Checksum-only reads take **89.08-89.55%** of real one-row prepared down
latency and **89.36%** of vocabulary latency. These diagnostics retain the
original four-column/eight-value work mapping and shared reduction, but
remove FP32 activation loads, dequantization and multiply/accumulate work.
The remaining surrogate gaps are about **13.75-14.36 us per down layer**
and **166.40 us for vocabulary**.

These differences are not proven removable arithmetic cost. The checksum
kernel has different arithmetic and register use. Its latency is not pure
DRAM time, a rigorous inference lower bound or a measured optimization.
Together with the [gate/up read result](2026-10-26-maxwell-ffn-read-screen.md),
this argues against another arithmetic-only probe without new evidence.

No inference backend, runtime selector, quality semantics or defaults change.
The exact qualified HTTP result remains **49.146 versus 43.106 official
Ollama tok/s**. The original **53.06** and **56.037 tok/s** requirements
remain unmet. No actual-512 or HTTP run is justified by checksum timing.

## Five-Pair Release Results

CUDA-event milliseconds per call, medians of five alternating rounds with
ten repeated enqueues per interval. Setup/readbacks/CPU checks and warmup
are excluded. Checksum output is never fed to any inference operation.

| Family | Layer | Rows | Read/checksum | Real projection | Complete real FFN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Down | 0 | 1 | 0.117421 | 0.131248 | 0.351930 |
| Down | 17 | 1 | 0.117133 | 0.131488 | 0.352464 |
| Down | 35 | 1 | 0.117837 | 0.131587 | 0.352086 |
| Down | 0 | 4 | 0.437552 | 0.544781 | 1.509299 |
| Down | 17 | 4 | 0.437469 | 0.578371 | 1.576301 |
| Down | 35 | 4 | 0.436682 | 0.571072 | 1.593974 |
| Vocabulary | - | 1 | 1.396787 | 1.563184 | - |
| Vocabulary | - | 4 | 5.558633 | 6.213578 | - |

Real down uses captured SwiGLU inputs in the isolated interval. Complete
FFN uses the unchanged split-f32x4-128 gate/up path followed by the original
prepared down. Its gate/up medians are 0.2213-0.2215 ms at one row. Four-row
down checksum fractions are 75.64-80.32%; vocabulary remains 89.46%. Real
four-row FFN varies more than the one-row path and retains its cross-stream
waits. It is not a single-request result or an alternative acceptance gate.
Vocabulary inputs are distinct frozen layer-0 normalized FFN surrogates,
not captured final output-norm hidden vectors or generation logits.

## Compiled Work And Memory

Contract `6a206ec`; clean source
`375c48898cf20517c88ef50f2d7525e1a91fde8e`. Release executable SHA-256
`fab4704d43f5b91923d90a9b7b67b03b075d0cd80ce988442c7272682aa0c1df`.
NVCC 12.8.93, `-O3`, sm_52 SASS/compute_52 PTX, no fast math, cuBLAS enabled,
with the exact frozen M02 environment. The original Qwen Q8_0 model and
all 37 prepared down/vocabulary identities remain unchanged.

The diagnostic loads the same original u16 scale and packed eight-byte
payload records. Each output sums eight original u32 payload words plus
four scale-bit contributions per Q8 block modulo 2^32. It retains four
columns per CTA and the original shared tree: one-row down uses 256 threads
and nine dynamic block barriers; four-row down and vocabulary use 128
threads and eight barriers. Both use 4096/2048 dynamic shared bytes.
The resource report lists static SHARED=0, not zero dynamic shared memory.

Checksum uses **32 registers** versus control's **40**, both with zero
stack/local storage. This resource difference further prevents attributing
the time difference solely to FP32 arithmetic. The original prepared
control's **552 encoded words** remain unchanged, SHA-256
`fafef036cf36582ad7c230dd9b2502575a53356ada1396aaeac8e9baf0872397`.

Each down tensor occupies 23,953,408 bytes; vocabulary occupies 330,612,736.
Source plus the qualified prepared cache totals 4,472,455,168 bytes, with
no KV. Down cases retain the same 47,906,816-byte gate/up pair and real FFN
scratch 201,152/803,264 bytes at rows 1/4. Guarded checksum storage adds
8,256/32,832 bytes, giving tracked totals 4,520,571,392/4,521,198,080 bytes.
Vocabulary real scratch is 616,064/2,463,872 bytes; checksums add
607,808/2,431,040 bytes, giving totals 4,473,679,040/4,477,350,080.
These are requested resident payloads, not peak VRAM or memory traffic.

## Validation And Isolation

- Exact CPU checksums from original interleaved GGUF blocks pass all eight
  cases, including repeated output/guard verification after primary timing.
  Both weight-plane hashes remain unchanged. Real FFN retains existing CPU
  bounds and intermediate-capture/normal-kernel bit equality; vocabulary CPU sample error is below 3.4e-8
  and complete real-output bits remain unchanged through checksum work.
- Primitive tests pass **3/3 with cuBLAS on and 3/3 off**, covering both
  streams, producer waits, rows 1/4, threads 128/256, inactive/tail columns,
  signed/wrapping payload patterns, zero/arbitrary scale bits, repeated scratch use, invalid
  spans/alignment/partial alias/overflow and final allocation cleanup.
- Shared benchmark **21 CPU tests pass**, with **14 hardware cases ignored**.
  Existing attention grid **11**, forward **20** with **4 ignored**,
  projection **2**, MLP **1**, warning-as-error non-CUDA **211**, CUDA/server
  all-target Clippy and formatting pass. Ignored cases are not counted passed.
- The independent audit reconciles 66 records, eight shapes, **40 pairs**,
  **140 serial primary intervals** and **80 GPU boundaries**. Every measured
  boundary has P0/1113 MHz SM/3004 MHz memory and only PID **3022210**.
  Session **55938** exits zero; subsequent compute-process query is empty
  and tracked allocation accounting returns zero to zero.
- No timed enqueue allocates/frees, copies or synchronizes a stream. Real
  FFN dependencies remain included; checksum has independent output and
  requires no per-call stream wait. Event completion is outside enqueue
  counters. No other GPU workload, build, test or MemPalace call overlaps.
- Ten negative controls reject missing pairs/stages, failed checksums,
  changed preparation, timed allocations, fabricated DRAM/inference claims,
  wrong dispatch, overlapping intervals and leaked bytes.

## Reproduction And Next Hypothesis

Source `scripts/dev-env.sh` and restore the frozen environment as in the
[gate/up read reproduction](2026-10-26-maxwell-ffn-read-screen.md#reproduction-and-evidence).
Check competing processes and request hardware access before running:

```bash
cargo build --release --features cuda --example q8_execution_bench
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --prepared-weight-read --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-prepared-read-screen.jsonl
```

The output must not exist. Warmup remains outside primary intervals.

- [Contract](2026-10-26-maxwell-prepared-read-hypothesis.md), [raw records](2026-10-26-maxwell-prepared-read-screen.jsonl), [audit](2026-10-26-maxwell-prepared-read-audit.jq) and [derived medians](2026-10-26-maxwell-prepared-read-audit.json)
- [Control SASS](2026-10-26-maxwell-prepared-read-control.sass.gz), [encoded words](2026-10-26-maxwell-prepared-read-control.words.gz), [checksum SASS](2026-10-26-maxwell-prepared-read-checksum.sass.gz) and [resources](2026-10-26-maxwell-prepared-read-resources.txt.gz)
- [Run log](2026-10-26-maxwell-prepared-read-screen.log.gz), [build log](2026-10-26-maxwell-prepared-read-build.log.gz) and [checksums](2026-10-26-maxwell-prepared-read.sha256)

Source and compiled control show **four separate 64-bit payload loads and
four U16 scale loads per eight-value task**, across distant output-column
planes. The next source-backed question is whether a lossless per-warp,
cross-column packet can reduce load/address issue work while preserving
per-thread task ownership, every FP32 accumulation and the control reduction.
Inspect its sector footprint and distinguish it from the closed scalar
input-interleaved C10 layout before implementation. A two-phase packet might
allow two coalesced 128-bit payload reads and one packed scale read without
changing logical bytes. That is an unmeasured hypothesis, not a speedup.
Do not combine closed activation/warp experiments or change any gate.

One possible fixed packet is
`values[output_tile][warp_task_tile][half][lane][column][byte]`, with two
halves, 32 lanes, four columns and four bytes per column; scales are
`scales[output_tile][qblock][column]`. Each half is 512 contiguous bytes.
For aligned full warps, four existing 8-byte-per-lane reads and two proposed
16-byte-per-lane reads both cover 1024 payload bytes and 32 distinct
32-byte address buckets summed over instructions. No payload amplification
is required. For these K-multiple-of-256 shapes, four scale instructions
each cover eight u16 values in one bucket; a packed scale instruction can
cover eight 8-byte records in two buckets. These are source-address counts,
not measured cache transactions, and do not prove a performance benefit.
Keep original eight scalar activation loads, per-column FMA order, task
ownership, thread counts and shared reduction in any first candidate. This
is distinct from C10's scalar-input-interleaved consumer and does not reuse
the rejected packed-activation or warp-finish changes.

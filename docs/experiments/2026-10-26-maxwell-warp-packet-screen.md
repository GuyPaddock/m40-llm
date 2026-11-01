# Lossless Warp Packets Miss Complete-FFN Gate

## Decision

The fixed packet improves isolated one-row down throughput **7.52-7.61%**,
but complete FFN improves only **2.46-2.62%**, below the unchanged **5%**
gate at all three layers. All complete-FFN interval upper bounds remain
below 2.77%; this is not a borderline pass needing more trials. Vocabulary
improves **9.28%**, clearing its own projection gate, but its standalone
optimistic profile-weighted E2E scope is only **0.6823%**, below 3%.

Keep implementation `966ead2` benchmark-only. Do not integrate either path,
combine failed down with the vocabulary result, or rescue the packet with
thread/width/unroll/cache/register changes. No actual-512, frozen generation
quality or HTTP run is authorized by these below-gate results. Primitive
and actual-weight microbenchmark parity do not establish whole-model output
qualification.

Qualified unchanged exact status remains **49.146 HTTP tok/s versus 43.106
official Ollama**. Both requirements, **53.06** and **56.037 tok/s** for
1.30x the fresh official median, remain unmet. No runtime/default, KV,
attention, sampling or scheduler change is made.

## Fixed Setup And Results

Contract `da427a4`; clean source
`966ead2439b95d56cd59f8a20ff1ec1ba5151b98`. Release executable SHA-256
`ff037b7b54937e4d91242b12cd265d300c192e443c006667e70fb1295d91a926`.
Original Qwen2.5-3B Q8_0 model, frozen M02 fixture/report, down layers
0/17/35 and real vocabulary weights. Vocabulary uses distinct normalized
FFN layer-0 surrogates, not captured output-norm hidden vectors. No KV is
allocated. NVCC 12.8.93, `-O3`, sm_52 SASS/compute_52 PTX, no fast math,
cuBLAS enabled and frozen environment.

Five alternating pairs per case, ten repeated enqueues per stage, separate
warmup. Event timings include original dependencies. Setup, packing, CPU
inverse/reference, upload and immutable-plane readbacks are excluded.
Control gate/up stays split-f32x4-128; complete FFN includes real down.

| Layer | Rows | Complete FFN control/packet ms | Complete gain | Down control/packet ms | Down gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.351955 / 0.342982 | +2.616% | 0.131494 / 0.122298 | +7.520% |
| 17 | 1 | 0.351898 / 0.343091 | +2.567% | 0.131917 / 0.122685 | +7.525% |
| 35 | 1 | 0.351834 / 0.343376 | +2.463% | 0.131750 / 0.122435 | +7.608% |
| 0 | 4 | 1.532774 / 1.510646 | +1.465% | 0.552253 / 0.520346 | +6.132% |
| 17 | 4 | 1.507853 / 1.509501 | -0.109% | 0.548144 / 0.530086 | +3.407% |
| 35 | 4 | 1.597270 / 1.564640 | +2.085% | 0.588010 / 0.569094 | +3.324% |

One-row complete-FFN paired 95% gain intervals are **[2.083%,2.650%]**,
**[2.166%,2.761%]**, **[2.074%,2.707%]**. Four-row intervals for layers
17/35 cross zero. Four rows are genuine distinct inputs, not a single-token
decode gate or evidence about server batching. Isolated-down generic
`passes_micro_screen` fields do not override the complete-FFN requirement.

| Vocabulary rows | Control ms | Packet ms | Gain | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.562826 | 1.430115 | +9.280% | [9.238%,9.349%] |
| 4 | 6.212461 | 5.691552 | +9.152% | [9.135%,9.172%] |

Only vocabulary independently qualifies. Its **7.9809%** middle-stage share
comes from the [current qualified profile](2026-10-26-maxwell-packed-stage-profile.md)
and its [audit](2026-10-26-maxwell-packed-stage-audit.json), step 255.
`1 / (1 - share * gain / (1 + gain)) - 1` gives **0.682338%** optimistic
standalone throughput scope. This is not measured E2E performance, and the
failed down path cannot be pooled in to bypass its own gate.

## Layout, Compiled Work And Memory

The versioned semantic key includes original model/tensor/offset/dtype/shape.
Payload order is `[output_tile][warp_task_tile][half][lane][column][byte]`;
scales are `[output_tile][qblock][column]`. K must be divisible by 256 and N
by four. Every scale bit and payload byte survives CPU inverse roundtrip,
without padding or quantization. This differs from C10's scalar-input
interleaving and does not combine its consumer or either rejected
packed-activation/warp-finish change.

The consumer retains original eight scalar FP32 activation loads, eight-value
thread tasks, per-column scale/FMA order, column guards, shared reduction
and launch geometry. Compiled weight reads change from four LDG64 plus four
U16 loads to **two LDG128 plus one LDG64 scale record**. The intended load
change is realized. Both control/packet use **40 registers, zero stack/local
storage**, and original **4096/2048 dynamic shared bytes**. Static SHARED=0
in resources does not describe dynamic capacity. There remain two static
barrier sites, nine/eight dynamic barriers at 256/128 threads.
Control's 552 encoded words remain unchanged, SHA-256
`fafef036cf36582ad7c230dd9b2502575a53356ada1396aaeac8e9baf0872397`.
No measured bandwidth, occupancy or cache-hit claim follows.

Original source plus the qualified 37-tensor down/vocabulary cache remains
**4,472,455,168 bytes**. Both timing arms retain the same original and
candidate allocations. Each down packet adds **23,953,408 bytes** beside
the shared **47,906,816-byte gate/up pair**; their combined 71,860,224 bytes
fit the 128 MiB case budget. The vocabulary packet adds **330,612,736 bytes**
within its 512 MiB budget. Payload is the same size as each original source;
coexistence, not padding, causes additional residency.

Down scratch is 201,152/803,264 bytes at rows 1/4, giving tracked case totals
**4,544,516,544/4,545,118,656 bytes**. Vocabulary scratch is
616,064/2,463,872, giving totals **4,803,683,968/4,805,531,776 bytes**.
Candidate host planes retain the same byte counts as device packets;
CPU inverse validation temporarily needs another source-sized vector.
Packing plus roundtrip takes about **66-72 ms** per down and **1053-1079 ms**
per vocabulary case; allocation/upload about **3.7-4.2/46.4 ms** respectively.
These are setup costs and requested payloads, not process peak VRAM/RSS.

## Validation And Isolation

- Host packing tests **3/3** cover bit-exact inverse, source identity, byte
  counts, unsupported shapes and failure after allocating either plane.
  The latter releases current and prior owners; success retains both.
- CUDA primitive tests **2/2 with cuBLAS on and 2/2 off** match original
  control bits and CPU bounds for nonbinary inputs, signed/zero weights,
  four-byte input alignment, long/short loops, rows 1/4, threads 128/256,
  both producer/consumer stream directions, guards/reuse and invalid spans.
- Actual-weight down cases preserve complete output, isolated down and
  gate/up bits; original CPU tolerances pass. Vocabulary preserves all
  control output bits, with sampled CPU error below 3.4e-8. Candidate and
  original planes/inputs and cache identities remain unchanged after timing.
- Shared benchmark **22 CPU tests** pass, with **14 hardware cases ignored**.
  Attention grid **11**, forward **20** with **4 ignored**, projection **2**,
  MLP **1**, non-CUDA warning-as-error **214**, CUDA/server all-target Clippy
  and formatting pass. Ignored tests are not counted as passes. The test-data
  precedence lint was corrected before source commit and release timing.
- Audit reconciles 68 records, eight cases, **40 pairs/200 serial intervals**,
  raw medians, reported paired bounds, counters/memory/dispatch and gates.
  All **80 GPU boundaries** show P0/1113 MHz SM/3004 MHz memory with only PID
  **3074139**. Session **17271** exits zero; tracked bytes return zero to
  zero and a post-exit GPU process query is empty. No other GPU workload,
  build, test or MemPalace call overlaps timing.
- Ten corruptions are rejected, including a false complete-FFN promotion,
  missing pairs, changed cache, failed roundtrip/bits, incorrect packet
  accounting, timed allocation, wrong dispatch, overlapping intervals and
  leaked bytes. No timed enqueue allocates/frees, copies or synchronizes;
  original stream waits and completion boundaries remain explicit.

## Reproduction And Artifacts

Source `scripts/dev-env.sh` and restore the frozen environment from the
[read diagnostic reproduction](2026-10-26-maxwell-ffn-read-screen.md#reproduction-and-evidence).
Request hardware access and check competing processes first:

```bash
cargo build --release --features cuda --example q8_execution_bench
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --prepared-warp-packet --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-warp-packet-screen.jsonl
```

The output must not exist; warmup is separate from primary pairs.

- [Contract](2026-10-26-maxwell-warp-packet-hypothesis.md), [raw report](2026-10-26-maxwell-warp-packet-screen.jsonl), [audit](2026-10-26-maxwell-warp-packet-audit.jq) and [derived gates](2026-10-26-maxwell-warp-packet-audit.json)
- [Control SASS](2026-10-26-maxwell-warp-packet-control.sass.gz), [words](2026-10-26-maxwell-warp-packet-control.words.gz), [packet SASS](2026-10-26-maxwell-warp-packet-candidate.sass.gz) and [resources](2026-10-26-maxwell-warp-packet-resources.txt.gz)
- [Run log](2026-10-26-maxwell-warp-packet-screen.log.gz), [build log](2026-10-26-maxwell-warp-packet-build.log.gz) and [checksums](2026-10-26-maxwell-warp-packet.sha256)

Close this fixed packet configuration without tuning rescue or combining
closed probes. Any next exact candidate needs independent evidence of enough
scope in the dominant FFN work, its own bounded contract and all unchanged
promotion gates. This result does not relax the original throughput goals.

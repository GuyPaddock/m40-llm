# Fixed Column Transpose Regresses Read Cost

## Decision

Close the [fixed visitation experiment](2026-10-26-maxwell-column-order-hypothesis.md).
Transpose32 regresses one-row read throughput **3.33-3.36% versus the
same-kernel sequential control** and **2.44-2.50% versus the original
reader**. All paired intervals are negative. Four-row regressions are
3.55-3.56% and 2.53-2.59%, respectively. Every unchanged read gate fails.

Keep `034be32` diagnostic-only, without an inference consumer or another
permutation, tile-size, layout, geometry, cache or register rescue. This
does not establish a general claim about block ordering, memory partitions,
cache misses or DRAM bandwidth. It only rejects this fixed mapping under
the recorded conditions. Checksum outputs are never passed to down projection.

Qualified exact status stays **49.146 HTTP tok/s** versus **43.106 official
Ollama**, not remeasured here. Both original requirements, **53.06** and
**1.30x fresh Ollama**, currently **56.037 HTTP tok/s**, remain unmet.
Prepared down/vocabulary, SGEMM prefix, attention and runtime/KV defaults
are unchanged. No actual512/quality/HTTP candidate evaluation is authorized.

## Five-Pair GPU Results

Five alternating pairs per frozen M02 shape use ten repeated enqueues per
stage. Values are median CUDA-event milliseconds. Gain is control/candidate
latency minus one. Paired intervals use 10,000 bootstrap resamples, seed zero.

| Layer | Rows | Original reader ms | New sequential ms | Transpose32 ms | Gain vs sequential | Gain vs original |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.208006 | 0.206237 | 0.213338 | -3.33% | -2.50% |
| 17 | 1 | 0.207994 | 0.206154 | 0.213322 | -3.36% | -2.50% |
| 35 | 1 | 0.208058 | 0.206141 | 0.213258 | -3.34% | -2.44% |
| 0 | 4 | 0.816211 | 0.807850 | 0.837597 | -3.55% | -2.55% |
| 17 | 4 | 0.815936 | 0.807904 | 0.837606 | -3.55% | -2.59% |
| 35 | 4 | 0.816470 | 0.807840 | 0.837658 | -3.56% | -2.53% |

One-row paired 95% gain intervals versus same-kernel sequential are
**[-4.03%,-3.24%]**, **[-3.44%,-3.22%]** and **[-3.40%,-3.21%]**.
Versus original reader they are **[-2.81%,-2.41%]**,
**[-2.84%,-1.99%]** and **[-2.72%,-1.96%]**. The slightly faster new
sequential reader is not a new inference optimization or reason to bypass
the >=5% gate. Both controls were required before observing any data.

Real one-row gate/up remains about **0.2212-0.2214 ms**, unchanged isolated
down **0.1314-0.1317 ms**, and complete real FFN **0.3521-0.3523 ms**.
Those are controls, not a completed candidate FFN. Distinct real input rows
remain in the FFN; checksum stages reread weights per row without activations.
Read costs are neither pure DRAM timing nor rigorous inference bounds.

## Correctness And Compiled Work

Clean source `034be328bbfbb766c58ee82e1128a3c2cf545218`, release executable
SHA-256 `74331e5dfa4fe5572dac27a99b80acb0cca4e2f0ca83ede2a0b99e156a365363`.
NVCC 12.8.93 builds `-O3`, sm_52 SASS/compute_52 PTX without fast math.

Both new arms use one kernel with a dynamic selector. CTA b owns column b
in sequential mode, or `(b % 32) * (N/32) + b/32` in transpose mode.
N=11008 gives width 344. Every row launches N CTAs of 128 threads with
1024 dynamic shared bytes. Original split planes, pitch, load widths,
per-thread tasks and reduction remain fixed. No weight copy or padding.

Every final checksum includes `blockIdx.x` as a modular tag. CPU validation
uses the independently derived inverse owner `(column % width)*32 + column/width`
to remove the tag before matching original GGUF sums. Both new modes pass
all real source checks, including repeated scratch reuse, guards and immutable
planes. Nontrivial GPU tests distinguish an ignored selector. This verifies
CUDA grid-index ownership, not physical SM affinity or chronological block
execution; CUDA does not promise that order.

Original and new readers both report **32 registers**, zero stack/local
storage, the same 1024 dynamic shared bytes and one block barrier. New SASS
contains the selector/address mapping, u32 payload and u16 scale loads, and
tagged stores. Unchanged encoded control hashes are:

- Original reader, 368 words: `0f50ce0eb3d2125737a0246294e95bb282cbe19607d92b07889a5e8b8e7b5388`.
- Qualified gate/up, 560 words: `31cd38f6ca0d4a0bebd03206f484bf8045834d8af592a645cdb168cfaf02e321`.
- Prepared down/head, 552 words: `fafef036cf36582ad7c230dd9b2502575a53356ada1396aaeac8e9baf0872397`.

## Ownership And Validation

Source plus 37 qualified prepared down/head identities remains
**4,472,455,168 bytes**, with the same **47,906,816-byte** gate/up pair in
all arms. Existing real scratch is **201,152 / 803,264 bytes**, checksum
storage **88,128 / 352,320 bytes**, for rows1/4. Tracked totals remain
**4,520,651,264 / 4,521,517,568 bytes**, ending zero to zero. No KV, new
weight storage, timed allocation/free, H2D/D2H copy, cuBLAS or internal
stream synchronization is introduced by the new reader.

CUDA checksum tests pass **7/7 with cuBLAS on and off**, including tagged
wrapping sums, N=32/64/96/128, K=32/160/2048/4096, rows1/4, async producers,
guards, unchanged planes, scratch reuse and cleanup. Unsupported N, invalid
dimensions/alignment, partial aliases and address overflow fail before GPU
work. CPU inverse coverage also tests actual N=11008. Shared benchmark CPU
tests pass **33**; **15 hardware tests ignored** are not counted as passes.
Attention **11**, forward **20**, projection **2**, MLP **1** and non-CUDA
warning-as-error **217** pass; four optional forward tests remain ignored.
CUDA/server Clippy and formatting pass.

Session **84942**, PID **3357843**, terminates zero. Thirty pairs contain
**180 nonoverlapping primary intervals** and **60 measured boundaries**,
all P0/1113 MHz SM/3004 MHz memory and only that process. No other inference,
build, test or MemPalace work overlaps timing. GPU compute is empty afterward.
The independent audit recomputes both control comparisons and gates; twelve
corruptions are rejected, including wrong ownership tags/dispatch, fabricated
gate success, missing trials, altered controls, allocations, overlaps and leaks.

## Reproduction And Artifacts

Source `scripts/dev-env.sh`, restore the frozen M02 environment from the
[task-pair reproduction](2026-10-26-maxwell-ffn-task-pair-screen.md#reproduction-and-artifacts),
request hardware access and verify idle GPU state. Use a new output path:

```bash
cargo build --release --features cuda --example q8_execution_bench
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-weight-read --ffn-weight-read-column-order \
  --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-column-order-screen.jsonl
```

- [Raw records](2026-10-26-maxwell-column-order-screen.jsonl), [audit](2026-10-26-maxwell-column-order-audit.jq), [derived results](2026-10-26-maxwell-column-order-audit.json), [corruption controls](2026-10-26-maxwell-column-order-negative-controls.sh) and [checksums](2026-10-26-maxwell-column-order.sha256).
- [New reader SASS](2026-10-26-maxwell-column-order-candidate.sass.gz), [original reader](2026-10-26-maxwell-column-order-original.sass.gz), [gate/up](2026-10-26-maxwell-column-order-control.sass.gz), [prepared control](2026-10-26-maxwell-column-order-prepared-control.sass.gz) and [resources](2026-10-26-maxwell-column-order-resources.txt.gz).
- [Run](2026-10-26-maxwell-column-order-screen.log.gz), [build](2026-10-26-maxwell-column-order-build.log.gz), [CUDA on](2026-10-26-maxwell-column-order-on.log.gz), [off](2026-10-26-maxwell-column-order-off.log.gz), [CPU](2026-10-26-maxwell-column-order-cpu.log.gz), [regressions](2026-10-26-maxwell-column-order-regressions.log.gz), [non-CUDA](2026-10-26-maxwell-column-order-noncuda.log.gz), [Clippy](2026-10-26-maxwell-column-order-clippy.log.gz) and [negative controls](2026-10-26-maxwell-column-order-negative-controls.log.gz) logs.

Next work needs independently supported dominant execution-cost scope,
not another permutation or a relabeling of this diagnostic as a speedup.

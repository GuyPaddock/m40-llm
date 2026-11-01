# Fixed Payload Column Gap Regresses Read Cost

## Decision

Keep `2e96ce7` diagnostic-only. The fixed 128-byte gap regresses one-row
read throughput **2.25-2.34%**, with negative paired intervals at all three
layers. Four-row throughput regresses **2.32-2.35%**. Do not implement an
inference consumer or rescue this layout with more gap sizes, geometry,
register/cache changes or rejected layouts.

This closes the [prespecified address-placement question](2026-10-26-maxwell-ffn-column-gap-hypothesis.md).
It does not identify a physical cache/DRAM partition mechanism, measure
DRAM traffic or establish a rigorous inference lower bound. In particular,
the result does not imply every possible stride is slower. It provides no
justification for expanding this bounded experiment.

The qualified exact runtime stays **49.146 HTTP tok/s** versus **43.106
official Ollama**, not remeasured here. Both original goals, **53.06** and
**1.30x fresh Ollama**, currently **56.037 HTTP tok/s**, remain unmet.
Generation defaults and the compressed-KV hierarchy are unchanged.

## Paired Read Results

Both columns below execute the **same dynamic-pitch kernel**. Only payload
pointer/pitch differs; original scale planes, four-value tasks, 128-thread
ownership, modular checksums and reduction remain fixed. No activation or
inference output is computed by either arm.

Five alternating pairs per shape use ten repetitions per stage. Medians are
GPU milliseconds. Throughput gain is zero-gap/gapped latency minus one;
paired 95% bootstrap intervals use 10,000 resamples and seed zero.

| Layer | Rows | Zero-gap ms | Gap128 ms | Gain | Paired 95% gain interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.207888 | 0.212704 | -2.26% | -2.35% to -2.08% |
| 17 | 1 | 0.207824 | 0.212803 | -2.34% | -2.66% to -2.09% |
| 35 | 1 | 0.207808 | 0.212602 | -2.25% | -2.39% to -1.95% |
| 0 | 4 | 0.817075 | 0.836490 | -2.32% | -2.34% to -2.31% |
| 17 | 4 | 0.816941 | 0.836451 | -2.33% | -2.37% to -2.28% |
| 35 | 4 | 0.816909 | 0.836538 | -2.35% | -2.37% to -2.33% |

The independently measured original checksum is about **0.208 ms** at one
row, close to the zero-gap pitched control. Real qualified gate/up remains
**0.2212-0.2213 ms**, isolated prepared down **0.1314-0.1316 ms**, and
complete FFN **0.3519-0.3521 ms**. Original checksum retains **93.97-94.06%**
of real gate/up time. Stage differences are not removable-cost estimates;
these operations have different arithmetic and cache histories.

No complete FFN candidate, actual-512 continuation, frozen task/logit
comparison or HTTP benchmark follows a failed read-only gate. There is
no new generation speedup claim.

## Source, Compiler And Memory

Clean source `2e96ce79a7287bd07a675ce419dcf8644dd0e17d`; release executable
SHA-256 `df5b2756625d3a270539091fed416f0a77d3e54a3636ea0709b0824d55f9b72e`.
NVCC 12.8.93 uses `-O3`, sm_52 SASS and compute_52 PTX, no fast math.
Original and pitched checksum kernels both use **32 registers**, zero
stack/local storage, 128 threads, 1024 dynamic shared bytes and one
publication barrier. Payload loads remain u32 and scale loads U16.
Both measured pitched arms share the same compiled instructions.

The original checksum instruction words match its earlier archived binary,
SHA-256 `0f50ce0eb3d2125737a0246294e95bb282cbe19607d92b07889a5e8b8e7b5388`.
Qualified gate/up's 560 words remain unchanged, SHA-256
`31cd38f6ca0d4a0bebd03206f484bf8045834d8af592a645cdb168cfaf02e321`.
Prepared down/head's 552 words remain unchanged, SHA-256
`fafef036cf36582ad7c230dd9b2502575a53356ada1396aaeac8e9baf0872397`.
No achieved-occupancy, cache-hit or memory-partition attribution is inferred;
legacy CUPTI counters remain unavailable.

Each gapped payload retains **2048 original bytes + 128 sentinel bytes**
per column, stride **2176**. Original scales stay in their live typed owners.
Both arms retain the same model, original gate/up pair and additional gapped
payloads. No device memory is allocated/freed during enqueue.

| Component | Bytes |
| --- | ---: |
| Source + unchanged 37 prepared down/head keys | 4,472,455,168 |
| Original gate/up pair | 47,906,816 |
| Gapped payload pair | 47,906,816 |
| Logical bytes in gapped payloads | 45,088,768 |
| Padding in gapped payloads | 2,818,048 |
| Real scratch, rows1 / rows4 | 201,152 / 803,264 |
| Guarded checksum scratch, rows1 / rows4 | 88,128 / 352,320 |
| Tracked case total, rows1 / rows4 | 4,568,558,080 / 4,569,424,384 |

Combined per-case preparation is **95,813,632 bytes**, below 128 MiB.
Retained additional host payload is 47,906,816 bytes. Inverse checking uses
iterators, not another reconstructed payload allocation. Packing/inverse
costs **63.49-64.74 ms**, allocation/upload **6.85-7.01 ms**, excluded from
primary intervals. All tracked device bytes return zero to zero after exit.
There is no KV cache or staging allocation in this projection diagnostic.

## Validation And Isolation

CUDA tests pass **5/5 with cuBLAS on and off**. They cover original/zero-gap/
gapped exact modular CPU sums, arbitrary scale/payload bits, short/long
dynamic K, odd output widths, rows1/4, padding and output guards, immutable
planes, asynchronous producers, scratch reuse and cleanup. Invalid gaps,
K/alignment/address overflow and aliasing inside padding fail before GPU
work. CPU benchmark tests pass **25**, with 14 hardware checks ignored in
that invocation, not counted as passes.

Regression checks pass attention **11**, forward **20**, projection **2**
and MLP **1**; four optional forward checks are ignored. Warning-as-error
non-CUDA tests pass **217**. CUDA/server all-target Clippy, formatting and
diff checks pass.

Session **37932**, PID **3161242**, exits zero. Thirty paired records contain
180 serial primary intervals: original checksum, zero-gap checksum, gapped
checksum, real gate/up, real down and real complete FFN. All sixty boundary
samples show only this process, P0, 1113 MHz SM / 3004 MHz memory. No other
inference, build, test or MemPalace operation overlaps timing; GPU compute
is empty after exit. Real FFN warms for 1000 GPU ms and each checksum for
100 ms using existing finite budgets, excluded from primary measurements.

The audit checks frozen provenance, source/cache identities, padding and
scratch accounting, CPU checksums/guards, real FFN bounds, actual dispatch,
timed side effects, raw medians/gains, possible bootstrap bounds, serial
intervals, clocks and cleanup. It does not independently regenerate the
seeded bootstrap sequence. Corruption controls verify rejection of missing
work, changed pitch/identity, fake scores/checksums, timed allocations,
competing GPU work and incomplete cleanup.

## Reproduction And Artifacts

Source `scripts/dev-env.sh`, restore the environment from the
[frozen capture](2026-10-25-maxwell-ffn-capture.jsonl) as shown in the
[previous screen](2026-10-26-maxwell-ffn-task-pair-screen.md#reproduction-and-artifacts),
request hardware access and verify the GPU is idle:

```bash
cargo build --release --features cuda --example q8_execution_bench
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-weight-read --ffn-weight-read-gap --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-column-gap-screen.jsonl
```

- [Raw report](2026-10-26-maxwell-ffn-column-gap-screen.jsonl), [audit](2026-10-26-maxwell-ffn-column-gap-audit.jq), [derived costs](2026-10-26-maxwell-ffn-column-gap-audit.json) and [corruption checks](2026-10-26-maxwell-ffn-column-gap-negative-controls.sh)
- [Pitched SASS](2026-10-26-maxwell-ffn-column-gap-checksum_split_weights_pitched.sass.gz), [original checksum SASS](2026-10-26-maxwell-ffn-column-gap-checksum_split_weights.sass.gz) and [resources](2026-10-26-maxwell-ffn-column-gap-resources.txt.gz)
- [Qualified gate/up SASS](2026-10-26-maxwell-ffn-column-gap-gate_up_split_f32x4.sass.gz), [words](2026-10-26-maxwell-ffn-column-gap-gate_up_split_f32x4.words.gz), [prepared SASS](2026-10-26-maxwell-ffn-column-gap-prepared-control.sass.gz) and [words](2026-10-26-maxwell-ffn-column-gap-prepared-control.words.gz)
- [Run log](2026-10-26-maxwell-ffn-column-gap-screen.log.gz), [build log](2026-10-26-maxwell-ffn-column-gap-build.log.gz) and [checksums](2026-10-26-maxwell-ffn-column-gap.sha256)

Next seek independent evidence of enough remaining cost in dominant FFN
execution, without treating a checksum surrogate as inference or reopening
the closed layout/geometry experiments. All original acceptance gates hold.

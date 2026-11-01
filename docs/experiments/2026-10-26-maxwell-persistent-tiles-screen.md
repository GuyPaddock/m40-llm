# Reusable FFN Blocks Regress

## Decision

The [fixed CTA-reuse experiment](2026-10-26-maxwell-persistent-tiles-hypothesis.md)
preserves qualified control bits but **reduces one-row complete-FFN
throughput by 9.79-9.85%**. Four-row throughput also regresses. Every
paired interval is negative, and no shape passes the unchanged 5% gate.
Keep implementation `2c8f753` benchmark-only. Do not search block caps,
thread counts, unrolling or register limits to rescue this mapping.

| Layer | Rows | Control complete ms | Reused-block complete ms | Throughput change | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 1 | 0.351830 | 0.390259 | -9.847% | [-9.999%, -9.628%] |
| 17 | 1 | 0.352339 | 0.390557 | -9.785% | [-9.931%, -9.450%] |
| 35 | 1 | 0.351866 | 0.390278 | -9.842% | [-10.224%, -9.708%] |
| 0 | 4 | 1.595046 | 1.689760 | -5.605% | [-9.055%, -5.605%] |
| 17 | 4 | 1.538870 | 1.644451 | -6.420% | [-8.951%, -3.900%] |
| 35 | 4 | 1.538800 | 1.665002 | -7.580% | [-8.817%, -6.059%] |

Five alternating pairs per shape, ten iterations per stage, follow bounded
warmup. These are CUDA-event complete-FFN measurements, not generated-token
or HTTP throughput. The loss is not borderline and needs no confirmation
extension. No runtime selector, default change or new E2E claim follows.

## What Changed

The control launches 11008 blocks per input row, one per output. The fixed
candidate launches 192 blocks per row and reuses each block across output
columns `blockIdx.x + iteration * gridDim.x`. At this hidden width, 64
workers process 58 columns and 128 workers process 57. Kernel launch
counts are unchanged; fewer blocks do not mean fewer kernel launches.

Each column still uses all 128 physical threads, the same four-value task
stride, two FP32 accumulators, split scale/value planes, FMA order, shared
publication and original reduction tree. Accumulators reset each column.
An additional block-local barrier protects shared partials before reuse.
SwiGLU and prepared down/vocabulary are unchanged. There is no global
queue, polling, atomic progress counter, device-wide barrier or decode
worker. This is original project code, not adapted upstream code.

Gate/up/SwiGLU alone grows from **0.2217-0.2218 ms** to **0.2590-0.2603
ms** at one row. Unchanged isolated down stays near **0.1311-0.1316 ms**.
Four-row isolated down varies more, and the raw pairs are retained rather
than attributing that variation to the candidate. No hardware-counter
measurement establishes how much of the loss comes from the loop, extra
barrier, registers or reduced block parallelism.

## Compiled And Numerical Evidence

The selected sm_52 entry uses **48 registers**, versus **32** for the
control, with zero reported stack/local bytes. Both use 1024 dynamic shared
bytes; the resource dump's zero static shared bytes does not mean no shared
allocation. Candidate SASS contains two `BAR.SYNC` and ten `SHFL.DOWN`
sites. These are static instruction counts, not measured dynamic costs.

The complete control SASS is unchanged, SHA-256
`7c2d3c8e46c37367a7ce14d14d865ea5a6476b4ab5baf4e2578b45adfa182b22`.
Candidate SASS SHA-256 is
`fb4b8243342fbe83c33a01a78c8507b36a1c43a45b1e3e0112dd56f6fdf3ce96`.
The earlier SwiGLU audit found hardware exponential/reciprocal arithmetic
on the normal path and cold exceptional-value handling in the long tail.
No approximate exponential or fast-math change was justified or tested.

All six real-input shapes preserve hidden/SwiGLU and complete-down control
bits, and intermediate capture matches normal enqueue. Maximum absolute
errors versus scalar FP64 references are 2.064e-6 gate, 3.409e-6 up,
1.713e-5 SwiGLU and 5.008e-5 complete down. These are reference errors,
not drift from the bit-identical GPU control. Behavioral CUDA tests also
check gate/up bits, signed extremes, zero scales, distinct rows, guards,
immutable planes, unsupported inputs and widths 191/192/193/577 across
the fixed cap and multiple worker iterations.

## Identity, Isolation And Memory

Clean source `2c8f753449495ba446eb0ae3036ad5410fc94069`, release benchmark
SHA-256 `0fba3c0644cf356f9c738fb1ea80c845411c0560f94cabf71c9a8d60d929f3b1`.
Frozen M02 model, activation fixture and capture identities are unchanged.
Inputs use actual Qwen layer 0/17/35 gate/up and down weights, with genuine
distinct row counts 1 and 4. CUDA uses `-O3`, sm_52/compute_52, no fast
math and toolkit 12.8.93, with the captured environment.

Both arms retain 37 qualified down/vocabulary owners, 1,192,935,424 prepared
bytes plus 3,279,519,744 source bytes. Each shape shares one 47,906,816-byte
prepared gate/up pair. Scratch is 201,152 bytes at one row and 803,264 at
four. Tracked totals are 4,520,563,136 and 4,521,165,248 bytes. No additional
candidate weight payload, scratch or KV allocation is introduced. Every
timed enqueue has zero allocation/free, H2D/D2H, cuBLAS calls or stream
synchronizations. Existing four-row stream waits remain included.

Thirty primary pairs contain 180 strictly serial stage intervals. All sixty
GPU boundary samples show P0, 1113 MHz SM and 3004 MHz memory, with only
benchmark PID 2675046. No other inference, tests, builds or MemPalace work
overlaps timing. Session 42319 exits zero; tracked payload returns zero to
zero, and the post-process GPU query is empty. Driver context memory in
the in-process completion record is not a tracked payload leak.

## Validation And Reproduction

Two FFN parity/rejection tests pass with cuBLAS enabled and disabled. The
enabled attention grid, projection, MLP and forward regressions pass; four
optional forward tests remain explicitly ignored, not passes. The new
hardware adapter test passes complete bits, scratch/cache reuse and actual
dispatch. Benchmark CPU tests pass 12 checks, with 11 hardware checks
ignored in that run; the new adapter is run explicitly afterward. The
warning-as-error non-CUDA suite passes 211 tests. CUDA/server all-target
Clippy, release build, formatting and diff checks pass.

The independent audit recomputes medians and gains from actual stage pairs
and checks coverage, frozen identities, memory, dispatch, bits and serial
GPU boundaries. It records the seeded benchmark's paired-bootstrap
intervals rather than claiming to reproduce its random stream. In every
shape even the fastest candidate is slower than the slowest control, so
every possible paired resample loses. Six negative audit cases reject
missing pairs, changed block cap, altered timing, broken bit parity,
incorrect scratch and overlapping primary intervals.

Source `scripts/dev-env.sh`, export the frozen environment from the first
record of `2026-10-25-maxwell-ffn-capture.jsonl`, and build release with CUDA.
Use a new output path:

```bash
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-f32-persistent-tiles --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-persistent-tiles-screen.jsonl
```

Audit the recorded run:

```bash
jq -s \
  --slurpfile capture docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --slurpfile previous docs/experiments/2026-10-26-maxwell-virtual-warp-screen.jsonl \
  -f docs/experiments/2026-10-26-maxwell-persistent-tiles-audit.jq \
  docs/experiments/2026-10-26-maxwell-persistent-tiles-screen.jsonl
```

- [Raw pairs](2026-10-26-maxwell-persistent-tiles-screen.jsonl)
- [Audit](2026-10-26-maxwell-persistent-tiles-audit.jq) and [derived results](2026-10-26-maxwell-persistent-tiles-audit.json)
- [Control SASS](2026-10-26-maxwell-persistent-tiles-control.sass.gz), [candidate SASS](2026-10-26-maxwell-persistent-tiles-candidate.sass.gz), [resources](2026-10-26-maxwell-persistent-tiles-resources.txt.gz)
- [Run log](2026-10-26-maxwell-persistent-tiles-screen.log.gz), [validation](2026-10-26-maxwell-persistent-tiles-validation.log.gz), [negative audit checks](2026-10-26-maxwell-persistent-tiles-negative-audit.log.gz), [checksums](2026-10-26-maxwell-persistent-tiles-screen.sha256)

Preserve `2c8f753` before any separate removal. Qualified runtime remains
**49.146 HTTP tok/s versus 43.106 official Ollama**, below both **53.06**
and **56.037** requirements. Defaults and the exact-output requirement are
unchanged. The next candidate needs independent source/measurement evidence;
do not reopen this block-reuse mapping or combine closed FFN probes.

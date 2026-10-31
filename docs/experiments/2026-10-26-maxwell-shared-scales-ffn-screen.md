# Shared Scale Reuse Misses The Complete-FFN Gate

## Decision

Keep `78716e1` benchmark-only. Loading each original FP16 scale once per
CTA into shared FP32 storage preserves all checked control bits and improves
complete one-row FFN throughput **1.99-2.19%**. All paired lower bounds are
positive, but every upper bound is below **2.34%**, well below the unchanged
5% gate. No runtime selector or actual512/HTTP qualification follows.

The [contract](2026-10-26-maxwell-shared-scales-ffn-hypothesis.md) fixes one
128-thread/output configuration. No geometry, staging-depth, cache, register
or unroll rescue is authorized, nor combining rejected candidates. This
experiment demonstrates some scale-reuse benefit, not enough complete-work
improvement to qualify. Four-row results cannot bypass the one-row gate.
No fresh borderline confirmation is warranted by the one-row intervals.

Qualified performance remains **49.146 HTTP tok/s** versus **43.106 official
Ollama**, not remeasured here. Both original requirements remain unmet:
**53.06 HTTP tok/s** and **1.30x fresh official Ollama**, currently
**56.037 HTTP tok/s**. Runtime/KV defaults, prepared down/vocabulary,
SGEMM prefix and qualified dense attention remain unchanged.

## Timing

Five alternating pairs repeat each stage ten times. CUDA-event GPU times
below are complete FFN milliseconds, not generated-token throughput. Gain
is control/candidate latency minus one. Paired bootstrap intervals use
10,000 resamples and seed zero.

| Layer | Rows | Control ms | Shared scales ms | Gain | Paired 95% gain interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.351661 | 0.344259 | +2.15% | +1.90% to +2.34% |
| 17 | 1 | 0.352224 | 0.344669 | +2.19% | +2.05% to +2.24% |
| 35 | 1 | 0.351987 | 0.345104 | +1.99% | +0.98% to +2.21% |
| 0 | 4 | 1.569414 | 1.561210 | +0.53% | +0.50% to +4.71% |
| 17 | 4 | 1.587440 | 1.549187 | +2.47% | +1.00% to +3.82% |
| 35 | 4 | 1.590042 | 1.552019 | +2.45% | +1.07% to +4.44% |

One-row gate/up/SwiGLU falls from **0.22168-0.22181 ms** to
**0.21359-0.21409 ms**. Unchanged isolated down remains about
**0.1314-0.1318 ms**. Separate-stage medians need not sum to complete FFN.
Four-row timing varies more and is not promotion evidence.

## Arithmetic, Compiler And Memory

Clean source `78716e1a2f229abb50b79914fe036a989f049330`; release executable
SHA-256 `a076dcd7f94b6dec403031ea611d7b045887a5a6d2739f65aab4a39b3e925ba8`.
NVCC 12.8.93 uses `-O3`, sm_52 SASS/compute_52 PTX, no fast math.

Both kernels retain original split planes, float4 activations, packed u32
payload, per-weight FP32 scale/FMA, task order, 128-partial reduction and
precise SiLU. Unlike scale factoring, multiplication association is unchanged.
The candidate loads/converts 128 scales per real-shape CTA instead of 1024
source scale uses. These are source operation counts, not DRAM transactions.
Global caches may already serve repeated control loads.

| Kernel | Registers/thread | Dynamic shared bytes at K=2048 | Block barriers |
| --- | ---: | ---: | ---: |
| Qualified split control | 32 | 1024 | 1 |
| Shared-scale candidate | 32 | 1536 | 2 |

Both have zero reported stack/local storage. Candidate SASS shows u16
scale loads and FP16 conversion before publication, shared FP32 scale reads
inside accumulation, and two `BAR.SYNC` sites. No measured occupancy,
bank conflict, stall, cache-hit or DRAM attribution is available.
Encoded-word counts/hashes are:

- Original gate/up, unchanged: 560 words, `31cd38f6ca0d4a0bebd03206f484bf8045834d8af592a645cdb168cfaf02e321`.
- Prepared down/head, unchanged: 552 words, `fafef036cf36582ad7c230dd9b2502575a53356ada1396aaeac8e9baf0872397`.
- Candidate: 568 words, `5097d8ec32ccb6f400445ecb549432e0df88f120b60d1e193bc1c8a6ed5e3a98`.

All six real layer/row cases preserve gate, up, SwiGLU and complete-down
bits, including capture/ordinary parity. Independent FP64 CPU references
pass unchanged `1e-3 + 2e-5 * abs(reference)` tolerances. Maximum errors are
2.07e-6 gate, 3.41e-6 up, 1.72e-5 SwiGLU and 5.01e-5 down. This is
layer-local correctness, not final-model qualification.

Both arms retain the same 37 prepared down/vocabulary identities and source
payload, **4,472,455,168 bytes**, plus the original **47,906,816-byte**
split gate/up pair. Guarded scratch remains **201,152 / 803,264 bytes** for
rows1/4; tracked totals are **4,520,563,136 / 4,521,165,248 bytes**. No new
global storage, quantization or KV is allocated. Timed enqueues add no
allocation/free, transfer, cuBLAS call or stream synchronization. Tracked
allocation returns zero to zero.

## Validation And Isolation

Primitive CUDA parity/guards/span tests pass **2/2 with cuBLAS on and off**.
The focused complete-FFN/CLI checks pass **2/2**, including strict bits,
actual dispatch, producer ordering, scratch reuse and immutable weights.
Coverage includes K=32/96/2048/4096, odd outputs, distinct rows, all byte
values, signed extremes, zero/negative-zero scales and invalid spans.
K=4128 is rejected before GPU work rather than silently selecting control.

Benchmark CPU tests pass **34**, with **16 hardware tests ignored** in that
invocation. The new hardware test was explicitly included separately.
Attention **11**, forward **20**, projection **2** and MLP **1** pass.
Four optional graph tests remain ignored; the optional real-model forward
diagnostic had no model supplied. Non-CUDA warning-as-error tests pass
**217**; CUDA/server all-targets Clippy and formatting pass.

Session **8246**, PID **3391278**, terminates zero. Thirty pairs contain
180 nonoverlapping primary intervals and 60 measured boundaries, all with
only that process, P0, 1113 MHz SM / 3004 MHz memory. No other inference,
build, test or MemPalace work overlaps timing. GPU compute is empty afterward.
The independent report audit passes; twelve corruptions are rejected,
including wrong barrier/scale counts and falsely claimed gate bit equality.
Raw pairs/medians/gates and bootstrap bounds are independently checked;
the seeded benchmark computes the reported bootstrap interval.

## Reproduction And Artifacts

Source `scripts/dev-env.sh`, request hardware access and verify idle GPU.
Restore the exact frozen environment before the build and timing:

```bash
while IFS=$'\t' read -r key value; do export "$key=$value"; done < <(
  head -n1 docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl |
  jq -r '.execution_configuration.environment|to_entries[]|[.key,.value]|@tsv')
cargo build --release --features cuda --example q8_execution_bench
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-f32-shared-scales --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-shared-scales-ffn-screen.jsonl
```

- [Raw pairs](2026-10-26-maxwell-shared-scales-ffn-screen.jsonl), [audit](2026-10-26-maxwell-shared-scales-ffn-audit.jq), [derived results](2026-10-26-maxwell-shared-scales-ffn-audit.json), [corruption controls](2026-10-26-maxwell-shared-scales-ffn-negative-controls.sh) and [checksums](2026-10-26-maxwell-shared-scales-ffn.sha256).
- [Candidate SASS](2026-10-26-maxwell-shared-scales-ffn-candidate.sass.gz), [control](2026-10-26-maxwell-shared-scales-ffn-control.sass.gz), [prepared down](2026-10-26-maxwell-shared-scales-ffn-down.sass.gz) and [resources](2026-10-26-maxwell-shared-scales-ffn-resources.txt.gz).
- [Run](2026-10-26-maxwell-shared-scales-ffn-screen.log.gz), [build](2026-10-26-maxwell-shared-scales-ffn-build.log.gz), [non-CUDA](2026-10-26-maxwell-shared-scales-ffn-noncuda.log.gz) and [Clippy](2026-10-26-maxwell-shared-scales-ffn-clippy.log.gz) logs.

Next work needs independent evidence of sufficient execution-cost scope,
not further shared-scale tuning. The full exact-output objective stays active.

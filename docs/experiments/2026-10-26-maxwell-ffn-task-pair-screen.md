# Thread-Owned FFN Packets: No Useful Gain

## Decision

Keep `65801ff` benchmark-only. Complete one-row FFN gains are **0.23-0.34%**,
below the unchanged **5%** gate at all three layers. All one-row upper
confidence bounds are below 0.62%; layer 35's interval crosses zero. This
is not borderline evidence for qualification. Four-row intervals all cross
zero. Do not rescue this fixed representation with threads, register caps,
unroll, cache policy, different packet widths or other rejected probes.

The [prespecified experiment](2026-10-26-maxwell-ffn-task-pair-hypothesis.md)
does realize wider payload loads while preserving original FP32 task order,
scale planes and reduction. It shows that the source-level reduction in
load instructions is insufficient here, not that it reduced DRAM traffic
or that another mapping will necessarily help.

No generation selector, actual-512 quality run, HTTP comparison or default
change follows this failed gate. The qualified runtime remains **49.146 HTTP
tok/s** versus **43.106 official Ollama**. Both original goals, **53.06**
and **1.30x fresh Ollama**, currently **56.037 HTTP tok/s**, remain unmet.
Prepared down/vocabulary, SGEMM prefix and dense exact control stay intact.

## Complete FFN Results

Five alternating pairs per shape use ten repetitions per gate/up/SwiGLU,
isolated down and complete FFN stage. Values are medians in milliseconds;
gain is control/candidate latency minus one. Paired 95% bootstrap intervals
use 10,000 resamples and seed zero. These are GPU stage measurements, not
generated-token or HTTP throughput.

| Layer | Rows | Split control ms | Task-pair ms | Gain | Paired 95% gain interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.351654 | 0.350595 | 0.30% | 0.18% to 0.37% |
| 17 | 1 | 0.351974 | 0.350794 | 0.34% | 0.23% to 0.54% |
| 35 | 1 | 0.351968 | 0.351146 | 0.23% | -0.13% to 0.62% |
| 0 | 4 | 1.543533 | 1.571984 | -1.81% | -2.94% to 1.98% |
| 17 | 4 | 1.597075 | 1.594608 | 0.15% | -0.79% to 0.60% |
| 35 | 4 | 1.592314 | 1.569763 | 1.44% | -3.00% to 2.62% |

One-row gate/up alone changes from roughly 0.2216-0.2218 ms to
0.2195-0.2199 ms. Unchanged isolated prepared down is roughly
0.1310-0.1316 ms. Stage sums need not equal the complete interval because
cache histories differ. The four-row variation does not reopen the failed
one-row screen or justify promoting a kernel-only result.

## Correctness And Representation

The versioned typed pair identifies both original GGUF sources by model
SHA, tensor name, offset, dtype and shape. Three RAII-owned planes retain
the original separate FP16 scales and interleave only payload words:
`[column][bundle][tid][G0,U0,G1,U1]`. A bundle covers 256 four-value tasks.
Each thread processes its original task, then its original task plus 128,
without changing summation ownership or scale/FMA association. Dynamic K
must be divisible by 1024; positive odd output widths are supported without
padding. This is distinct from the closed adjacent-eight-value mapping.

Independent host inverse packing reconstructs every original source byte.
Synthetic CUDA tests compare gate, up, SwiGLU and ordinary/capture outputs
bit-for-bit with the original split consumer at K=1024/2048/4096, odd/tiny
outputs and distinct rows. Signed payloads, negative-zero scales, zero
blocks, guards, immutable planes, reusable scratch and explicit cross-stream
input producers pass. Bad spans/aliasing fail before GPU work or mutation.
Partial upload failures release the current and all previous plane owners.

All six actual-model cases preserve timing-control SwiGLU and completed-down
bits. Their gate/up intermediates are checked against the existing full FP64
CPU reference, not separately recorded as actual-model control-bit deltas.
Maximum CPU errors across rows/layers are gate **2.06e-6**, up **3.41e-6**,
SwiGLU **1.71e-5**, complete down **5.01e-5**. The original
`1e-3 + 2e-5 * abs(reference)` bounds and finite checks remain unchanged.
Frozen inputs, guarded scratch, original planes and prepared cache identities
are checked after measurement. These checks are not final model logits.

## Compiler And Memory Evidence

Clean source `65801ff67d30f5e2726d005220ac2feb7828bf94`; release ELF SHA-256
`60dddbbfad5f0bde3d60b6a7a3ca0c72a66d6454ace78888b9ba2c5a980e3903`.
NVCC 12.8.93 uses `-O3`, sm_52 SASS and compute_52 PTX, no fast math.
Both candidate and control have **32 registers**, zero reported stack/local
storage, 128 threads, 1024 dynamic shared bytes and one publication barrier.
The candidate emits one `LDG.E.CI.128` payload load, two activation LDG128
loads and four U16 scale loads per two-task body. No native DP4A, Tensor
Cores, FP16 accumulation or activation quantization is involved.

Original control instruction words remain byte-identical, SHA-256
`31cd38f6ca0d4a0bebd03206f484bf8045834d8af592a645cdb168cfaf02e321`.
Prepared down/head words remain byte-identical, SHA-256
`fafef036cf36582ad7c230dd9b2502575a53356ada1396aaeac8e9baf0872397`.
Static loads/registers do not prove measured bandwidth, cache hits, stalls
or achieved occupancy. Legacy CUPTI counters remain unsupported.

Both timing arms retain the same source and original 37 prepared down/head
identities, totaling **4,472,455,168 bytes** before per-case preparation.
The original gate/up pair adds **47,906,816 bytes** and the candidate adds
another **47,906,816 bytes**, combined **95,813,632**, below 128 MiB.
Candidate planes are **1,409,024 / 1,409,024 / 45,088,768 bytes**. Host
retained candidate payload and temporary inverse each use 47,906,816 bytes.
Packing plus inverse costs **134.55-140.63 ms**, upload/allocation
**7.62-8.32 ms**, all outside primary timing.

Guarded scratch stays **201,152 bytes** for one row and **803,264** for four.
Tracked device totals are **4,568,469,952 / 4,569,072,064 bytes** during
the cases and return zero to zero after final cleanup. There is no KV cache
in this projection screen. No per-enqueue allocation/free, copy, cuBLAS call
or stream synchronization is introduced; existing four-row dependency waits
remain included. Completion uses one event wait after each repeated stage.

## Validation And Isolation

Host tests pass 3/3; benchmark CPU tests pass 23, with 14 hardware checks
ignored in that invocation, not counted as passes. New CUDA tests pass 2/2
with cuBLAS enabled and disabled. Regressions pass attention 11, forward 20,
projection 2 and MLP 1; four optional forward checks are ignored. Non-CUDA
warning-as-error tests pass 217. CUDA/server all-target Clippy, formatting
and diff checks pass.

Session **58874**, PID **3128051**, exits zero. Thirty pairs contain 180
serial primary intervals; all sixty boundary samples show only this process,
P0, 1113 MHz SM / 3004 MHz memory. No other inference, builds, tests or
MemPalace work overlaps timing. GPU compute is empty after exit. Warmup
reaches 1000 ms control / 100 ms candidate under the original finite budget
and is excluded from the paired results.

The audit reconciles frozen provenance, immutable cache, source identities,
plane/scratch bytes, numerical checks, dispatch/counters, raw medians, gains,
possible bootstrap bounds, coverage, clocks and cleanup. It does not
independently regenerate the seeded bootstrap sequence. Ten corruption
controls must reject incomplete work, wrong dispatch, altered bits/cache,
timed allocation, false summaries, bad identities, cleanup failure and a
competing process.

## Reproduction And Artifacts

Source `scripts/dev-env.sh` and restore the environment from the first
[frozen capture record](2026-10-25-maxwell-ffn-capture.jsonl):

```bash
while IFS=$'\t' read -r key value; do export "$key=$value"; done < <(
  head -n 1 docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl |
  jq -r '.execution_configuration.environment|to_entries[]|[.key,.value]|@tsv')
cargo build --release --features cuda --example q8_execution_bench
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-f32-task-pairs --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-task-pair-screen.jsonl
```

Request hardware access, verify an idle GPU and use a new output path.

- [Raw pairs](2026-10-26-maxwell-ffn-task-pair-screen.jsonl), [audit](2026-10-26-maxwell-ffn-task-pair-audit.jq), [derived results](2026-10-26-maxwell-ffn-task-pair-audit.json) and [negative controls](2026-10-26-maxwell-ffn-task-pair-negative-controls.sh)
- [Candidate SASS](2026-10-26-maxwell-ffn-task-pair-candidate.sass.gz), [control SASS](2026-10-26-maxwell-ffn-task-pair-control.sass.gz), [resources](2026-10-26-maxwell-ffn-task-pair-resources.txt.gz) and [control words](2026-10-26-maxwell-ffn-task-pair-control.words.gz)
- [Prepared control SASS](2026-10-26-maxwell-ffn-task-pair-prepared-control.sass.gz) and [words](2026-10-26-maxwell-ffn-task-pair-prepared-control.words.gz)
- [Run log](2026-10-26-maxwell-ffn-task-pair-screen.log.gz), [build log](2026-10-26-maxwell-ffn-task-pair-build.log.gz) and [checksums](2026-10-26-maxwell-ffn-task-pair.sha256)

Next work requires an independent hypothesis with enough measured scope in
dominant FFN execution, not more tuning of this packet or a runtime claim
from its instruction counts. All exact-output goals remain intact.

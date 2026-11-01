# Gate/Up Reads Remain Close To Real FFN Cost

## Finding

The fixed checksum-only read takes **93.94-94.00%** of the real one-row
gate/up/SwiGLU time. Removing activation loads, per-weight conversion,
FP32 multiplication/accumulation and SiLU leaves only a **13.27-13.42 us
surrogate gap per layer**. This does not establish pure memory time or a
rigorous lower bound on inference. It argues against expecting a large win
from another arithmetic-only change on the same weight-access mapping.

There is no inference candidate, complete-candidate FFN, token-quality
qualification or HTTP measurement in this experiment. The qualified exact
configuration stays at **49.146 HTTP tok/s versus 43.106 official Ollama**.
The **53.06** and **56.037 tok/s** requirements remain unmet. Runtime defaults
and the preferred compressed-KV hierarchy do not change.

## Five-Pair Release Results

CUDA-event milliseconds per call, medians of five alternating rounds with
ten repeated enqueues per interval. All four stages run separately; checksum
output is never passed to down projection.

| Layer | Rows | Read/checksum | Real gate/up/SiLU | Isolated down | Complete real FFN |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.207923 | 0.221344 | 0.131142 | 0.352157 |
| 17 | 1 | 0.208134 | 0.221462 | 0.131434 | 0.352445 |
| 35 | 1 | 0.208006 | 0.221277 | 0.131373 | 0.352144 |
| 0 | 4 | 0.815904 | 0.955040 | 0.565434 | 1.540499 |
| 17 | 4 | 0.815667 | 0.965792 | 0.564157 | 1.529549 |
| 35 | 4 | 0.816080 | 0.955885 | 0.558003 | 1.511594 |

At four rows, checksum takes 84.46-85.43% of real gate/up time. This is not
the one-request gate, batched server work or four requests sharing weights.
The benchmark has genuine distinct M02 activation rows for the real FFN;
checksum has no activations and deliberately rereads weights per row.

Both paths read the same actual gate/up pair, 47,906,816 unique source bytes.
Payload/time is approximately 229-235 logical GB/s for checksum intervals.
That is **not measured DRAM bandwidth**: scales repeat across tasks, caches
may serve reads, and stores/checksum arithmetic are included. No occupancy,
cache-hit or DRAM counter attribution is available. Do not subtract the
times and call the difference removable inference cost or an E2E prediction.

## Source, Compiled Work And Memory

Prespecified in `4b72521`; implementation/source
`1001976189a7c67e6ca9e82d630c52a5845609af`, clean release executable SHA-256
`8674b75c59725f93a5301c41c7828cf52f033822f06d6f11f13e9027b878b6d5`.
NVCC 12.8.93, `-O3`, sm_52 SASS plus compute_52 PTX, no fast math, cuBLAS
enabled. Restore the frozen M02 environment, not a different geometry.

The diagnostic reads original u32 gate/up payload words and original u16
scale bits with four-value tasks, 128 threads per output column/row. Every
word plus its scale contributes modulo 2^32; scale bits contribute eight
times per Q8 block. Separate gate/up checksums retain every plane bit and
are verified against an independent parser of original interleaved GGUF
blocks. Compiler output contains packed payload and U16 scale loads,
integer additions, shared partials, one block barrier, warp shuffles and
checksum stores. There is no activation load or FP32 inference arithmetic.

Both kernels report **32 registers, zero stack/local storage**, and request
**1024 dynamic shared bytes** per CTA. The resource report's static SHARED=0
does not mean zero dynamic shared memory. Qualified control's 560 encoded
words and prepared down/vocabulary's 552 words match the prior experiment:
`31cd38f6ca0d4a0bebd03206f484bf8045834d8af592a645cdb168cfaf02e321`
and `fafef036cf36582ad7c230dd9b2502575a53356ada1396aaeac8e9baf0872397`.

Source plus qualified 37-tensor down/vocabulary cache remains 4,472,455,168
bytes. Both paths retain the same prepared gate/up pair. Existing real FFN
scratch is 201,152/803,264 bytes at rows 1/4. Additional guarded checksum
storage is 88,128/352,320 bytes, including 64 guard bytes. Tracked case totals
are 4,520,651,264/4,521,517,568 bytes. These are resident requested payloads,
not peak VRAM or traffic. Model/KV semantics are unchanged and no KV is used.

## Validation And Isolation

- Exact source-derived CPU checksums, repeated output/guard checks and
  immutable plane/input verification pass for all six actual-model cases.
  Existing real FFN CPU tolerances and capture-vs-normal checks also pass.
- New primitive tests pass **2/2 with cuBLAS on and 2/2 off**. They cover
  wraparound, signed payload patterns, arbitrary/zero scale bits, short/long
  loops, odd widths, rows 1/4, ordered prefill producers, scratch reuse,
  invalid dimensions/alignment, partial overlap, address overflow and cleanup.
- Shared benchmark tests pass **20 CPU cases**, with **14 hardware cases
  ignored**, not counted as passed. Attention grid **11**, forward **20**
  with **4 ignored**, projection **2**, MLP **1**, non-CUDA warning-as-error
  **211**, CUDA/server all-target Clippy and formatting pass.
- The independent audit reconciles 56 records, six shapes, 30 pairs and
  **120 serial primary intervals**. All 60 measured GPU boundaries show
  P0/1113 MHz SM/3004 MHz memory and only benchmark PID **2985464**. Session
  **6880** exits zero. Subsequent `nvidia-smi` shows no compute process;
  tracked allocation accounting returns zero to zero.
- Checksum enqueue issues ten launches per interval and no allocation/free,
  transfer, cuBLAS call, stream wait or stream synchronization. Real FFN
  records its original dependencies, including four-row stream waits.
  Completion uses events outside enqueue counters. No other GPU workload,
  build, test or MemPalace activity overlaps timing.
- The audit rejects nine corruptions: missing pairs, failed CPU sums,
  allocations in enqueue, changed prepared cache, missing stages, fabricated
  DRAM claims, wrong dispatch, overlapping intervals and leaked bytes.

The initial implementation used an unavailable CUDA macro, and adding keys
to the pre-existing large JSON macro hit its recursion limit. Both were
fixed before source commit and before this release run. No failed timing
run was pooled; no recursion limit was raised.

## Reproduction And Evidence

Source `scripts/dev-env.sh`; restore the exact captured environment:

```bash
while IFS=$'\t' read -r key value; do
  export "$key=$value"
done < <(head -n1 docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl |
  jq -r '.execution_configuration.environment|to_entries[]|[.key,.value]|@tsv')
cargo build --release --features cuda --example q8_execution_bench
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-weight-read --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-ffn-read-screen.jsonl
```

Request hardware access and check competing processes first. The output
must not exist; warmup is separate from primary intervals.

- [Contract](2026-10-26-maxwell-ffn-read-hypothesis.md), [raw report](2026-10-26-maxwell-ffn-read-screen.jsonl), [audit](2026-10-26-maxwell-ffn-read-audit.jq) and [derived results](2026-10-26-maxwell-ffn-read-audit.json)
- [Checksum SASS](2026-10-26-maxwell-ffn-read-candidate.sass.gz), [control SASS](2026-10-26-maxwell-ffn-read-control.sass.gz), [control words](2026-10-26-maxwell-ffn-read-control.words.gz) and [resources](2026-10-26-maxwell-ffn-read-resources.txt.gz)
- [Prepared control SASS](2026-10-26-maxwell-ffn-read-prepared-control.sass.gz) and [words](2026-10-26-maxwell-ffn-read-prepared-control.words.gz)
- [Run log](2026-10-26-maxwell-ffn-read-screen.log.gz), [build log](2026-10-26-maxwell-ffn-read-build.log.gz) and [checksums](2026-10-26-maxwell-ffn-read.sha256)

Next audit independent weight-access/work-ownership costs, including the
qualified prepared down/vocabulary mapping, before proposing another exact
candidate. Do not rescue the closed FFN arithmetic/geometry probes or promote
checksum as inference. No result here authorizes changing any existing gate.

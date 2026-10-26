# Physical FFN Cohorts Do Not Qualify

## Decision

Keep `ac505bf` benchmark-only. Two independent physical 128-thread output
groups in one 256-thread CTA preserve control bits, but complete one-row
FFN throughput regresses **0.49-0.52%** at layers 0/17/35, with all paired
intervals negative. Four-row results also miss the fixed 5% gate. Do not
vary cohort count, block size, publication layout, register caps or combine
this mapping with closed probes. No runtime/quality/HTTP integration follows.

The experiment tests [one prespecified mapping](2026-10-26-maxwell-ffn-cohorts-hypothesis.md),
not per-thread multi-column accumulation, virtual partial registers or a
looping worker grid. It halves CTA count while preserving two accumulators
per thread and the original FP32 sum tree. That does not establish lower
admission cost or better effective occupancy. Registers rise 32 to 40;
the timing does not attribute the regression to that observation alone.

Qualified runtime stays **49.146 HTTP tok/s** versus **43.106 official
Ollama**, not remeasured here. Both original goals remain unmet: **53.06**
and **1.30x fresh official Ollama**, currently **56.037 HTTP tok/s**.
Runtime defaults and the preferred compressed-KV hierarchy are unchanged.

## Complete FFN Results

Milliseconds are medians of five alternating contemporaneous pairs, ten
iterations per stage. Gain is control/candidate latency minus one. Intervals
use 10,000 paired bootstrap resamples, seed zero. The unchanged one-column
split-f32x4-128 path is control; all candidates retain prepared down/vocabulary.

| Layer | Rows | Control ms | Two-cohort ms | Gain | Paired 95% gain interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.351907 | 0.353763 | -0.52% | -0.64% to -0.38% |
| 17 | 1 | 0.352154 | 0.353872 | -0.49% | -0.89% to -0.26% |
| 35 | 1 | 0.351910 | 0.353699 | -0.51% | -0.75% to -0.38% |
| 0 | 4 | 1.583363 | 1.567101 | 1.04% | -0.79% to 2.26% |
| 17 | 4 | 1.577629 | 1.597040 | -1.22% | -2.64% to -0.10% |
| 35 | 4 | 1.542794 | 1.547638 | -0.31% | -3.30% to 0.78% |

One-row gate/up/SwiGLU is approximately 0.222-0.223 ms in both arms;
isolated down remains approximately 0.131-0.132 ms. There is no meaningful
isolated gate/up gain hidden by the unchanged down path. Stage medians need
not sum to complete timing because their cache histories differ. Four-row
isolated-down differences do not come from a changed projection implementation.

All six complete screens fail. None is borderline against 5%; more trials
are not a reason to rescue this fixed mapping. These are CUDA-event
microbenchmarks, not generated-token throughput or a measured HTTP change.

## Compiled Evidence And Ownership

Clean source `ac505bfe3ddeec4db877be1b863bfc630749ff28`; release executable
SHA-256 `c2a25cc36383fa0dc8f1f628540f68e614e59f9855ad8f8084faf14b67cb2e09`.
NVCC 12.8.93 uses `-O3`, sm_52 SASS plus compute_52 PTX, no fast math.
The same executable is inspected with cuobjdump 12.8.90 before timing.

| Property | Control | Candidate |
| --- | ---: | ---: |
| Threads per CTA | 128 | 256 |
| Physical threads per output | 128 | 128 |
| Outputs per CTA | 1 | 2 |
| CTAs per Qwen row | 11008 | 5504 |
| Dynamic shared bytes per CTA | 1024 | 2048 |
| Registers per thread | 32 | 40 |
| Stack / local bytes | 0 / 0 | 0 / 0 |
| Publication barrier sites | 1 | 1 |

Both variants retain the same four-value tasks, per-thread FMA order,
per-output reduction tree and ordinary SwiGLU expression. The inactive
cohort at odd widths publishes initialized partials and joins the barrier,
without reading or writing an invalid output column. No layout, cache,
activation quantization, approximate exponential or new precision is used.

The old control's 560 encoded instruction/control words match the archived
pre-experiment control, SHA-256
`31cd38f6ca0d4a0bebd03206f484bf8045834d8af592a645cdb168cfaf02e321`.
Disassembler text formatting differs, not encoded words. Prepared down/head
instruction words also match the qualified profile archive, SHA-256
`fafef036cf36582ad7c230dd9b2502575a53356ada1396aaeac8e9baf0872397`.
No occupancy, stall or DRAM-traffic measurements are available from those
comparisons. Legacy counter limitations remain unchanged.

The microbenchmark model retains 3,279,519,744 source bytes and
1,192,935,424 prepared down/vocabulary bytes, with all 37 identities fixed.
Each case adds one shared gate/up pair of 47,906,816 bytes to **both arms**.
This pair already exists in the fully qualified runtime; the candidate adds
no new execution-weight layout or full-model cache. Guarded scratch remains
201,152 bytes at rows1 and 803,264 at rows4. Dynamic shared memory is separate
from these heap allocations. Tracked bytes return zero to zero at exit.

## Validation And Timing Isolation

All eighteen native/control/candidate validation rows pass independent CPU
tolerances and guarded immutable-input checks. Candidate capture/normal
SwiGLU bits, captured gate/up, hidden output and completed down bits match
the timing control. Maximum candidate CPU absolute errors are 2.06e-6 gate,
3.41e-6 up, 1.71e-5 SwiGLU and 5.01e-5 completed down.

CUDA primitive tests pass 2/2 with cuBLAS on and off, covering tiny/odd/even
widths, ragged legal K, every signed byte position, zero scales, ordered
prefill-stream input producers, guards, immutable planes and early argument
rejection. The new complete-FFN adapter test passes explicitly on/off with
unchanged scratch/cache and actual cohort dispatch. The CPU benchmark suite
passes 18 tests; 13 hardware tests are ignored in that CPU invocation rather
than counted as passes. Affected enabled CUDA regressions pass attention 11,
forward 20, projection 2 and MLP 1, with four optional forward tests ignored.
Warning-as-error non-CUDA tests pass 211; CUDA/server all-target Clippy,
formatting and diff checks pass.

Thirty pairs produce 180 non-overlapping primary stage intervals. Arm and
stage order alternate. Standard bounded warmup reaches 1000 ms control /
100 ms candidate with 32 iterations per batch; it is not pooled with primary
timing. All sixty pair-boundary GPU samples show P0, 1113 MHz SM / 3004 MHz
memory, only benchmark PID 2902422. No other agent GPU workloads, builds,
tests or MemPalace operations overlap timing. Session 57538 exits zero, and
the post-process GPU query is empty.

Ten repeated stages issue ten gate/up or down launches, twenty for complete
FFN. Candidate cohort operation counts are ten or zero for isolated down.
One-row enqueue has no stream waits; four-row work retains the existing
reuse/dependency waits, ten per isolated stage and twenty complete. There
are no timed device alloc/free, copies, cuBLAS calls or stream synchronizations.
Event completion occurs once after each repeated interval, outside enqueue.

The independent audit verifies raw medians/gains, recorded confidence bounds,
source/configuration/cache identities, dispatch, geometry, numerical flags,
residency, warmup, clocks, serial intervals and cleanup. Seven negative
controls reject missing samples, timed allocation, wrong cohort geometry,
changed prepared identity, failed bit parity, false medians and a competing
GPU process. The original acceptance gates are unchanged.

## Reproduction And Artifacts

Restore the exact frozen environment using the
[reduction-screen shell setup](2026-10-26-maxwell-decode-reductions-screen.md#reproduction-and-artifacts),
source `scripts/dev-env.sh`, then build and run:

```bash
cargo build --release --features cuda --example q8_execution_bench
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-f32-cohorts --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-physical-cohorts-screen.jsonl
```

Request hardware access, use a new output file and inspect competing GPU
processes first. No new environment or runtime generation selector is added.

- [Raw report](2026-10-26-maxwell-ffn-cohorts-screen.jsonl), [audit](2026-10-26-maxwell-ffn-cohorts-audit.jq) and [derived results](2026-10-26-maxwell-ffn-cohorts-audit.json)
- [Control SASS](2026-10-26-maxwell-ffn-cohorts-control.sass.gz), [candidate SASS](2026-10-26-maxwell-ffn-cohorts-candidate.sass.gz), [resources](2026-10-26-maxwell-ffn-cohorts-resources.txt.gz) and [control words](2026-10-26-maxwell-ffn-cohorts-control.words.gz)
- [Prepared control SASS](2026-10-26-maxwell-ffn-cohorts-prepared-control.sass.gz) and [words](2026-10-26-maxwell-ffn-cohorts-prepared-control.words.gz)
- [Run log](2026-10-26-maxwell-ffn-cohorts-screen.log.gz), [release build log](2026-10-26-maxwell-ffn-cohorts-build.log.gz) and [checksums](2026-10-26-maxwell-ffn-cohorts.sha256)

Next inspect the qualified four-value task/load granularity for independent
evidence of removable work. Do not retune this cohort mapping, combine
rejected probes or substitute approximate inference for the exact goal.

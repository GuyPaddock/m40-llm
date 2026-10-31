# Rotated activation preparation result

The bounded experiment is complete. Fusing sign multiplication, Walsh128
rotation and Q8_1 encoding passes the complete-operation gate in all 24
captured-input cases. It saves 5.24 to 9.88 microseconds per row batch when
followed by the fixed 256-channel consumer. Exact canonical values, encoded
bytes and consumer output bits match the staged path throughout.

This is a positive isolated result with a missing-model prerequisite. The
selected Qwen runtime has no Hadamard rotation to remove, and its existing
Q8_1 gate/up path already shares one encoded activation. Keep runtime dispatch
unchanged. No complete-FFN, token-throughput or model-quality gain is claimed.

## Provenance and boundaries

The clean release source is `0749dcfa38f97d017d364d30d51b7547690d4e39`.
The executable SHA-256 is
`f1668d9ea3a8c1d5355836e8dcd8ca74d887f59cbf916305231071f254c74f0f`.
The run used the Tesla M40, CUDA 12.8, driver 580.159.04 and the exact retained
capture environment. The manifest records model, capture and source hashes.
Model setup took 27.236 seconds outside timing.

Inputs are the first eight distinct retained rows from layers 0, 17 and 35.
K2048 uses normalized FFN inputs; K11008 uses captured SwiGLU values. Both
paths feed the same existing 128-thread Q8_1 gate/up/SwiGLU consumer, one row
per launch. K2048 weights are the first 256 gate/up rows; K11008 uses two
disjoint 256-row slices of down weights as synthetic gate/up projections.
The same deterministic signs and normalized rotation apply to inputs and
weight fixtures. The fitted weights add quantization; these are not a
released rotated-model checkpoint or a full model architecture.

The compared complete boundary includes preparation, all launches and the
consumer. Both execute on decode with same-stream dependencies. Canonical
FP32 and Q8_1 output stores remain in both arms. Allocation, upload, reference
calculation, diagnostic reads and pre-sample synchronization are outside the
timing boundary. Raw samples record CUDA-event time, host enqueue time, host
enqueue-plus-completion time and dispatch counters.

## Timing evidence

Each arm and boundary warms for at least one GPU second. The experiment
contains 24 preparation-only and 24 complete comparisons. Each sample
contains 100 operations. Five alternating pairs form each screen; three
near-gate complete cases receive the prescribed fresh twenty-pair
confirmation. There are 300 pairs in total, with no variation extensions.
All screens and confirmations are stable; the largest throughput MAD/median
is 3.0714%. All final comparisons clear their variation-adjusted gain gate
and have positive paired confidence bounds. Bootstrap intervals use 10,000
paired resamples with seed zero.

The table gives ranges across layers 0, 17 and 35. Times cover the whole row
batch, in microseconds. Gains are throughput changes, `staged / fused - 1`.
The eight-row K11008 results use their fresh confirmations.

| Width | Rows | Staged complete, us | Fused complete, us | Throughput gain |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 1 | 17.001 to 17.081 | 11.575 to 11.643 | 46.07 to 47.57% |
| 2048 | 2 | 24.349 to 24.548 | 18.949 to 19.114 | 27.40 to 29.55% |
| 2048 | 4 | 38.670 to 38.883 | 33.052 to 33.165 | 17.00 to 17.41% |
| 2048 | 8 | 67.304 to 67.688 | 61.607 to 61.744 | 9.25 to 9.69% |
| 11008 | 1 | 47.629 to 47.971 | 41.707 to 41.851 | 13.94 to 15.02% |
| 11008 | 2 | 84.943 to 85.245 | 78.462 to 78.522 | 8.22 to 8.56% |
| 11008 | 4 | 159.918 to 160.166 | 152.708 to 152.959 | 4.55 to 4.88% |
| 11008 | 8 | 312.462 to 312.667 | 302.635 to 302.788 | 3.24 to 3.26% |

For the three eight-row K11008 confirmations, the 95% lower gain bounds are
3.212%, 3.177% and 3.192%. They exceed zero and the 3% point-estimate floor
is also satisfied. No repeated search for a favorable trial was performed.
The raw records preserve each screen, confirmation and decision. Only the
fresh confirmation enters the summary's final aggregate for these three cases.

Preparation alone takes 10.031 to 20.166 us staged and 4.842 to 11.807 us
fused across the grid, with 70.74% to 112.33% throughput gains. Those are
diagnostic producer timings, not full projection or model gains.

## Numerical and ownership evidence

All 24 cases pass before and after timing. Dense FP64 Walsh comparison has
maximum absolute error 0.000002064. Every encoded byte matches the CPU Q8_1
reference. The independent encoded-consumer comparison has maximum absolute
error 0.000011351 and uses at most 0.271% of its numerical bound.
Staged/fused canonical values, encoded bytes and consumer outputs match
exactly. Input/scratch guards, immutable signs/weights, all six width cleanup
checks and final model allocation cleanup pass.

The terminal completion record reports 24 cases and 48 comparisons. After
process exit the M40 has no compute processes and zero MiB used. The terminal
in-process snapshot still includes CUDA context memory after model teardown;
it is not a model allocation leak.

The [implementation checkpoint](2026-10-31-rotated-preparation-implementation.md)
preserves hardware coverage with cuBLAS on/off, existing Q8_1 and runtime
regressions, attention/ownership checks, CPU tests, Clippy and the initial
test-reference failure and correction. This run uses that fixed correction;
it does not alter any tolerance after observing retained-input timings.

## Storage and prerequisite costs

The two rotated weight slices occupy 1,114,112 bytes at K2048 or 5,988,352
bytes at K11008. Sign storage is 2048 or 11008 bytes. Guarded producer/input/
output scratch ranges from 28,224 to 1,164,352 bytes. Peak tracked residency
is 4,479,618,880 bytes with the original and prepared model retained equally
in both arms. These small weight slices do not establish the cache behavior
or relative gains of full-width FFN projections.

Source validation, rotation/fitting and report hashing take 182.55 to 183.79 ms
per K2048 pair and 344.86 to 354.68 ms per K11008 pair. Upload takes 0.558 to
0.631 ms and 1.385 to 1.511 ms respectively. The largest recorded weight-fit
absolute error is 0.000828843. Each source slice and encoded representation
has its own hash and error report; fit error is not model-quality evidence.

Fusion removes the signed scratch write/read and the canonical quantizer read,
12 logical bytes per activation element. No measured DRAM-sector or instruction
count is available. Both experimental arms retain all scratch allocations;
the measured gain is not attributed to lower resident memory.

Runtime adoption requires the following evidence beyond this completed study:

- A compatible rotated model with documented weight/input transformation
  semantics. The current unrotated Qwen path offers no matching operation.
- Full-width consumer and whole-model comparisons. The measured absolute
  batch-one saving is only 5.36 to 6.26 us here, even where the small consumer
  gives a large relative gain.
- Model-quality validation for the weight representation and Q8_1 activations,
  including held-out loss, logits and complete generations. Exact fusion
  parity does not make those quantizations lossless.
- Integration of the validated invocation owner into that model's execution
  graph, with explicit producer/consumer dependencies and storage lifetime.

This satisfies the fixed protocol's stopping condition for a successful
isolated experiment with a priced prerequisite report. Retain the candidate
and benchmark for a compatible future model; proceed to legacy graphs as the
next independent topic.

## Artifacts and reproduction

The [raw records](2026-10-31-rotated-preparation-timing.jsonl.gz),
[run log](2026-10-31-rotated-preparation-timing.log.gz) and
[release build log](2026-10-31-rotated-benchmark-release.log.gz) preserve the run.
The raw archive SHA-256 is
`31d3e859f6e9be479bd333bd14a7a0f31f3fee1587f428ba31b175386110d2a4`.
The [summary](2026-10-31-rotated-preparation-summary.json) is reproducible:

```bash
gzip -dc docs/experiments/2026-10-31-rotated-preparation-timing.jsonl.gz |
  jq -s -f docs/experiments/2026-10-31-rotated-preparation-summary.jq
```

Build the release example following the implementation checkpoint, then restore
the capture manifest's exact environment. Invocation arguments were:

```text
--model <Qwen2.5-3B Q8 blob with SHA-256 4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d>
--output target/rotated-preparation-timing-01.jsonl
--ffn-inputs <retained m02-ffn-inputs.json>
--ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl
--ffn-rotated-preparation --rows 1,2,4,8 --trials 5 --iterations 100
```

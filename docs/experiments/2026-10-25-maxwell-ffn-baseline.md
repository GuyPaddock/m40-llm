# Initial Complete FFN Baseline

## Result and Limitation

Clean release `d4d6716d3c448c4f82e74baafeb83c139bd77bab` completes all six
real FFN cases: Qwen2.5-3B Q8_0 layers 0/17/35, genuine rows 1/4, five trials
and ten iterations per measured stage. All numerical, capture, guard, input,
enqueue-accounting and allocation-cleanup checks pass.

This first timing run exposes a measurement limitation. Three FFN warmup
iterations do not keep clock conditions consistent across the matrix. Layer 0
starts at 949 MHz SM clock, and the device reaches 1113 MHz in later cases.
The early four-row case also includes a clock-ramp outlier. Preserve all
trials; do not attribute differences across layers to their weights or inputs.
A bounded active warmup is needed before finalizing the baseline and before
screening candidate kernels. M02 remains in progress at this checkpoint.

## Measured GPU Latency

Milliseconds per operation, median of five trials. Each four-row number covers
the whole matrix, not one token or four sequential decode calls.

| Original Layer | Rows | Gate/Up/SwiGLU | Isolated Prepared Down | Complete FFN | Complete MAD/Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.374483 | 0.137866 | 0.511821 | 0.224% |
| 0 | 4 | 4.165110 | 0.590365 | 4.755805 | 0.241% |
| 17 | 1 | 0.326842 | 0.131677 | 0.458122 | 0.092% |
| 17 | 4 | 4.161088 | 0.574253 | 4.761818 | 0.099% |
| 35 | 1 | 0.328413 | 0.131530 | 0.456858 | 0.085% |
| 35 | 4 | 4.170950 | 0.585011 | 4.758800 | 0.071% |

Small median absolute deviations do not erase the initial clock ramp. For
layer-0 rows4, complete FFN spans 4.744361 to 5.461357 ms; gate/up spans
4.158032 to 4.844637 ms. No observations are discarded. GPU boundary samples
are observations, not per-kernel clock counters or proof of constant frequency.

Gate/up remains the larger FFN cost and the existing multi-row path is not an
automatic efficiency win at four rows. These are baseline diagnostics, not a
candidate speedup, speculative-verification result, token throughput or a
measured bandwidth claim. Later candidates need fresh paired native controls,
not ratios against this historical table.

## Correctness and Scope

The [frozen paired inputs](2026-10-25-maxwell-ffn-inputs.md) bind the original
model, calibration corpus and execution identity. Each case uses the first
one/four captured rows; each of the four is distinct. Gate/up weights are
`K=2048,N=11008`; down is `K=11008,N=2048`.

All outputs are compared, not only selected logits. CPU references use scalar
Q8_0 dequantization, FP64 accumulation and SiLU. The largest gate/up/SwiGLU
CPU-reference difference is 0.000295679, and the largest down difference is
0.000046606, within the existing `1e-3 + 2e-5 * abs(reference)` tolerance.
All three single-row native SwiGLU vectors match the frozen captures bitwise.
Four-row SwiGLU differs from sequential capture by at most 0.000328064 due to
the existing multi-row arithmetic path; this is recorded, not called bit parity.

Native one-row gate/up/SwiGLU is fused on decode, with prepared down four
columns/256 threads. Four-row gate/up uses two real matrix projections on
prefill, waits to decode SwiGLU and back to prepared down four columns/128
threads. Counts per iteration are 1/1/2 launches for the three one-row stages
and 3/1/4 for four rows, with four-row waits 2/1/3. All 90 repeated-stage
intervals have zero device allocations/frees, H2D/D2H copies, cuBLAS calls or
stream synchronizations inside enqueue. The final event wait is separate.

Guarded reusable scratch is 201,152 bytes for rows1 and 803,264 for rows4.
Original weights are 3,279,519,744 bytes plus 1,192,935,424 prepared bytes.
Vocabulary preparation remains resident but is not executed. No KV, attention,
embedding, RMSNorm or vocabulary work is timed. Tracked device bytes return
zero to zero; the process's remaining CUDA context is not a weight leak.

## Reproduction and Evidence

Use [the complete-FFN command](../ffn_benchmark_inputs.md#complete-ffn-baseline)
with the exact environment in the frozen capture companion. The output used
`target/research-fixtures/m02-ffn-baseline.jsonl`. The implementation was
committed before building/running, with a clean worktree. Rust release 1.95.0,
NVCC 12.8.93, g++ 14.3.1, CUDA `-O3`, sm_52 SASS and compute_52 PTX, no fast
math. Executable SHA-256 is
`ae0a7028127cc38e8b9bf0aad1006b07a30fc4b80d5c81701e55ebc23fae276e`.

- [Raw JSONL](2026-10-25-maxwell-ffn-baseline.jsonl), 38 records: manifest,
  six validations, 30 timing rows and completion.
- [Raw stderr](2026-10-25-maxwell-ffn-baseline.stderr.log).
- [Summary and audit](2026-10-25-maxwell-ffn-baseline-summary.json), including
  all individual timings, host times, ranges, MADs and GPU clock samples.

All summary audit checks pass, including non-overlapping timed intervals,
preserved captured bits and cleanup. Source/fixture identities are retained in
the manifest. GPU process queries are empty before and after the single
benchmark; no tests, builds, other inference or MemPalace work overlap it.

Six benchmark tests pass on M40 with cuBLAS enabled; the hardware pipeline
also passes with it disabled. Full warning-as-error non-CUDA tests, CUDA/server
all-targets Clippy and formatting pass. Hidden-device NVCC runs pass four CPU
tests and explicitly ignore two hardware tests; explicit inclusion correctly
fails context creation without a device. This is not missing-hardware parity.
No production kernel or runtime default changed. Both original E2E goals remain
unmet. Next preserve this run and add bounded warmup, then repeat the baseline.

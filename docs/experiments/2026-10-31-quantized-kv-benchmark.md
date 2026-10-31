# Quantized KV operator benchmark implementation checkpoint

Status: all seventeen runner tests pass on M40, with cuBLAS enabled and
disabled. The [qualification record](2026-10-31-quantized-kv-zero-reference.md)
retains affected runtime tests and the corrected staged-reference oracle.
The runner is committed separately after that correction. Full real-input
release admission and paired measurements remain pending. No operator,
model-speed or quality result is claimed.

The [admitted capture reader](2026-10-31-quantized-kv-fixtures.md) supplies layers
0/17/35, contexts 1K/2K/4K/8K, eight distinct sequences, 16 query heads and two
KV heads. Head64 is a diagnostic slice of the original head128 data. Queries
remain deterministic synthetic vectors. An incomplete capture is rejected.

## Execution and numerical gates

Each case owns one encoded cache, one full-context dense FP16 cache, persistent
staging for eight sequences, one guarded query/output allocation, device
sequence/length arrays and reusable CUDA events. Every comparison shares that
encoded cache and query representation. Recent window1024, block size32, plain
top8, score order and no anchors/masks are frozen. The 1K control has no old KV.

The query upload crosses the prefill/decode stream boundary with an explicit
dependency. Captured K is already rotated, so append uses zero RoPE frequency;
the [CPU reference](2026-10-31-quantized-kv-reference.md) accounts for signed-zero
effects. Full sequences upload once per cache-setup phase and append by explicit
position. Allocation, uploads, quantizing append and dense append are reported
outside attention timing, with separate host/CUDA-event intervals where applicable.

Before timing, validation checks all eight sequences' lengths, selected positions,
staged FP16 words, query/output guards, inactive output rows, and deterministic
repeated execution. FP64 output parity uses the existing attention-grid absolute
`1e-3` tolerance. Reports retain observed errors and separate full-versus-selected
and fixed-selection quantization differences. A failed case records its error
and stops before timing. Numerical failures cannot become speed claims.

The ignored hardware test covers Q8/head64, mixed/head64 and mixed/head128 at
recent-only length5 and wrapped/partial-block length23. Eight distinct sequence
fixtures exercise batches1/2/4/8, GQA, serial and batched direct consumers. The
separate [staged-control test](2026-10-31-quantized-kv-staging.md) covers reset
lifetimes, physical layer/sequence slots and guard regions around staging.
Both tests now pass on hardware with cuBLAS enabled and disabled.

## Comparisons and timing

For each active batch1/2/4/8:

1. Compare staged and direct Q8/head64 or mixed/head64/head128 attention.
2. For mixed backing, compare serial direct rows with the existing batched
   direct consumer. Attribute this separately from removal of staging.
3. Record a paired full-context dense FP16 cost reference. Its token coverage
   differs, so it has no operator-adoption gate and no borderline confirmation.
   Its variation extension still applies.

The complete grid has 36 cases and 384 comparison groups. Three warm calls per
arm precede five alternating pairs, with ten complete B-row operations per
interval by default. Each interval includes selection, gather/reconstruction
and attention. CUDA events and completed host wall time are both retained;
the primary gate uses wall operations/second. Event submission and completion
overhead are amortized over the configured iteration count.

The shared paired protocol controls variation extension to ten pairs and one
fresh twenty-pair borderline confirmation for matching-coverage comparisons.
The HTTP target field is removed. Dense coverage differences never qualify as
a speed-only replacement. Each timed output is read after the timing interval
and must match its own untimed reference exactly. Readbacks and their counters
are outside both arm timings. Input guards are rechecked after each comparison.

The operation must allocate/free no tracked device buffers and perform no host
copies inside its measured interval. Counters describe host C ABI submissions,
not physical kernel counts; a staged wrapper submits more than one kernel.
Logical byte counts include all-old-key selection scans, selected old KV,
recent KV and intermediate staging traffic. Sector figures sum aligned vector
and scale footprints; they omit cache reuse and duplicated warp requests.
They are not observed DRAM traffic. No hardware counters are collected.

Both arms retain the same scratch and cache allocations. Reports show encoded
payload/scale/recent/summary bytes, staging, the dense reference and temporary
upload buffers separately. The Rust allocation tracker covers auxiliary buffers;
native cache payload estimates exclude allocator/context overhead. Final driver
residency still needs an external post-process check after the experiment.

## Serial execution and remaining checks

`--benchmark` requires a clean release build and admits only the selected M40.
It checks for another compute process before creating its context and before
each case. Environment overrides affecting attention, selection, graphs or
logging are rejected. `--correctness-only` validates the same complete grid
without timing pairs. Inspection remains the default CPU-only CLI mode.

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/quantized-kv/target
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test quantized_kv_staging -- --include-ignored --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda --example kv_attention_bench \
  -- --include-ignored --test-threads=1
# Run the affected regressions listed in the staged-control checkpoint.
# Commit the verified staged control before the dependent benchmark unit.
M40LLM_ENABLE_CUBLAS=1 cargo build --release --features cuda \
  --example kv_attention_bench --locked -j 4
target/release/examples/kv_attention_bench \
  --capture-report COMPLETE_CAPTURE.jsonl --kv-inputs COMPLETE_CAPTURE.bin \
  --benchmark --correctness-only --output NEW_CORRECTNESS.jsonl
target/release/examples/kv_attention_bench \
  --capture-report COMPLETE_CAPTURE.jsonl --kv-inputs COMPLETE_CAPTURE.bin \
  --benchmark --iterations 10 --output NEW_TIMINGS.jsonl
```

Nine CPU reader/reference tests pass. CUDA/server all-target Clippy and the GPU
test executable's compile/link step pass. NVCC emits its existing legacy-target
deprecation notice. GPU tests, release real-input validation, paired timing,
result analysis and the final branch decision remain pending. The ongoing
lossless capture retains exclusive GPU use; do not restart or overlap it.

# Direct versus staged quantized KV on the M40

**PASS at the operator level.** All 144 comparisons using identical
encoded history and selected positions exceeded the frozen speed gate.
Direct reads reduced median wall latency by **3.53–20.45%**, compared
with 3.53–20.40% in October. CUDA-event spans fell by 3.52–20.45%.
The smallest lower 95% bound on *throughput* improvement was 3.65%.
The largest within-arm MAD/median in these comparisons was 0.71%.

| Encoded history | Head size | Passed | Wall latency reduction |
|---|---:|---:|---:|
| Q8 K and V | 64 | 48/48 | 6.43–19.53% |
| FP16 K and Q4 V | 64 | 48/48 | 3.53–19.13% |
| FP16 K and Q4 V | 128 | 48/48 | 4.35–20.45% |

At Qwen's head width and batch one, the advantage narrowed at long
contexts. Each range below spans the three real source layers.

| Context | Staging median ms | Direct median ms |
|---:|---:|---:|
| 1,024 | 3.105–3.114 | 2.470–2.484 |
| 2,048 | 4.445–4.458 | 3.719–3.732 |
| 4,096 | 5.604–5.610 | 5.092–5.109 |
| 8,192 | 7.890–7.905 | 7.543–7.554 |

**Keep main's existing single-sequence direct consumer.** The benchmark
used real Qwen layer 0/17/35 caches, synthetic queries and eight distinct
input sequences; head64 samples use the first 64 channels. Head128
staging uses the research comparator. The 1,024-position cases exercise
only the recent window. The operator generated zero tokens. Dense FP16
attention covers different positions, so its 144 timing groups are
descriptive. Full-model staging versus direct execution is a separate
TinyLlama HTTP study.

**Correctness and costs.** All 36 fixtures passed independently scored
selection, reconstructed staged words, output guards, inactive rows and
repeatability. Maximum observed direct/staged and batched/serial error
was zero. Maximum direct versus matched FP64 error was 0.00001048.
Full-history selection error reached 1.67479, and quantization at fixed
positions reached 0.01937. Model-quality evidence belongs to the
[separate generation study](../2026-11-21-kv-quality-result/README.md).

All arms retained the cache and staging workspace during this comparison.
Staging reserved 34,079,232–84,541,952 bytes. Both timed paths had zero
wrapper-counted device allocations and host transfers. The sampled
operator process peaked at 326 MiB with no model weights resident.
Allocating, uploading, and appending are outside the attention endpoint;
their separate measurements are in each setup record. CUDA event spans
cover a complete submitted operation. Physical DRAM traffic, individual
kernel service time and CPU-only submission latency were unmeasured.

**Batching is a separate comparison.** In the 96 mixed batched-versus-
serial direct groups, 6/24 batch-one groups passed and all 72 groups at
batches 2/4/8 passed. The respective multirow latency reductions were
44.34–50.77%, 73.79–74.83%, and 86.30–86.88%. These measurements provide
a control for batched attention. Scheduler throughput and queue latency
need independent HTTP evidence.

## Evidence and reproduction

The [comparison](comparison.json) uses the original October 36-fixture,
384-group admission program. An independent [pair replay](pairs.json)
recovered both arm medians on both clocks from 2,340 alternating pairs.
There were 21 fresh confirmations, zero ten-pair extensions and zero
final timing errors. Every case ended at 1,113 MHz SM clock. The recorded
end-of-case temperatures were 47–64 °C. Post-process residency was
0 MiB. The [raw archive](measurements.tar.gz) contains every interval,
setup, correctness, trial ordering, variability, device state, host load,
and final report. From a repository worktree run
`bash docs/experiments/2026-11-22-direct-kv-operator/replay.sh NEW_DIRECTORY`
to repeat both analyses and compare the stored summaries.

Runtime source e100c549e9f40654df3f8fb587f939ac69712ae7, measured executable
SHA-256 bae74da7023db8351175d025693e28526cdde3ea287e3f50d8c7fece258da24c.
The reused benchmark, CUDA and runtime sources match 97102a1. The release
build used CUDA 12.8.93 targeting sm_52 and driver 580.159.04. The Qwen
Q8_0 weight hash is 4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d.
The 192 MiB FP16 fixture remains in the project's qualified local artifacts
with SHA-256 da01d6328aef717902edbc93c215f0468f2bc9d789211effbc1b61b1569d1242.

With the hardware lock held and the clean measured source checked out,
use the preflight environment and run:

```sh
source scripts/dev-env.sh
export M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
cargo build --release --locked --features cuda --example kv_attention_bench
git show 73a9417:docs/experiments/2026-10-31-lossless-capture-complete.jsonl.gz |
    gzip -dc > "$OUT/capture.jsonl"
target/release/examples/kv_attention_bench --benchmark --iterations 10 \
    --capture-report "$OUT/capture.jsonl" --kv-inputs "$QUALIFIED_FP16_FIXTURE" \
    --output "$OUT/report.jsonl"
```

# Exact Head128 Attention, 2026-10-24

C07 replaces serial context traversal only when explicitly requested. The
experiment retains dense full-context FP16 KV, FP32 computation, GQA mapping,
and absolute RoPE. No weights, sampling, KV selection, or model semantics are
approximated. All eight tested launch choices remain available for reference.

## Implementation and Scope

Each warp processes independent context tokens with coalesced K/V reads. Shared
reductions occur per partition rather than twice per context token. One
partition writes the normalized result directly. Split2/4/8 stores FP32 maxima,
exponential sums, and weighted V vectors, then launches a numerically stable
merge using a global maximum and rescaled sums.

`Head128AttentionWorkspace` owns the partial buffer through `DeviceBuffer`.
The normal Rust attention entrypoint caches one workspace in the KV cache and
reuses it across layers/requests on the owning context's decode stream. A
mutex protects preparation/enqueue, and the common stream orders reuse.
The explicit API rejects incompatible workspace capacity/context. Tests verify
tracked cleanup and no steady-call allocation, free, or synchronization.

At 16 query heads, capacity is 66,560 bytes, or 65 KiB:
`16 heads * 8 partitions * (2 statistics + 128 values) * sizeof(f32)`.
Active partial bytes are 16,640/33,280/66,560 for split2/4/8. One partition
requires no partial buffer or merge. These bytes are reported separately from
the 37,748,736-byte dense KV payload. The model does not allocate a buffer
per layer or per token.

Both new kernels compile to 32 registers per thread with zero stack/local
memory reported by `cuobjdump --dump-resource-usage`; the merge has 48 static
shared bytes. Split8/256 uses 4392 dynamic shared bytes at length 512 and
4648 at length 1024. These are compiled resource counts, not measured achieved
occupancy; legacy CUPTI counters remain unavailable on this stack.

The M40 has 24 SMs, while this Qwen model has 16 query heads. One-block/head
therefore offers 16 blocks; split8 offers 128 partial blocks followed by 16
merge blocks. The extra merge cost is included in all reported timings.

The experiment is opt-in for serialized dense head128 attention. Packed server
scheduler and device-parameter graph paths are unchanged. Runtime/global KV
defaults are unchanged. The production reference remains the existing fast
kernel unless `M40LLM_ATTN_HEAD128_KERNEL=parallel` is explicitly set.

## Bounded Geometry Screen

Two thread counts, 128/256, crossed with 1/2/4/8 partitions give eight new
configurations, below the twelve-configuration limit. The existing fast kernel
is the control. All requested lengths 1/32/64/128/256/512/1024 and irregular
319/575/1025 are measured, using query heads 16, KV heads two, head dimension
128, and nonzero sequence slot two.

The release-only test runs five trials per configuration with alternating
order. CUDA events surround forty repeated calls, including merge launches,
then one stream completion. Scratch setup is outside timing. KV is cache-hot
and no model weights are streamed, so these are isolated diagnostic costs,
not E2E speed claims. Its fixture keeps the same 65 KiB scratch resident for
every row; `active_partial_bytes` records actual use. The manifest captures
an initial SM-clock observation of 948 MHz, not continuous per-trial clocks.
Do not equate that observation with the HTTP run's 1113 MHz boundary samples.

Median CUDA-event microseconds per call:

| KV Length | Existing Fast | 1 Part/128 Threads | 1/256 | 2/256 | 4/256 | 8/256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.72 | 6.28 | 6.48 | 11.67 | 11.37 | 13.35 |
| 32 | 34.66 | 12.48 | 9.29 | 13.01 | 11.90 | 14.42 |
| 64 | 64.92 | 19.72 | 13.01 | 14.96 | 13.18 | 14.95 |
| 128 | 125.16 | 34.16 | 20.47 | 18.79 | 15.22 | 16.67 |
| 256 | 218.88 | 56.12 | 31.56 | 23.93 | 17.61 | 17.40 |
| 512 | 433.62 | 103.22 | 55.63 | 36.07 | 24.30 | 21.83 |
| 1024 | 829.08 | 201.25 | 105.89 | 62.20 | 37.85 | 31.79 |

All eight configurations and irregular lengths are in the
[raw microbenchmark](2026-10-24-head128-micro.jsonl) and
[median summary](2026-10-24-head128-micro-summary.json). Split8/256 is the
candidate taken to HTTP comparison, not a claim that it wins at every length.
For example, the reference wins at length one and one-partition/256 is cheaper
at lengths 32/64. No extra dynamic-dispatch tuning was added in this round.

## Paired HTTP Result

Use the controlled Qwen2.5-3B Q8_0 blob, SHA-256
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
The rendered browser-loading prompt is the same 65 ChatML IDs, with the same
system message and assistant prefix. Dense FP16 KV capacity is 1024. All rows
use greedy argmax and actually generate 512 IDs with length stops and fresh
prompt KV. Rust is release optimized; CUDA 12.8 `-O3`/`sm_52` flags and Q8
projection/fusion settings are unchanged from C03/C06.

The new `--compare-head128-attention` mode alternates complete m40-llm HTTP
requests, restoring the env switch on return. Ollama is used only for untimed
model/tokenizer verification and is idle during measurement. Both models are
resident, but inference is strictly sequential. No GPU test overlaps timing.
Boundary telemetry shows driver 580.159.04, P0, 1113/3004 MHz SM/memory clocks,
46-68 C, and only the two permitted benchmark GPU processes.

| Pair | Existing Fast HTTP Tok/s | Split8/256 HTTP Tok/s |
| --- | ---: | ---: |
| 0 | 23.8394 | 31.9620 |
| 1 | 23.7961 | 31.9351 |
| 2 | 23.7737 | 31.8255 |
| 3 | 23.7506 | 31.8239 |
| 4 | 23.7410 | 31.8520 |
| Median | **23.7737** | **31.8520** |

Throughput improves **33.980%**, with paired-bootstrap 95% interval
**[33.868%, 34.203%]**. MAD/median variation is 0.097% control and 0.088%
candidate. The 3% promotion threshold and positive confidence lower bound
pass; neither the ten-pair extension nor twenty-pair borderline confirmation
is indicated. HTTP latency falls from 21.536 s to 16.074 s, about 25.36%.

| Median Phase | Existing Fast | Split8/256 |
| --- | ---: | ---: |
| Internal prefill | 1.736 s | 1.683 s |
| Internal decode | 19.762 s | 14.338 s |
| Internal total | 21.485 s | 16.021 s |
| Client HTTP E2E | 21.536 s | 16.074 s |

All timed rows have identical output hashes. Untimed companions reproduce
their measured outputs and match **all 512 generated IDs**, with no divergence.
See [raw HTTP evidence](2026-10-24-head128-http.jsonl) and
[separate actual token traces](2026-10-24-head128-token-traces.jsonl).

This is a contemporaneous **m40-llm kernel comparison**, not a fresh Ollama
performance run. The earlier controlled Ollama median remains 43.035 tok/s;
31.852 is still below Ollama and both project targets, 53.06 and 55.945.
Do not present the 33.98% gain as a margin over Ollama.

## Separate Stage Diagnostic

A single profiled CLI run uses the same prompt, 1024-context dense KV, 512
greedy predictions, and split8/256. Its full decoded output matches both HTTP
token traces. It is diagnostic only, not another E2E timing trial.

| Stage, All 36 Layers | Step 0 | Step 255 | Step 511 |
| --- | ---: | ---: | ---: |
| Attention | 0.687 ms | 0.791 ms | 0.973 ms |
| Gate/up/SwiGLU | 11.842 ms | 11.804 ms | 11.803 ms |
| Down | 6.351 ms | 6.298 ms | 6.311 ms |
| Vocabulary/argmax | 3.438 ms | 3.421 ms | 3.426 ms |
| Measured GPU interval union | 28.479 ms | 28.245 ms | 28.428 ms |

The earlier C04 medians were 11.421/20.387 ms attention at middle/late samples.
That is a five-request baseline versus this one-request diagnostic; do not
derive a repeated E2E gain from that stage comparison. FFN now accounts for
about 64% of this diagnostic's middle GPU interval union.

Each sampled token records 435 launches instead of 399, exactly one merge
per layer. It still records one completion sync, zero waits, zero allocations/
frees, one 8192-byte H2D, and two D2H copies totaling 2180 bytes. Extra launches
are worthwhile here because they remove much more serial barrier work.
The first event-pool capture still has an unexplained 86.74 ms wall versus
28.48 ms union; reused middle/late wall is 29.12/29.28 ms. Do not call that
first gap ordinary CPU inference cost. Raw
[stage events](2026-10-24-head128-stage-events.jsonl.gz) retain every layer.

## Correctness and Limitations

Numerical tests compare every geometry against the current CUDA kernel and
an FP64 CPU softmax reference over FP16-rounded K/V. They cover lengths
1/7/32/33/64/127/128/255/256/257/511/512/513/1023/1024/1025, slots zero/two,
absolute split-half RoPE offset 17, both GQA groups, empty partitions, uniform
and extreme scores, and a poisoned future KV token. The test tolerance is
`1e-3 + 1e-4 * abs(reference)`. Outputs must also be finite.
Across 256 geometry/length/slot comparisons of 2048 outputs each, measured
max/mean absolute error is `3.6061e-6 / 9.2537e-9` versus the CPU reference
and `3.1739e-6 / 9.2570e-9` versus legacy CUDA. The aggregate
[numerical record](2026-10-24-head128-parity.json) retains those measurements.

Workspace tests verify no scratch allocation for one partition, one tracked
allocation on the first split call, reuse without per-call allocation/sync,
capacity/context rejection, invalid configs/ranges, and tracked RAII cleanup.
The existing attention grid passes with cuBLAS disabled and enabled. Forward
smoke, projection/MLP parity, buffered profiler, server metrics, and server
batching smokes pass with cuBLAS enabled. Warning-as-error non-CUDA tests,
CPU benchmark protocol tests, CUDA/server and non-CUDA/server Clippy, and fmt
also pass.

The explicit real-model smoke uses the same Qwen system/ChatML format as
the benchmark. Both kernels return `4`, `OK`, and `37`; all six
[quality records](2026-10-24-head128-generation-smokes.jsonl) are retained.
The first user-only Auto-format trial returned immediate EOS on the legacy
exact-OK row before testing the new kernel. Those
[earlier records](2026-10-24-head128-user-only-smoke.jsonl) remain visible.
The final smoke changes the wrapper to match the measured workload, not the
expected answers or attention math. This checkpoint makes no automatic-prompt
formatting fix or universal Qwen capability claim.

The current build script selects CUDA through `--features cuda` and requires
NVCC; it does not interpret `M40LLM_ENABLE_NVCC=0` as a stub switch. A separate
build invoked with that variable still used NVCC and is not stub evidence.
The actual non-CUDA test run exercises the normal C stub build. Build behavior
was not changed in this experiment.

## Reproduction and Decision

Use the [attention experiment commands](../diagnostics.md#exact-head128-attention-experiment)
for numerical/model checks and the microbenchmark. Use the complete C03 Q8
backend environment from [the HTTP protocol](../controlled_benchmark.md), plus:

```bash
M40LLM_ATTN_HEAD128_PARTITIONS=8 M40LLM_ATTN_HEAD128_THREADS=256 \
target/release/examples/compare_ollama \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ollama-pid OLLAMA_SERVER_PID \
  --ollama-log /tmp/m40-ollama-controlled.log \
  --compare-head128-attention --output /tmp/head128-http.jsonl
```

The HTTP binary SHA-256 is
`ecde08da5df8c1d9609bc67c3b7da576c0377778e9bb9481f35e1889cd3f58a1`;
its manifest pins base `be44ed5` and the C07 worktree. Later changes in this
commit extend tests, provenance, and docs, not the measured kernel/runtime.
The separate profiled CLI binary SHA-256 is
`07de947ed042731e8466d8a8f5b860aa0e467530b7722c6820dd55bca6e153f9`.

Artifact SHA-256 values:

- HTTP JSONL: `2de237fe26c4dfa2f051d7b541fdca2141edd1672a9cf73469b5e01d54f8b1bc`.
- Micro JSONL: `fc4d9dbb5371df8e2ba654348369307db937109bddfc5d26deafa0da97d6e3e9`.
- Gzip stage JSONL: `86b18eddf616c3134f11edbde443e6f6e0ef85c7107b0136e031b8aa8a78478f`.

**Keep split8/256 as the selected opt-in benchmark configuration.** It passes
numerical/model checks and the predeclared paired HTTP promotion gate. Do not
change global runtime or compressed-KV defaults. Preserve the other measured
geometries as reference, including the losing short-context cases.

Next is C08's benchmark-only typed execution-weight ownership, then independent
C09 separated scale/value and C10 input-tiled Q8 layout experiments. FFN is
now the main measured opportunity. Do not spend another unbounded attention
tuning round before those experiments and the C11/C11A checkpoints.

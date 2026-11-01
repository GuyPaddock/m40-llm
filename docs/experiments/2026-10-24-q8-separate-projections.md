# Separate Q8 Scales and Values

## Decision

C09 is a promising **isolated projection** experiment. Keep its lossless
repacker and four fixed launch configurations for the C10/C10A comparison.
Do not change runtime dispatch, loading, caching, or defaults from these
measurements. In particular, ordinary inference's fused QKV, fused gate/up/
SwiGLU, and fused vocabulary argmax are not replaced by this benchmark.

The separate-plane representation copies the two original FP16 scale bytes
and all 32 signed-value bytes per Q8_0 block. It does not requantize weights.
The CUDA consumer reads aligned eight-byte value groups, reuses eight input
values across one/four columns, and accumulates/reduces in FP32. It uses no
Tensor Cores, WMMA, or integer dot-product instructions.

## Method

- Tesla M40 24GB, sm_52, driver 580.159.04; official Qwen2.5-3B Q8_0 blob
  SHA-256 `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
- Optimized Rust release build, NVCC 12.8 V12.8.93, existing `-O3`, sm_52 SASS
  and compute_52 PTX flags, cuBLAS enabled at build time.
- Benchmark executable SHA-256
  `93a013d7446447eea9ca3d3275e65ab784ad6617f4abdc69bc4cea7defb97e2f`.
  Built from the C09 working tree on `33891e8`; manifests retain dirty status.
- Real layer-0 Q/K/V/output/gate/up/down tensors and tied vocabulary weight,
  not newly quantized synthetic approximations. Deterministic FP32 activations
  differ across input rows. All outputs compare to the current GGUF path;
  sixteen row/column samples per configuration also compare to FP64 CPU sums.
- Fixed candidates are columns 1/4 by threads 128/256, four configurations.
  All use four lanes per Q8 block. The current decode control is GGUF
  split4/coltile4, 256 threads. Multi-row controls use existing GGUF dispatch.
- Three untimed warmup pairs, five alternating serial pairs per configuration,
  with CUDA events around repeated calls. Single/four-row batches repeat
  10/3 calls per event bracket; the bounded 64-row check repeats two calls.
  Preparation, data readback, `nvidia-smi`, and parity checks are outside timing.
- No Ollama inference or CUDA tests overlap timing. The executable rejects
  any other GPU compute PID; every timing boundary shows only its own process.
- Boundary samples are P0, memory clock 3004 MHz, 29-37 C. SM clocks are
  949 MHz for the initial small/FFN rows, 1113 MHz for vocabulary rows, and
  949/1037/1113 MHz across the separate 64-row run. These are paired isolated
  conditions, not the sustained 1113 MHz HTTP baseline from C07.

Event brackets can include GPU idle gaps if host submission cannot keep up
with tiny kernels. Do not interpret these numbers as instruction-only kernel
time or measured DRAM bandwidth. Comparing the best candidate also changes
launch geometry and vector loads, so gains are not attributed solely to
separating scales. The same 4-column/256-thread candidate independently gives
Q/output/gate/up/vocabulary ratios around 1.22-1.25x and down around 1.34x
in this screen, before selecting the best configuration.

## Single-Row Results

Median event milliseconds per projection. K/V use the separate ten-pair,
100-call confirmation below; all other rows use the initial five-pair run.

| Projection | K x N | Best Columns/Threads | GGUF ms | Separate ms | Ratio |
| --- | --- | --- | ---: | ---: | ---: |
| Q | 2048 x 2048 | 4/128 | 0.05460 | 0.03670 | 1.488x |
| K | 2048 x 256 | 1/128 | 0.01115 | 0.00975 | 1.144x |
| V | 2048 x 256 | 1/128 | 0.01099 | 0.00963 | 1.141x |
| Attention output | 2048 x 2048 | 4/128 | 0.05477 | 0.03685 | 1.486x |
| FFN gate | 2048 x 11008 | 4/128 | 0.22867 | 0.13907 | 1.644x |
| FFN up | 2048 x 11008 | 4/128 | 0.22989 | 0.14028 | 1.639x |
| FFN down | 11008 x 2048 | 4/128 | 0.19063 | 0.14302 | 1.333x |
| Vocabulary projection | 2048 x 151936 | 4/128 | 2.58428 | 1.62364 | 1.592x |

The initial K/V decode screen has up to 18.22% baseline and 8.09% candidate
MAD/median variation, so its apparent best timings are not the confirmation.
One bounded extension uses ten pairs and 100 calls per bracket. Variation
falls below 0.96% baseline / 1.19% candidate. One-column/128-thread is best
there, around 14% faster; four-column/128-thread is effectively neutral and
one-column/256-thread is slower. Preserve those negative configurations too.
Small K/V speed is not a reason to dismantle the production fused QKV path.

## Multi-Row Results

Best 4-column/128-thread candidate unless noted. All values are medians from
five pairs; the complete artifact contains every candidate.

| Projection | Rows | GGUF ms | Separate ms | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Q | 4 | 0.52743 | 0.10985 | 4.802x |
| FFN gate | 4 | 2.34322 | 0.51986 | 4.507x |
| FFN down | 4 | 1.90612 | 0.48513 | 3.929x |
| Vocabulary projection | 4 | 25.06256 | 6.45781 | 3.881x |
| Q | 64 | 6.98186 | 1.51742 | 4.601x |
| FFN gate | 64 | 29.81693 | 7.48147 | 3.985x |
| FFN down | 64 | 20.42931 | 7.35272 | 2.778x |

The candidate parallelizes each row's reduction across a CUDA block, rather
than introducing a tuned multi-row GEMM. It does not reuse weights between
rows. These large ratios partly expose the existing GGUF multi-row baseline;
they are not comparisons to materialized cuBLAS, packed full-model prefill,
server batching, or speculative verification.

## Preparation, Memory, and Numerical Checks

The execution payload is exactly the source Q8_0 payload size, without padding
growth. This does **not** mean adding a cache alongside the uploaded model is
free. The benchmark retains both representations, explicitly reporting their
sum; C10A must account for any duplicate resident model weights.

| Tensor | Execution Bytes | Repack ms | Upload ms |
| --- | ---: | ---: | ---: |
| Q | 4,456,448 | 2.65 | 0.62 |
| K | 557,056 | 0.36 | 0.29 |
| V | 557,056 | 0.16 | 0.25 |
| Attention output | 4,456,448 | 2.80 | 0.57 |
| Gate | 23,953,408 | 15.18 | 4.04 |
| Up | 23,953,408 | 15.08 | 3.65 |
| Down | 23,953,408 | 14.85 | 3.93 |
| Vocabulary | 330,612,736 | 206.38 | 50.50 |

These are single setup observations, not load-time medians. Source file reads
are outside them. At vocabulary setup, the two device representations request
661,225,472 bytes and the three host payloads occupy 991,838,208 bytes.
CUDA context/allocator granularity and process VRAM are separate overheads.

`cuobjdump` reports 28 registers for one-column and 45 for four-column kernels,
zero stack/local bytes for both, and dynamic shared memory of
`columns * threads * 4` bytes, at most 4096. Actual occupancy/traffic counters
remain unavailable as documented in C05.

Every real-tensor parity check passed. Across single/four-row configurations,
worst max/mean difference from GGUF is `1.01328e-5 / 9.70206e-7`; worst
sampled FP64 reference error is `2.86847e-7`. The 64-row run's worst max
difference is `1.23978e-5`. The lossless host test reconstructs source bytes,
including signed zero/NaN scale bits and nonzero inactive tail bytes. The
CUDA regression covers K/N tails, signed extremes, zero/negative scales,
distinct rows, invalid layout/rows/buffer sizes, output preservation on
rejection, and zero enqueue-time allocation/synchronization.

Warning-as-error non-CUDA tests, CUDA/server and non-CUDA/server Clippy,
the existing CUDA attention grid, projection/MLP wrappers, and execution-owner
tests pass. No quality, HTTP throughput, or model-level speedup is claimed.

## Reproduction and Next Work

```bash
source scripts/dev-env.sh
M40LLM_ENABLE_CUBLAS=1 cargo build --release --features cuda \
  --example q8_execution_bench
MODEL="$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d"
target/release/examples/q8_execution_bench --model "$MODEL" \
  --output /tmp/q8-separate.jsonl --rows 1,4 --trials 5 --iterations 10
target/release/examples/q8_execution_bench --model "$MODEL" \
  --output /tmp/q8-separate-prefill64.jsonl --rows 64 --trials 5 \
  --tensors blk.0.attn_q.weight,blk.0.ffn_gate.weight,blk.0.ffn_down.weight
target/release/examples/q8_execution_bench --model "$MODEL" \
  --output /tmp/q8-separate-kv.jsonl --rows 1 --trials 10 --iterations 100 \
  --tensors blk.0.attn_k.weight,blk.0.attn_v.weight
gzip -dc docs/experiments/2026-10-24-q8-separate-projections.jsonl.gz | \
  jq -s -f docs/experiments/2026-10-24-q8-separate-summary.jq
```

Run GPU commands with hardware access, sequentially. Output paths must not
already exist. Raw [single/four-row pairs](2026-10-24-q8-separate-projections.jsonl.gz),
[64-row pairs](2026-10-24-q8-separate-prefill64.jsonl.gz),
[K/V confirmation](2026-10-24-q8-separate-kv-confirm.jsonl.gz),
[derived summary](2026-10-24-q8-separate-projections-summary.json), and
[compiler resources](2026-10-24-q8-separate-resources.txt) preserve the results.

Next is the independent C10 input-tiled layout experiment. Only after its
comparison should C10A integrate useful per-family layouts, preserving fused
operations where beneficial and measuring preparation/resident memory. Any
runtime candidate must pass model parity and repeated paired HTTP E2E gates
against C07 split8/256. The unchanged-model goals remain unmet; do not project
these ratios onto the 31.852 tok/s HTTP result or claim an Ollama win.

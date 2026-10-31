# Controlled Qwen Baseline Checkpoint, 2026-10-24

C06 consolidates the C03 matched HTTP comparison, C04 sampled stage events,
and C05 counter-availability result. It does not introduce a kernel or run a
new timing matrix. This is the fixed reference for the first bounded round
of exact-kernel work, not a claim that the speed target has been reached.

## Matched Workload

Both engines load Qwen2.5-3B-Instruct Q8_0 with SHA-256
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
The actual inventory has 36 layers, hidden width 2048, FFN width 11008,
16 query heads, two KV heads, head dimension 128, and vocabulary 151936.
All 253 matrix tensors are Q8_0; 181 norms/biases are F32.

The browser-loading question is unchanged. Both raw interfaces receive the
same rendered system/user/assistant ChatML prompt and 65 verified token IDs.
Prompt text SHA-256 is
`3005a73b43160b265f25b4545853c3fdab0ab75014d315c701ec6bef62648f9a`;
token-ID JSON SHA-256 is
`395153b63b714fd6b481f4cc148988649d1220f4c788915ded1e6c3caf3c6ad2`.
Both use greedy argmax, no repetition penalties, context capacity 1024,
full dense FP16 KV, and a limit of 512 generated tokens. Natural stopping
remains enabled. All measured rows actually generate 512 tokens with length
stops; no EOS suppression or output padding is used.

Ollama 0.40.11 is NVIDIA-CUDA-12-backed and fully GPU-offloaded, using the
official prebuilt release. The system Ollama installation is unchanged.
m40-llm uses release Rust 1.95.0, NVCC 12.8.93, g++ 14.3.1, unchanged CUDA
`-O3`/`sm_52` SASS plus `compute_52` PTX, large-model Q8 execution, no FP32
materialization, and no graphs. The complete fixed environment and command
are in the [controlled benchmark protocol](../controlled_benchmark.md).

Each engine completes before the next begins, with alternating order across
pairs. Both models are resident, but inference is never concurrent. No GPU
tests overlap the timed trials. Each request starts with fresh prompt KV.
Ollama's RAM prompt cache is disabled, and its native log proves zero reused
tokens and 65 actual evaluated tokens after a zero-common-prefix replacement.
m40-llm resets KV per request. Nominal `prompt_eval_count` alone is not the
cache-reset evidence.

All observations are from M40 UUID
`GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852`, driver 580.159.04. Boundary samples
show P0, SM/memory clocks 1113/3004 MHz, temperature 62-81 C, and power
72.60-106.00 W. These are boundary observations, not continuous telemetry.
Only the two allowed benchmark processes own GPU allocations during trials.

## Timing and Targets

Client HTTP wall time includes request send through complete buffered body
receipt, before JSON parsing. Model load and untimed verification are outside
this warm boundary. Prefill and decode metrics are secondary diagnostics.

| Runtime | Median HTTP Tok/s | Range | Median HTTP Wall | Internal Prefill | Internal Decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| m40-llm | 23.735 | 23.715-23.776 | 21.571 s | 1.731 s | 19.794 s |
| Ollama 0.40.11 | 43.035 | 43.001-43.108 | 11.897 s | 0.230 s | 11.661 s |

Each runtime has five paired observations. MAD/median is 0.059% and 0.032%.
The runtime-to-Ollama gain is -44.846%, with paired-bootstrap 95% interval
[-44.894%, -44.793%], using 10000 resamples and seed zero. The noise-extension
and borderline-confirmation gates do not trigger. All rows remain in the data.

The unchanged-model target is both **53.06 HTTP E2E tok/s** and **1.30x the
fresh Ollama median**, now **55.945 tok/s**. The latter is the binding target,
requiring 2.357x current throughput, or about 57.57% less E2E latency.
Neither target is reached. The historical 24.95/40.81 figures used different
measurement conditions and are not the control for the next experiment.

m40-llm prefill includes setup through first sampling; Ollama reports native
prompt evaluation. Do not subtract these phase timings as if their boundaries
were identical. Dense m40-llm prefill is sequential in this configuration:
`M40LLM_PREFILL_CHUNK_SIZE` is unset. There are 576 one-token forwards and
512 vocabulary predictions, counting the prompt and generated continuation.
Faster prompt processing remains useful, but the 19.8-second decode phase
dominates this particular E2E workload.

The m40-llm process's construction/upload measurement was 5.256 s. The recorded
2.933 ms Ollama load probe reached an already resident runner; it is **not**
an equivalent cold-load measurement. No cold-loading speed comparison is made.
The original cold discovery/JIT timeout is compatibility history only.

## Weight Inventory and Bandwidth Estimates

The tensor inventory contains 3,085,938,688 parameters and 3,279,519,744 bytes
of tensor payload, excluding GGUF metadata/alignment. FFN matrices contain
2,434,793,472 parameters, **78.90%** of the total. Q8_0 uses 34 bytes per
32 elements, preserving its original FP16 scales and signed values.

| Family | Tensors | Stored Bytes |
| --- | ---: | ---: |
| Gate/up | 72 | 1,724,645,376 |
| Down | 36 | 862,322,688 |
| Q/K/V | 108 | 200,540,160 |
| Attention output | 36 | 160,432,128 |
| Tied embedding/vocabulary matrix | 1 | 330,612,736 |
| Norms and Q/K/V biases | 181 | 966,656 |

The Q8 matrices sum to 3,278,553,088 bytes. A full vocabulary prediction reads
the tied embedding matrix as its output projection; do not count it twice
as two resident matrices. The additional input-embedding row, norms/biases,
activation traffic, KV traffic, cache effects, and repeat loads are outside
the following simplified matrix-bytes-times-rate estimates.

| Rate | Estimated Once-Per-Token Matrix Bandwidth |
| --- | ---: |
| m40-llm, 23.735 tok/s | 77.82 GB/s |
| Ollama, 43.035 tok/s | 141.09 GB/s |
| Original target, 53.06 tok/s | 173.96 GB/s |
| Refreshed target, 55.945 tok/s | 183.42 GB/s |

These are not measured DRAM bandwidth. NVIDIA's
[M40 24GB datasheet](https://images.nvidia.com/content/tesla/pdf/78071_Tesla_M40_24GB_Print_Datasheet_LR.PDF)
specifies 288 GB/s peak memory bandwidth. The simplified target is below that
peak, but that does not prove it is attainable E2E. CUPTI 12.8.90 catalogs
metrics but cannot create legacy event groups on this driver, even as root.
DRAM bytes, cache efficiency, and achieved occupancy remain unknown. See the
[counter preflight](../maxwell_counters.md), not an inferred utilization claim.

## Bottleneck Ranking

C04's five profiled requests use the same workload. Profiling reduces observed
HTTP throughput by 0.47%; actual generated IDs match the unprofiled companions.
Only measured profiler sessions 2-6 contribute here, not warmup or trace runs.

| Sampled Stage | Middle Step 255 | Late Step 511 |
| --- | ---: | ---: |
| Gate/up/SwiGLU | 11.857 ms | 11.858 ms |
| Attention | 11.421 ms | 20.387 ms |
| Down | 6.315 ms | 6.318 ms |
| Vocabulary/argmax | 3.432 ms | 3.434 ms |
| QKV | 2.490 ms | 2.495 ms |
| Attention output | 1.713 ms | 1.720 ms |
| All measured GPU interval union | 38.889 ms | 47.898 ms |
| Token wall | 39.724 ms | 48.713 ms |

The combined FFN spans are about 47% of the middle GPU interval union;
attention is about 29%, increasing to 43% late. Gate/up is the largest fixed
FFN cost. The old down-only emphasis is no longer supported by this profile.
This supports attention and FFN work, not approximate attention semantics.

The following Amdahl calculation extrapolates the middle projection costs
across 576 sequential forwards and the vocabulary cost across 512 predictions.
Attention uses a linear fit between sampled sequence lengths 65 and 576,
including extrapolation over the earlier prompt positions. It divides those
estimated stage totals by the observed 21.571-second HTTP median. The last
column assumes a stage costs zero while everything else stays unchanged.

| Stage Removed | Estimated E2E Fraction | Idealized E2E Speedup |
| --- | ---: | ---: |
| Gate/up/SwiGLU | 31.66% | 1.463x |
| Attention | 27.58% | 1.381x |
| Down | 16.86% | 1.203x |
| Both FFN stages | 48.52% | 1.943x |
| Vocabulary/argmax | 8.15% | 1.089x |
| QKV | 6.65% | 1.071x |
| Attention output | 4.57% | 1.048x |

These are idealized **estimates**, not measured whole-workload stage totals
or hard bounds. Samples do not prove every intermediate token has the same
cost. Do not add the combined-FFN row to its constituent rows. The estimates
explain why one small fusion cannot plausibly close a 2.357x gap and why even
attention alone is insufficient. They do not predict an optimization's speed.

Event spans can include host enqueue gaps and existing wrapper waits. Host
synchronization overlaps GPU work. The first fresh-pool capture has an
unexplained 97.55 ms wall versus 29.99 ms measured union; do not attribute
that gap to ordinary model CPU overhead. Reused middle/late captures are much
closer to the measured GPU intervals. Each sampled token has 399 launches,
one completion sync, no stream waits, and no device allocations/frees.

## Next Experiments and Acceptance

Proceed with C07 and C08-C10 as bounded, independent exact-kernel work. Keep
the current dense benchmark command as control; project compressed-KV defaults
remain unchanged. No new kernel is preferred before parity and repeated E2E
acceptance. Preserve working failed experiments before removing them.

C07 compares context-parallel one-block/head and split-context 2/4/8 attention
with stable FP32 softmax merging. A fresh driver-attribute query confirms
24 SMs, 2048 maximum threads/SM, 98304 shared bytes/SM, and 65536 registers/SM.
One block per Qwen query head provides only 16 blocks, so at least eight SMs
cannot receive one of those blocks in an isolated launch. That is a geometry
limit, not measured occupancy. Include merge launch and reusable scratch cost.
Cover 1/32/64/128/256/512/1024 and irregular lengths/nonzero slots.

C08 first adds minimal typed RAII execution-weight views for benchmarks.
C09 separated scales/values and C10 input-tiled layouts both depend on C08,
not on each other succeeding. Benchmark real gate/up, down, QKV, and vocabulary
shapes before loader/cache integration. Preserve values/scales exactly and
select useful layouts per projection family. Bound the first round to twelve
launch configurations per kernel family, then publish C11.

For promotion, require matched actual 512-token length-limited output,
unchanged-model correctness, E2E gain at least `max(3%, 2 * larger MAD/median)`,
and a positive 95% paired-bootstrap lower bound. Five pairs are a screen;
noise over 5% extends once to ten. A borderline result schedules exactly
twenty fresh confirmation pairs, not sampling until significant. A kernel
microbenchmark gain alone does not qualify.

After C11, run the formal history-only n-gram acceptance gate C11A before
building verification. The remaining research sequence is unchanged even
if these kernel experiments do not reach the ultimate target.

## Evidence and Reproduction

The [derived JSON](2026-10-24-qwen-baseline-checkpoint.json) retains the tensor
totals, sampled stages, arithmetic, targets, and explicit assumptions. Its
[jq program](2026-10-24-qwen-baseline-checkpoint.jq) checks actual counts, stops,
cache evidence, matching traces, tensor types, sampled sessions, and medians.

```bash
gzip -cd docs/experiments/2026-10-24-qwen-stage-events.jsonl.gz \
  > /tmp/m40-c06-events.jsonl
jq '.hardware' docs/experiments/2026-10-24-qwen-baseline-checkpoint.json \
  > /tmp/m40-c06-device-attributes.json
jq -n \
  --slurpfile control docs/experiments/2026-10-24-qwen-ollama-controlled.jsonl \
  --slurpfile events /tmp/m40-c06-events.jsonl \
  --slurpfile hardware /tmp/m40-c06-device-attributes.json \
  --slurpfile perturbation docs/experiments/2026-10-24-qwen-profile-comparison.jsonl \
  --slurpfile counters docs/experiments/2026-10-24-maxwell-counter-preflight.jsonl \
  -f docs/experiments/2026-10-24-qwen-baseline-checkpoint.jq
```

The hardware resources were queried independently with `cuDeviceGetAttribute`
on device 0 using `MULTIPROCESSOR_COUNT`, `MAX_THREADS_PER_MULTIPROCESSOR`,
`MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`, `MAX_REGISTERS_PER_MULTIPROCESSOR`, and
`GLOBAL_MEMORY_BUS_WIDTH`. The reproduction above reuses that observation;
it does not claim to re-query hardware without access.

Raw artifact SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| C03 controlled JSONL | `ffa9ccbcb4d4043bde36c10df3f59fa1a613b7d6745b4a1947065f998c475732` |
| C04 HTTP JSONL | `6f3e46317fb3a2ab6d048b6b150732d1acfab5a33166865785ac8ae18aae10e4` |
| C04 gzip events | `6add55fe7ca4bdfea47c69f5a73379c728c7c1f0ee99972d7b0be83999409110` |
| C05 preflight JSONL | `b30a58202736fb00ce7ad17c757077fde6bc93632a2b6aa931dcb977a913f594` |

The C03 manifest pins base commit `2f4d7ca` plus benchmark worktree and binary
hash; C04 pins `663c414` plus profiler worktree and binary hash. Implementations
are committed in `663c414` and `76905e1`; the counter checkpoint is `4594560`.
The rejected extra-BOS attempt remains linked in the historical baseline.
Both engines' outputs are coherent but differ, so this is not cross-engine
logit equivalence. Ollama's extra cataloged EOG markers were not emitted in
these length-limited rows; no universal stop-token equivalence is claimed.

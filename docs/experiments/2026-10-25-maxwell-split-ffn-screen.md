# Split-Plane FP32 FFN Micro Gate

## Decision

The fixed 128-thread/four-value split-plane candidate passes the complete-FFN
micro gate at all three original layers. One-row throughput improves
**15.54-15.65%** over qualified packed/warp128, with positive paired confidence
intervals. Genuine four-row work improves 13.47-15.91%, reported independently.
Retain it for conditional model-local runtime evaluation, not as a runtime or
default recommendation.

No actual 512-token, whole-model quality or HTTP run of this candidate exists
yet. The last qualified runtime remains packed FP32/128 plus dequant SGEMM at
41.331 HTTP tok/s against fresh Ollama 42.946. Both original targets remain
unmet: 53.06 and 55.829 tok/s for 1.30x that fresh Ollama median. Do not multiply
these FFN gains into the prior HTTP rate or change acceptance thresholds.

## Complete FFN Results

Milliseconds are CUDA-event medians of five alternating pairs, with ten
repetitions per stage. Gain is control latency / candidate latency minus one.
Confidence intervals resample contemporaneous pairs 10,000 times, seed zero.

| Layer | Rows | Packed128 Control ms | Split-Plane ms | Gain | Paired 95% Gain Interval | Micro Gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 1 | 0.406986 | 0.351933 | +15.64% | +15.10 to +16.05% | Pass |
| 17 | 1 | 0.406918 | 0.351853 | +15.65% | +14.15 to +15.89% | Pass |
| 35 | 1 | 0.407485 | 0.352675 | +15.54% | +14.70 to +16.03% | Pass |
| 0 | 4 | 1.753968 | 1.513155 | +15.91% | +11.19 to +18.80% | Pass |
| 17 | 4 | 1.728570 | 1.511392 | +14.37% | +12.36 to +16.58% | Pass |
| 35 | 4 | 1.714230 | 1.510672 | +13.47% | +11.58 to +14.44% | Pass |

One-row gate/up/SwiGLU falls from 0.27733-0.27766 ms to
0.22116-0.22127 ms. Isolated prepared down remains about 0.131-0.132 ms.
Stage sums need not equal complete timing because their cache histories differ.
Both arms are fused at rows 1 and 4, so four-row gains are not a new fusion
opportunity or server/batched-token throughput measurement. This is a clear
pass against the fixed 5% micro gate, not a borderline case needing extension.

## Representation, Memory and Setup

The candidate uses the existing lossless `SeparateScalesValues` representation.
FP16 scale bit patterns and signed Q8 bytes are copied without requantization.
Each lane loads four FP32 activations and four payload bytes from each weight
matrix. Per-weight FP32 scaling, one shared publication, final-warp reduction
and fused SwiGLU remain. Summation ownership changes, so neither unchanged
logits nor actual generated tokens can be inferred from unchanged precision.

Only the current layer's gate/up pair is prepared. It remains resident for
both timing arms; packed128 still reads original GGUF gate/up. Original
weights and qualified prepared down/vocabulary remain 3,279,519,744 and
1,192,935,424 requested device bytes, total 4,472,455,168. The added pair is
**47,906,816 bytes / 45.6875 MiB**, comprising:

- FP16 scales: 2,818,048 bytes.
- Signed payloads: 45,088,768 bytes.
- Retained host planes for verification: another 47,906,816 host bytes.
- Guarded activation/output scratch: 201,152 / 803,264 device bytes at rows 1/4.

Observed tracked totals are 4,520,563,136 / 4,521,165,248 bytes. Tracked payload
returns zero to zero after the model/pipelines drop. CUDA context/driver
residency is separate, and the final process query is empty after exit.

Across six separately prepared cases, median host packing is **30.615 ms**
and allocation/upload **7.950 ms**. These setup costs and full-plane readback
verification are outside warmup/primary intervals. There is no per-call or
per-iteration repacking. The per-case 128 MiB bound is checked before packing.

All 36 same-shaped gate/up pairs would add **1,724,645,376 bytes**. This is an
estimate, not observed whole-model residency. It would bring source plus
prepared weight payload to 6,197,100,544 bytes before workspace/KV; runtime
integration must validate its actual allocation and budget behavior. The
microbenchmark does not demonstrate that full-model memory or setup cost.

## Protocol and Correctness

- Implementation/source `3420c332b180fb0f4836a64e5d64380468916290`, executable
  SHA-256 `a616cd1f5d7967e83a05d026e57e9d873223919cddae95953692e373936440bc`.
  Source was clean before the run; the manifest lists only its newly created
  JSONL report as untracked. Rust 1.95.0, NVCC 12.8.93, CUDA `-O3`, sm_52
  SASS plus compute_52 PTX, no fast math, cuBLAS enabled.
- Original Qwen2.5-3B Q8_0 model SHA-256
  `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
  Frozen real calibration inputs, source identities and original layers
  0/17/35 are unchanged. Rows 1/4 are genuine distinct captured inputs.
  Native remains an independent numerical reference, not the timing control.
- Eighteen native/control/candidate validations pass the unchanged
  `1e-3 + 2e-5 * abs(reference)` CPU bound. Candidate maximum absolute errors
  are 2.07e-6 gate, 3.41e-6 up, 1.72e-5 SwiGLU and 5.01e-5 completed down.
  Maximum native-reference differences are 0.00031281 hidden and 0.00039673
  completed output. Candidate capture and normal SwiGLU bits match. Full
  readback confirms prepared planes remain byte-identical.
- Thirty pairs cover 180 non-overlapping primary intervals. Arm/stage order
  alternates, with bounded 1000 ms control / 100 ms candidate GPU warmup.
  All sixty clock boundaries show P0, SM 1113 MHz, memory 3004 MHz and 35-41 C.
  Only benchmark PID 1192568 appears. No tests, builds, MemPalace embeddings,
  Ollama or other inference work overlaps timing. Exit is zero; a final
  cuBLAS-enabled validation rerun occurs afterward, not during primary work.
- Timed enqueue has zero allocation/free, H2D/D2H transfer, cuBLAS call or
  stream synchronization. Per ten repetitions, launches are 10/10/20 for
  gate-up/isolated-down/complete. Stream waits are 0/0/0 at one row and
  10/10/20 at four rows. One event completion wait follows each repeated stage.
- Generated sm_52 code has 128-bit activation and 32-bit payload loads.
  Both candidate and control use 32 registers, no stack/local storage,
  1024 bytes of dynamic shared memory, one barrier site and ten shuffle sites.
  Compiler unrolling/prologues make whole-function instruction-site counts
  different from dynamic counts. No measured DRAM bandwidth or occupancy is
  available; the [CUPTI limitation](2026-10-25-maxwell-counter-capability.md)
  remains, including the prior administrator retry.
- Release CUDA primitive 2/2 and benchmark 12/12 tests pass with cuBLAS on/off.
  Enabled affected attention/forward/projection/MLP/runtime tests pass 46/46;
  warning-as-error non-CUDA tests pass 208/208. CUDA/server all-target Clippy
  and formatting pass. Coverage includes signed extremes, zero scales, tails,
  alignment rejection, output guards, immutable planes, scratch reuse and
  unchanged qualified down/vocabulary cache metadata.

## Evidence and Next Step

- [Raw JSONL](2026-10-25-maxwell-split-ffn-screen.jsonl),
  [stderr](2026-10-25-maxwell-split-ffn-screen.log.gz),
  [paired audit](2026-10-25-maxwell-split-ffn-audit.json),
  [memory audit source](2026-10-25-maxwell-split-ffn-memory-audit.jq) and
  [memory/setup summary](2026-10-25-maxwell-split-ffn-memory-audit.json).
- [Shared paired audit](../../scripts/audit_ffn_followup.jq) checks source,
  validations, pairs, ordering, counters, medians, gates, clocks and isolation.
  Use source `3420c332b180fb0f4836a64e5d64380468916290`, candidate
  `split-f32x4-128`, this report path and PID `1192568`. Wrong control,
  timed allocation, fabricated median and omitted extra-memory negative
  controls are rejected. Shared audit source is unchanged from prior results.
- [sm_52 resources](2026-10-25-maxwell-split-ffn-resources.txt),
  [SASS](2026-10-25-maxwell-split-ffn-sass.txt.gz),
  [static instruction sites](2026-10-25-maxwell-split-ffn-instructions.txt),
  [CUDA tests](2026-10-25-maxwell-split-ffn-cuda-tests.log.gz),
  [cuBLAS-disabled tests](2026-10-25-maxwell-split-ffn-no-cublas.log.gz),
  [affected regressions](2026-10-25-maxwell-split-ffn-regressions.log.gz),
  [non-CUDA tests](2026-10-25-maxwell-split-ffn-noncuda.log.gz),
  [Clippy](2026-10-25-maxwell-split-ffn-clippy.log.gz) and
  [artifact hashes](2026-10-25-maxwell-split-ffn.sha256).

Next add immutable model-local selection and cached ownership for this one
qualified geometry, with explicit unsupported/budget errors and no per-token
preparation. Preserve prepared down/vocabulary and the qualified dequant-SGEMM
prefill path; adding gate/up to the cache must not accidentally change prefill
dispatch. Validate actual full-model allocation, then compare actual 512 IDs
and the frozen dense-valid tasks against the current packed128/SGEMM runtime.
Only passing quality permits incremental controlled HTTP qualification and a
fresh Ollama comparison. Do not substitute the old native control alone for
that incremental comparison. No global default or original objective changes.

# Native QKV Warp Reduction Screen

## Decision

The one-barrier warp finish passes the **complete-QKV micro gate**:
one-request throughput improves **27.47-27.76%**, with positive paired
intervals, unchanged control bits and no extra storage. Keep `499d2de` as
a positive benchmark candidate, not a failed numerical/kernel experiment.

It does **not** pass the independently prespecified standalone runtime
feasibility gate. With the qualified profile's 11.72% QKV share, the
optimistic whole-token gain is only **2.59-2.61%**, below 3%. That is a
model-based ceiling, not measured HTTP throughput. Do not lower the gate,
integrate it alone, or call this a runtime win. No actual-512 quality or
HTTP run was performed for this candidate.

The last qualified split-FFN/SGEMM runtime remains 45.075 HTTP tok/s versus
official Ollama 43.119; both original goals, 53.06 and 1.30x fresh Ollama,
currently 56.055, remain unmet. Runtime defaults are unchanged.

## Fixed Setup And Paired Results

Clean release source `499d2dea3988ec4418856c6d779cec75d991cd85`, executable
SHA-256 `04aab98908b357e522b4aaefecf96a0969459b37eef2919ee796af5b0dc4a04b`.
The [pre-implementation hypothesis](2026-10-25-maxwell-qkv-warp-hypothesis.md)
fixes weights, scalar FP32-activation math, split4 assignment, optional
biases, one output column per block and 256 threads. Only reduction changes.

Use the original frozen M02 environment, corpus binding and inputs in the
existing shared runner:

```bash
source scripts/dev-env.sh
while IFS=$'\t' read -r key value; do
  export "$key=$value"
done < <(head -n 1 docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl |
  jq -r '.execution_configuration.environment | to_entries[] | [.key,.value] | @tsv')
target/release/examples/q8_execution_bench \
  --model "$MODEL" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --qkv-f32-warp --rows 1,4 --trials 5 --iterations 10 \
  --output /tmp/new-qkv-warp-screen.jsonl
```

Real layer 0/17/35 Q/K/V tensors and biases are used. Q shape is
2048x2048, K/V each 2048x256. Frozen actual normalized **FFN** vectors are
projection surrogates, not newly captured attention-norm inputs. This
distinction is in the JSONL manifest. Rows 4 uses four distinct input
vectors and four independent single-token fused launches, not one batched
GEMM, four copies of one vector, or a prefill benchmark.

Each latency is the median of five alternating pairs, ten repeated groups
per arm. Gain is control/candidate minus one; paired bootstrap uses 10,000
resamples and seed zero. Rows-4 latency is for the four-request group;
JSONL also reports per-request time.

| Layer | Requests | Original ms | Warp ms | Throughput gain | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.069056 | 0.054051 | +27.76% | +23.00% to +31.92% |
| 17 | 1 | 0.068851 | 0.054013 | +27.47% | +24.85% to +27.87% |
| 35 | 1 | 0.068867 | 0.053955 | +27.64% | +21.12% to +30.97% |
| 0 | 4 | 0.275869 | 0.217677 | +26.73% | +25.50% to +27.62% |
| 17 | 4 | 0.275347 | 0.217747 | +26.45% | +23.63% to +26.63% |
| 35 | 4 | 0.275405 | 0.217334 | +26.72% | +26.16% to +27.05% |

One-request latency falls about 21.6-21.7%; that is not a 27% latency
reduction. The point gains and all intervals clearly pass the stage's
5% screen. The separate profile-weighted gate fails, so no further trial
extension or geometry search is used to promote it.

## Exactness, Instructions And Memory

All six cases preserve Q/K/V output bits against the original shared-tree
control, including the real Qwen biases. Guard regions and original input,
weight and bias spans remain unchanged. Maximum CPU absolute errors are
9.0029e-7 Q, 3.8485e-6 K and 1.1768e-6 V, within the existing scalar
FP64-reference tolerance. The primitive tests additionally cover signed
extremes, zero scales, unequal/tail widths, absent/mixed/all biases,
explicit producer waits on both streams, invalid arguments and cleanup.

The release sm_52 candidate has one barrier site and five `SHFL.DOWN`
instructions. The control has one publication barrier plus a loop barrier
executed for eight strides at 256 threads. Both use 32 registers, no
reported stack/local storage and 1024 dynamic shared bytes. Control SASS
remains byte-identical, SHA-256
`b40f55cf110eeff0d125b246b1e8fc54fd517e12cd98d1ca51677f7d9dca509f`.
This proves the reduction change and measured latency difference, not
measured DRAM traffic or occupancy.

Source weights remain 3,279,519,744 bytes; prepared down/vocabulary remains
1,192,935,424 bytes, with no QKV or gate/up preparation in this isolated
probe. Guarded scratch is 18,688 / 73,984 bytes for one/four requests.
Tracked totals are 4,472,473,856 / 4,472,529,152 bytes. No KV or quantized
activation storage is allocated. Both arms retain the same allocations;
primary intervals have no allocation, free, copy, cuBLAS call, stream wait
or stream synchronization. Actual operation-name launch deltas are checked,
not just arm labels. Event completion after each group is included in GPU
elapsed time but excluded from enqueue counters.

Tracked payload returns zero to zero. The final in-process sample still
includes driver context memory; the subsequent compute-process query is
empty after exit. This is not a zero-driver-RSS or peak-VRAM claim.

## Validation And Evidence

CuBLAS on/off each pass two native-QKV tests, one retained prepared-QKV test
and thirteen shared-runner tests, including the new adapter's dispatch,
numerical, buffer-reuse and cleanup test. Warning-as-error non-CUDA tests
pass 210 cases; CUDA/server all-target Clippy, formatting and 52 affected
CUDA FFN/runtime/attention/forward/projection tests pass. Any optional
external real-Qwen fixture was unconfigured, not full-generation evidence.
The initial build-only adapter type errors were corrected before these
checks and before committing; they produced no timing or numerical sample.

Thirty pairs / sixty primary intervals are nonoverlapping. Sixty GPU
boundary samples show only process 1405396, P0 and 1113/3004 MHz clocks.
No other GPU workload, build, tests or MemPalace embedding ran during timing.
The benchmark handle is terminal and the GPU is empty after exit.

- [Raw JSONL](2026-10-25-maxwell-qkv-warp-screen.jsonl), [audit](2026-10-25-maxwell-qkv-warp-audit.json) and [auditor](2026-10-25-maxwell-qkv-warp-audit.jq)
- [SASS](2026-10-25-maxwell-qkv-warp-sass.txt.gz), [compiler resources](2026-10-25-maxwell-qkv-warp-resources.txt), [log](2026-10-25-maxwell-qkv-warp-screen.log.gz) and [checksums](2026-10-25-maxwell-qkv-warp.sha256)

The auditor recomputes actual-pair medians and both gates, binds source and
executable, validates QKV identities/biases, checks actual dispatch and
memory/cleanup/isolation, and rejects altered dispatch, timed allocations
and fabricated E2E ceilings. Historical evidence remains unchanged.

## Next Independent Opportunity

Retain this positive stage result for a future combined exact candidate,
not standalone integration. Current native output projection is 1.773 ms /
8.25% of the [qualified middle-token profile](2026-10-25-maxwell-split-stage-profile.md).
Its selected `gemm_f32xq8_0_gguf_f32_decode_coltile4_split4_kernel` also uses
a publication barrier and eight shared-tree barriers, reducing four output
columns in parallel at 256 threads. A separately scoped exact-tree warp
finish can be tested without changing the four-column mapping, loads,
arithmetic, raw layout or prepared down/head. Only if that independent
stage passes its numerical/micro gate and the combined profile-weighted
ceiling clears 3% should a combined opt-in runtime undergo actual-token
quality and measured HTTP qualification. Do not retune this QKV geometry,
revive losing prepared layouts or lower either original whole-engine target.

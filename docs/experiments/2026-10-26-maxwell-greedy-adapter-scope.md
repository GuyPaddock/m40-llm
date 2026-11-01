# Synthetic Greedy Logits Have Little Measured Scope

## Decision

The host adapter in `generate_text` is unnecessary, but its isolated CPU
cost is only **6.072 ms per actual512 trace**, about **11.86 us/token**.
Across five pairs, the range is **6.046-6.161 ms**. Direct forwarding costs
about **0.000394 ms per trace** in this diagnostic. This does not justify
runtime work as the next performance candidate, or combining rejected
kernel changes to manufacture an E2E gain. Keep `a2b8458` diagnostic-only.

The qualified HTTP interval is **10,417.843 ms**. The adapter's observed
median/max costs are **0.0583% / 0.0591%** of that interval, far below the
unchanged 3% promotion floor. These ratios are screening context, **not an
inference speedup forecast or rigorous bound**. The CPU replay has different
cache/scheduling history from interleaved generation, and CPU/GPU work can
overlap. No new HTTP measurement, runtime optimization or default change
is claimed.

The [prespecified diagnostic](2026-10-26-maxwell-greedy-adapter-hypothesis.md)
measures the existing production sampler: after CUDA argmax, generation
allocates `token_id + 1` FP32 logits filled with negative infinity, writes
zero at the selected index and scans them with CPU greedy sampling to
recover the original token. The benchmark does not include CUDA argmax,
device-to-host token transfer, forward execution or output decoding.

## Fixed Release CPU Measurement

Clean source `a2b8458c02c5d9c13b6e934b9252b9753f9b2897`, executable SHA-256
`546c86757ce194fbe1d44f1da6d85970aa8fd222d6b485f706ddb447f83d946a`.
Rust 1.95.0 release; Intel Xeon E5-2697A v4, pinned to allowed CPU 0.
CPU affinity is not exclusive ownership; frequency and system interference
are not measured. GPUs are explicitly hidden with `CUDA_VISIBLE_DEVICES=-1`.
No model, cache search, CUDA context or GPU workload is constructed.

The input is the qualified packed-projection quality report, SHA-256
`8b499787179f484aeb32e37caf837208e0994da4995918e18e34ba922d3d949e`.
All 512 actual candidate IDs are replayed in their original order. This is
not a worst-case high-index token repeated. Five pairs alternate arm order,
with **256 full-trace repetitions per arm**, after untimed individual-token
verification and two warmup traces. Inputs/results are black-boxed. All
ten measured intervals are serial and preserve the expected token sum.

| Pair | Synthetic adapter ms/trace | Direct forwarding ms/trace |
| --- | ---: | ---: |
| 0 | 6.106116 | 0.000395 |
| 1 | 6.046303 | 0.000393 |
| 2 | 6.068216 | 0.000394 |
| 3 | 6.161277 | 0.000394 |
| 4 | 6.072234 | 0.000634 |

The synthetic vectors request **18,854,388 payload bytes per trace** in
aggregate; the largest individual request is **364,808 bytes**. These are
computed requested `Vec<f32>` payloads, not measured allocator traffic,
allocator-rounded storage, peak memory or VRAM. Tracked device bytes stay
zero. Session **5243** terminates zero; the explicitly included ignored
release diagnostic passes without accessing a GPU.

Behavioral tests pass **2/2** for individual zero/high vocabulary IDs,
real sampler behavior and rejection of stale/unqualified/truncated/invalid
traces. The release timing case is ignored in ordinary test invocation,
not counted as a pass there. Non-CUDA target compilation with warnings
denied passes and runs **zero tests**, as expected for this CUDA-gated
target. CUDA/server all-targets Clippy and formatting pass. No CUDA kernel
or inference behavior changed, so GPU numerical suites are not claimed
as rerun for this CPU diagnostic.

The independent audit checks source/build identity, actual trace IDs,
requested payload arithmetic, fixed repetitions, alternating arms, token
checksums, elapsed/per-trace arithmetic, serial timestamps, hidden GPUs and
final tracking. Nine corruption controls are rejected. Qualified HTTP
context is independently recomputed from its five primary M40 requests,
not taken from rounded prose or pooled with these CPU observations.

## Pinned Upstream Check

The pinned `9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e` source does not
provide an alternate Q8/FP32-activation matvec through `mmvf.cu`: its
dispatch switches at lines 706 and 765 accept F32/F16/BF16 weights.
`mmvq.cu` maps Q8_0 to `vec_dot_q8_0_q8_1`; that dot takes Q8_1 activation
blocks, not original FP32 activations. This source inspection does not
invent another executed M40 kernel or attribute measured speed to it.
No upstream implementation is copied. See the existing
[source/dispatch and license record](2026-10-25-maxwell-q8-dispatch.md) and
[additional file hashes](2026-10-26-maxwell-greedy-adapter-upstream-sources.sha256).

## Reproduction And Evidence

Source `scripts/dev-env.sh`, request test escalation, build the release
`qwen_decode_throughput` target, then run only the CPU diagnostic on one
allowed core. Use a new output path; no model input is needed:

```bash
cargo test --release --features cuda --test qwen_decode_throughput --no-run
CUDA_VISIBLE_DEVICES=-1 \
M40LLM_GREEDY_ADAPTER_TRACE=docs/experiments/2026-10-25-maxwell-packed-projection-runtime-quality.jsonl \
M40LLM_GREEDY_ADAPTER_REPORT=/tmp/new-greedy-adapter-cost.jsonl \
taskset -c "$ALLOWED_CPU" cargo test --release --features cuda \
  --test qwen_decode_throughput \
  greedy_adapter_cost::test_greedy_adapter_cost_from_frozen_tokens \
  -- --exact --include-ignored --nocapture --test-threads=1
```

- [Raw pairs and IDs](2026-10-26-maxwell-greedy-adapter-cost.jsonl), [audit](2026-10-26-maxwell-greedy-adapter-audit.jq), [derived costs](2026-10-26-maxwell-greedy-adapter-audit.json), [corruption controls](2026-10-26-maxwell-greedy-adapter-negative-controls.sh) and [checksums](2026-10-26-maxwell-greedy-adapter.sha256).
- [Run](2026-10-26-maxwell-greedy-adapter-run.log.gz), [build](2026-10-26-maxwell-greedy-adapter-build.log.gz), [behavioral tests](2026-10-26-maxwell-greedy-adapter-tests.log.gz), [non-CUDA compilation](2026-10-26-maxwell-greedy-adapter-noncuda.log.gz), [Clippy](2026-10-26-maxwell-greedy-adapter-clippy.log.gz) and [negative controls](2026-10-26-maxwell-greedy-adapter-negative-controls.log.gz) logs.

Exact qualified performance remains **49.146 HTTP tok/s** versus **43.106
official Ollama**. Both original requirements, **53.06** and **56.037
HTTP tok/s**, remain unmet. Next work needs independently supported scope
in dominant execution costs, not this tiny host adapter or closed probes.

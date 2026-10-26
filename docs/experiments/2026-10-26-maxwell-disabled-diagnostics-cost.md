# Disabled Forward Diagnostics Cost Is Below Standalone Scope

## Decision

The actual disabled forward diagnostic wrapper costs a median **246.311 ms
per nominal 512-token replay**, or **481.08 us per nominal token**. This
observed CPU cost is **2.3643%** of the qualified 10417.843301 ms HTTP
interval; the largest trial is **2.4283%**. It misses the prespecified 3%
standalone cost threshold. Keep `a592deb` test-only without a runtime fast
path, logging/environment cache or combination with closed kernel probes.

This is not a measured E2E speedup or an inference-latency bound. CPU work
overlaps GPU execution. The replay includes no model, actual generation,
CUDA enqueue or profile counter recording. Stage labels are preconstructed,
not dynamically recreated per layer/token. Repeating one middle-token
operation shape 512 times does not reproduce a real 512-token execution.
Its purpose is to measure this wrapper's cost, not total host overhead.

Qualified performance remains **49.146 HTTP tok/s** versus **43.106 official
Ollama**, not rerun. Both **53.06** and **56.037 HTTP tok/s** requirements
remain unmet. No projection, KV/default, prefill, attention or scheduling
behavior changes.

## CPU Measurements

Intel Xeon E5-2697A v4, Rust 1.95.0 release. Core affinity is exactly CPU 0;
the core is not exclusive and CPU frequency is not controlled. After fixed
warmups, five pairs alternate traversal-only and actual-wrapper order.
Each arm contains eight complete replays. Raw nanosecond totals are retained
and divided by eight for this table.

| Trial | Order | Traversal ms/replay | Disabled wrapper ms/replay |
| --- | --- | ---: | ---: |
| 0 | traversal, wrapper | 0.074256 | 246.311202 |
| 1 | wrapper, traversal | 0.078343 | 243.134312 |
| 2 | traversal, wrapper | 0.074026 | 252.389618 |
| 3 | wrapper, traversal | 0.079510 | 252.977191 |
| 4 | traversal, wrapper | 0.081340 | 241.640699 |

Median traversal is **0.078343 ms/replay** and median paired excess is
**246.236946 ms/replay**. Each arm traverses **1,622,016 operation boundaries**
and returns checksum **321,970,176**. The wrapper arm calls the real private
`log_profiled_op`, `profile::snapshot_if_enabled`, and CPU clock pair, with
all timing/profile logs disabled. The function still formats two labels
and queries environment settings. There is no alternative implementation.

No confidence interval or E2E gain is inferred from this CPU timing.
All five measured costs are below 3%; this is not a borderline runtime
promotion study. The known GPU profile's roughly 6 ms host enqueue interval
must not be treated as wholly removable or added to GPU duration.

## Evidence And Validation

Contract `48bd684`; clean source `a592debdfd46f974f84acace7256451b3adc0ed6`.
Executable SHA-256
`dd18d1521904cacf4c166f018feeb4ca4126a487477a044c202dec1007265e06`.
The qualified profile SHA is
`4eeec0020ed346341cfd816cc3ef28ae1e98ae88f2b7a6f24a378b75927b126c`.
Session 2, step 255, absolute position 319 supplies exactly 36 ordered
layers with eleven stages each. Non-layer operations are deliberately
excluded. Input SHA, stage order and multiplicity are checked before timing.

Two behavior tests pass: exact traversal work is preserved and incomplete,
duplicated or reordered coverage is rejected. The timing test is ignored
by default and explicitly included once in release. Session **54914**, PID
**3418174**, exits zero after 11.93 seconds. CUDA visibility is `-1`; no model
or CUDA context is constructed. Tracked bytes and raw counter maps remain
zero/empty before and after. Ten primary CPU intervals do not overlap.
No other build, test, inference or MemPalace activity overlaps timing.

Independent audit verifies frozen coverage, both arm checksums, ordering,
raw duration arithmetic, clean identities, affinity and unchanged counters.
Ten corruptions are rejected. Non-CUDA warning-as-error tests pass **217**,
CUDA/server all-targets Clippy and formatting pass. CUDA numerical tests
are not rerun for this test-only code; the preceding shared-scale checkpoint
retains their latest recorded results. No quality improvement is claimed.

## Reproduction

Source `scripts/dev-env.sh`, build the release library tests with CUDA/cuBLAS
enabled, then run the emitted executable with its actual path:

```bash
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --release --features cuda --lib --no-run
env -u M40LLM_LAUNCH_LOG -u M40LLM_SYNC_LOG -u M40LLM_COPY_LOG \
  -u M40LLM_PROFILE_LOG -u M40LLM_TIMING_LOG \
  CUDA_VISIBLE_DEVICES=-1 \
  M40LLM_DIAGNOSTICS_COST_TRACE=docs/experiments/2026-10-26-maxwell-packed-stage-events.jsonl \
  M40LLM_DIAGNOSTICS_COST_REPORT=/tmp/new-disabled-diagnostics-cost.jsonl \
  taskset -c 0 target/release/deps/m40_llm-0218b30a8d8dc496 \
  infer::forward::diagnostic_cost::test_disabled_diagnostic_cost_from_frozen_operations \
  --exact --include-ignored --nocapture --test-threads=1
```

The output must not already exist. Do not time a debug executable or run
other work concurrently. This command does not access a model cache.

- [Raw report](2026-10-26-maxwell-disabled-diagnostics-cost.jsonl), [audit](2026-10-26-maxwell-disabled-diagnostics-audit.jq), [derived results](2026-10-26-maxwell-disabled-diagnostics-audit.json), [corruption controls](2026-10-26-maxwell-disabled-diagnostics-negative-controls.sh) and [checksums](2026-10-26-maxwell-disabled-diagnostics.sha256).
- [Run](2026-10-26-maxwell-disabled-diagnostics-run.log.gz), [release build](2026-10-26-maxwell-disabled-diagnostics-build.log.gz), [behavior tests](2026-10-26-maxwell-disabled-diagnostics-behavior.log.gz), [non-CUDA](2026-10-26-maxwell-disabled-diagnostics-noncuda.log.gz) and [Clippy](2026-10-26-maxwell-disabled-diagnostics-clippy.log.gz) logs.

Next requires independent dominant execution-cost evidence. Do not turn
this small observed CPU cost into a runtime optimization claim.

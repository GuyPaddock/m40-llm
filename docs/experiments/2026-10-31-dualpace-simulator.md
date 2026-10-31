# Cooperative decode simulator checkpoint

The CPU-only `cooperative_decode` example implements the
[bounded scheduling contract](2026-10-31-dualpace-audit.md). It compares
GPU-only execution, independently progressing CPU/GPU requests, and a fixed
dense-assist policy. This is tested scheduling tooling. CPU operator costs,
transfer measurements and the priced feasibility decision remain pending.

## Run and input meaning

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/dualpace/target
cargo run --no-default-features --locked --example cooperative_decode -- \
  --trace docs/experiments/2026-10-31-dualpace-synthetic.json \
  --output target/cooperative-synthetic-new.jsonl
```

The [synthetic input](2026-10-31-dualpace-synthetic.json) documents schema 1.
Its nanosecond durations and byte counts are hand-calculation fixtures, not
hardware timings or estimates for this host. Output files must be new.

Each request supplies its decode-ready timestamp, frozen ownership, final
host/GPU KV footprints, and one named profile for each token position. Profiles
contain ordered model stages, each with positive integer CPU/GPU nanoseconds
and cost evidence. Changing the profile across positions models context growth.
An optional assist entry fixes that stage's assisted routing and charges H2D,
D2H, transfer bytes and final CPU synchronization overhead. The GPU operation
uses the same stage's GPU cost. That cost must include relevant submission and
finalization service; synchronization cost excludes GPU execution and queue
waiting, which the simulator already charges separately.

Evidence is either an explicit assumption or a described artifact with a path
and SHA-256. Artifact paths resolve relative to the trace file. The CLI verifies
artifact bytes before opening an output. It does not extract cost values or
prove that declared durations agree with the cited measurements. The input
trace, its digest and executable digest are retained in the manifest. Every
report identifies itself as simulation and records zero generated model tokens.

The CPU thread count and NUMA node identify the supplied cost profile; the
simulator does not pin a worker or extrapolate performance across thread counts.
It schedules individual requests, not dynamic batches. Fixed memory must include
resident weights, scratch, transfer buffers and retained copies. Capacity uses
the sum of all admitted requests' final KV footprints, even for non-overlapping
arrivals. There is no eviction or migration to rescue an over-budget scenario.

## Scheduling and reporting

One CPU worker group owns a token through all stages and assist waits. One GPU
main request owns a token, with assistance taking priority between stages.
Neither preempts a running operation. The shared copy resource is half-duplex;
it may overlap compute on a different resource. Inputs must therefore price a
transfer path that supports that overlap. Isolated profiles do not predict
shared memory-bandwidth contention or host launch-thread interference.

All events completing at one timestamp become visible before dispatch. Finished
tokens rejoin their engine's FIFO using completion time; ties use input order.
The next token cannot start before the current one completes. Initial prefill
and KV migration must have completed before a request's ready timestamp.

Outputs retain every resource interval, queue wait, copy byte count and token
completion. CPU compute time and CPU-worker occupancy are separate, so waiting
for GPU assistance does not become free CPU capacity. Throughput covers the
first ready timestamp through the last completion, including intervening idle
time. ITL pools successive completions within each request and uses nearest-rank
p50/p95/p99. First-decode latency is reported separately. A single-token request
has no ITL sample; this tool does not report prompt TTFT.

## Validation

Nine behavioral tests pass. Hand calculations cover independent progress,
non-preemption, assist priority, simultaneous completion, CPU ownership during
assist waits, FIFO token fairness, context-dependent costs, arrivals, memory
limits and overflow. A workload grid verifies causal completion and non-overlap
on each resource. Invalid or unpriced costs and unknown schema fields fail.

The CLI fixture gives completion windows of 40, 24 and 30 synthetic nanoseconds
for GPU-only, independent and assisted scheduling respectively. In the assisted
case the CPU request finishes at 23 instead of 24, while GPU completion moves
from 20 to 30. This checks that CPU acceleration can reduce total throughput.
It is not an M40 break-even result.

All 220 non-CUDA regression tests pass with warnings denied. Non-CUDA all-target
Clippy and formatting checks pass. CLI checks confirm the calculated outputs,
reject an existing output without changing its digest, and reject a bad cost
artifact hash before creating output. No GPU tests or hardware timings ran for
this CPU-only unit. The active lossless KV capture retains exclusive GPU use.

Raw logs and the synthetic output are compressed alongside this report; the
[checksum manifest](2026-10-31-dualpace-simulator.sha256) covers all six artifacts.
The common five-pair performance gate does not apply to deterministic event
simulation. It remains required for subsequent measured performance comparisons.

Next: measure the bounded AVX2 primitive and, after GPU capture completion,
transfers and synchronization. Then construct a sourced scheduling workload
and explicitly price the missing full CPU decode path. This branch is not yet
complete, and no server or runtime default changes follow from the simulator.

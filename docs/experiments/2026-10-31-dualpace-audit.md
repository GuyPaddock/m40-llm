# DualPace source audit and bounded simulation contract

The full-model CPU decode prerequisite is missing. The current study will
price that prerequisite and model scheduling before proposing a server change.
No CPU or GPU performance result is established by this audit.

## Source and local execution inventory

[DualPace v1](https://arxiv.org/html/2610.33501v1), posted October 24,
uses an independently scheduled AMX CPU engine. CPU-owned requests retain
host KV even when the GPU executes selected dense stages. Assistance runs
at non-preemptive GPU stage boundaries. Its routing score charges both CPU
completion time and GPU service displaced by assistance. Host admission
reserves capacity for growth and migration as well as materialized KV.
These mechanisms require complete CPU-local decoding; an offloaded attention
operator alone cannot provide independent request progress.

Local inspection at `2c71b21`, based on `0eee578`, finds:

| Component | Available behavior | Missing requirement |
| --- | --- | --- |
| Host | Two Xeon E5-2697A v4 sockets, 16 cores each, two threads/core; AVX2/FMA/F16C, no AMX in `lscpu` flags | AMX implementation cannot execute here |
| NUMA | Node 0 CPUs 0-15/32-47; node 1 CPUs 16-31/48-63 | Affinity and first-touch placement must accompany CPU timings |
| `src/infer/forward.rs` | A non-CUDA scalar single-layer reference exists | `forward_one_token_all_layers_for_sequence` explicitly errors without CUDA buffers |
| `src/infer/backend.rs` | Chooses fast-fits or large-model CUDA projection paths | It does not select an independent CPU execution engine |
| Production Rust operators | No explicit AVX2 implementation found under `src/` | One bounded AVX2 projection probe, not a new runtime backend |
| Server scheduling | Existing GPU decode/session ownership | CPU-owned KV, complete CPU generation and stage-boundary assist dispatch are unqualified |

The scalar reference must not be described as a complete CPU decoder. A
successful non-CUDA build does not establish CPU inference capability.

## Implementation sequence

1. `feat(bench): model cooperative scheduling`: CPU-only discrete-event
   simulator and behavioral tests. Consume explicit per-token stage costs
   and arrival traces. Preserve cost provenance and mark every output as a
   simulation, even when component timings are measured.
2. `feat(bench): price AVX2 decode primitives`: one bounded projection probe
   on retained Qwen weight/activation fixtures, with scalar numerical control.
   Record thread count, affinity, NUMA placement, bytes and setup. Measure
   transfers and synchronization serially after the active KV capture exits.
3. `docs(perf): record cooperative decode results`: replay the frozen workload
   using measured primitives and explicit missing-cost sensitivity. Report
   the break-even conditions and a priced prerequisite list. Do not implement
   a CPU backend or modify the server scheduler in this study.

## Simulator contract

Compare GPU-only, independent CPU/GPU, and CPU/GPU with selective dense assist
on the same post-prefill request trace. Trace time zero is a decode-ready
boundary. Prompt processing and initial KV migration are excluded and must
already have completed; outputs are not TTFT or whole-request serving results.

One CPU worker group holds one active request through a token, including its
assist waits. One GPU main request holds its token across native stages;
ready assists take priority only between stages. One shared, half-duplex copy
resource charges explicit transfers in both directions. Synchronization
consumes CPU-worker time. A request starts its next token only after its
previous token completes. There is no kernel preemption, dynamic batching,
request migration or invented compute/copy overlap on the same resource.

Use explicit stage-cost profiles per token so context growth can change the
costs. Routing is frozen in the input, not optimized against observed output.
The primitive study covers distinct rows at batches 1/2/4/8. The first simulator
models individually scheduled requests; batched primitive costs must not be
substituted as single-request costs without an explicit workload extension.

Reserve each request's final KV footprint plus fixed resident weights and
scratch against the appropriate memory budget. Reject a scenario that cannot
hold all admitted requests at their finish peak. This conservative admission
has no eviction or oversubscription mechanism. CPU workers and NUMA node are
input metadata; separate measured profiles are required for different thread
or NUMA settings. Memory bandwidth interference is not predicted by isolated
primitive timings and remains a limitation.

Output all resource intervals, token completion times, queue waits, resource
busy time, transfer bytes, simulated throughput, first-decode latency and
p50/p95/p99 inter-token latency. Single-token requests have no ITL sample.
Include input hashes and cost evidence. Reject non-finite/negative costs,
missing profiles, duplicate request identities, invalid memory accounting and
unbounded workloads. Never turn a modeled gain into a measured performance
gate or a claim of numerical/model correctness.

## Current checkpoint

The lossless KV capture owns the M40. This audit uses source inspection and
CPU inventory only. Simulator, primitive measurements and priced decision are
pending. The nine-topic goal remains active; six other bounded studies are
complete. Runtime defaults and the separate drafter admission remain unchanged.

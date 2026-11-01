# Cooperative scheduling cost scenarios

The remaining study will combine measured projection and completed-transfer
costs with explicit assumptions for the missing CPU decoder. It can establish
conditions under which cooperation would pay, but cannot qualify a scheduler
or estimate the performance of an implemented CPU engine. Freeze the following
workload before collecting CPU and transfer timings.

## Retained GPU costs

The completed ShiftAdd study measured native and prepared Q8 projections on
the same nine original Qwen matrices and captured input rows used by the CPU
probe. Its source revision is `2c8e985`. Retain the original compressed JSONL
as `2026-10-31-dualpace-gpu-source.jsonl.gz`; its bytes must match the source
digest `e2d107a2457ba4089e0a5e5049fcefad98a68e15d0a99ae767f43a81111ee610`.
The separate study's candidate format is irrelevant to these control costs.

The [cost audit](2026-10-31-dualpace-gpu-costs.jq) checks source identities,
all nine weight records, the 36 before/after validations and the 360 paired
Q8 control observations. It extracts 72 series covering both GPU paths and
batches 1/2/4/8, preserving input hashes, row positions, tensor identities,
raw event/submission/wall measurements and five-sample ranges. Verify the
compressed digest before extraction; the jq program checks record semantics,
not the containing file's digest.

Use batch-one wall medians as nominal service costs. The 100-operation
intervals amortize completion and omit host/device copies. They are isolated
projection costs, not single-assist latency, fused FFN cost or current complete
decode performance. Native and prepared paths are separate sensitivity cases;
do not choose the winner per operation. Prepared residency must include its
extra weight planes. Never add submission or event time to wall time.

Match each CPU/GPU join on model, tensor bytes, layer, projection family,
dimensions, input digest and captured row identities. Missing or inconsistent
joins fail the scenario build. Batch 2/4/8 costs remain descriptive operator
results because the simulator schedules individual requests.

## Fixed scheduling workload

The [placement amendment](2026-10-31-dualpace-placement-amendment.md) replaces
unavailable CPU 31 with allowed CPU 30 before node-1 measurements. The original
failed launch and completed node-0 evidence remain preserved.

Use one CPU worker group with the measured single-core profile, first CPU
15/node 0 and then CPU 30/node 1. These are separate runs and scenarios.
No cross-socket parallel CPU throughput is inferred. Use schema 2 with copy
and GPU compute serialized, because the transfer probe measures completed
pageable APIs. CPU compute may proceed while the GPU runs, subject to the
unmodeled host interference below.

Each request has eight simulated generated tokens and is already prefilled
to 1,024 tokens. Reserve dense FP16 KV through position 1,032. The model has
36 layers, two KV heads and dimension 128, giving 36,864 bytes per stored
token and 38,043,648 bytes per request at the reserved endpoint. These bytes
are a geometry calculation, not measured allocation overhead. Fixed storage
separately includes weights, scratch, transfer buffers and retained copies.
Charge the 3,279,519,744-byte original model on both host and GPU. The prepared
GPU scenario additionally retains all 108 projection planes, calculated as
108 times 23,953,408 bytes. Do not substitute the source experiment's smaller
prepared cache, whose runtime families did not include every FFN matrix.
Reserve another 1 GiB on each processor for unimplemented workspace, transfer
buffers and overhead. This is an assumed implementation budget, not measured
peak use. Compare against the selected NUMA node's and M40's actual capacities.
Failure to implement within those budgets would invalidate memory admission.

Run three arrival/ownership traces:

1. One GPU-owned and one CPU-owned request, both ready at time zero.
2. Three GPU-owned and one CPU-owned request, all ready at time zero.
3. Two GPU-owned and two CPU-owned requests. One of each is ready at zero;
   the other two become ready after four nominal GPU token service intervals.

Ownership remains fixed across independent and assisted modes. GPU-only mode
executes the same arrivals and token counts on the GPU. Compare total simulated
throughput, first-decode latency, pooled p50/p95/p99 ITL and per-owner ITL.
Eight tokens produce few tail samples; report sample counts and do not treat
these percentiles as production latency estimates. Arrival offsets are derived
once from the unassisted GPU profile and shared by all three scheduling modes.

Use 36 ordered layer groups. Map layers 0-11 to measured layer 0, 12-23 to
layer 17 and 24-35 to layer 35. This mapping extrapolates costs from sampled
weights and must be marked as an assumption. Keep gate, up and down as separate
stages in layer order; they are complete projection service intervals.
Optional assistance targets down projection only, fixed before measurement.
It transfers one FP32 width-11008 vector to the GPU and one width-2048 vector
back, charging 44,032 H2D bytes and 8,192 D2H bytes per assisted layer.
No selection from observed timing gains changes this routing.

## Missing-work sensitivity

Full CPU attention, QKV/output/vocabulary projections, norms, activation,
sampling and end-to-end state ownership are unimplemented or unmeasured.
Do not give these operations a measured-looking constant. Use a nine-point
sensitivity grid: missing CPU work at 0, 1 and 4 times that profile's summed
108 projection costs; independently vary missing GPU work at the same three
multipliers of its own projection sum. Zero means an explicitly incomplete
projection-only lower-bound model, not a zero-cost full decoder.

Distribute each nonzero missing-work budget equally over the 36 layer groups,
before their projections, using exact integer division with the remainder
charged to the final group. This is assumed scheduling granularity; it does
not locate the real vocabulary/attention work. Omit missing-work stages for
zero budgets instead of inserting artificial one-nanosecond operations.
All eight token positions use the same nominal profile. This freezes costs
over a short context range; it makes no long-context growth claim.

This gives 108 nominal input traces: two NUMA placements, two complete GPU
path choices, nine missing-work combinations and three workloads. Retain all
three scheduling modes, including regressions and rejected memory admissions.
Use measured CPU, GPU and transfer minima/maxima for a second conservative
envelope only if nominal outcomes would otherwise justify proposing a scheduler
change. A decision that is already blocked by the missing CPU engine needs no
additional tuning sweep. Any new measured comparison still follows the common
paired timing protocol.

Completed H2D/D2H costs already include their completion waits. Additional
synchronization is zero only with that cited evidence. Price a stream handoff
only if a proposed implementation actually uses it. Isolated transfer costs
assume input readiness; they do not predict contention, asynchronous overlap
or CPU submission delay behind another token. The sourced simulation remains
conditional on a dispatcher that can submit ready stages promptly. Amortized
projection completion, memory-bandwidth interference and missing complete
CPU generation are explicit prerequisites, not newly measured benefits.

## Required decision report

Keep four evidence classes separate: measured operators, calculated storage,
assumed scheduling costs and missing engineering. Report local assistance
break-even as CPU down cost minus GPU down and completed transfer costs, before
queueing. Then report the simulator's queueing and displaced GPU work on the
same inputs. A positive local saving alone cannot justify a server change.

Price initial matrix/KV movement from the corresponding transfer cases, listing
bytes and operation counts. Do not extrapolate arbitrary byte sizes as measured
PCIe bandwidth. State required host/GPU resident weights and per-request finish
KV reserves. Include a separate list of unpriced prerequisites: full AVX2
generation, numerical/token validation, host/GPU state ownership, asynchronous
submission and measured concurrent contention. This study will not implement
those systems or estimate developer hours without evidence.

## Audit checkpoint and reproduction

The retained GPU source passes the cost audit, producing 72 five-pair cost
series. All 36 input cases and nine original weight identities match the
already validated CPU fixture. This comparison preserves both GPU paths;
it does not qualify CPU/GPU numerical or generated-token identity.

Two positive admission checks and eight corrupted-record checks pass. Missing
completion, changed model identity, failed guards/arithmetic, duplicate trial
indices, zero times, wrong launch counts and missing input rows all reject
without emitting a partial cost library. The tests use actual serialized
results, not assertions about implementation text.

```bash
bash docs/experiments/2026-10-31-dualpace-gpu-costs-check.sh
gzip -dc docs/experiments/2026-10-31-dualpace-gpu-source.jsonl.gz |
  jq -s -f docs/experiments/2026-10-31-dualpace-gpu-costs.jq
sha256sum --check docs/experiments/2026-10-31-dualpace-gpu-costs.sha256
```

The extracted output and validation log are retained as compressed artifacts.
No new hardware measurements ran for this audit. CPU timing, completed transfer
costs, generated simulator inputs and the final prerequisite decision remain
pending while the lossless capture owns the GPU.

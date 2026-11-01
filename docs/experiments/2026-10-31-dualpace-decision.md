# Cooperative scheduling does not justify integration

The bounded DualPace study is complete with a negative scheduler decision.
Measured single-core projections and completed transfers support 108 sourced
simulation inputs and all 324 scheduling modes. Independently progressing CPU
requests lose to GPU-only scheduling throughout this grid. GPU assistance
helps those CPU requests, but exceeds GPU-only throughput in only one
optimistic case, by 0.072%. A usable complete CPU decoder remains missing.
Keep the existing server scheduler and runtime defaults.

## Measured components

The CPU study uses original Qwen Q8 weights and distinct retained inputs for
gate/up/down at layers 0/17/35 and batches 1/2/4/8. CPU 15/node 0 and CPU
30/node 1 each complete all 36 numerical checks and 180 timing pairs. Every
series is stable without extension or confirmation. The scalar and AVX2
implementations satisfy the fixed `1e-3 + 2e-5 * abs(FP64 reference)` gate.
Maximum absolute errors are 0.000996087425 and 0.000148677535 respectively.
These are isolated single-core projections, not a complete CPU decoder.

| Placement | Scalar batch-one projection, ms | AVX2 batch-one projection, ms |
| --- | ---: | ---: |
| CPU 15, node 0 | 21.726 to 24.490 | 5.384 to 5.635 |
| CPU 30, node 1 | 23.685 to 24.514 | 5.370 to 5.655 |

The two runs use the same CPU executable SHA-256
`199d649c5f30da058ab9d487497ef58b167969f8a46a6b634ff9e6d5c1583659`.
Node 0 ran at `8171d7f`; node 1 ran at `93ded84` after the
[documented placement amendment](2026-10-31-dualpace-placement-amendment.md).
The first attempted CPU 31 launch was rejected by the environment before
measurement. Its raw failure and the valid node-0 result remain preserved.
Actual CPU affinity and local memory policy are recorded for each run.

Each projection observation contains three completed operations with an
untimed 128 MiB eviction pass before each operation. Matrix reads are logical
traffic estimates; no hardware DRAM counters establish complete eviction.
AVX2 reduction order differs from scalar. These numerical checks do not
establish generated-token or long-sequence equivalence.

## Completed transfers

The final transfer cohort uses clean source `4b7bb61`, both placements and
thirteen representative sizes. All payload, guard and cleanup checks pass.
Each of the 54 arm warmups per node lasts at least one wall-clock second
and three complete calls. Each transfer interval contains 100 completed
operations; idle synchronization uses 1,000 calls. Device timelines include
host gaps and are not active DMA time. Completed host wall cost is primary.

Node 0 records 190 pairs and eleven extensions; node 1 records 175 pairs
and eight extensions. All 26 directional-copy groups and both idle-sync
groups pass stability admission. Ten node-0 and eight node-1 stream-handoff
groups remain unstable. Their raw values and error labels are retained, and
no qualified point estimate is assigned to those handoff costs. They are
not inputs to the frozen model's serialized H2D/GPU/D2H assistance.

The [original transfer cohort](2026-10-31-dualpace-transfer-instability.md)
remains intact, including the required node-0 copy cost that failed admission.
The one frozen follow-up increased interval length and added sustained warmup.
Both new directional-copy cohorts are used consistently; no favorable rows
are selected across attempts. No further handoff tuning follows this study.

Host wall medians for completed copies, in microseconds:

| Payload | Node 0 H2D / D2H | Node 1 H2D / D2H |
| --- | ---: | ---: |
| 8,192-byte output activation | 7.950 / 8.313 | 9.833 / 11.129 |
| 44,032-byte input activation | 15.969 / 15.016 | 17.580 / 19.283 |
| 1 MiB, one layer of 1K FP16 KV | 171.821 / 172.913 | 177.676 / 251.795 |
| 8 MiB, one layer of 8K FP16 KV | 832.143 / 769.533 | 911.448 / 1609.698 |
| 23,953,408-byte original Q8 matrix | 2221.441 / 2024.967 | 2643.834 / 4461.538 |

The selected assist pays 44,032 bytes H2D and 8,192 bytes D2H per layer,
about 24.281 and 28.708 microseconds on the two nodes before rounding to
integer nanoseconds. Completion is already included. Idle decode/prefill
synchronization medians are 765/775 ns on node 0 and 737/780 ns on node 1;
adding them again would double-charge the completed copy path.

## Fixed simulation

The simulator uses retained native/prepared Q8 GPU costs from the independent
ShiftAdd controls, joined to CPU costs on exact model, tensor and input
identities. GPU observations amortize completion across 100 projections and
exclude copies. They are not measured single-assist or complete-token costs.
The source library retains all 72 series across nine matrices and four batches.
Only batch-one costs drive individually scheduled requests.

Each sampled layer represents twelve of the model's 36 layers. Gate, up and
down remain separate stages, with GPU assistance fixed to down only. There
is one CPU worker and one GPU engine. Copy and GPU compute serialize; CPU
compute can advance independently. Requests retain their declared ownership.
Queues, displaced GPU work and completion dependencies are charged explicitly.

Three workloads cover two simultaneous requests, three GPU-owned plus one
CPU-owned request, and four staggered requests with both owners represented.
Each request starts after prefill at 1K context and has eight simulated tokens.
CPU and GPU missing-work budgets independently take values zero, one or four
times their summed projection costs. Zero is an incomplete projection-only
lower bound. These assumptions cover missing attention, other projections,
normalization, activation, sampling and dispatch; they are not measured costs.

The two placements, two GPU paths, nine missing-work combinations and three
arrival traces produce 108 inputs. Every input retains all three scheduling
modes. Per-owner first-decode latency subtracts arrival time. ITL percentiles
use within-request intervals and exclude the first token; raw sample counts
remain in each summary. Queue wait sums describe waiting and are not added
again to elapsed completion time. These are simulated outcomes, not measured
serving throughput, TTFT or model-generated tokens.

## Scheduling outcome

All 324 modes satisfy the declared physical-capacity budgets. This is simulated
admission, not a successful real allocation or concurrent serving test.
The native projection-only token sums are 589.120 ms on CPU node 0,
600.485 ms on node 1, and 21.244 ms on the GPU. Prepared GPU projections sum
to 14.032 ms. These extrapolate sampled operators and omit other decoder work.

| Mode | Simulated throughput / GPU-only | Cases above GPU-only |
| --- | ---: | ---: |
| Independent CPU/GPU | 0.009347 to 0.721205 | 0/108 |
| CPU/GPU with down-projection assistance | 0.009998 to 1.000716 | 1/108 |

Assistance saves 5.181 to 5.448 ms per down projection before queueing and
improves throughput over the independent mode by 6.62% to 48.17%. That local
benefit does not remove the slow CPU work. The single positive GPU-only
comparison uses node 0, native Q8 GPU costs, three GPU-owned plus one CPU-owned
request, zero missing CPU work, and missing GPU work four times its measured
projection sum. Its +0.0716% is conditional simulation arithmetic, not a
measured gain. The result and missing full CPU engine do not justify a
scheduler proposal or the optional extra timing-envelope sweep.

For a concrete latency example, node 0 with native GPU costs, two simultaneous
requests and zero missing-work budgets gives the following per-owner ITLs.
Each row has seven within-request samples. First-decode latency is separate.

| Scheduling mode and declared owner | ITL p50, ms | ITL p95, ms | ITL p99, ms |
| --- | ---: | ---: | ---: |
| GPU-only, either owner | 42.488 | 42.488 | 42.488 |
| Independent, GPU | 21.244 | 21.244 | 21.244 |
| Independent, CPU | 589.120 | 589.120 | 589.120 |
| Assisted, GPU | 21.647 | 21.648 | 21.648 |
| Assisted, CPU | 402.138 | 402.138 | 402.138 |

Assisted first-decode latencies in this example are 21.446 ms for the GPU owner
and 403.970 ms for the CPU owner. Recorded queue waits total 1.832 ms on the
copy resource and 3.027 ms on the GPU; these sums are descriptive and already
reflected in completion times. Full-grid per-owner results, first-decode
latencies, pooled percentiles, resource occupancy, transfer bytes and every
queue interval remain in the evidence bundle. Small eight-token traces do
not characterize production tails.

## Storage and initial movement prerequisites

Each processor reserves 3,279,519,744 bytes for the original model plus an
assumed 1 GiB workspace, totaling 4,353,261,568 fixed bytes. Prepared GPU
scenarios add 108 planes of 23,953,408 bytes, for 6,940,229,632 fixed bytes.
Each request reserves 38,043,648 bytes of dense FP16 KV through token 1032.
These are calculated budgets. Physical node/GPU capacity admission does not
prove allocation availability on a host with other users and reserved pages.

Multiplying measured equal-size copy costs gives the following initial
movement estimates. Each row specifies its operation count; none is a measured
whole-model load or migration. Setup, allocation, preparation and unlisted
model tensors remain additional work.

| Movement | Bytes and operations | Node 0 H2D / D2H, ms | Node 1 H2D / D2H, ms |
| --- | --- | ---: | ---: |
| 1K FP16 KV over 36 layers | 37,748,736 bytes; 36 copies of 1 MiB | 6.186 / 6.225 | 6.396 / 9.065 |
| 8K FP16 KV over 36 layers | 301,989,888 bytes; 36 copies of 8 MiB | 29.957 / 27.703 | 32.812 / 57.949 |
| 108 original FFN matrices | 2,586,968,064 bytes; 108 equal matrix copies | 239.916 / 218.696 | 285.534 / 481.846 |

The scenarios begin after prefill and initial residency. These estimates are
reported separately and cannot be silently omitted from a future whole-request
or migration claim. Arbitrary remaining byte sizes are not assigned invented
bandwidth-derived costs.

A complete AVX2 decoder remains unimplemented. Further prerequisites include
CPU attention/QKV/output/vocabulary and sampling, full-token numerical and
sequence validation, CPU-owned KV lifetime, host/GPU handoff ownership,
asynchronous submission, actual contention, and available-memory qualification.
No development-hour estimate is assigned without evidence. The 18 unresolved
handoff diagnostics require separate investigation if a future design uses
that path. They do not justify a scheduler or runtime-default change here.

## Evidence

The original source plan, source audit and fixed cost-scenario contract remain
linked in this branch's earlier checkpoints. The final component-price table,
full simulation overview and every per-input summary retain distinctions
between measured, calculated and assumed values. Original raw costs, warmup
records, validation, timing errors, all scheduling modes and artifact hashes
remain available for reanalysis. No GPU timings overlap; every completed
transfer process leaves zero tracked storage and zero external M40 residency.
The study generates zero real model tokens and changes no runtime defaults.

The [component prices](2026-10-31-dualpace-component-prices.json) and
[simulation overview](2026-10-31-dualpace-simulation-overview.json) are readable
summaries. The [portable replay bundle](2026-10-31-dualpace-simulations.tar.gz)
contains all 108 traces, per-input summaries, copied and hashed cost sources,
exact builder scripts and `artifacts-review.sha256`. All 237 files pass an
extracted-bundle checksum check. The 21 MB
[full archive record](2026-10-31-dualpace-simulation-archive.json) gives the
protected local path and hash for all 108 five-record results, including every
resource interval. Its 344 files also pass extracted-bundle checks. The large
archive is outside Cargo cleanup and Git; raw results can also be regenerated
from the committed inputs with the retained simulator. The original full
checksum manifest remains in the replay bundle as a reference.
The simulator executable SHA-256 is
`0684486604cd5620cf18c26d2d17220df352ae7396fcc1b612532f1554b41042`.

The [transfer follow-up script](2026-10-31-dualpace-transfer-followup.sh),
[transfer audit](2026-10-31-dualpace-transfer-followup-summary.jq),
[simulation runner](2026-10-31-dualpace-simulations.sh) and
[overview extractor](2026-10-31-dualpace-simulation-overview.jq) retain the
executed commands and complete-grid checks. The
[final checksum manifest](2026-10-31-dualpace-final.sha256) covers the follow-up
raw observations, validation, placement/build/exit logs, summaries and bundle.
The original failed placement and unstable timing cohorts have their own
retained manifests and commits.

# Short transfer intervals fail stable cost admission

The resumed run completes CPU 30/node 1 and both transfer grids at `93ded84`.
Both CPU placements now have 36 numerically valid cases and 180 stable pairs.
They use the same CPU executable SHA-256
`199d649c5f30da058ab9d487497ef58b167969f8a46a6b634ff9e6d5c1583659`.
No CPU timing will be repeated in the transfer follow-up.

Both transfer validation grids pass all thirteen sizes. Timed payloads and
guards remain exact, tracked allocation totals return to zero, and each
process leaves the M40 at zero MiB with no compute process. The transfer
executable is `32320efe85905ae3bff47b2f0da3aa78c762b08233736ce6f78e7fee486a6cf4`.

Node 0 records 190 pairs over 27 groups, with eleven extensions and eight
unresolved variation errors. Node 1 records 160 pairs, five extensions and
one unresolved handoff group. The node-0 8,192-byte directional-copy group
is a required simulator input; the builder rejects it without publishing
an input bundle. No nominal scheduling result is admitted from that input.

At 8,192 bytes, node-0 H2D sample means span 8.923 to 11.038 microseconds
and D2H spans 8.182 to 9.976 microseconds. Twenty operations therefore occupy
only about 0.16 to 0.22 ms per interval. Rate MAD/median remains 6.55% and
5.45% after ten pairs. The first small cases report GPU clocks 949/3004 MHz;
large cases later report 1113/3004 MHz. This does not identify a single cause
for the variation. Three counted warm calls do not establish sustained
warm operation for these short host/device APIs.

## Bounded follow-up

Before collecting new data, fix one follow-up protocol for both placements:
use 100 completed operations per transfer interval and warm each arm for at
least one wall-clock second and three complete calls. Record warmup duration
and call count. Idle synchronization retains 1,000 calls per interval.
The consumer, placement, data sizes, numerical/guard checks, five/ten-pair
rule and instability gate remain unchanged. The longer interval and sustained
warmup address inadequate sampling duration; no claim attributes all observed
variation to that mechanism.

Preserve this entire cohort. Use the new cohort consistently for both nodes
if its required inputs pass admission; do not select favorable rows across
attempts. Any remaining unstable diagnostic stays unresolved. This is one
bounded measurement follow-up, not repeated retries until a passing result.

The [checksum manifest](2026-10-31-dualpace-transfer-first.sha256) retains both
complete transfer grids, independent validation, CPU node-1 results, summaries,
executed script, build/placement/exit logs and the rejected admission result.

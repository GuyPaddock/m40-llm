# Use an allowed core for the second NUMA placement

The node-1 cost measurements now use CPU 30 with memory bound to node 1.
CPU 15/node 0 remains unchanged. This amendment follows the preserved
[CPU 31 launch restriction](2026-10-31-dualpace-placement-stop.md), before
any node-1 result exists. It does not widen the process affinity boundary.
CPU 30 and sibling 62 were idle in the recorded three-second admission sample;
the pricing script will check them again immediately before execution.

Both projection and transfer probes already validate the actual affinity and
memory policy. Their implementation is unchanged. Scenario admission now
requires CPU 30/node 1 in both reports, rejecting obsolete CPU 31 evidence.
The existing node-1 behavioral fixture exercises the new placement and still
checks node capacity and relocation of source evidence. A rejected obsolete
placement must leave no output bundle.

The retained CPU 15 result from `8171d7f` supplies node 0 without repetition.
The remaining run will measure only CPU 30 projections, then both placements'
transfer grids. It keeps the same original weights and inputs, numerical
tolerances, iteration counts, paired policy, cache eviction, single-thread
scope and fixed scheduling scenarios. Source revisions and executable hashes
will make the resumed run distinguishable from the completed node-0 run.

This is an execution-placement amendment within the authorized two-NUMA
study. The full CPU decoder, concurrent serving and model-quality evaluation
remain outside this bounded experiment.

The amended scenario checks pass all 162 node-0 scheduling modes, node-1
placement/capacity and relocated-evidence checks, eleven malformed-input
rejections, overwrite protection and source-tampering rejection. The original
successful check log remains intact; the new
[check log](2026-10-31-dualpace-placement-check.log.gz) and current scripts are
covered by the [scenario manifest](2026-10-31-dualpace-scenarios.sha256).

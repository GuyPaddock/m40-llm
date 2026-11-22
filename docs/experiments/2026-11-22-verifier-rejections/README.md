# Rejected drafts and committed progress

The [October measurements](../2026-10-28-direct-verifier-measurement.jsonl)
establish the control for C1's same-model four-row target. The existing
packed verifier with a serial down projection spent about 75.8 ms from
target submission through rollback and the ordinary correction. Useful
progress ranged from one token on immediate rejection to four when three
proposals were accepted. At those endpoints, the observed median cost was
75.825 and 18.971 ms per committed token. This interval excludes drafting,
session setup and prefill.

| Accepted prefix | Complete median ms | Median ms per committed token |
|---:|---:|---:|
| 0 | 75.825 | 75.825 |
| 1 | 75.849 | 37.924 |
| 2 | 75.887 | 25.296 |
| 3 | 75.883 | 18.971 |

The [replay](replay.jq) accepts the original one-schedule capture and
the new three-schedule measurement. It checks each correction against
the actual 512-token trace, all four acceptance lengths in every round,
the number of committed tokens, and the elapsed time charged for target,
rollback and correction. Corrupting the correction token in one round
invalidates the analysis. The [historical summary](historical.json)
includes per-component medians and ranges. These measurements implement
neither a drafter nor a request-level speed claim.

```sh
d=docs/experiments/2026-11-22-verifier-rejections
jq -es -f "$d/replay.jq" docs/experiments/2026-10-28-direct-verifier-measurement.jsonl |
    diff - "$d/historical.json"
jq -es -f "$d/replay.jq" NEW_COMPLETE_THREE_SCHEDULE_MEASUREMENT.jsonl
```

## Four accepted proposals

The [target replay](target.jq) checks complete four-row transactions,
the three single-row tails, chronological trial order, allocated scratch,
dispatch counts, clean teardown and all 512 IDs against the independent
October 27 output. It binds the November three-schedule timings to their
three correctness captures. Each must match all 511 vocabulary-row
digests across schedules and the final valid KV digest. October's
correctness capture predates per-row digests. It retains the per-row
151,936-logit comparisons and final valid-KV digest.

The [original five direct-verifier trials](october-target.json) averaged
57.820 ms per four-row transaction at the median request. The three tails
cost another 56.329 ms per request at the median. The resident generation
took 7,781.267 ms including setup, prefill and teardown. Each request
used 1,521,188 bytes of tracked session scratch and returned to its
initial device allocation after teardown. The target and tails made
208,569 counted launches, including 18,396 down projections. The same
interval included 511 host-to-device copies, 1,022 device-to-host copies
and 1,022 stream synchronizations. Those are counts. The measured
target time includes their cost.

On the two multi-row schedules the dispatch check expects 4,680 down
launches, including 108 single-row tails. The existing multi-row
control and the new shared-weight kernel therefore save the same
13,716 launches against serial down. An advantage over the existing
multi-row control has to survive a comparison at the same launch count.
These expectations carry no performance result for the November verifier.

Run from the consolidation checkout. For a completed November run,
concatenate its three `correctness/*.jsonl` captures into one input
and pass its `measurement.jsonl` as the final argument.

```sh
d=docs/experiments/2026-11-22-verifier-rejections
q=docs/experiments/2026-10-27-maxwell-compound-runtime-quality.jsonl
base=docs/experiments/2026-10-28-direct-verifier
jq -es --slurpfile quality "$q" --slurpfile correct "$base-correctness.jsonl" \
  -f "$d/target.jq" "$base-measurement.jsonl" | diff - "$d/october-target.json"
```

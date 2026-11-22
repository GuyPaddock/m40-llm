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

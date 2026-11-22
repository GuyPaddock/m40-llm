# Idle compressed KV on the M40

**PASS for active-row isolation on this fixture. Close the proposal to
add another active-row index.** In all six configurations the 2,048 values
of the first output row matched bit-for-bit. Unwritten output guards
remained intact. Adding 4,096 idle tokens to an existing second slot
did not increase the active row's latency. A second active row wrote a
distinct output and increased median paired event time by 5.76%.

| Slots | Active rows | Second slot tokens | CUDA event ms | Host wall ms | Paired event change vs empty second slot, % [95% interval] |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | — | 3.6856 | 3.6861 | −0.52 [−0.89, −0.49] |
| 2 | 1 | 0 | 3.7047 | 3.7055 | 0 |
| 2 | 1 | 512 | 3.6745 | 3.6758 | −0.81 [−0.83, −0.78] |
| 2 | 1 | 2,048 | 3.6688 | 3.6694 | −0.99 [−1.00, −0.95] |
| 2 | 1 | 4,096 | 3.6738 | 3.6744 | −0.82 [−0.87, −0.75] |
| 2 | 2 | 2,048 | 3.9177 | 3.9184 | +5.76 [+5.71, +5.78] |

The active history stayed at 2,048 tokens. Every two-slot case reserved
7,996,168 cache bytes, compared with 3,998,084 for one slot. Each case
owned its own allocation throughout timing. Sub-percent differences
between allocations cannot establish a benefit from retaining idle
tokens. The [dispatch trace and frozen workload](../2026-11-21-active-row-kv.md)
explain why the active sequence never scans the idle slot.

Each case ran three warmups followed by five alternating rounds of 16
calls. The table uses medians per dispatch and 10,000 seed-zero bootstrap
resamples of paired round changes. Event spans include the work on the
decode stream during submission. The two-active positive control also uses
the batched wrapper. This synthetic operator measures neither physical
DRAM bytes nor scheduling effects under concurrent HTTP traffic.

Source `2228ea7095c4c8513faab7dac02e5ac6897d72ee`, release CUDA 12.8.93,
real sm_52, 1,113 MHz. The [raw capture](measurements.tar.gz) retains
the preflight, executable hash, deterministic input formulas, parity
records, all trials, host load and process ownership. Reproduce the operator
from a clean worktree by running `cargo run --locked --release --features
cuda --example active_row_kv -- --output NEW_FILE` with the manifest's
environment. Reproduce the [summary](summary.json) offline:

```sh
out=$(mktemp -d)
tar -xzf measurements.tar.gz -C "$out"
python3 replay.py "$out/active-row-01/report.jsonl" | diff - summary.json
```

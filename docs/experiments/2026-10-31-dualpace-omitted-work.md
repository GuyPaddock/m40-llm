# Explicitly omitted work in scheduling sensitivity

Schema 3 lets a sensitivity scenario omit work on one processor while
retaining the same stage on the other. This is needed by the zero-missing-work
cases in the [frozen scenario contract](2026-10-31-dualpace-cost-scenarios.md).
Use schema 3 for that contract's generated inputs. Its workload, cost grid,
arrival traces, routing, memory budgets and serialized-copy policy stay fixed.
The original schema-2 protocol and checksum remain preserved.

An omitted CPU or GPU cost must be zero and cite explicit assumption evidence.
A measurement artifact cannot justify zero processor service. The simulator
skips that processor's stage, producing no service or queue interval. It does
not insert a fabricated nanosecond or run a zero-duration event loop.

Every token profile must retain positive work on both processors. A stage
with zero work on both must be omitted entirely. Assistance requires positive
work on both processors; transfers remain positive. Already-included
synchronization keeps the schema-2 zero-extra-cost rule. Schemas 2 and 3 both
require an explicit copy/compute overlap policy. Schemas 1 and 2 retain their
original rejection of zero CPU/GPU work.

Thirteen CPU behavioral tests pass, including all eleven prior cases. The
two added cases check omitted CPU/GPU intervals, exact completion times,
assistance and invalid admission. Non-CUDA all-target Clippy and formatting
pass. No production runtime or GPU implementation changes.

The [synthetic schema-3 fixture](2026-10-31-dualpace-omitted-synthetic.json)
omits the CPU prefix while retaining the GPU prefix. Its GPU-only, independent
and assisted completion windows are 40, 20 and 34 synthetic nanoseconds.
The assisted CPU request finishes at 14 and the GPU request at 34. The CLI
output confirms these values and contains only positive-duration intervals.
These are hand-calculation checks, not processor measurements.

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/dualpace/target
RUSTFLAGS="-D warnings" cargo test --no-default-features --locked \
  --example cooperative_decode
cargo run --no-default-features --locked --example cooperative_decode -- \
  --trace docs/experiments/2026-10-31-dualpace-omitted-synthetic.json \
  --output target/dualpace-omitted-synthetic-new.jsonl
```

The [checksum manifest](2026-10-31-dualpace-omitted-work.sha256) retains the
test/Clippy/CLI logs, synthetic input and serialized result. CPU/transfer
measurements and generated sourced scenarios remain pending. The lossless
capture and its queued hardware qualification retain exclusive GPU scheduling.

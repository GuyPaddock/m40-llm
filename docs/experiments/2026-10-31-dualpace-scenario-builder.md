# Reproducible cooperative scheduling inputs

The scenario builder converts completed CPU and transfer reports into the
54 inputs per NUMA placement fixed by the
[cost scenario contract](2026-10-31-dualpace-cost-scenarios.md).
It uses the [schema-3 omitted-work rules](2026-10-31-dualpace-omitted-work.md)
and the retained GPU projection costs. The two placements yield 108 inputs;
each input compares GPU-only, independent and assisted scheduling.

CPU timing and transfer measurements are still pending. The checks below
use synthetic durations and establish builder behavior only. They do not
establish a cooperative throughput gain or qualify a complete CPU decoder.

The builder requires complete reports, matching CPU/NUMA placement, successful
numerical and transfer validation, and stable final timing series. It joins
all 36 CPU cases to both GPU paths using original model, tensor and activation
identities. It calculates service costs from the raw trial medians and rounds
up to integer nanoseconds. Batch-one costs drive the simulator; other batches
remain descriptive operator measurements.

Each output directory contains copied source reports, the exact builder
scripts, source hashes, the complete scenario index and 54 standalone traces.
Evidence paths resolve relative to that directory. Generation publishes a
complete directory only after admission succeeds and refuses existing output.
The simulator checks artifact hashes again before opening its result file.

The [capacity inventory](2026-10-31-dualpace-capacity.json) records read-only
NUMA and M40 totals with the original command output. These are physical
capacity bounds, not reservations or proof of currently available memory.
Other host users and reserved huge pages can reduce allocatable memory.
The frozen workspace, resident weights and finish-KV budgets remain explicit
assumptions. Any eventual implementation must separately qualify allocation
and concurrent contention.

## Behavioral checks

The [fixture](2026-10-31-dualpace-scenario-fixture.jq) retains the audited
weight and input identities but replaces durations with labeled synthetic
values. CPU projections cost 10,000 ns; native and prepared GPU projections
cost 100 and 50 ns; H2D and D2H cost 1,000 and 500 ns.

The [check script](2026-10-31-dualpace-scenarios-check.sh) passes:

- All 54 node-0 inputs conserve both processor budgets and use down-only
  assistance. All 162 scheduling modes run through the actual simulator.
- The balanced projection-only fixture produces the independently calculated
  GPU-only and independent completion windows of 172,800 and 8,640,000 ns.
- Node-1 generation selects the second capacity and CPU placement. Moving
  the output directory preserves artifact resolution and simulation.
- Existing output remains byte-identical after rejected overwrite.
- Eleven invalid reports reject without publishing output. They cover obsolete
  CPU 31 placement, incomplete
  or validation-only reports, mismatched placement or input bytes, failed
  arithmetic, zero costs, missing trials, unstable transfers, wrong transfer
  sizes and failed guards.
- Altering a copied source artifact rejects before simulator output opens.

The [checksum manifest](2026-10-31-dualpace-scenarios.sha256) retains the
scripts, inventory and original check log. The
[placement amendment](2026-10-31-dualpace-placement-amendment.md) also retains
the successful current check log for CPU 30/node 1. These CPU-only checks use
the schema-3 simulator; no GPU workload or model generation ran.

## Reproduction and next measurements

From this worktree, build and check the CPU simulator:

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/dualpace/target
cargo build --no-default-features --locked --example cooperative_decode
bash docs/experiments/2026-10-31-dualpace-scenarios-check.sh \
  "$CARGO_TARGET_DIR/debug/examples/cooperative_decode"
sha256sum --check docs/experiments/2026-10-31-dualpace-scenarios.sha256
```

After serial hardware qualification and the frozen CPU/transfer timing runs,
build one directory for each placement:

```bash
gzip -dc docs/experiments/2026-10-31-dualpace-gpu-costs.json.gz \
  > target/dualpace-gpu-costs.json
bash docs/experiments/2026-10-31-dualpace-build-scenarios.sh \
  CPU.jsonl TRANSFERS.jsonl target/dualpace-gpu-costs.json \
  docs/experiments/2026-10-31-dualpace-capacity.json NEW_DIRECTORY
```

Use the reports for the same placement in each invocation. Retain all modes
and regressions, report measured components separately from missing-work
assumptions, and finish the priced prerequisite report. The lossless capture
and its queued qualification retain the current and next GPU slots.

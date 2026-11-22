# Dense head128 attention at the long-context boundary

This is the operator diagnostic frozen in the long-cohort protocol at
`6113c490e72663615e09436464e63ae0f1560f72`. It adds one release-only
test to the existing dense head128 FP64 fixture. Kernel and runtime
sources match the pinned Phase 4 reference. The synthetic input uses
the Qwen benchmark's 16 query heads, two KV heads and head dimension
128. The exact prefix inputs and RoPE positions are deterministic.

From the worktree, run under the common lock on the Tesla M40.

```bash
flock --exclusive ../raw/m40-benchmark.lock \
    bash docs/experiments/2026-11-22-long-context-boundary/run.sh \
    ../raw/head128-dense-boundary-01
```

Retain the device-selection log, all six event and wall intervals,
per-sample CPU-reference checks, binary hash and actual test exit. A
15-minute cap bounds the operator run. Use the result to explain the
existing dispatch boundary, separate from whole-model measurements.

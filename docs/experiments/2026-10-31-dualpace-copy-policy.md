# Cooperative scheduling with completed blocking transfers

Schema 2 adds an explicit `has_copy_compute_overlap` policy. Use `false`
when pricing the runtime's completed pageable-copy APIs. Copy service then
shares execution time with GPU compute: neither starts while the other runs.
A ready assist copy has priority at the next GPU stage boundary, preserving
the existing non-preemptive assistance rule.

This is a conservative scheduling choice for the available transfer path,
not a claim that the M40 cannot overlap DMA and compute. The current host
copy methods hold the shared context lock and return completed data. Their
isolated costs do not qualify a separately scheduled asynchronous copy engine.
Pinned asynchronous transfers and host-submission concurrency remain separate
prerequisites for an overlap-based implementation.

Schema 1 retains its original overlap contract for historical traces. Schema
2 requires an explicit Boolean; omitting it fails validation. A schema-1 trace
cannot set the new field. The manifest reports the effective policy, so a
legacy assumption is visible even if its input omitted the field.

Schema 2 also permits zero *additional* synchronization cost when the cited
transfer or GPU service already includes completion. Such a cost still requires
evidence. Other work and transfer durations remain positive. The simulator
omits the zero-duration synchronization step instead of inventing a nanosecond
or counting completion overhead twice.

Eleven CPU tests pass. Two new cases verify resource exclusion, assist priority,
explicit schema admission and included synchronization. The existing nine
tests retain their original behavior. Non-CUDA all-target Clippy passes.

The [new synthetic input](2026-10-31-dualpace-serialized-synthetic.json) retains
the old hand-calculation costs and declares serialized copies with already
included synchronization. Its GPU-only, independent and assisted completion
windows are 40, 24 and 34 synthetic nanoseconds. In the assisted scenario,
the CPU request finishes at 24, and the GPU request finishes at 34. The H2D
copy waits until the first GPU stage ends at 10, runs through 12, and then
the assist runs through 22. D2H runs through 24 before native GPU work resumes.
These are simulator checks, not host measurements.

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/dualpace/target
RUSTFLAGS="-D warnings" cargo test --no-default-features --locked \
  --example cooperative_decode
cargo run --no-default-features --locked --example cooperative_decode -- \
  --trace docs/experiments/2026-10-31-dualpace-serialized-synthetic.json \
  --output target/dualpace-serialized-synthetic-new.jsonl
```

The [checksum manifest](2026-10-31-dualpace-copy-policy.sha256) retains the test,
Clippy and CLI logs plus the synthetic result. The earlier schema-1 artifacts
remain unchanged. This independent simulator extension belongs in a new commit;
it does not rewrite the original overlap-based contract or its results.

Next measure the completed host transfers and stream handoffs on the idle M40,
then construct schema-2 workloads with their actual costs. CPU timings and
the full CPU decode prerequisite report are still pending.

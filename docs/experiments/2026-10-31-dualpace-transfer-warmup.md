# Sustained warmup for the bounded transfer follow-up

The transfer probe now warms each arm for at least one wall-clock second
and three complete calls before collecting pairs. A warmup record retains
call count, operation count and elapsed wall time. Preparation and verification
are included in that warmup duration; it is not a GPU-active duration.
The manifest states the warmup requirements. Paired timing boundaries,
consumer operations and correctness/variation gates are unchanged.

This implements the [frozen follow-up](2026-10-31-dualpace-transfer-instability.md)
after preserving the first timing cohort at `3e5a58a`. Both placements will
use 100 completed transfers per interval. Idle synchronization keeps 1,000
calls. CPU projection results remain the original stable measurements.

The eight affected tests pass on the idle M40, including all six paths and
two allocation lifetimes. CUDA/server and non-CUDA example Clippy pass, as
does formatting. External residency returns to zero MiB. These checks qualify
the consumer and build; the subsequent release records must still establish
actual warmup duration and stable costs.

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/dualpace/target
export CUDA_VISIBLE_DEVICES=GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852
export M40LLM_ENABLE_CUBLAS=1 M40LLM_ENABLE_NVCC=1
cargo fmt --all -- --check
cargo test --features cuda --locked --example cooperative_transfer_probe \
  -- --include-ignored --test-threads=1 --nocapture
cargo clippy --features cuda,server --locked \
  --example cooperative_transfer_probe -- -D warnings
cargo clippy --no-default-features --locked \
  --example cooperative_transfer_probe -- -D warnings
```

The [qualification manifest](2026-10-31-dualpace-transfer-warmup.sha256)
retains the tested source hash and all three result logs. The original
three-call implementation, full raw costs and rejected scenario admission
remain in prior commits and experiment artifacts. No runtime path changes.

# Quantized KV staged reference checkpoint

Status: the first release hardware execution
[fails exact reconstructed V bytes](2026-10-31-quantized-kv-initial-failure.md).
The staged implementation and failing test are preserved before diagnosis.
This is not a qualified attention or performance result.

The mixed staged C ABI accepts head128 through a new compile-time
specialization. Its gather and attention kernels retain a head64
specialization for existing callers. The head width and inverse-square-root
attention scale are the changed assumptions. Runtime backend admission and
all direct consumers remain unchanged. No model default selects this control.

The ignored `quantized_kv_staging` hardware test covers both widths, four
physical slots representing two sequences and two layers, two reset/reuse
lifetimes, and lengths 1/5/9/14/23. It independently computes selected token
positions, FP16 reconstructed values and FP64 attention. It also checks
guarded buffers, query immutability, repeat calls without allocation, invalid
staging capacity, recent-ring wraparound, partial blocks and an explicit
prefill-to-decode query-copy dependency. These are intended checks; the test
has compiled and linked but has not run on hardware yet.

Current checks:

- All 220 non-CUDA tests pass with warnings denied.
- CUDA/server all-target Clippy passes. NVCC emits its existing legacy-target
  deprecation notice; the new shared-memory linkage warnings are corrected.
- The new test and attention parity, batched attention, transfer ownership,
  full-forward and MLP regression binaries compile and link.
- Formatting and diff whitespace checks pass.
- No GPU test or timing run has started on this branch. The lossless-KV capture
  remains the sole active M40 workload.

Logs are retained temporarily at `/tmp/quantized-kv-staging-clippy.log`,
`/tmp/quantized-kv-staging-clippy-final.log`,
`/tmp/quantized-kv-staging-noncuda.log` and
`/tmp/quantized-kv-staging-build.log`. The first Clippy run found three test
style errors and CUDA shared-symbol linkage warnings; all are fixed in the
current uncommitted unit. Preserve these logs with the eventual validation
report and commit the implementation only after its hardware checks complete.

Next: verify capture process `318282` / exec session `95030` is terminal and
the GPU has no other compute process before serial hardware validation.
Use the project environment and this worktree's disk-backed target directory:

```bash
source scripts/dev-env.sh
export CARGO_TARGET_DIR=/tmp/m40-research-1030/quantized-kv/target
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test quantized_kv_staging -- --include-ignored --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test attention_parity_cuda_grid --test attention_batched_varlen \
  --test cuda_transfer_order --test forward_with_layer_smoke \
  --test mlp_wrappers -- --include-ignored --test-threads=1
```

The new fixture requires selection-policy/order/anchor/mask environment
overrides to be absent. Check actual execution and device availability; a
skipped legacy test does not pass. Check affected cuBLAS-disabled parity too.
The paired context-grid benchmark from the
[source audit](2026-10-31-quantized-kv-audit.md) remains to be implemented.

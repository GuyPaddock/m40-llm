# Q8 synchronization carry-forward

The two barrier additions from research commit `1e445a5` apply directly to
feature base `a7479ba`. Both head64 quantizers load a shared K maximum and
then reuse the reduction array for V. A block barrier now separates those
operations so every warp reads K's maximum before any warp overwrites it.
Quantization arithmetic, representation and dispatch remain unchanged.

The direct Q8 and FP16-K/Q4-V attention consumers already exist on the
feature branch. The other quantized-KV research additions provide staged
comparison tools, not a missing production direct consumer. They are not
imported with this fix.

## Current-branch validation

The standalone [regression](../../tests/q8_kv_scale_reuse.rs) uses head64,
two KV heads, four query heads, four independent sequence slots and 129
tokens with a one-token recent window. Four resets reuse the same cache.
Large positive keys and tiny signed values have deliberately different
maxima. All four old blocks are selected so the check inspects every old
token, not only a favorable subset.

The check compares staged FP16 K/V bits against independently reconstructed
Q8 values, verifies the untouched recent token, counts/unique positions,
staged/direct attention agreement, input/output guards and tracked allocation
cleanup. It fails if the M40 cannot be opened, rather than skipping.

- Corrected streaming-eviction regression: passes with cuBLAS on and off.
- Affected attention, batched attention, stream-transfer and allocation tests:
  20 passes including the new regression, zero skipped.
- Projection, MLP and compound forward/ownership checks: six passes, plus
  another pass of the regression after its Clippy allocation correction.
- CUDA/server all-target Clippy: passes with Rust warnings denied. NVCC
  reports its existing deprecated-architecture notice for `sm_52`.
- Non-CUDA suite: 220 passes, zero failed or skipped, with warnings denied.

These are debug correctness runs on Tesla M40 UUID
`GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852`, not performance measurements.
The M40 was idle before execution; GPU tests ran serially.

## Limits and retained unsuccessful checks

One run of the new regression against the original uncorrected kernels also
passes. It does not independently reproduce the intermittent race. The
earlier branch's failing captured-cache evidence motivates the correction;
the new check protects reconstructed-value invariants when the race manifests.
Do not describe this negative control as a reproduced failure or claim a
dynamic race-detector result.

The bulk dense-to-Q8 kernel receives the matching barrier and compiles. Its
public Rust builder currently rejects every normally constructed dense
cache because those owners have `exact_old_backing=dense`. No new API or
dense-Q8 allocation path is introduced merely to reach this dormant kernel.
Hardware validation here covers streaming eviction, not the bulk builder.

The first invocation stopped before compilation because the disk-backed
target symlink's destination did not exist. Creating that destination fixed
the setup. The first Clippy run rejected a fixed-size `Vec<bool>` in the
regression; an array replaces it. Neither failure is a kernel result.

Raw logs and source checksums are listed in
[the checksum manifest](2026-10-31-q8-scale-carry-forward.sha256). Full
generation quality and throughput remain separate investigations.

## Reproduction

Source `scripts/dev-env.sh`, select the stated idle M40 with
`CUDA_VISIBLE_DEVICES`, and run serially:

```bash
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test q8_kv_scale_reuse --test attention_parity_cuda_grid \
  --test attention_batched_varlen --test cuda_transfer_order \
  --test weight_allocation_accounting -- --include-ignored --test-threads=1
M40LLM_ENABLE_CUBLAS=0 cargo test --features cuda \
  --test q8_kv_scale_reuse -- --include-ignored --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test q8_kv_scale_reuse --test q8_compound_runtime \
  --test mlp_wrappers --test proj_wrappers -- --include-ignored --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo clippy --features cuda,server --all-targets -- -D warnings
RUSTFLAGS='-D warnings' cargo test --no-default-features --locked
cargo fmt --all -- --check
```

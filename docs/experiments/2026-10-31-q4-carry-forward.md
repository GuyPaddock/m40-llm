# Selected Q4 execution carried onto the feature lineage

The host layout from `2705ef2` and isolated CUDA execution from `1aaaddc`
apply to feature base `a7479ba` without source changes. The host component
is committed separately as `1e3a6d3`. The CUDA component includes the build
dependency, FFI declaration, unavailable-backend stub and hardware tests.
No ternary screen, fixture converter, benchmark dispatch or runtime mode is
imported. Existing direct quantized-KV attention remains in place.

## API decision

`Q4ChannelWeights` preserves every original Q4_0 code and FP16 scale while
arranging down weights by channel. It validates dimensions and payload
lengths and can reconstruct the original bytes. It does not convert Q8 into
Q4 or guarantee finite values for arbitrary scale bits.

`DeviceQ4SelectedFfn` owns all uploads, immutable sorted per-row indices,
compact hidden vectors and down-reduction scratch. It supports rows 1/2/4/8,
positive block-aligned dimensions and nonempty unique in-range selections.
The unsafe enqueue requires disjoint live FP32 inputs/outputs, context
identity, producer ordering and serialized scratch use through completion.
Both native and channel-major down layouts remain resident for comparison.

This API is applicable as an opt-in research component on the feature branch.
It is not a dynamic selector or production FFN integration. Any later
integration must separate immutable weights from session-local selection
and scratch, define device-index updates and charge their cost, and avoid
duplicating down storage when only one layout is needed. No warm per-token
owner construction is justified by these operator results.

## Fresh validation

- Three host tests preserve all 65,536 FP16 scale bit patterns, payloads,
  scattered channel values, odd-output padding and invalid-source rejection.
- Two M40 tests pass with cuBLAS enabled and disabled. They check every
  selected FFN output against an independent masked FP64 calculation and
  exact agreement between native and repacked down addressing.
- Cases include rows 1/2/4/8, irregular selections and full retention,
  K=64/H=96, and synthetic Qwen-sized K=2048/H=11008. Guards, immutable
  readbacks, scratch reuse, cross-stream producer ordering, invalid spans
  and tracked allocation cleanup pass.
- The cuBLAS-enabled command passes 20 tests total: selected-Q4 two,
  attention parity eleven, projection two, MLP one, compound forward and
  ownership three, and the carried Q8 scale regression one. None skip.
- CUDA/server all-target Clippy passes with warnings denied. The complete
  non-CUDA suite passes 223 tests with warnings denied and none skipped.
  Formatting and whitespace checks pass.

Hardware was the idle M40 UUID `GPU-e590a24b-e805-dbcd-2e8b-b991dfe1f852`.
GPU runs were serial and tracked device usage returned to zero MiB after
completion. These are debug correctness checks; no new latency, whole-model
quality or physical memory-traffic measurement is claimed. Source and raw
log hashes are in [the manifest](2026-10-31-q4-carry-forward.sha256).

Reproduce after sourcing `scripts/dev-env.sh` and selecting the idle M40:

```bash
M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test q4_selected_ffn --test q8_kv_scale_reuse --test q8_compound_runtime \
  --test mlp_wrappers --test proj_wrappers --test attention_parity_cuda_grid \
  -- --include-ignored --test-threads=1
M40LLM_ENABLE_CUBLAS=0 cargo test --features cuda --test q4_selected_ffn \
  -- --include-ignored --test-threads=1
M40LLM_ENABLE_CUBLAS=1 cargo clippy --features cuda,server --all-targets -- -D warnings
RUSTFLAGS='-D warnings' cargo test --no-default-features --locked
cargo fmt --all -- --check
```

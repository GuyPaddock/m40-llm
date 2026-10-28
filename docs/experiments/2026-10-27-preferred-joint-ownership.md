# Prefer Joint-Only Compound Ownership

The user authorized joint-only as the preferred configuration after the
[independent storage qualification](2026-10-27-joint-only-ownership.md).
Implementation `a37cedc` makes it the ownership default for the opt-in
`joint-shared-scales-128` consumer. The exact 1,724,645,376-byte saving and
neutral measured performance justify the preference; no new timing was run.

## Scope

`Q8FfnConfig::joint_shared_scales()` and environment selection of that kernel
choose joint-only when ownership is omitted. Explicit
`M40LLM_Q8_FFN_OWNERSHIP=joint-only` is equivalent. Use `retain-split` or
`.with_ownership(Q8FfnOwnership::RetainSplit)` for the original physical owners.
Startup logs include the selected ownership.

Original uploaded GGUF, prepared down/head, kernel geometry, arithmetic,
prefill and cache semantics are unchanged. Preparation and compound FFN
selection remain opt-in. Native and other FFN defaults still retain split
ownership; KV defaults are untouched. Unsupported configurations still fail.

Historical compound choices in `model_quality` and `compare_ollama` now pin
retained ownership explicitly. The independent `joint_only` runner already
selects both arms explicitly. No historical variant or 109-owner reference
audit is silently repurposed. Existing measured reports and binaries remain
unchanged, including the immutable 50.910426 HTTP tok/s reference and the
56.153149 target. This is not a dense-speed optimization or new qualification.

## Validation

All Rust commands source `scripts/dev-env.sh`. Tests run serially with approved
GPU access; no model downloads, compatibility sweeps or timing runs occur.

| Check | Result |
| --- | --- |
| Warning-as-error non-CUDA suite | 219 pass |
| Compound ownership/runtime, cuBLAS enabled | 3 pass |
| Compound ownership/runtime, cuBLAS disabled | 3 pass |
| Packed-prefix prefill, hardware cases included | 5 pass |
| Attention parity grid | 11 pass |
| Compound selection/preparation unit checks | 2 pass |
| Historical compound control selection | 1 pass |
| CUDA/server all-target Clippy, warnings denied | pass |
| Formatting and diff checks | pass |

The existing ownership test now exercises an omitted environment ownership
flag versus explicit `retain-split`, on head64 and biased Qwen head128
fixtures. Both models keep identical logical sources, joint payloads and
logits. The preferred path has eight fewer fixture device allocations,
no split gate/up owners and no warm allocation churn. Budget failures and
model teardown return tracked allocations to baseline. An omitted ownership
flag with native FFN still selects the unchanged native configuration.

The initial combined attention/prefill invocation ran eleven attention
cases but ignored four hardware-only prefill cases. The explicit subsequent
`--include-ignored` invocation ran all five prefill tests successfully; ignored
cases are not counted as passes. Tests use explicit dense reference KV.

Representative commands:

```bash
source scripts/dev-env.sh
RUSTFLAGS='-D warnings' cargo test --no-default-features --locked
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo test --features cuda \
  --test q8_compound_runtime --test q8_prefill_runtime \
  --test attention_parity_cuda_grid -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=0 cargo test --features cuda \
  --test q8_compound_runtime -- --include-ignored --nocapture --test-threads=1
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo clippy \
  --features cuda,server --all-targets -- -D warnings
```

The [validation archive](2026-10-27-preferred-joint-validation.tar.gz) preserves
the exact invocations' build/test output and reference checksum checks.
The historical benchmark-control test checks the resolved typed ownership,
not source text. No paired HTTP remeasurement is needed for this selection
change; the earlier completed storage comparison remains its performance
evidence, not a measurement of this later revision.

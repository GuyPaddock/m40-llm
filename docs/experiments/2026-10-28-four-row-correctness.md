# Fixed Four-Row Operator Correctness Gate

Implementation `cac4c290afe8ac0482f36d7c6d6b339faf090eb4` passes the
pre-measurement correctness/resource gate. No performance trials are in this
checkpoint. The release executable SHA-256 is
`a19efd16c9a46790776729dc4bc83e4713b6dd1a535569a7f36a1a994a8cebf1`.

The [compiled audit](2026-10-28-four-row-compiled-audit.json) preserves all 137
original CUDA instruction-word sequences against the immutable reference.
Exactly one new entry has 656 words, SHA-256
`849d6f417421ed615d6a7c0f6a5810def6d4743148a0e6fac5459af6a4ebc43d`.
It uses 36 registers/thread, no reported stack/local storage, and 4608 dynamic
shared bytes. This is static resource evidence, not measured occupancy.

The [frozen-input observations](2026-10-28-four-row-correctness.jsonl) and
[independent audit](2026-10-28-four-row-correctness-audit.json) cover the first
four distinct rows at layers 0/17/35. All three arms have bit-identical gate,
up, SwiGLU and completed-down outputs. Ordinary and intermediate-capture
execution agree. CPU maxima are 2.065e-6 gate, 3.409e-6 up, 1.713e-5 hidden
and 3.483e-5 down, within the unchanged `1e-3 + 2e-5*abs(reference)` rule.

Each arm uses the same four qualified single-row down calls. All three gates
finish before those four down calls; this is an operator comparison, not four
autoregressive target steps. Original 37 down/vocabulary owners occupy
1,192,935,424 bytes; original uploaded sources remain 3,279,519,744 bytes.
A benchmark-local joint pair adds 47,906,816 bytes, without split gate/up
duplicates. Guarded scratch is 594,240 bytes and tracked residency is
4,520,956,224 bytes. Every layer releases its pair/scratch, returning to
4,472,455,168 bytes. Model teardown returns to zero. Actual joint planes and
sampled-layer prepared-down payloads are checked after use; guards and frozen
inputs remain intact.

Checks before primary measurement:

- New primitive: 2/2 with cuBLAS enabled and 2/2 disabled. Signed extremes,
  zero/negative-zero scales, producer-stream waits, captures, guards,
  unchanged payload/input bytes and rejected requests are covered.
- New CPU CLI/order checks: 2/2. CUDA/server all-target Clippy denies warnings.
- Existing attention 11, compound ownership 3, execution weights 2 and joint
  primitive 2 pass. Forward suites pass 1/20/23, with four optional graph
  cases still ignored. Projection 2 and MLP 1 pass.
- Non-CUDA warning-as-error suite: 219 pass. Formatting/diff checks pass.
- An initial benchmark compile used nonexistent event method names; it was
  corrected to the existing `elapsed_sync` API before any real-input run.
  No geometry, arithmetic, resource ceiling or acceptance rule changed.

Run the explicit diagnostic through `scripts/dev-env.sh` with exactly the
environment from `.execution_configuration.environment` in the frozen capture.
Arguments are `--ffn-four-row-reuse --ffn-four-row-correctness-only --rows 4
--iterations 10 --trials 5`, with the original model, expanded frozen inputs,
capture report and a new output path. No default or generation selector changed.

The next authorized action is the fixed five-trial release screen under the
[unchanged plan](2026-10-28-four-row-operator-plan.md). No kernel changes or
parameter search follow from this correctness pass.

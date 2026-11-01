# Compound FFN Runtime Qualification

## Admission

The [single compound consumer](2026-10-27-maxwell-compound-ffn-screen.md)
passes >=5% complete one-row FFN improvement at layers0/17/35 after twenty
fresh pairs, with unchanged gate/up/SwiGLU/down bits. The user's scoped
exception permits runtime evaluation after that gate. No new CUDA consumer,
geometry, arithmetic association, reduction or search is admitted here.
Exact verification has [stopped before EAGLE](2026-10-27-maxwell-exact-verification-screen.md).

## Fixed Runtime Design

- Add immutable opt-in `M40LLM_Q8_FFN_KERNEL=joint-shared-scales-128` and an
  explicit typed benchmark choice. Keep native and the qualified
  split-f32x4-128 control unchanged. Reject unsupported CUDA/device/dtype,
  row counts and K>4096 instead of silently selecting another kernel.
- Reuse `DeviceQ8FfnPair` and the exact consumer from `c629471`. Prepare
  pairs once, keyed by both original source pointers and semantic identities,
  under existing model-owned execution-cache lifetime. No hot-path packing,
  upload, allocation or synchronization.
- Retain all 109 original split prepared identities, especially down and
  vocabulary. For Qwen, joint pairs add 1,724,645,376 bytes; total prepared
  bytes become 4,642,226,176, within the existing 6144 MiB budget. Report
  joint pair count/keys, resident bytes, largest temporary host pair,
  preparation time and combined preparation budget separately. Original
  GGUF remains resident/needed for prefix work. Do not hide duplicate owners.
- Check every pair plan and combined budget before candidate uploads; use
  RAII for partial failure. Keep this accounting separate from original
  per-tensor statistics so old prepared identities remain directly comparable.
- Keep SGEMM prefix, packed-warp-256 QKV/output, prepared down/vocabulary,
  dense reference KV, context1024, greedy IDs and all other flags fixed.

## Atomic Sequence

1. `feat(infer): load opt-in compound FFN pairs`: typed selector, budgeted
   model-local owners, existing-consumer dispatch and behavioral ownership,
   invalid-config/budget, stream and complete-forward parity tests. Depends
   on the measured compound checkpoint. Defaults stay unchanged.
2. `test(bench): qualify compound runtime choice`: extend existing quality
   and HTTP comparison tools with an explicit qualified packed-projection
   control, exact dispatch checks and joint memory/preparation reporting.
   Keep all prior reference guards. Depends on tested runtime selection.
3. `docs(perf): record compound runtime gates`: run release actual512 and
   frozen dense-valid tasks first, then five paired HTTP trials only if
   quality passes. Record raw evidence, memory and cold setup separately.
   Apply the existing variation and fresh-confirmation rule without pooling.

## Acceptance And Stop

Require actual512 greedy token equality and no frozen dense-valid task
regressions. Run CUDA parity/ownership/prefill/forward checks with cuBLAS
on/off where applicable, non-CUDA warning-as-error tests, CUDA/server
Clippy, formatting and compiled-control checks. Escalate hardware/tests;
serialize all GPU work, including official Ollama oracles.

HTTP gain must exceed max(3%, twice variation), with a positive paired 95%
bootstrap lower bound. Five initial pairs, one ten-pair variation extension,
twenty fresh borderline confirmation pairs, no pooling or rescue. A losing
runtime remains opt-in/research-only with code/results preserved; do not
promote from the passing FFN microbenchmark. Preserve qualified 49.146 HTTP
tok/s until a replacement qualifies. Both original 53.06 and 1.30x fresh
official Ollama targets remain binding and distinct from this checkpoint.

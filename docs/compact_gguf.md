# Compact Layer Export

C25 is research-only model modification. Its first candidate is C24's
quality-evaluated `[8]` mask, not an automatically preferred model. That mask
preserves the small dense-valid task set but raises held-out PPL 3.46%.
See [the complete quality evidence](experiments/2026-10-25-ranked-layer-tradeoffs.md).

## Atomic Sequence

1. `feat(gguf): plan compact layer exports`: checked original-to-compact layer
   mapping, tensor ranges/alignment, explicit supported metadata and CPU tests.
2. `feat(gguf): write separate compact variants`: byte-preserving streaming
   writer, create-new outputs, provenance/quality bindings, metadata updates
   and round-trip/file-safety tests. Depends on the validated copy plan.
3. `test(quality): compare compact and masked models`: masked-source versus
   exported-model full-logit/task parity, real allocation accounting and a
   bounded runner. Depends on safe export and C24 frozen evidence.
4. `docs(perf): record compact model costs`: clean release measurements,
   preparation time, memory and quality limitations. Depends on parity.

## Copy Plan

`CompactGgufPlan::build` accepts a parsed source, actual source file length
and a checked nonempty `LayerMask`. It does not create files or claim that
a candidate passed quality checks. Retained layers preserve original order
and become `blk.0` through `blk.N-1`. Global embedding/output/norm/RoPE
tensors retain their names. Copy offsets specify actual tensor payloads,
not neighboring tensors' alignment gaps. Report payload and padding separately.

The initial scope is single-file little-endian GGUF v3, dense `llama` or
`qwen2`, standard `blk.ID.*` tensor families, and F32/F16/Q8_0 storage.
Quantized rows must contain whole 32-value blocks. Reject malformed,
duplicate, overlapping, aliased, truncated or overflowing ranges. Protect
first/last layers and require a complete matching tensor family in every
original layer, including layers being removed. No unsupported source is
silently rewritten.

Sharded sources, expert metadata, architecture-prefixed array metadata,
nonstandard global names and already compacted sources require separate
support and are rejected. Tokenizer arrays do not depend on layer IDs.
Preserve their original serialized types and bytes in the future writer,
including empty-array element types, merges, control tokens and templates.
Do not round-trip them through a representation that loses wire information.

The [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
defines aligned tensor offsets and typed metadata. Export will update the
architecture block count and model identity, record source/ranking/quality
hashes and original layer mapping, and keep the original file immutable.

## Remaining Gates

The copy-plan contract alone is not an exporter. Before writing a real model,
bind the mask to the completed frozen ranking and its evaluated quality row.
The destination must load as a distinct model and match the masked source,
with identical tokenizer behavior and preserved absolute RoPE positions.
Then measure actual uploaded weight and allocated KV reductions independently
of logical active-layer counts. No original model, runtime default or
unchanged-engine performance claim changes through this work.

The four CPU copy-plan tests pass with both CUDA feature configurations.
Both warning-as-error CI lanes, non-CUDA and CUDA/server all-targets Clippy,
fmt and whitespace checks pass. These validate the contract, not file export
or GPU inference equivalence.

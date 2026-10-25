# Compact Layer Export

C25 is research-only model modification. Its first candidate is C24's
quality-evaluated `[8]` mask, not an automatically preferred model. That mask
preserves the small dense-valid task set but raises held-out PPL 3.46%.
See [the complete quality evidence](experiments/2026-10-25-ranked-layer-tradeoffs.md).

## Atomic Sequence

1. `feat(gguf): plan compact layer exports`: checked original-to-compact layer
   mapping, tensor ranges/alignment, explicit supported metadata and CPU tests.
2. `refactor(eval): share research content hashes`: reuse the existing quality
   runner's coreutils identity routines in CPU export without new dependencies.
3. `feat(gguf): require evaluated export masks`: frozen ranking and complete
   paired quality evidence, with explicit adverse results. Depends on identities.
4. `feat(gguf): write separate compact variants`: byte-preserving streaming
   writer, create-new outputs, provenance/quality bindings, metadata updates
   and round-trip/file-safety tests. Depends on the validated copy plan.
5. `feat(cuda): report owned weight allocation`: query successful context-owned
   uploads rather than infer device storage from host file size. Test replacement
   and cleanup before relying on the comparison's memory report.
6. `test(quality): compare compact and masked models`: masked-source versus
   exported-model full-logit/task parity, real allocation accounting and a
   bounded runner. Depends on safe export and C24 frozen evidence.
7. `docs(perf): record compact model costs`: clean release measurements,
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
The writer preserves their original serialized types and bytes,
including empty-array element types, merges, control tokens and templates.
Do not round-trip them through a representation that loses wire information.

The [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
defines aligned tensor offsets and typed metadata. Export updates the
architecture block count and model identity, records source/ranking/quality
hashes and original layer mapping, and keeps the original file immutable.

## Quality-Bound Streaming Writer

The [evidence contract](compact_export_evidence.md) binds the frozen ranking
to completed paired quality measurements and the exact corpus. Regressions,
PPL costs and dense-failed inconclusive rows stay visible; an evaluated mask
is not quality-approved.

`export_compact_gguf` hashes the source before and after copying. It retains
the original bytes of unchanged metadata, rather than serializing parsed
tokenizer values. Only the block count, display name and stored parameter
count change; the stale `general.size_label` is removed. Namespaced
`m40llm.pruning.*` fields record the four evidence hashes, original layer count,
retained/removed IDs, model-change warning and both quality summaries.
Source license, author, base lineage and tokenizer metadata remain intact.

The writer buffers at most a 64 MiB header and streams tensor data. A unique
same-directory temporary file owns partial output. After data fsync, source
rehash, descriptor round-trip and output hashing, a create-new hard link
publishes the completed GGUF. Existing files, source hard links and even
dangling symlinks are never overwritten. Temporary files are cleaned up on
normal/error unwinding; an interrupted process can leave a `.partial` file,
not a successfully published incomplete GGUF. Same-directory hard-link support
and GNU coreutils `sha256sum` are required. No CUDA device is needed to export.

```bash
source scripts/dev-env.sh
cargo build --release --no-default-features --example export_compact
mkdir -p target/research-exports
target/release/examples/export_compact \
  --source "$model" \
  --layer-ranking docs/experiments/2026-10-25-layer-ranking.json \
  --quality-report docs/experiments/2026-10-25-ranked-layer-one.jsonl \
  --remove-layer-count 1 \
  --output target/research-exports/qwen-q8-pruned-layer8.gguf \
  --report target/research-exports/qwen-q8-pruned-layer8-export.json
```

Both output and report must be new paths. The ignored `target` directory keeps
large generated models out of Git; avoid RAM-backed `/tmp` for these files.
The report records build profile, separate evidence/export preparation wall
times, file/hash identities, payload/padding and
source/output tensor-data bytes. It does not measure GPU allocation, quality
of a newly loaded artifact, peak host memory or engine throughput. An export
error can leave an empty/incomplete report; only `compact_export_complete`
with `is_complete=true` is successful. Source and output are never substituted
into normal CLI/server defaults.

## Remaining Gates

The destination must load as a distinct model and match the masked source,
with identical tokenizer behavior and preserved absolute RoPE positions.
Then measure actual uploaded weight and allocated KV reductions independently
of logical active-layer counts. No original model, runtime default or
unchanged-engine performance claim changes through this work.

## Live Masked-Source Comparison

The shared `model_quality` runner accepts `--compact-export-report` to compare
the original source with its checked frozen mask against the compact artifact
with every retained layer active. Do not use `--skip-layers 8` on the compact
file; its layer IDs have already been renumbered. Other model transforms and
fitting flags conflict with this comparison mode.

Use the selected native-Q8 CUDA environment from
[the quality-runner instructions](model_quality.md#release-runner), then run
serially with the original source in `$model`:

```bash
cargo build --release --features cuda --example model_quality
target/release/examples/model_quality \
  --reference-model "$model" \
  --model target/research-exports/qwen-q8-pruned-layer8.gguf \
  --compact-export-report docs/experiments/2026-10-25-compact-layer8-export.json \
  --variant-id compact-layer8-equivalence --split all --context-tokens 1024 \
  --step-timing-trials 5 --output /tmp/compact-layer8-equivalence.jsonl
```

Source/output hashes, copied metadata, tensor descriptors and evidence lineage
are checked before load. After load, the runner checks source-only masking,
matching dimensions/RoPE/tokenizers, successful owned weight allocations and
fewer physical KV slots at unchanged context. It verifies identical task
tokens/text/stops, zero full-logit drift on task-prefix boundaries and loss
passages, and equal forced-prefix predictions in the paired timing trials.
Diagnostic full-prefix readback/scoring is not generation timing or corpus
PPL. The final dummy target is never forwarded. No complete record is written
if a check fails; retain the partial rows for diagnosis.

This reference is the masked source, not the original full-layer model.
Compact summaries name it explicitly and retain the existing C24 quality
costs. Requested weights and native KV payload capacity are separate; native
KV does not appear in the Rust allocation-tracker total. Driver/library
overhead and peak VRAM are not measured by these fields. Preflight file hashing
is timed separately from model load and is not fresh-process cold I/O.

The ignored real-model CUDA smoke requires explicit
`M40LLM_COMPACT_SOURCE_MODEL`, `M40LLM_COMPACT_MODEL` and
`M40LLM_COMPACT_EXPORT_REPORT` paths. It never scans caches or downloads a model.
An explicit request fails if the device or model paths are unavailable.
The Qwen-only parallel head128 override is scoped to that test, not the suite's
head64 fixtures. CPU preflight checks and a same-top-token/different-logit
negative case protect against weak equivalence claims.

Four CPU copy-plan, two evidence and four writer tests cover both supported
architectures, scalar block-count widths, mixed payloads, custom/default
alignment, exact unchanged metadata bytes, invalid claims and filesystem
safety. These are format/export checks, not GPU inference equivalence.
Both warning-as-error CI lanes, non-CUDA and CUDA/server all-targets Clippy,
fmt and whitespace checks pass.

The [real Qwen export checkpoint](experiments/2026-10-25-compact-layer8-export-checkpoint.md)
constructs the separate 35-layer artifact with unchanged source identity and
2.498% fewer tensor-data file bytes. Live masked-source numerical parity and
actual device-memory measurements remain; no inference speedup is established.

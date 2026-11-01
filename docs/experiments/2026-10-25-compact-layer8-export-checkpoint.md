# Compact Qwen Layer-8 Export Checkpoint

The CPU export step succeeds on clean `890b9c3`, using the frozen C24 `[8]`
candidate and its completed quality report. This is a file construction and
preparation-cost checkpoint. C25 remains incomplete until masked-source versus
compact-model inference and actual GPU allocations are verified.

## Artifact and Identity

The source is Qwen2.5-3B-Instruct Q8_0, SHA-256
`4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d`.
Its local Ollama blob is unchanged. The writer hashes it before and after
copying; a separate post-export checksum also verifies the original identity.

The separate output is
`target/research-exports/qwen-q8-pruned-layer8.gguf`, SHA-256
`7d04260214a3b57379310e963c4f6b4d39208be9b644d45c83aefa4430d03070`.
It has 35 layers and 422 tensors. Original layer 8 is absent; retained original
IDs remain chronological and are renumbered contiguously. Original first/last
IDs 0/35 map to 0/34. Header parsing and descriptor/offset round-trip succeed.
The GPU model loader has not yet been exercised with this artifact.

The GGUF records source/ranking/quality/corpus hashes, retained/removed IDs,
the model-change warning and C24 quality summaries. Unchanged metadata bytes
are copied directly, including the Qwen tokenizer/template and source
authorship/license metadata. The display name identifies a compact 35-of-36
layer variant, and the stored parameter count is recomputed. It is not a
renamed unmodified source model.

Both generated model and local export report stay under ignored `target`,
on NVMe. `/tmp` is RAM-backed and had only 4.3 GiB free, so it was not used
for the multi-gigabyte model. No binary model is committed.

## Storage and Preparation

| Field | Source | Compact Output |
| --- | ---: | ---: |
| File bytes | 3,285,476,256 | 3,203,563,552 |
| Tensor-data bytes | 3,279,519,744 | 3,197,605,888 |
| Header/alignment bytes before data | 5,956,512 | 5,957,664 |
| Original/retained layers | 36 | 35 |
| Tensors | 434 | 422 |

The payload reduction is 81,913,856 bytes, 78.119 MiB or 2.498%. File reduction
is 81,912,704 bytes; provenance grows the header by 1,152 bytes. These are
actual file/layout bytes, not measured device allocation or peak memory.
Q8_0 here describes model weights, not the KV format.

The release CPU-only exporter reports 10.264 ms evidence validation,
38,664.475 ms export and 38,683.148 ms preparation through publication.
Preparation includes source hashing twice, metadata scanning, streaming
copy, data fsync, header verification and output hashing. It excludes
process startup and report serialization/sync. One export only, without
a cold-page-cache experiment or timing promotion claim. No other agent-run
inference, build, test or MemPalace work overlaps the measured export.

The CPU-only release build uses:

```bash
cargo build --release --no-default-features --locked --example export_compact
```

No CUDA export path is used. Executable SHA-256 is
`fad61d280480af8b431f831cd3f43d9fbff2c5e970c144f3c7d2664ce998364c`.
See [the build log](2026-10-25-compact-layer8-export-build.txt) and
[the complete export record](2026-10-25-compact-layer8-export.json).
The record SHA-256 is
`fcd50b5a972c80404c8f8368ccda90334941013cbd10cd99cd20281feefd49b7`.
The [separate post-export checksums](2026-10-25-compact-layer8-export-sha256.txt)
match both source and published-output identities. No temporary `.partial`
file remains after export.

## Quality Boundary and Next Gate

The [frozen C24 ranking](2026-10-25-layer-ranking.json) SHA-256 is
`a9f0f2cc8c952eacaf96d6fe2576e30a35ef225a5f85f4574630545af3ab4d3c`.
The [one-layer quality report](2026-10-25-ranked-layer-one.jsonl) SHA-256 is
`204d620cb7f7963978e2c0e9e5cb9cbd4430b33ec64aebca69d9e8e47af1f847`.
No new mask, prompt, rubric or threshold is selected by export.

Existing C24 evidence preserves all dense-valid task passes, but held-out
PPL rises from 39.6122 to 40.9846. This is a quality cost, not unchanged-model
inference. Exported summaries recombine the recorded passage losses and can
differ from historical aggregate floats by final-rounding bits; they are
not new quality measurements. Dense-failed tasks remain inconclusive.

Next compare a live source model masked at original ID 8 with the compact
model running all 35 layers, using identical tokenizer inputs, absolute
positions and explicit dense KV. Require full-logit and generated-token
equivalence and verify actual uploaded weight/KV bytes independently.
At context 1024, 35-layer dense KV is expected to request 36,700,160 bytes,
but that is only an estimate until allocation is measured. Also preserve the
C24 quality cost when comparing against the unchanged full-layer source.

Ten CPU format/evidence/export checks pass across both feature configurations;
both warning-as-error CI lanes, all-targets Clippy and fmt pass. GPU-less CI
does not prove inference parity. Normal runtime defaults, original model
files, exact-engine performance claims and the postponed research gates are
unchanged.

# Complete FP16 KV capture and storage results

The clean release capture at `82a118a` completed all eight disjoint input
sequences. Each consumed 8,192 fixed input tokens and generated zero tokens.
The report contains 96 head streams, 384 compression measurements and 24 layer
readbacks. Every measured prefix reconstructs its original FP16 bits exactly.
This is a bounded compression fixture drawn from technical-document token
spans, not a task-quality or workload-representativeness result.

The binary contains 201,326,592 bytes with SHA-256
`da01d6328aef717902edbc93c215f0468f2bc9d789211effbc1b61b1569d1242`.
The capture executable hash is
`ec18befbaa2d5e1eaf9915a9c807447b521dbb28e15647a896ea1dd249a2c0e1`.
The raw report retains model/source identities, all token IDs and their hashes,
dispatch settings, per-stream digests, memory snapshots and sequence stop
reasons. Full-artifact hash verification and CPU fixture admission pass.

## Storage result

All 384 measured stream prefixes save bytes after encoded headers and block
metadata. Raw fallback remains enabled. At 8K context across all eight
sequences and sampled layers 0/17/35, the encoded streams occupy 192,117,184
bytes. Their 768-byte device pointer directories bring the representation to
192,117,952 bytes, saving 9,208,640 bytes, or **4.574%**, against original FP16.

The following aggregates include all three sampled layers and pointer
directories. They describe the encoded representation, not peak process VRAM.

| Context | Batch 1 saved | Batch 8 saved |
| --- | ---: | ---: |
| 1,024 | 4.628% | 4.575% |
| 2,048 | 4.607% | 4.577% |
| 4,096 | 4.593% | 4.574% |
| 8,192 | 4.591% | 4.574% |

The [storage data](2026-10-31-lossless-capture-storage.json) retains batches
1/2/4/8, per-role and per-layer breakdowns, raw fallback counts, metadata and
packed widths. At 8K/batch 8, encoded-stream savings before device pointers
are 4.150% for K and 4.999% for V. Layer 0 saves 3.562%, layer 17 saves 5.103%
and layer 35 saves 5.058%. Nearly all packed blocks need four exponent-offset
bits; these results do not reproduce a BF16 codec's published compression ratio.

The CPU fixture inspector admits all twelve layer/context cases and verifies
exact reconstruction again after converting capture views into head streams.
All twelve have a positive device-representation saving, including their
directories. Resident original copies, staging, upload temporaries and attention
workspace remain separate costs in the pending GPU comparison.

## Qualification checkpoint

Capture session `95030` exited successfully. Qualification session `68704`
also exited successfully after verifying the full capture, all frozen source
hashes and idle M40 state between phases. Both lossless CUDA tests pass,
including exhaustive FP16 reconstruction and attention/ownership checks.
All fourteen runner tests pass, including the hardware append test. These
results use the [frozen source manifest](2026-10-31-lossless-qualification-source.sha256).
The [qualification log manifest](2026-10-31-lossless-initial-qualification.sha256)
retains both test logs, fixture-inspection output and the completed queue log.

Tracked capture allocation returns from zero to zero. The capture's final
in-process driver snapshot still contains its CUDA context; the subsequent
external queue check reports zero M40 memory use and no compute process.
The same external check passes after the first hardware qualification.

Supported affected correctness checks also pass with cuBLAS enabled and
disabled. The broader command failures and optional model skips remain
separate from passing execution and will accompany the consumer validation
record. No performance claim or runtime promotion follows.

## Retained evidence and next step

The [capture checksum manifest](2026-10-31-lossless-capture-complete.sha256)
binds the compressed raw capture, run log, fixture inspection, storage summary
and full binary. The binary has a protected copy outside Cargo build outputs:

```text
/mnt/array-fastest/home/guyep/playground/2025-12-20-m40-llm/m40-llm/local-artifacts/qualified-references/research-1030-lossless/kv-inputs-01.bin
```

The working copy remains at `target/lossless-kv-inputs-01.bin`. The source
report is [retained compressed](2026-10-31-lossless-capture-complete.jsonl.gz),
as is the [CPU fixture inspection](2026-10-31-lossless-fixture-inspection.jsonl.gz).
Reproduce the aggregate data with the committed capture-summary jq program:

```bash
gzip -dc docs/experiments/2026-10-31-lossless-capture-complete.jsonl.gz |
  jq -s -f docs/experiments/2026-10-31-lossless-capture-summary.jq > NEW_SUMMARY.json
sha256sum --check docs/experiments/2026-10-31-lossless-capture-complete.sha256
```

The positive storage result admits the frozen staged/fused attention study.
Commit the hardware-checked consumer before its runner, then run
the clean release correctness and paired attention/append grids. Encoding and
replacement costs remain required; storage reduction alone is not a speedup.

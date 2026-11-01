# Immutable Compound Runtime Reference

## Identity And Scope

Reference ID `compound-runtime-63240c5-v1` fixes the qualified compound
runtime at **50.910426 HTTP tok/s**, versus official Ollama **43.194730**.
The [machine-readable manifest](2026-10-27-compound-reference.json) binds
the exact measured source, model, executables, settings and 27 evidence files.
This is a preservation checkpoint, not another benchmark or promotion.

Clean measured source is `63240c52b64e06870922c8b6c0cbae8b0fe2e6f4`, tree
`b5d09d66af0477d4ecbfc16af337e79291d055a7`. The source remains in Git and
also in an ignored local archive. Later documentation commits do not become
the measured source revision.

| Measured input | SHA-256 |
| --- | --- |
| Qwen2.5-3B Q8_0 GGUF | `4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d` |
| Release model_quality | `9a659afbdfbf02249588b9669b42f14e18cc86266ef24227b4f05ad0de1738d3` |
| Release compare_ollama | `eb6dc69f94a2e67ecd62bc07a02a12e47c26067236ac6f7df40e2c4f78b5b86f` |
| Official Ollama 0.40.11 | `b0a4becfaf872e3ad5d2e9051244e0dfe55409f03d6e2cd9bd9c34400e201b4c` |
| Official llama-server | `234b05b2138264f8fb263c3205e85f4c290e8afe5067e280a4f6f90cdac5696b` |
| Official CUDA 12 backend | `7a7b6696ffb43ca00f78504a0b2e76db0c6554320ed64e405559e3433580d9b3` |

## Frozen Configuration

Keep all 36 layers, canonical 65 ChatML prompt IDs, greedy `top_k=1`, 512
actual generated tokens and natural `length` stop. KV is explicitly dense
`off`, context 1024. Use compound `joint-shared-scales-128`, fixed 128-thread
geometry and FP32 arithmetic/reduction order, packed-warp-256 QKV/output,
bounded dequant-SGEMM prefix and the original prepared down/vocabulary path.
Dense attention uses the qualified head128 kernel, eight partitions and
256 threads. Graphs, materialized FP32 weights and activation quantization
remain off. All other environment values and prompt IDs are in the manifest.

The benchmark loader explicitly prepares split `gate-up,down,vocabulary`
weights with a 6144 MiB combined budget. The raw environment's execution
selector is `off`; the named loader installs the effective preparation
configuration. Replaying only environment variables through an unrelated
CLI does not reproduce that model. Keep the archived commands and the
effective preparation/dispatch reports together.

Release builds use `cuda,server`, NVCC 12.8.93, `-O3`, sm_52 SASS and
compute_52 PTX, without fast math. Keep the recorded compiler, sysroot,
driver, GPU UUID and clocks. The controlled process snapshots show isolated
serial inference at P0, 1113 MHz SM and 3004 MHz memory.

The 109 split prepared owners occupy 2,917,580,800 bytes; 36 additional joint
pairs occupy 1,724,645,376 bytes. Combined preparation is 4,642,226,176 bytes,
and uploaded GGUF plus every prepared owner is 7,921,745,920 bytes. Dense KV
adds 37,748,736 bytes and temporary prefix capacity is 90,177,536 bytes.
These are component allocations, not standalone peak VRAM measurements.

## Evidence And Gates

Use the [qualification report](2026-10-27-maxwell-compound-runtime.md),
[quality JSONL](2026-10-27-maxwell-compound-runtime-quality.jsonl),
[twenty fresh incremental pairs](2026-10-27-maxwell-compound-runtime-http.jsonl),
and [five fresh official pairs](2026-10-27-maxwell-compound-runtime-ollama-http.jsonl).
The incremental result is 50.798730 versus 49.221402 HTTP tok/s, +3.2046%,
95% paired interval +3.1565% to +3.3032%. The independent official comparison
is +17.8626%, interval +17.6394% to +18.0302%. Do not pool these comparisons.

Quality preserves all 512 generated IDs and twelve frozen task outputs.
Seven dense-valid tasks pass without regressions; five remain inconclusive.
Full-vocabulary logits at steps 0/1/8/32/128/256/511 are bit-identical and
top-10 overlap is 10. The original 137 compiled CUDA entries are unchanged.
The manifest also retains the audit programs, corruption controls, raw
orchestration/logs and the independent stage-cost evidence.

The [controlled protocol](../controlled_benchmark.md) remains binding:
five initial alternating pairs, ten if variation exceeds 5%, and exactly
twenty fresh confirmation pairs for the specified borderline cases. Require
gain above `max(3%, 2 * larger variation)` and a positive paired 95% lower
bound. Do not pool screen and confirmation, force continuation, retune
thresholds or repeatedly extend confirmation. Only matched actual-token
counts, stop reasons, work, cache-reset evidence and isolated GPU runs count.

Targets remain 53.06 and 56.153149 HTTP tok/s, the latter 1.30 times the
fresh official result. The stronger target needs 938.958 ms less per
512-token request. The [stage report](2026-10-27-maxwell-compound-stage-profile.md)
does not establish DRAM saturation or a hardware ceiling. A new speed kernel
needs a distinct supported avoided-work hypothesis, accounted overhead,
credible request-level savings and fixed stop conditions. Closed FFN
geometry/layout experiments and the rejected verifier/EAGLE path stay closed.

## Local Preservation And Verification

The original model remains at its content-addressed Ollama path. Do not
prune that blob. Five independently copied measured binaries and the source
archive are retained read-only under
`local-artifacts/qualified-references/compound-63240c5/`, outside Cargo's
`target/` cleanup paths. Git intentionally ignores these large files.
Upstream license notices accompany the local copies. They are not a portable
Ollama/CUDA distribution; retain the original runtime libraries and toolchain.

A clone contains the evidence/hashes, not the local binaries or model. Missing
files are a preservation gap, not proof that a rebuild reproduced their
hashes. Never overwrite retained binaries with a candidate build. Copy this
directory and the model separately when moving the checkpoint to another
machine; local disk storage alone is not an off-machine backup.

From the repository root, verify the manifest's 27 evidence files, five
preserved binaries, source archive and original model:

```bash
jq -r '
  (.artifact_sha256 | to_entries[] | "\(.value)  \(.key)"),
  (.executables[] | "\(.sha256)  \(.preserved_path)"),
  "\(.source.archive_sha256)  \(.source.local_archive)",
  "\(.model.sha256)  \(.model.path)"
' docs/experiments/2026-10-27-compound-reference.json | sha256sum --check -
```

All original runtime/stage checksum sets and the live model/executable hashes
were verified while preserving this checkpoint. No GPU access, build, new
inference or performance requalification was needed. New candidates must
write distinct reports referencing this manifest, not revise these measured
records. Project defaults and every acceptance gate remain unchanged.

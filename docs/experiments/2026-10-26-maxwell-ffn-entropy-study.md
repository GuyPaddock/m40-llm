# Lossless Byte Coding Has Too Little FFN Scope

## Decision

Do not implement a CUDA byte-Huffman decoder from this study. All three
[prespecified indexed size models](2026-10-26-maxwell-ffn-entropy-hypothesis.md)
miss the **10% stored-byte saving** feasibility threshold at every tensor.
The best, 1024-symbol chunks, save only **2.37-2.69%** before decoder
storage/execution costs. Close these models without a codebook, predictor
or chunk-size tuning round.

This is CPU-only feasibility evidence, not a new representation, actual
encoded stream, reversible codec, GPU bandwidth measurement or inference
optimization. It does not rule out predictive or other lossless formats.
The zero-order bound applies only to the measured independent-byte model.

Qualified performance stays **49.146 HTTP tok/s** versus **43.106 official
Ollama**, not remeasured here. Both original targets, **53.06** and **1.30x
fresh Ollama**, currently **56.037 HTTP tok/s**, remain unmet. Runtime
defaults, model weights, dense benchmark semantics and prepared down/head
control are unchanged.

## Actual Tensor Counts

All six Qwen2.5-3B Q8_0 gate/up tensors use K=2048, N=11008 and contain
**23,953,408 original bytes**: **22,544,384 signed payload bytes** plus
**1,409,024 unchanged FP16 scale bytes**. Every byte is read from its
GGUF-validated absolute file range and checked against frozen M02 tensor SHA.

The study counts a global 256-symbol histogram and optimal binary byte-prefix
lengths per tensor. Indexed models use 32/256/1024 payload-symbol chunks,
16-byte chunk alignment, a 4-byte start offset per chunk, a 256-byte code-length
table and literal original scales. Negative savings mean larger storage.

| Layer | Tensor | Entropy bits/payload byte | Ideal zero-order total saving | Chunk32 saving | Chunk256 saving | Chunk1024 saving |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Gate | 7.65094 | 4.11% | -15.06% | -1.39% | 2.65% |
| 0 | Up | 7.64898 | 4.13% | -14.95% | -1.38% | 2.69% |
| 17 | Gate | 7.66887 | 3.90% | -16.01% | -1.33% | 2.54% |
| 17 | Up | 7.66253 | 3.97% | -15.73% | -1.35% | 2.61% |
| 35 | Gate | 7.68322 | 3.73% | -16.20% | -1.43% | 2.37% |
| 35 | Up | 7.67837 | 3.78% | -15.88% | -1.43% | 2.43% |

The ideal zero-order column includes unchanged scales but no indexing,
alignment or codebook. Even that model saves only 3.73-4.13%. The Huffman
models add actual code-length, alignment and index costs, not a measured
compressed byte stream. Decoder lookup/storage, prefix-search and execution
costs are deliberately excluded and cannot be assumed free.

These are per-tensor representation counts, not resident-VRAM predictions.
The qualified runtime also retains original GGUF source weights alongside
execution-weight copies. No such owner is replaced or released here. Byte
savings cannot be relabeled as complete-FFN or HTTP throughput improvement.

## Provenance And Validation

Clean source `fc7f377b3bf5dd55fddd76d01c157e5f507542a9`; release executable
SHA-256 `d270a313357a1b1dc29bf415e3f26bf0262cba5317e4545fdc5c27d2545eeec6`.
The shared `q8_execution_bench` adds only `--ffn-payload-entropy`, requiring
captured provenance and rejecting combinations with timing experiments.
It returns before model upload or GPU inspection. Each tensor read is bounded
to 64 MiB; actual raw buffers are 23,953,408 bytes and are released serially.
Model/fixture/capture/tensor hashes, dtype, shape and offsets are validated.

Session **47677** exits zero with eight records: one manifest, six tensors,
one completed decision. No CUDA context/model is constructed; tracked device
bytes remain zero to zero. No inference or GPU timings are collected. Host
read/hash and analysis durations are labeled separately in the raw report;
they are not performance qualification data.

CPU tests pass **30**, with 14 hardware tests ignored in that invocation,
not counted as passes. New tests cover known optimal Huffman costs, empty/
overflowing frequencies, uniform and constant alphabets, scale-bit exclusion,
all stored components, malformed blocks and tail chunks. Non-CUDA
warning-as-error tests pass **217**; CUDA/server all-target Clippy, formatting
and diff checks pass. No CUDA/runtime code changes, so no new GPU parity
result is claimed. An initial usize/u64 GGUF shape comparison compile error
was corrected within the implementation before successful checks/commit.

The JSON audit independently recomputes entropy, weighted optimal prefix
cost through repeated minimum-frequency merges, Kraft bounds, index/scale/
table costs, possible alignment bounds and the all-six-tensor decision.
It binds every source identity to frozen M02. A separate `dd`/`od`/`awk`
audit rereads all six original tensor ranges, verifies their SHA, recounts
all 256 symbols and recomputes every chunk's aligned code-length cost.
It consumes the reported code lengths, whose optimal cost/prefix bounds
are checked independently by jq. This is not an independent codec test.
The verifier also rejects incomplete six-tensor coverage before scanning
tensor ranges; the completed scan confirms all six records.
Corruption controls reject missing coverage, wrong identities/counts/codes,
bad indexing/accounting, fake savings and false runtime eligibility.

## Reproduction And Artifacts

Source `scripts/dev-env.sh` and restore the frozen M02 environment as shown
in the [earlier reproduction](2026-10-26-maxwell-ffn-task-pair-screen.md#reproduction-and-artifacts).
Use the explicit original model and a new output file:

```bash
cargo build --release --features cuda --example q8_execution_bench
target/release/examples/q8_execution_bench \
  --model "$HOME/.ollama/models/blobs/sha256-4420ccb0f1d9e12811d04ae2a28ec881469305f813c62d86c10e595ef8e0111d" \
  --ffn-inputs target/research-fixtures/m02-ffn-inputs.json \
  --ffn-capture-report docs/experiments/2026-10-25-maxwell-ffn-capture.jsonl \
  --ffn-payload-entropy --output /tmp/new-ffn-entropy-study.jsonl
```

- [Raw counts and code lengths](2026-10-26-maxwell-ffn-entropy-study.jsonl), [audit](2026-10-26-maxwell-ffn-entropy-audit.jq), [derived sizes](2026-10-26-maxwell-ffn-entropy-audit.json) and [negative controls](2026-10-26-maxwell-ffn-entropy-negative-controls.sh)
- [Independent source-byte audit](2026-10-26-maxwell-ffn-entropy-verify-bytes.sh) and [verification records](2026-10-26-maxwell-ffn-entropy-independent.jsonl)
- [Build log](2026-10-26-maxwell-ffn-entropy-build.log.gz), [CPU tests](2026-10-26-maxwell-ffn-entropy-cpu.log.gz), [Clippy](2026-10-26-maxwell-ffn-entropy-clippy.log.gz), [non-CUDA checks](2026-10-26-maxwell-ffn-entropy-noncuda.log.gz) and [checksums](2026-10-26-maxwell-ffn-entropy.sha256)

Next work needs independently supported execution-cost scope, not a decoder
whose byte budget already misses feasibility. All original actual-output,
complete-FFN, quality and HTTP acceptance rules remain intact.

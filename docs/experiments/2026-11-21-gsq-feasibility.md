# GSQ checkpoint feasibility on M40-LLM

**FAIL for present runtime feasibility. INCONCLUSIVE for M40 quality, VRAM and performance.** The authors publish both requested GSQ formats. The local five-model inventory contains neither Q2_K nor Q3_K_M, and obtaining the files alone would leave the supported standard-layer mapping unable to run them. Defer to checkpoint preparation and compatibility evaluation. There is no GSQ-specific hot-path change to port.

## Admission and evidence

The frozen C5 scope requires identical model architecture, tokenizer and quantization for each quality pair, higher-precision reference, direct supported inference, actual residency and whole-request timing. Stop before downloading full weights or training if the execution backend cannot consume the checkpoint. Small differences in container size at the same quantization do not establish a memory saving on the GPU.

In Phase 4 48c63332462e009964e891be1cabaf139e3b50f8, src/gguf.rs recognizes GGML tensor type IDs 10 and 11 and calculates their block storage. src/infer/meta.rs handles their byte sizes. src/infer/mapping.rs rejects both types for ordinary layer projections before GEMM. Embeddings and output heads have their own narrower type lists. src/infer/gemm.rs supplies separate Q4_0/Q8_0 consumers and a restricted Q6_K consumer, with no corresponding Q2_K/Q3_K arithmetic. A mixed Q3_K_M file requires support for the actual tensor mix and roles. Qwen3 also needs its architecture semantics validated before serving. The detached main e1403e0 has the same lack of Q2_K/Q3_K layer dispatch. Reproduce the screen on each commit with rg 'Q2K|Q3K|Q6K|Q4_0' src/gguf.rs src/infer/{mapping,gemm,meta}.rs. The local five checksums are preserved in consolidation baac624.

The November 10 revision compared Qwen3-8B GSQ with Unsloth checkpoints. Its Table 4 average rose from 50.03 to 56.28 at Q2_K and from 60.52 to 61.61 at Q3_K_M. These are the authors' three-benchmark averages. No local inference or timing underlies these numbers. [Paper, Table 4](https://arxiv.org/html/2604.18556v3#S4.SS4)

| Artifact | Published bytes | LFS SHA-256 |
|---|---:|---|
| Unsloth Q2_K | 3,281,733,440 | 7226e0183d31dca14d81c6f799ada2944be62160b8b7549a70254fba4124a5cf |
| GSQ Q2_K | 3,281,732,960 | 2bc740416074ee7a7c26189d82d1b1c75274287ec70877c945a0548f2e561858 |
| Unsloth Q3_K_M | 4,124,161,856 | 4924cf38a3b3c4b27ead5ccb93e27027f9418738506ac50a24a70dfe8581a007 |
| GSQ Q3_K_M | 4,124,161,376 | 81894ed9b0e0245af542019a941e0aac21f3ec8b5a2386b82486db19c0170e77 |
| Unsloth BF16 | 16,388,044,384 | 5e416a2020fe63e76ea13c8979be35fc6070aaf3578f7876400c55c2f5c3eb30 |

Read-only model API requests with blobs=true returned the above pointers. Model names correspond to the public reports. These metadata checks do not establish byte-identical tokenizers, the paper's exact starting-file revision, or local file integrity. The BF16 file is a reference checkpoint for a capable backend. [GSQ files](https://huggingface.co/ISTA-DASLab/Qwen3-8B-GGUF-GSQ/tree/5605b36840bc26d62b34a6adb518109f55c412d2) · [Unsloth files](https://huggingface.co/unsloth/Qwen3-8B-GGUF/tree/a6adef130ffb23ddaf1a62fec9dced968c9bc482)

Reopen once a supported model/backend covers the complete tensor mix, including embeddings, projections and output. Validate tokenization and same-model control provenance first. Charge any device repacking or expansion to FP16, then compare quality at fixed format and compare resource costs across formats separately.

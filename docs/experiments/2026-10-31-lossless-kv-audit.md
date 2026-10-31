# Lossless FP16 KV experiment

## Source review

[ZipFusion v1](https://arxiv.org/html/2610.33719v1), submitted October 24,
uses BF16 KV on H100. It retains sign/fraction bits, stores bounded exponent
offsets and restores values inside attention. Its example uses 512-value
blocks and 16-token pages. Eligible payloads use 12 bits per value before
metadata; ineligible blocks remain raw. Partially filled pages stay raw.

The evaluation uses precompressed prefixes. Initial compression and prefill
are excluded, and newly generated KV stays uncompressed. Its throughput
result therefore does not establish online compression/append cost. The
attention implementation uses layouts for Tensor Core operands. The paper
does not establish Maxwell performance or FP16 compression ratios.

The inspected paper links no implementation repository. This experiment
implements the following FP16 adaptation independently; it does not copy
the paper's text, diagrams or an unlicensed CUDA implementation.

## Fixed FP16 candidate

FP16 has five exponent bits and ten fraction bits. Retaining sign/fraction
already costs eleven bits. A four-bit exponent offset therefore leaves only
one bit per value to save before metadata. This is a format calculation,
not an observed compression result.

Use 512 consecutive values within one KV head, ordered by token then channel.
For each block, find the minimum stored exponent and choose the smallest
offset width from zero through four that represents the exponent range.
Pack eleven sign/fraction bits plus that offset, least-significant bit first.
Blocks containing exponent zero or 31 remain raw, preserving zeros,
subnormals, infinities and all NaN payloads. A packed block is eligible only
when its payload plus eight directory bytes is smaller than its raw payload.
Otherwise store the original little-endian words.

The serialized stream starts with four magic bytes and a u32 value count.
Each block has an eight-byte directory entry: u32 payload offset, u16 count,
u8 exponent base and u8 offset width. Width 255 means raw. Payloads follow
contiguously. Header, every directory entry, tail blocks and raw blocks all
count toward storage. Raw fallback can still leave net directory overhead;
the experiment must report that overhead rather than claiming zero cost.
No entropy coding, outlier side table, block-size search or data-dependent
choice of layout is included.

## Capture and gates

There is no retained raw FP16 KV artifact in this branch. Existing debug
readback supports explicit layer/sequence addressing, and `model_quality`
already owns checked model loading, dense sessions and report writing.
Extend that tool for the capture rather than adding a separate inference
driver. Pin source/model/tokenizer/environment hashes and token IDs.

Use eight disjoint 8192-token spans from the repository's retained performance
history at base `0eee578`. This supplies distinct real model inputs without
new data acquisition or training. These technical-document inputs are a
bounded compression fixture, not representative workload or task-quality
evidence. Freeze token spans before observing compression. Capture layers
0/17/35 with the actual Qwen GQA head layout, explicit dense FP16 KV, graphs
off and unchanged projection controls. Preserve all bit patterns. Derive
1K/2K/4K/8K prefixes from each full causal cache, and report each sequence,
layer, head and K/V separately. Do not treat duplicated rows as concurrency.

CPU qualification requires bit-exact reconstruction, including all 65,536
half bit patterns, width transitions, raw exceptional blocks and tails.
Malformed serialized offsets, dimensions, exponent codes and padding fail.
Before any CUDA codec work, compare original bytes to the entire encoded
stream on real captures. If there is no net benefit, close the dependent
attention experiment. If savings survive metadata, freeze the staged/fused
attention comparison and include compression, append and raw-page costs.
The original 1K/2K/4K/8K and batch 1/2/4/8 scope remains in force.

Commit sequence:

1. `feat(kv): add lossless FP16 block probe`: CPU format, independent decode
   and behavioral checks, with this source/design audit.
2. `feat(research): capture lossless KV inputs`: extend `model_quality`,
   preserve real bits and identities, and report CPU compression results.
3. Conditional `feat(cuda): consume lossless KV blocks`: only after a net
   storage benefit, followed by controlled attention measurements.

Runtime defaults, compressed-Q4 KV behavior and drafter restrictions remain
unchanged. This is an exact FP16 experiment, not a new quantization mode.

# Bound Input-Channel Redundancy Before A New Weight Format

## Question

The qualified reader consumes each logical payload word once. A different
lossless representation could avoid reading some of those bytes, but only
if the actual matrices have enough redundancy after metadata and decoding
costs. The prior independent-byte study did not find sufficient savings.
Its failed indexed byte-code models remain closed.

Test one different statistical question: is byte entropy substantially
lower when conditioned on the weight's known **input-channel coordinate**?
That coordinate is already available to the consumer; it is not a learned
model approximation or an activation-dependent prediction. Per-channel
distributions can differ even when the pooled distribution is broad.

This is a CPU-only information/size feasibility scan, not a new codebook,
encoder, decoder, Huffman parameter search, CUDA kernel or runtime selector.
Do not change the closed models or try alternate conditioning schemes if
this fixed one fails. No performance or physical-traffic claim follows
from entropy alone.

## Fixed Scan And Accounting

- Use original Qwen2.5-3B Q8_0 gate/up tensors at layers 0/17/35. Recheck
  the full model and every tensor SHA against the existing frozen source
  records. Require K=2048, N=11008 and validated Q8_0 byte offsets/shapes.
- Count all payload bytes by input channel across output columns. The
  original layout is `[output][input_block32][scale16,payload32]`.
  Skip scale bytes; retain their full literal storage in size estimates.
- Report every per-channel histogram and Shannon entropy, the pooled
  histogram/entropy and mean conditional entropy. The pooled counts must
  exactly reproduce the independently validated earlier scan.
- Use one fixed optimistic accounting model: ideal conditional payload
  bits, K*256 bytes for code lengths, four-byte offsets for each 256-output
  chunk per channel, and all original FP16 scale bytes. No alignment,
  integral-code-length or decoder lookup/execution costs are included.
  This deliberately optimistic model is not an encoded stream, reversible
  format or universal compression lower bound.
- Require at least **10% total stored-byte saving at every tensor** even
  under that optimistic model before considering a separate format design.
  Passing size does not qualify a kernel. A reader still needs safe random
  access, original FP32 arithmetic, full memory/setup/decoding costs,
  >=5% complete FFN and original actual512/quality/HTTP gates.
- Keep each source buffer below 64 MiB, serial analysis and process memory
  bounded to 512 MiB. Use a temporary CPU-only script, no new dependencies,
  model downloads, CUDA contexts, uploads or hardware settings. Preserve
  its exact source/hash and behavioral checks with the result.

The potential avoided work is fewer stored payload bytes read, not a
different arithmetic expression over the same bytes. Whether compressed
access actually saves physical traffic or time remains unmeasured. Original
GGUF weights remain resident for prefill; any future format would replace
execution copies, not silently remove that source allocation. Decode
tables/workspace are additional costs, not free memory.

## Atomic Delivery And Stop Rule

1. `docs(research): bound channel redundancy scan`: this contract before
   counting actual data or selecting a representation.
2. `docs(perf): record channel redundancy scope`: source-verified raw
   histograms, tested scanner, independent arithmetic audit, exact metadata
   accounting and decision. No inference implementation.

If the optimistic size gate fails, stop without encoder/decoder or another
channel/chunk/codebook/predictor round. Preserve qualified 49.146 HTTP tok/s
and both unmet original targets, 53.06 and 56.037 HTTP tok/s. No defaults,
model semantics or previously closed kernel experiments change.

# Offline Lossless FFN Payload Feasibility

## Question And Bound

Read/checksum and native-warp screens do not justify another instruction or
geometry sweep. Before implementing a different representation, measure how
much information redundancy is actually present in the original Q8 payload.
This is an offline gate, not a CUDA optimization or a model change.

Use the shared Q8 benchmark with frozen M02 source identities and actual
gate/up tensors from layers 0/17/35. Validate model/capture/tensor SHA,
dtype, dimensions and offsets before analyzing read-only bytes. Do not
create a CUDA context, upload weights, change defaults or download a model.

Report each tensor's full 256-symbol payload histogram, signed zero/negative
counts, zero-order Shannon entropy and optimal binary byte-Huffman code
lengths. Count exact code-length bits, not a measured encoded stream.
Model independent chunks of **32, 256 and 1024 payload bytes**, each with
128-bit alignment and one 32-bit offset. Include original two scale bytes
per Q8 block and a 256-byte code-length table per tensor. Do not pretend
decoder lookup storage, prefix-search cost or CUDA execution has been tested.

No encoder/decoder or execution format is implemented in this task. All size
models must say so explicitly. The entropy bound is a zero-order symbol
bound, not a universal lower bound for predictive/model-aware compression.
Huffman counts and alignment/index costs are size evidence, never throughput.

Require at least **10% total stored-byte saving at all six tensors** in one
of the indexed size models before considering any CUDA decoder design.
This is only an offline feasibility threshold, not a promotion criterion.
A passing model still needs a reversible, safely indexed format, bounded
Maxwell consumer and original complete-FFN >=5%, actual512/frozen quality
and HTTP gates. A failure closes these indexed byte-code models without
codebook/chunk-size/predictor tuning. Do not substitute lossy quantization,
sparsity, pruning, approximate attention or speculative decoding.

## Atomic Delivery

1. `docs(research): bound FFN entropy study`: this contract before code.
2. `feat(bench): inspect lossless Q8 payload scope`: shared-runner CPU-only
   mode, provenance/bounds, deterministic histogram/code-cost and behavior
   tests; depends on 1. No CUDA changes or new dependencies.
3. `docs(perf): record FFN payload feasibility`: clean release raw counts,
   independent audit, size/cost limitations and decision; depends on tested 2.

Both exact-output targets, 53.06 HTTP tok/s and 1.30x fresh Ollama, remain
required. Qualified runtime and prepared down/vocabulary stay unchanged.

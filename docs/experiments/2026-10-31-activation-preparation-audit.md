# Activation preparation source audit

The existing Q8_1 FFN path already quantizes each input once for both gate and
up. The remaining experiment is sign/rotation/quantization fusion for a
rotated input, with an explicit consumer and invocation lifetime. It cannot
accelerate the selected unrotated Qwen path by removing a rotation that path
does not execute.

## Upstream evidence

[PR 31398](https://github.com/ggml-org/llama.cpp/pull/31398) was open and unmerged
when inspected on October 31. The reviewed head is
`985b4d7cf8391cc44c156d5cf54a8df9e6bc74f1`, against base
`3c809f44383bcebc318b8cddd8c6d11aaa861e9b`.

The patch fuses sign multiplication into the Hadamard transform and optionally
writes Q8_1 output for later matvec consumers. Its cache resets at graph
evaluation boundaries and before another fused rotation. Reuse requires the
same data address and element count, a contiguous tensor, supported row
padding, and a reshape-only ancestry leading to the cached tensor. Concurrent
CUDA streams disable caching. Fusion also checks graph uses and allocation
dependencies so the eliminated intermediate cannot alias a live input.

The contributor's byte-identity and speed claims apply to the tested rotated
model and newer GPUs. They establish neither Maxwell correctness nor a
benefit for the current Qwen model. The implementation reference is
[the pinned CUDA source](https://github.com/devvexus/ms-llama.cpp/tree/985b4d7cf8391cc44c156d5cf54a8df9e6bc74f1/ggml/src/ggml-cuda).

## Local evidence

At branch checkpoint `2a517e3`, the `Q8FfnKernel::Q81` arm in
[`src/infer/gemm.rs`](../../src/infer/gemm.rs) calls `quantize_q8_1_async` once,
then passes that scratch to one `ffn_q8_1_async` call consuming both weights.
The scratch belongs to the invocation's workspace. There is no second gate/up
conversion to remove in this path.

The retained [Q8_1 screen](2026-10-25-maxwell-q81-ffn-screen.md) separates
encoded-input numerical correctness from changed model outputs. Reusing that
encoding does not make the candidate mathematically exact relative to FP32
activations. Its historical timing also predates the qualified compound FFN;
it is not a contemporary speed comparison.

## Bounded next experiment

Use the branch protocol's isolated fixture, with a fixed 128-coordinate
normalized rotation, widths 2048 and 11008, and rows 1, 2, 4 and 8. Both widths
contain complete rotation blocks. Compare staged sign multiplication,
rotation and Q8_1 encoding with a fused producer. Feed the resulting bytes to
the same existing Q8_1 consumer, with identical rotated weight fixtures.
Record producer-only and producer-plus-consumer latency separately.

Validate encoded bytes and consumed outputs, including changed inputs at the
same address, different shapes, separate workspace owners and explicit stream
dependencies. Keep preparation ownership within one invocation; no global
pointer-keyed cache is needed for this probe. Charge canonical FP32 output
writes if the staged contract still requires them.

No hardware measurement or runtime change follows from this audit. The topic
remains open for the isolated experiment. A preparation-only win requires a
real rotated consumer before integration into model execution.

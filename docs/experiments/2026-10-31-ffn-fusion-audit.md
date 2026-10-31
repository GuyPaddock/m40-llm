# FFN fusion source audit

The batch-one gate/up/SwiGLU mechanism already exists in this repository.
The remaining implementation candidate is multi-row Q4 fusion with the
down consumer held fixed. This audit admits that bounded comparison; it
does not close the branch or establish a new speed result.

## Upstream evidence

[llama.cpp PR 31188](https://github.com/ggml-org/llama.cpp/pull/31188)
merged on October 28. Its head is
`eb4eee1d57a1c3638d1977504a75da9c1278adc9`, base
`a4699400e3ccca756297ef4433eb1d7998fca525`, and merge commit
`f5222f25c18da062ab2b8a2dbbfbf4feae398566`. The metadata and complete diff
were inspected through the GitHub connector.

The change extends existing dense MMQ gate/up/GLU fusion to expert
`MUL_MAT_ID` operations. Dispatch uses expert count and nominal token tiles;
the fusion path excludes cases selected for MMVQ. It passes existing fusion
arguments through the expert path and adds expert-oriented tests. This is
not evidence of a new batch-one Maxwell kernel.

The contributor's measurements use 16,384 prompt tokens, zero generated
tokens, batch and microbatch 2,048, FP16 KV and five repetitions. The stated
5.7% to 17.2% gains apply to those prefill tests on RTX 5090 and DGX Spark.
They provide no matched M40 decoding baseline.

## Local implementation at base 0eee578

In [gemm.rs](../../src/infer/gemm.rs), the Q4 fused dispatch requires `m == 1`,
positive dimensions and Q4_0 gate/up weights. It is enabled by
`M40LLM_FUSED_Q4_MLP_SWIGLU` or the existing Q8 fusion flag. The CUDA entry
`m40llm_mlp_gate_up_swiglu_f32xq4_0_gguf_decode_async` computes both projections
and writes the final SwiGLU vector in one launch. It reads both matrices;
it does not remove their weight traffic. The Rust/CUDA path already avoids
separate gate/up global output buffers for that operation.

The opt-in Q8 `JointSharedScales` path also performs fused gate/up/SwiGLU.
The [existing compound screen](2026-10-27-maxwell-compound-ffn-screen.md)
and [runtime qualification](2026-10-27-maxwell-compound-runtime.md) establish
its implementation and earlier measured benefit. Those retained results
must not be relabeled as October 31 improvements.

Multi-row Q4 cannot take the `m == 1` fused dispatch. Its existing alternatives
are repeated single-row fused operations or separate block-loop gate/up,
SwiGLU and down operations with stream dependencies. Compare both controls.
Use the shared source-bound Q4 fixture encoder, preserving hashes and the
additional-quantization distinction.

## Next experiment

Add one fused multi-row Q4 gate/up/SwiGLU launch for distinct rows 2, 4 and 8.
Preserve the existing single-row gate/up accumulation order and FP32 math.
Keep native down weights and the applicable down consumer unchanged so the
experiment measures fusion rather than a different down layout. Record
launches, scratch writes, complete FFN timings and input reuse. Include batch
one as an existing-implementation control. Require the original numerical,
ownership and paired complete-FFN gates before any runtime integration.

The channel-selection branch's repacked full-retention result changes down
addressing and reduction geometry too. It cannot isolate a fusion benefit
and is not a substitute for this comparison. No CUDA changes or new hardware
measurements were made on the fusion branch during this audit.

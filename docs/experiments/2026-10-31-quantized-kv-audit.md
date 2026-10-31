# Quantized KV source audit and comparison prerequisites

The direct-consumption mechanism is already implemented in M40-LLM. This
study must measure its current benefit before proposing another kernel change.
The audit uses branch checkpoint `3241dde`, based on `0eee578`. No hardware
measurement or runtime change accompanies this checkpoint.

## Upstream evidence

[llama.cpp PR 31125](https://github.com/ggml-org/llama.cpp/pull/31125)
merged on October 29, 2026. The inspected head is
`e81b227c972bb14dfda4e0c66d92671d5070e2f4`; the merge commit is
`f7c01c5226cac1de1052c17920ff38cdefea0ae6`.
The PR description and patch add a Xe2 SYCL attention path for Q4_0/Q8_0
cache data. Its dispatcher chooses direct consumption before existing paths;
the kernel uses integer DPAS for QK and FP16 DPAS for value accumulation.
Those instructions do not provide a Maxwell implementation.

The reported B580 Llama 3.1 8B result rises from 31.3 to 54.2 tokens/s at
16K context. The contributor explicitly reports one end-to-end repetition
and no perplexity or KL evaluation. It motivates a local controlled comparison;
it does not establish a Maxwell speedup or accuracy result. The local mixed
format also differs from GGML Q4_0: old K remains FP16 and old V uses signed
nibbles with FP32 scales per token and KV head.

## Existing local execution

The C ABI implementations are in [kernels.cu](../../cuda/kernels.cu), with
Rust wrappers in [cuda.rs](../../src/cuda.rs). Runtime admission is enforced
by [kv_compression.rs](../../src/kv_compression.rs).

| Consumer | Old storage | Head dimensions | Preparation inside the operation |
| --- | --- | --- | --- |
| Staged Q8 | Q8 K and V | 64 | Select/gather/dequantize to FP16, then attention |
| Direct Q8 | Same Q8 K and V | 64 | Select and reconstruct during attention |
| Staged mixed | FP16 K and signed Q4 V | 64 | Select/gather/dequantize to FP16, then attention |
| Direct mixed | Same FP16 K and signed Q4 V | 64, 128 | Select and reconstruct V during attention |

The mixed staged entry point rejects head dimensions other than 64. It
launches `gather_block_select_exact_q8_old_head64_kernel`, followed by
`attention_last_token_gqa_staged_exact_head64_kernel`, using caller-owned
reusable buffers. Direct mixed dispatch supports 64 and 128, including a
batched entry point. Direct mixed value accumulation explicitly rounds
reconstructed values through FP16 before accumulating in FP32.

The Qwen model used by the other topic branches has head dimension 128.
Consequently, selecting the staged or Q8-direct runtime backend cannot yield
a valid same-model control. Do not bypass its admission check, report a
rejected launch as a timing result, or label a 64-channel fixture as Qwen.
The head128 comparison needs a narrowly scoped staged reference before it
can measure the existing direct consumer's advantage.

## Retained evidence and its limits

[Attention parity tests](../../tests/attention_parity_cuda_grid.rs) cover
staged/direct Q8 and mixed consumers at head64, partial old blocks, and a
separate mixed-direct head128 case. Their staged/direct comparisons use a
numerical tolerance, not a bit-identical output contract. The
[batched attention test](../../tests/attention_batched_varlen.rs) compares
mixed direct batched execution with individual calls. Read these tests before
adding coverage; their existence is not a fresh passing hardware run.

The May 14 direct-Q8 section of [performance history](../perf_baselines.md)
records faster passing 2048-token retrieval rows than an earlier staged sweep.
It also records failures at 4096 and early-stop outputs. These historical
sweeps are not the contemporaneous paired 1K/2K/4K/8K operator comparison
required by this branch. Their quality failures remain failures. A direct
consumer can preserve a staged consumer's output without making selection
or quantization equivalent to dense FP16 attention.

## Next experiment

1. Reuse existing cache append and attention entry points. Start with head64,
   where both controls exist, and independent sequence inputs at batches
   1, 2, 4 and 8. Use contexts 1024, 2048, 4096 and 8192; record recent-window,
   block-size, top-k, GQA and selected-block order explicitly.
2. Freeze plain top-k selection and disable diagnostic masks, anchors and
   alternate policies. Check selected token identities and ordering before
   timing. Within each backing format, use the same cache allocation and
   encoded bytes for staged/direct calls, with identical queries and lengths.
3. Time the complete staged gather plus attention against complete direct
   selection plus attention. Allocate staging outside timing but report its
   residency. Report append/quantization and upload separately. Serial rows
   and the batched mixed dispatcher are separate comparisons.
4. Add a head128 staged reference with the same selection and FP16 rounding
   semantics only after inspecting the existing gather kernel's fixed-width
   assumptions. Keep it experimental. Its correctness checks must cover
   recent-ring wraparound, partial blocks, GQA, multiple physical sequence and
   layer slots, stream completion and cleanup.
5. Include dense FP16 attention as a separate approximation reference. Keep
   full-context and selected-context results distinct. Report numerical
   differences; require the existing model-quality gates before any adoption
   claim. An operator benchmark alone cannot qualify generated sequences.
6. Follow the [frozen branch protocol](2026-10-30-quantized-kv-protocol.md)
   for paired trials and advancement. Implement a performance change only
   if profiling identifies a remaining removable cost. Otherwise retain the
   measured already-implemented result.

The lossless-KV branch currently owns the serial GPU capture. This audit
does not start another GPU workload or close the quantized-KV study.

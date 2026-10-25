# Diagnostic Layer Masks

Layer skipping changes the model. It is research-only, disabled by default,
and must not be reported as an unchanged-model engine improvement.

`LayerMask::new(layer_count, skipped_ids)` accepts unique interior original
layer IDs. It rejects first/last layers, duplicates, out-of-range IDs, and
reuse with a different layer count. Input order is canonicalized. Default or
an explicit empty list runs every layer.

Configure a model with `LoadedModel::set_diagnostic_layer_mask(&[17, 18])`
before KV allocation or session creation. No environment-global mask is used.
The public field supports existing in-memory model construction; normal callers
must use the checked setup method rather than replace it during generation.

All full-model sequential, batched decode, packed-prefill and device-parameter
forward entrypoints retain original tensor/KV layer IDs and skip masked work.
Return values report active execution counts. Single-token scratch alternates
by executed layer index, not original layer parity. The low-level explicit
single-layer routines still execute the requested layer; they provide the
independent retained-layer oracle used in tests.

Source weights and physical KV capacity remain allocated for every original
layer. A mask only removes forward computation. It is not compact model export
and does not reclaim weight/KV memory. Later C24/C25 work will rank candidates
and export separately identified compact artifacts after quality evaluation.

Masked session CUDA Graph replay is rejected; keep `M40LLM_DECODE_GRAPH=0`.
Masked packed-then-compress reconstruction is rejected because it currently
rebuilds every physical slot without a mask. Sequential and packed-prefix paths
remain available. Tests cover dense and preferred FP16-K/q4-V compressed
head64/head128 paths, mixed sequence lengths, biases and absolute RoPE positions.
Packed-prefix parity is tested within the exact recent window, then decode
crosses into old q4-V storage. Packed full-value prefill and sequential prefill
that has already quantized old V are not claimed numerically equivalent.

Validation checks empty-mask full logits, poisoned skipped weights, unchanged
skipped KV, original-ID multi-sequence routing, matched launch/cuBLAS counts
against explicit retained-layer execution, packed/batched outputs and compressed
snapshots, unsupported-mode errors, subsequent usability and RAII cleanup.

```bash
source scripts/dev-env.sh
cargo test --no-default-features --test layer_mask
M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 \
  cargo test --features cuda --test layer_mask_cuda -- --test-threads=1
```

The C20 evaluation-runner integration and fixed one/two/three-layer results
remain required before this checkpoint is complete. No mask is preferred yet.

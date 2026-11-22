# GGUF projection shape validation

Start at current remote main e1403e0. While checking the GSQ tensor-type path, the loader and the tensor-view builder each multiplied dimensions with an unchecked u64 product. The view builder later computes checked strides from the same dimensions. In debug builds the earlier multiplication can panic. In optimized builds the later stride check can still reject an unrepresentable result. Scope the study to consistent recoverable errors for tensor shapes on the ordinary loader entry point.

Use exactly the current layouts, offsets, alignment rules and dtype whitelist. Compute the element count from the checked stride calculation. Preserve valid view strides/bytes and the existing offset, zero-dimension, and dtype rejections. Compare debug and release CPU behavior, run current GGUF device-view/offset suites, and compile the CUDA/server path. Stop at the checked arithmetic repair. This issue does not qualify Qwen3 architecture or K-Quant execution, and supplies no speed claim.

## Disposition

**PASS for checked shape handling, Tier 1.** Candidate 541cec1 is independently based on e1403e0. It touches src/infer/load.rs, src/infer/tensor_views.rs and tests/gguf_device_views.rs. The regular LoadedModel::from_gguf call now returns a contextual error naming the tensor for an unrepresentable element count. No additional crate, flag, CUDA kernel, architecture or weight representation is required. Test in both arithmetic profiles and retain the existing valid byte-count, stride, offset and alignment cases.

The unfixed debug run panicked during multiplication. The first consolidation of the two size checks changed which error appeared for a tensor simultaneously misaligned and out of bounds. The original offset regression caught it. 541cec1 preserves the existing bounds-first behavior. All 15 GGUF view/offset tests passed in both debug and release. The complete warning-as-error non-CUDA suite passed 122 tests. CUDA/server all-target Clippy compiled real sm_52 with cuBLAS. No GPU inference or speed claim is part of this input-validation result. The retained [logs](2026-11-21-gguf-shape/) include the initial failure and the complete corrected runs.

Replay with cargo test --locked --no-default-features --test gguf_device_views --test gguf_device_views_offsets in debug and with --release. Use RUSTFLAGS="-D warnings" for the full non-CUDA suite, and M40LLM_ENABLE_NVCC=1 M40LLM_ENABLE_CUBLAS=1 cargo clippy --features cuda,server --all-targets -- -D warnings for the CUDA build check.

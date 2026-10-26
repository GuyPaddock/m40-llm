// Isolate public-library headers/host features from all inference kernels.
#pragma once
#include <climits>

extern "C" int m40llm_cub_checksum_scratch_bytes(void* stream, size_t* bytes);
extern "C" int m40llm_cub_checksum_launch(
    void* stream, const void* const* scales, const void* const* values,
    void* output, void* scratch, size_t bytes, int rows, int hidden, int k);

extern "C" int m40llm_ffn_q8_cub_checksum_scratch_bytes(
    M40llmCudaContext* ctx, size_t* bytes) {
    if (!ctx || !bytes) return -1;
    return m40llm_cub_checksum_scratch_bytes(ctx->decode_stream, bytes);
}

extern "C" int m40llm_ffn_q8_cub_checksum_async(
    M40llmCudaContext* ctx, const void* const* scales, const void* const* values,
    void* output, void* scratch, size_t bytes, int rows, int hidden, int k) {
    if (!ctx || !scales || !values || !output || !scratch) return -1;
    if ((rows != 1 && rows != 4) || hidden <= 0 || k <= 0 || k % 32 ||
        static_cast<uint64_t>(rows) * hidden * (k / 4) > INT_MAX ||
        reinterpret_cast<uintptr_t>(output) % 8 ||
        reinterpret_cast<uintptr_t>(scratch) % 256 || bytes == 0) return -2;
    for (int i = 0; i < 2; ++i) {
        if (!scales[i] || !values[i] || reinterpret_cast<uintptr_t>(scales[i]) % 2 ||
            reinterpret_cast<uintptr_t>(values[i]) % 4) return -2;
    }
    return m40llm_cub_checksum_launch(ctx->decode_stream, scales, values,
                                     output, scratch, bytes, rows, hidden, k);
}

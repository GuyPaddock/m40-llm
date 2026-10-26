// Activation-load-only probe; keep the qualified packed weights/shared tree.
#pragma once

__global__ void q8_separate_packed_activation_kernel(
    const float* __restrict__ input, const __half* __restrict__ scales,
    const int8_t* __restrict__ values, float* __restrict__ output, int N, int K) {
    const int col0 = blockIdx.x * 4;
    const int row = blockIdx.y;
    const int blocks_per_col = K / 32;
    const float* row_input = input + static_cast<size_t>(row) * K;
    float acc[4] = {};
    for (int task = threadIdx.x; task < blocks_per_col * 4; task += blockDim.x) {
        const int qblock = task / 4;
        const int offset = (task % 4) * 8;
        const int k0 = qblock * 32 + offset;
        const float4 lo = *reinterpret_cast<const float4*>(row_input + k0);
        const float4 hi = *reinterpret_cast<const float4*>(row_input + k0 + 4);
        const float activation[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        q8_separate_accumulate<4>(scales, values, activation, col0, N,
                                  blocks_per_col, qblock, offset, acc);
    }
    q8_separate_reduce<4>(acc);
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (col0 + c < N) output[static_cast<size_t>(row) * N + col0 + c] = acc[c];
        }
    }
}

extern "C" int m40llm_gemm_f32xq8_separate_packed_activation_async(
    M40llmCudaContext* ctx, const void* input, const void* scales,
    const void* values, void* output, int M, int N, int K, int threads,
    uint32_t stream_kind) {
    if (!ctx || !input || !scales || !values || !output || input == output ||
        scales == output || values == output) return -1;
    if (M <= 0 || M > 65535 || N <= 0 || K <= 0 || K > INT32_MAX - 31 || K % 32 ||
        (threads != 128 && threads != 256) || stream_kind > 1 ||
        reinterpret_cast<uintptr_t>(input) % 16 || reinterpret_cast<uintptr_t>(output) % 4 ||
        reinterpret_cast<uintptr_t>(scales) % 2 || reinterpret_cast<uintptr_t>(values) % 8) return -4;
    const cudaStream_t stream = stream_kind == 1 ? ctx->decode_stream : ctx->prefill_stream;
    const dim3 grid((static_cast<unsigned>(N) + 3) / 4, M);
    q8_separate_packed_activation_kernel<<<grid,threads,4 * threads * sizeof(float),stream>>>(
        static_cast<const float*>(input),static_cast<const __half*>(scales),
        static_cast<const int8_t*>(values),static_cast<float*>(output),N,K);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

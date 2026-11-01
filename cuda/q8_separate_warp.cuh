// Fixed-geometry experiment on existing separate scale/value planes.
#pragma once

template<int Threads>
__global__ void q8_separate_projection_warp_kernel(
    const float* __restrict__ input, const __half* __restrict__ scales,
    const int8_t* __restrict__ values, float* __restrict__ output, int N, int K) {
    const int col0 = blockIdx.x * 4;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int blocks_per_col = (K + 31) / 32;
    const float* row_input = input + static_cast<size_t>(row) * K;
    float acc[4] = {};
    // Keep the control's tasks, vector loads and FP32 association unchanged.
    for (int task = tid; task < blocks_per_col * 4; task += blockDim.x) {
        const int qblock = task / 4;
        const int offset = (task % 4) * 8;
        const int k0 = qblock * 32 + offset;
        float activation[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            activation[i] = k0 + i < K ? row_input[k0 + i] : 0.0f;
        }
        q8_separate_accumulate<4>(scales, values, activation, col0, N,
                                  blocks_per_col, qblock, offset, acc);
    }
    extern __shared__ float published[];
    #pragma unroll
    for (int c = 0; c < 4; ++c) published[c * Threads + tid] = acc[c];
    __syncthreads();
    if (tid < 32) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float value = m40llm_q8_ffn::reduce_published<Threads>(
                published + c * Threads, tid);
            if (tid == 0 && col0 + c < N) {
                output[static_cast<size_t>(row) * N + col0 + c] = value;
            }
        }
    }
}

extern "C" int m40llm_gemm_f32xq8_separate_warp_async(
    M40llmCudaContext* ctx, const void* input, const void* scales,
    const void* values, void* output, int M, int N, int K, int threads,
    uint32_t stream_kind) {
    if (!ctx || !input || !scales || !values || !output || input == output ||
        scales == output || values == output) return -1;
    if (M <= 0 || M > 65535 || N <= 0 || K <= 0 || K > INT32_MAX - 31 ||
        (threads != 128 && threads != 256) || stream_kind > 1 ||
        reinterpret_cast<uintptr_t>(input) % 4 || reinterpret_cast<uintptr_t>(output) % 4 ||
        reinterpret_cast<uintptr_t>(scales) % 2 || reinterpret_cast<uintptr_t>(values) % 8) return -4;
    const cudaStream_t stream = stream_kind == 1 ? ctx->decode_stream : ctx->prefill_stream;
    const dim3 grid((static_cast<unsigned>(N) + 3) / 4, M);
    const size_t shared = 4 * threads * sizeof(float);
    if (threads == 128) {
        q8_separate_projection_warp_kernel<128><<<grid,128,shared,stream>>>(
            static_cast<const float*>(input),static_cast<const __half*>(scales),
            static_cast<const int8_t*>(values),static_cast<float*>(output),N,K);
    } else {
        q8_separate_projection_warp_kernel<256><<<grid,256,shared,stream>>>(
            static_cast<const float*>(input),static_cast<const __half*>(scales),
            static_cast<const int8_t*>(values),static_cast<float*>(output),N,K);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

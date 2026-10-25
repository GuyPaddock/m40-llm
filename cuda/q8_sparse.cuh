// C21 experimental 16-of-32 Q8 blocks. No runtime model dispatch.
#pragma once

template<int Columns>
__global__ void q8_sparse_projection_kernel(
    const float* __restrict__ input, const unsigned char* __restrict__ weights,
    float* __restrict__ output, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const int warps = blockDim.x / 32;
    const int col0 = blockIdx.x * Columns;
    const size_t row = blockIdx.y;
    const int blocks = (K + 31) / 32;
    float acc[Columns] = {};
    for (int qblock = warp; qblock < blocks; qblock += warps) {
        const int k = qblock * 32 + lane;
        const float a = k < K ? input[row * K + k] : 0.0f;
        #pragma unroll
        for (int c = 0; c < Columns; ++c) {
            if (col0 + c < N) {
                const unsigned char* block = weights +
                    (static_cast<size_t>(col0 + c) * blocks + qblock) * 24;
                const uint32_t mask = *reinterpret_cast<const uint32_t*>(block);
                const uint32_t bit = 1u << lane;
                const int rank = __popc(mask & (bit - 1));
                const float scale = __half2float(*reinterpret_cast<const __half*>(block + 4));
                if ((mask & bit) && k < K) {
                    const int8_t value = reinterpret_cast<const int8_t*>(block + 8)[rank];
                    acc[c] += a * (static_cast<float>(value) * scale);
                }
            }
        }
    }
    __shared__ float partial[Columns][8];
    #pragma unroll
    for (int c = 0; c < Columns; ++c) {
        for (int offset = 16; offset > 0; offset /= 2) {
            acc[c] += __shfl_down_sync(0xffffffffu, acc[c], offset);
        }
        if (lane == 0) partial[c][warp] = acc[c];
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int c = 0; c < Columns; ++c) {
            float sum = lane < warps ? partial[c][lane] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            }
            if (lane == 0 && col0 + c < N) output[row * N + col0 + c] = sum;
        }
    }
}

extern "C" int m40llm_gemm_f32xq8_sparse_f32_async(
    M40llmCudaContext* ctx, const void* input, const void* weights, void* output,
    int M, int N, int K, int columns, int threads) {
    if (!ctx || !input || !weights || !output || input == output || weights == output) return -1;
    if (M <= 0 || M > 65535 || N <= 0 || K <= 0 || K > INT32_MAX - 31 ||
        (columns != 1 && columns != 4) || (threads != 128 && threads != 256)) return -4;
    const dim3 grid((static_cast<unsigned>(N) + columns - 1) / columns, M);
    if (columns == 1) {
        q8_sparse_projection_kernel<1><<<grid, threads, 0, ctx->decode_stream>>>(
            static_cast<const float*>(input), static_cast<const unsigned char*>(weights),
            static_cast<float*>(output), N, K);
    } else {
        q8_sparse_projection_kernel<4><<<grid, threads, 0, ctx->decode_stream>>>(
            static_cast<const float*>(input), static_cast<const unsigned char*>(weights),
            static_cast<float*>(output), N, K);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

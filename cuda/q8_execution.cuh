// Benchmark-only lossless Q8 execution layouts. Included after the context type.
#pragma once

template<int Columns>
__global__ void q8_separate_projection_kernel(
    const float* __restrict__ input,
    const __half* __restrict__ scales,
    const int8_t* __restrict__ values,
    float* __restrict__ output,
    int N, int K) {
    const int col0 = blockIdx.x * Columns;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int blocks_per_col = (K + 31) / 32;
    float acc[Columns] = {};
    for (int task = tid; task < blocks_per_col * 4; task += blockDim.x) {
        const int qblock = task / 4;
        const int offset = (task % 4) * 8;
        const int k0 = qblock * 32 + offset;
        float activation[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            activation[i] = k0 + i < K ? input[static_cast<size_t>(row) * K + k0 + i] : 0.0f;
        }
        #pragma unroll
        for (int c = 0; c < Columns; ++c) {
            if (col0 + c < N) {
                const size_t block = static_cast<size_t>(col0 + c) * blocks_per_col + qblock;
                const float scale = __half2float(scales[block]);
                // Every 32-byte value block is aligned, unlike GGUF's 34-byte stride.
                const uint2 packed = *reinterpret_cast<const uint2*>(values + block * 32 + offset);
                const int8_t* q = reinterpret_cast<const int8_t*>(&packed);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[c] += activation[i] * (static_cast<float>(q[i]) * scale);
                }
            }
        }
    }
    extern __shared__ float reduction[];
    #pragma unroll
    for (int c = 0; c < Columns; ++c) {
        reduction[c * blockDim.x + tid] = acc[c];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int c = 0; c < Columns; ++c) {
                reduction[c * blockDim.x + tid] += reduction[c * blockDim.x + tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        #pragma unroll
        for (int c = 0; c < Columns; ++c) {
            if (col0 + c < N) {
                output[static_cast<size_t>(row) * N + col0 + c] = reduction[c * blockDim.x];
            }
        }
    }
}

extern "C" int m40llm_gemm_f32xq8_separate_f32_async(
    M40llmCudaContext* ctx, const void* input, const void* scales,
    const void* values, void* output, int M, int N, int K, int columns, int threads) {
    if (!ctx || !input || !scales || !values || !output) return -1;
    if (M <= 0 || M > 65535 || N <= 0 || K <= 0 || K > INT32_MAX - 31 ||
        (columns != 1 && columns != 4) || (threads != 128 && threads != 256)) return -4;
    const dim3 grid((static_cast<unsigned>(N) + columns - 1) / columns, M);
    const size_t shared = columns * threads * sizeof(float);
    if (columns == 1) {
        q8_separate_projection_kernel<1><<<grid, threads, shared, ctx->decode_stream>>>(
            static_cast<const float*>(input), static_cast<const __half*>(scales),
            static_cast<const int8_t*>(values), static_cast<float*>(output), N, K);
    } else {
        q8_separate_projection_kernel<4><<<grid, threads, shared, ctx->decode_stream>>>(
            static_cast<const float*>(input), static_cast<const __half*>(scales),
            static_cast<const int8_t*>(values), static_cast<float*>(output), N, K);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

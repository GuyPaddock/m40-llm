// Four output columns share each decoded weight across up to four input rows.
#pragma once

template<int Rows>
__global__ void q8_small_rows_kernel(
    const float* __restrict__ input, const __half* __restrict__ scales,
    const int8_t* __restrict__ values, float* __restrict__ output,
    int rows, int columns, int width) {
    const int row0 = blockIdx.y * Rows;
    const int column0 = blockIdx.x * 4;
    const int blocks = (width + 31) / 32;
    float sums[Rows][4] = {};
    for (int task = threadIdx.x; task < blocks * 4; task += blockDim.x) {
        const int index = task * 8;
        float activation[Rows][8];
        #pragma unroll
        for (int row = 0; row < Rows; ++row) {
            #pragma unroll
            for (int lane = 0; lane < 8; ++lane) {
                activation[row][lane] = row0 + row < rows && index + lane < width
                    ? input[static_cast<size_t>(row0 + row) * width + index + lane] : 0;
            }
        }
        #pragma unroll
        for (int col = 0; col < 4; ++col) {
            if (column0 + col >= columns) continue;
            const size_t block = static_cast<size_t>(column0 + col) * blocks + task / 4;
            const float scale = __half2float(scales[block]);
            const uint2 packed = *reinterpret_cast<const uint2*>(values + block * 32 + (task % 4) * 8);
            const int8_t* codes = reinterpret_cast<const int8_t*>(&packed);
            #pragma unroll
            for (int lane = 0; lane < 8; ++lane) {
                const float weight = static_cast<float>(codes[lane]) * scale;
                #pragma unroll
                for (int row = 0; row < Rows; ++row) {
                    sums[row][col] += activation[row][lane] * weight;
                }
            }
        }
    }
    #pragma unroll
    for (int row = 0; row < Rows; ++row) {
        q8_separate_reduce<4>(sums[row]);
        if (threadIdx.x == 0 && row0 + row < rows) {
            #pragma unroll
            for (int col = 0; col < 4; ++col) {
                if (column0 + col < columns) {
                    output[static_cast<size_t>(row0 + row) * columns + column0 + col] = sums[row][col];
                }
            }
        }
        if (row + 1 < Rows) __syncthreads();
    }
}

template<int Rows>
static void launch_q8_small_rows(const void* input, const void* scales, const void* values,
                                  void* output, int m, int n, int k, cudaStream_t stream) {
    const dim3 grid((static_cast<unsigned>(n) + 3) / 4, (m + Rows - 1) / Rows);
    q8_small_rows_kernel<Rows><<<grid, 256, 4 * 256 * sizeof(float), stream>>>(
        static_cast<const float*>(input), static_cast<const __half*>(scales),
        static_cast<const int8_t*>(values), static_cast<float*>(output), m, n, k);
}

extern "C" int m40llm_q8_small_rows_async(M40llmCudaContext* ctx, const void* input,
    const void* scales, const void* values, void* output, int m, int n, int k) {
    if (!ctx || !input || !scales || !values || !output) return -1;
    if (m <= 0 || m > 16 || n <= 0 || k <= 0 || k > INT32_MAX - 31) return -4;
    switch (m) {
        case 1: launch_q8_small_rows<1>(input, scales, values, output, m, n, k, ctx->decode_stream); break;
        case 2: launch_q8_small_rows<2>(input, scales, values, output, m, n, k, ctx->decode_stream); break;
        case 3: launch_q8_small_rows<3>(input, scales, values, output, m, n, k, ctx->decode_stream); break;
        default: launch_q8_small_rows<4>(input, scales, values, output, m, n, k, ctx->decode_stream);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

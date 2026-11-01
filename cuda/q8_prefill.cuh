// Temporary Q8_0 -> column-major FP32 conversion for bounded prefill SGEMM.
// Included after the CUDA context definition, outside the C ABI linkage block.
#pragma once

namespace m40llm_q8_prefill {

__global__ void dequantize_colmajor(const unsigned char* __restrict__ source,
    float* __restrict__ output, int input_dim, int output_dim) {
    __shared__ float tile[32][33];
    const int lane = threadIdx.x;
    const int stripe = threadIdx.y;
    const int input = blockIdx.x * 32 + lane;
    const int blocks_per_column = input_dim / 32;
    #pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        const int column = blockIdx.y * 32 + stripe + offset;
        if (input < input_dim && column < output_dim) {
            const size_t base =
                (static_cast<size_t>(column) * blocks_per_column + blockIdx.x) * 34;
            const float scale = __half2float(*reinterpret_cast<const __half*>(source + base));
            const int8_t value = *reinterpret_cast<const int8_t*>(source + base + 2 + lane);
            tile[stripe + offset][lane] = scale * static_cast<float>(value);
        }
    }
    __syncthreads();
    // Transpose through a bank-padded tile: contiguous packed reads become
    // contiguous FP32 writes in the N-leading layout consumed by cuBLAS.
    const int column = blockIdx.y * 32 + lane;
    #pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        const int row = blockIdx.x * 32 + stripe + offset;
        if (row < input_dim && column < output_dim) {
            output[static_cast<size_t>(row) * output_dim + column] = tile[lane][stripe + offset];
        }
    }
}

} // namespace m40llm_q8_prefill

extern "C" int m40llm_dequantize_q8_0_to_f32_colmajor_async(
    M40llmCudaContext* ctx, const void* source, void* output, size_t capacity_bytes,
    int input_dim, int output_dim) {
    if (!ctx || !source || !output || source == output) return -1;
    if (input_dim <= 0 || input_dim % 32 != 0 || output_dim <= 0) return -2;
    constexpr size_t max_bytes = 128 * 1024 * 1024;
    if (static_cast<size_t>(output_dim) > max_bytes / (sizeof(float) * input_dim)) return -3;
    const size_t required = static_cast<size_t>(input_dim) * output_dim * sizeof(float);
    if (capacity_bytes < required) return -3;
    if (reinterpret_cast<uintptr_t>(source) % 2 || reinterpret_cast<uintptr_t>(output) % 4) return -4;
    const dim3 threads(32, 8);
    const dim3 blocks(input_dim / 32, (output_dim + 31) / 32);
    m40llm_q8_prefill::dequantize_colmajor<<<blocks, threads, 0, ctx->prefill_stream>>>(
        static_cast<const unsigned char*>(source), static_cast<float*>(output), input_dim, output_dim);
    return cudaGetLastError() == cudaSuccess ? 0 : -5;
}

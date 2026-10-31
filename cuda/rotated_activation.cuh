#pragma once
#include "q8_activation.cuh"

namespace m40llm_rotated_activation {

__global__ void apply_signs(const float* __restrict__ input,
    const int8_t* __restrict__ signs, float* __restrict__ output, int count, int columns) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] * static_cast<float>(signs[index % columns]);
}

template<bool IsFused>
__global__ void rotate(const float* __restrict__ input, const int8_t* __restrict__ signs,
    float* __restrict__ output, m40llm_q8_activation::Block* __restrict__ quantized,
    int columns) {
    const int lane = threadIdx.x;
    const int column = blockIdx.x * 128 + lane;
    const size_t index = static_cast<size_t>(blockIdx.y) * columns + column;
    __shared__ float values[128];
    float value = input[index];
    if constexpr (IsFused) value *= static_cast<float>(signs[column]);
    values[lane] = value;
    __syncthreads();
    #pragma unroll
    for (int distance = 1; distance < 128; distance *= 2) {
        const float own = values[lane];
        const float other = values[lane ^ distance];
        __syncthreads();
        values[lane] = (lane & distance) ? other - own : own + other;
        __syncthreads();
    }
    value = values[lane] * 0.08838834764831844f;
    output[index] = value;
    if constexpr (IsFused) {
        m40llm_q8_activation::encode_value(value, lane % 32, quantized[index / 32]);
    }
}
}

extern "C" int m40llm_prepare_rotated_q8_1_async(M40llmCudaContext* ctx,
    const void* input, const void* signs, void* signed_input, void* rotated,
    void* quantized, int rows, int columns, int is_fused, uint32_t stream_kind) {
    if (!ctx || !input || !signs || !signed_input || !rotated || !quantized) return -1;
    if ((rows != 1 && rows != 2 && rows != 4 && rows != 8) || columns <= 0 ||
        columns % 128 || columns > (INT_MAX - 255) / rows || stream_kind > 1 ||
        (is_fused != 0 && is_fused != 1) || reinterpret_cast<uintptr_t>(input) % 4 ||
        reinterpret_cast<uintptr_t>(signed_input) % 4 || reinterpret_cast<uintptr_t>(rotated) % 4 ||
        reinterpret_cast<uintptr_t>(quantized) % 4) return -2;
    const cudaStream_t stream = stream_kind == 0 ? ctx->prefill_stream : ctx->decode_stream;
    using namespace m40llm_rotated_activation;
    auto* encoded = static_cast<m40llm_q8_activation::Block*>(quantized);
    if (is_fused) {
        rotate<true><<<dim3(columns / 128, rows),128,0,stream>>>(
            static_cast<const float*>(input), static_cast<const int8_t*>(signs),
            static_cast<float*>(rotated), encoded, columns);
    } else {
        const int count = rows * columns;
        apply_signs<<<(count + 255) / 256,256,0,stream>>>(static_cast<const float*>(input),
            static_cast<const int8_t*>(signs), static_cast<float*>(signed_input), count, columns);
        if (cudaGetLastError() != cudaSuccess) return -3;
        rotate<false><<<dim3(columns / 128, rows),128,0,stream>>>(
            static_cast<const float*>(signed_input), static_cast<const int8_t*>(signs),
            static_cast<float*>(rotated), encoded, columns);
        if (cudaGetLastError() != cudaSuccess) return -4;
        // Flatten complete blocks; no quantization block crosses a true row boundary.
        m40llm_q8_activation::quantize<<<(count + 255) / 256,256,0,stream>>>(
            static_cast<const float*>(rotated), encoded, count, count);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -5;
}

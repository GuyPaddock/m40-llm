// Experiment-only FP32 sign-add consumer for the SA4_128 format from ggml
// 93a0b5111eb71c90f302f1268f1535f83e2e1f13. This does not reproduce upstream's
// Q8_1/DP4A execution. Format notice: Copyright (c) 2023-2026 The ggml authors.
// SPDX-License-Identifier: MIT; full notice in ../src/shiftadd/LICENSE.txt.
#pragma once
#include <cstdint>

namespace m40llm_shiftadd {

__global__ void project(const float* __restrict__ input,
    const unsigned char* __restrict__ weights, float* __restrict__ output,
    int output_width, int input_width) {
    const unsigned int channel = blockIdx.x * 4u + threadIdx.x / 32u;
    if (channel >= static_cast<unsigned int>(output_width)) return;
    const int lane = threadIdx.x % 32;
    const size_t row = blockIdx.y;
    const int blocks = input_width / 256;
    const float* activation = input + row * input_width;
    float accumulated = 0.0f;
    for (int block_index = 0; block_index < blocks; ++block_index) {
        const unsigned char* block = weights + (static_cast<size_t>(channel) * blocks + block_index) * 138;
        #pragma unroll
        for (int group = 0; group < 2; ++group) {
            float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int part = 0; part < 4; ++part) {
                const int index = group * 128 + lane + part * 32;
                const float value = activation[block_index * 256 + index];
                #pragma unroll
                for (int plane = 0; plane < 4; ++plane) {
                    const bool is_positive = (block[plane * 32 + index / 8] >> (index % 8)) & 1;
                    sums[plane] += is_positive ? value : -value;
                }
            }
            #pragma unroll
            for (int distance = 16; distance > 0; distance >>= 1) {
                #pragma unroll
                for (int plane = 0; plane < 4; ++plane) {
                    sums[plane] += __shfl_down_sync(0xffffffffu, sums[plane], distance);
                }
            }
            if (lane == 0) {
                const float scale = __half2float(*reinterpret_cast<const __half*>(block + 136));
                #pragma unroll
                for (int plane = 0; plane < 4; ++plane) {
                    const float amplitude = scale * static_cast<float>(block[128 + group * 4 + plane]);
                    accumulated += amplitude * sums[plane];
                }
            }
        }
    }
    if (lane == 0) output[row * output_width + channel] = accumulated;
}

inline bool has_valid_span(const void* pointer, size_t bytes, uintptr_t alignment) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(pointer);
    return start != 0 && start % alignment == 0 && bytes <= UINTPTR_MAX - start;
}

inline bool has_overlap(const void* left, size_t left_bytes, const void* right, size_t right_bytes) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(left);
    const uintptr_t b = reinterpret_cast<uintptr_t>(right);
    return a < b + right_bytes && b < a + left_bytes;
}
}

extern "C" int m40llm_gemv_f32xsa4_128_async(M40llmCudaContext* ctx,
    const void* input, const void* weights, void* output,
    int rows, int output_width, int input_width, uint32_t stream_kind) {
    if (!ctx || stream_kind > 1) return -1;
    if ((rows != 1 && rows != 2 && rows != 4 && rows != 8) ||
        output_width <= 0 || input_width <= 0 || input_width % 256 != 0) return -2;
    const size_t input_bytes = static_cast<size_t>(rows) * input_width * sizeof(float);
    const size_t weight_bytes = static_cast<size_t>(output_width) * (input_width / 256) * 138;
    const size_t output_bytes = static_cast<size_t>(rows) * output_width * sizeof(float);
    if (!m40llm_shiftadd::has_valid_span(input,input_bytes,4) ||
        !m40llm_shiftadd::has_valid_span(weights,weight_bytes,2) ||
        !m40llm_shiftadd::has_valid_span(output,output_bytes,4)) return -3;
    if (m40llm_shiftadd::has_overlap(output,output_bytes,input,input_bytes) ||
        m40llm_shiftadd::has_overlap(output,output_bytes,weights,weight_bytes)) return -4;
    cudaStream_t stream = stream_kind == 0 ? ctx->prefill_stream : ctx->decode_stream;
    const unsigned int channel_blocks = (static_cast<unsigned int>(output_width) - 1u) / 4u + 1u;
    m40llm_shiftadd::project<<<dim3(channel_blocks,rows),128,0,stream>>>(
        static_cast<const float*>(input), static_cast<const unsigned char*>(weights),
        static_cast<float*>(output), output_width, input_width);
    return cudaGetLastError() == cudaSuccess ? 0 : -5;
}

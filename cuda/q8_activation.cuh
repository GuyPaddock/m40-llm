// Q8_1 encoding adapted from llama.cpp 9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e:
// ggml/src/ggml-common.h block_q8_1; ggml-cuda/quantize.cu quantize_q8_1;
// ggml-cuda/common.cuh warp_reduce_max/warp_reduce_sum, sm_52 branches only.
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT; complete notice in licenses/llama.cpp-MIT.txt.
#pragma once
#include <climits>

namespace m40llm_q8_activation {

struct alignas(4) Block {
    __half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Block) == 36, "Q8_1 activation storage must be 36 bytes");

__global__ void quantize(const float* __restrict__ input, Block* __restrict__ output,
                         int columns, int padded_columns) {
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    // Complete 32-lane warps are either active or inactive under this bound.
    if (position >= padded_columns) return;
    const int row = blockIdx.y;
    const int lane = position % 32;
    const float value = position < columns ? input[static_cast<size_t>(row)*columns+position] : 0.0f;
    float maximum = fabsf(value);
    float sum = value;
    #pragma unroll
    for (int distance = 16; distance > 0; distance >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, distance));
        sum += __shfl_xor_sync(0xffffffffu, sum, distance);
    }
    const float scale = maximum / 127.0f;
    const int8_t q = maximum == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));
    Block& block = output[static_cast<size_t>(row)*(padded_columns/32)+position/32];
    block.qs[lane] = q;
    if (lane == 0) block.ds = __floats2half2_rn(scale, sum);
}
}

extern "C" int m40llm_quantize_q8_1_async(M40llmCudaContext* ctx, const void* input,
    void* output, int rows, int columns, int padded_columns) {
    if (!ctx || !input || !output || input == output) return -1;
    if ((rows != 1 && rows != 4) || columns <= 0 || padded_columns < columns ||
        padded_columns > INT_MAX - 255 || padded_columns % 32 ||
        reinterpret_cast<uintptr_t>(input) % 4 || reinterpret_cast<uintptr_t>(output) % 4) return -2;
    m40llm_q8_activation::quantize<<<dim3((padded_columns+255)/256,rows),256,0,ctx->decode_stream>>>(
        static_cast<const float*>(input),static_cast<m40llm_q8_activation::Block*>(output),columns,padded_columns);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

// Q8_1 encoding adapted from llama.cpp 9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e:
// ggml/src/ggml-common.h block_q8_1; ggml-cuda/quantize.cu quantize_q8_1;
// ggml-cuda/common.cuh warp_reduce_max/warp_reduce_sum, sm_52 branches only.
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT; complete notice in licenses/llama.cpp-MIT.txt.
#pragma once
#include <climits>
#include "q8_ffn.cuh"

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

// Packed access and signed dot are adapted from pinned vecdotq.cuh::get_int_b2
// and common.cuh::ggml_cuda_dp4a's pre-sm_61 fallback. No DP4A instruction.
__device__ __forceinline__ uint32_t load_weight_word(const unsigned char* values) {
    const uint16_t* halves = reinterpret_cast<const uint16_t*>(values);
    return static_cast<uint32_t>(halves[0]) | (static_cast<uint32_t>(halves[1]) << 16);
}

__device__ __forceinline__ int dot_four(uint32_t a, uint32_t b) {
    return static_cast<int8_t>(a)*static_cast<int8_t>(b)
        + static_cast<int8_t>(a >> 8)*static_cast<int8_t>(b >> 8)
        + static_cast<int8_t>(a >> 16)*static_cast<int8_t>(b >> 16)
        + static_cast<int8_t>(a >> 24)*static_cast<int8_t>(b >> 24);
}

template<int Threads, bool HasIntermediates>
__global__ void gate_up_swiglu(
    const Block* __restrict__ input, const unsigned char* __restrict__ gate,
    const unsigned char* __restrict__ up, float* __restrict__ output,
    float* __restrict__ gate_output, float* __restrict__ up_output,
    int hidden, int columns, int padded_columns) {
    const int col = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int blocks = columns / 32;
    const Block* activation = input + static_cast<size_t>(row)*(padded_columns/32);
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int task = tid; task < blocks*4; task += Threads) {
        const int block = task/4;
        const int offset = (task%4)*8;
        const size_t weight_base = (static_cast<size_t>(col)*blocks+block)*34;
        const float a_scale = __half2float(__low2half(activation[block].ds));
        const float g_scale = __half2float(*reinterpret_cast<const __half*>(gate+weight_base));
        const float u_scale = __half2float(*reinterpret_cast<const __half*>(up+weight_base));
        int g_dot = 0;
        int u_dot = 0;
        #pragma unroll
        for (int word = 0; word < 2; ++word) {
            const uint32_t a = *reinterpret_cast<const uint32_t*>(activation[block].qs+offset+word*4);
            const uint32_t g = load_weight_word(gate+weight_base+2+offset+word*4);
            const uint32_t u = load_weight_word(up+weight_base+2+offset+word*4);
            g_dot += dot_four(g,a);
            u_dot += dot_four(u,a);
        }
        gate_acc += (g_scale*a_scale)*static_cast<float>(g_dot);
        up_acc += (u_scale*a_scale)*static_cast<float>(u_dot);
    }
    extern __shared__ float published_q8_1[];
    published_q8_1[tid] = gate_acc;
    published_q8_1[tid+Threads] = up_acc;
    __syncthreads();
    if (tid < 32) {
        const float g = m40llm_q8_ffn::reduce_published<Threads>(published_q8_1,tid);
        const float u = m40llm_q8_ffn::reduce_published<Threads>(published_q8_1+Threads,tid);
        if (tid == 0) {
            const size_t index = static_cast<size_t>(row)*hidden+col;
            if constexpr (HasIntermediates) {
                gate_output[index] = g;
                up_output[index] = u;
            }
            output[index] = g*(1.0f/(1.0f+expf(-g)))*u;
        }
    }
}

template<int Threads, bool HasIntermediates>
void launch_ffn(M40llmCudaContext* ctx, const void* input, const void* gate, const void* up,
    void* output, void* gate_output, void* up_output, int rows, int hidden, int columns, int padded_columns) {
    gate_up_swiglu<Threads,HasIntermediates><<<dim3(hidden,rows),Threads,2*Threads*sizeof(float),ctx->decode_stream>>>(
        static_cast<const Block*>(input),static_cast<const unsigned char*>(gate),static_cast<const unsigned char*>(up),
        static_cast<float*>(output),static_cast<float*>(gate_output),static_cast<float*>(up_output),hidden,columns,padded_columns);
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

extern "C" int m40llm_ffn_q8_1_async(M40llmCudaContext* ctx, const void* input,
    const void* gate, const void* up, void* output, void* gate_output, void* up_output,
    int rows, int hidden, int columns, int padded_columns, int threads) {
    if (!ctx || !input || !gate || !up || !output || input == output) return -1;
    if ((rows != 1 && rows != 4) || hidden <= 0 || columns <= 0 || columns % 32 ||
        padded_columns < columns || padded_columns % 32 ||
        reinterpret_cast<uintptr_t>(input) % 4 || reinterpret_cast<uintptr_t>(output) % 4 ||
        reinterpret_cast<uintptr_t>(gate) % 2 || reinterpret_cast<uintptr_t>(up) % 2) return -2;
    if ((gate_output == nullptr) != (up_output == nullptr) ||
        reinterpret_cast<uintptr_t>(gate_output) % 4 || reinterpret_cast<uintptr_t>(up_output) % 4) return -3;
    using namespace m40llm_q8_activation;
    if (threads == 128) {
        if (gate_output) launch_ffn<128,true>(ctx,input,gate,up,output,gate_output,up_output,rows,hidden,columns,padded_columns);
        else launch_ffn<128,false>(ctx,input,gate,up,output,nullptr,nullptr,rows,hidden,columns,padded_columns);
    } else if (threads == 256) {
        if (gate_output) launch_ffn<256,true>(ctx,input,gate,up,output,gate_output,up_output,rows,hidden,columns,padded_columns);
        else launch_ffn<256,false>(ctx,input,gate,up,output,nullptr,nullptr,rows,hidden,columns,padded_columns);
    } else return -4;
    return cudaGetLastError() == cudaSuccess ? 0 : -5;
}

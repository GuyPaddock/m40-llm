// Bounded FP32-activation ablations of the native split4 Q8_0 fused FFN.
// Included after the CUDA context definition, outside the C ABI linkage block.
#pragma once

namespace m40llm_q8_ffn {

template<int Threads>
__device__ __forceinline__ float reduce_published(const float* values, int lane) {
    float partial[Threads / 32];
    #pragma unroll
    for (int warp = 0; warp < Threads / 32; ++warp) {
        partial[warp] = values[lane + warp * 32];
    }
    // Match native block-wide strides (128,64,32 at 256 threads), including
    // FP32 association, before finishing the same tree inside one warp.
    #pragma unroll
    for (int stride = Threads / 64; stride > 0; stride >>= 1) {
        #pragma unroll
        for (int index = 0; index < stride; ++index) {
            partial[index] += partial[index + stride];
        }
    }
    float sum = partial[0];
    #pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
        const float other = __shfl_down_sync(0xffffffffu, sum, stride);
        if (lane < stride) sum += other;
    }
    return sum;
}

template<bool HasPackedLoads, bool HasWarpReduction, int Threads>
__global__ void gate_up_swiglu(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate,
    const unsigned char* __restrict__ up,
    float* __restrict__ output, int hidden, int k) {
    const int column = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int blocks = k / 32;
    const float* activation = input + static_cast<size_t>(row) * k;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    // Identical split4 task assignment and local FMA sequence at 256 threads.
    for (int task = tid; task < blocks * 4; task += Threads) {
        const int block = task / 4;
        const int offset = (task - block * 4) * 8;
        const int input_base = block * 32 + offset;
        const size_t weight_base = (static_cast<size_t>(column) * blocks + block) * 34;
        const float gate_scale = __half2float(*reinterpret_cast<const __half*>(gate + weight_base));
        const float up_scale = __half2float(*reinterpret_cast<const __half*>(up + weight_base));
        if constexpr (HasPackedLoads) {
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                // A 34-byte Q8_0 stride guarantees only two-byte alignment.
                // Never reinterpret this payload as an aligned uint32 vector.
                const size_t position = weight_base + 2 + offset + pair * 2;
                const uint16_t gate_pair = *reinterpret_cast<const uint16_t*>(gate + position);
                const uint16_t up_pair = *reinterpret_cast<const uint16_t*>(up + position);
                #pragma unroll
                for (int lane = 0; lane < 2; ++lane) {
                    const float a = activation[input_base + pair * 2 + lane];
                    const int8_t gate_q = static_cast<int8_t>(gate_pair >> (lane * 8));
                    const int8_t up_q = static_cast<int8_t>(up_pair >> (lane * 8));
                    gate_acc += a * (static_cast<float>(gate_q) * gate_scale);
                    up_acc += a * (static_cast<float>(up_q) * up_scale);
                }
            }
        } else {
            #pragma unroll
            for (int index = 0; index < 8; ++index) {
                const float a = activation[input_base + index];
                const int8_t gate_q = *reinterpret_cast<const int8_t*>(gate + weight_base + 2 + offset + index);
                const int8_t up_q = *reinterpret_cast<const int8_t*>(up + weight_base + 2 + offset + index);
                gate_acc += a * (static_cast<float>(gate_q) * gate_scale);
                up_acc += a * (static_cast<float>(up_q) * up_scale);
            }
        }
    }
    extern __shared__ float published[];
    float* gate_partial = published;
    float* up_partial = published + Threads;
    gate_partial[tid] = gate_acc;
    up_partial[tid] = up_acc;
    __syncthreads();
    if constexpr (HasWarpReduction) {
        if (tid < 32) {
            const float g = reduce_published<Threads>(gate_partial, tid);
            const float u = reduce_published<Threads>(up_partial, tid);
            if (tid == 0) output[static_cast<size_t>(row) * hidden + column] = g * (1.0f / (1.0f + expf(-g))) * u;
        }
    } else {
        for (int stride = Threads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                gate_partial[tid] += gate_partial[tid + stride];
                up_partial[tid] += up_partial[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            const float g = gate_partial[0];
            const float u = up_partial[0];
            output[static_cast<size_t>(row) * hidden + column] = g * (1.0f / (1.0f + expf(-g))) * u;
        }
    }
}

template<bool HasPackedLoads, bool HasWarpReduction, int Threads>
void launch(M40llmCudaContext* ctx, const void* input, const void* gate, const void* up,
            void* output, int rows, int hidden, int k) {
    gate_up_swiglu<HasPackedLoads,HasWarpReduction,Threads><<<dim3(hidden,rows),Threads,2*Threads*sizeof(float),ctx->decode_stream>>>(
        static_cast<const float*>(input),static_cast<const unsigned char*>(gate),
        static_cast<const unsigned char*>(up),static_cast<float*>(output),hidden,k);
}
}

extern "C" int m40llm_ffn_q8_f32_ablation_async(
    M40llmCudaContext* ctx, const void* input, const void* gate, const void* up,
    void* output, int rows, int hidden, int k, int variant) {
    if (!ctx || !input || !gate || !up || !output || input == output) return -1;
    if ((rows != 1 && rows != 4) || hidden <= 0 || k <= 0 || k % 32 != 0 ||
        reinterpret_cast<uintptr_t>(gate) % 2 || reinterpret_cast<uintptr_t>(up) % 2) return -2;
    using namespace m40llm_q8_ffn;
    switch (variant) {
        case 0: launch<false,false,256>(ctx,input,gate,up,output,rows,hidden,k); break;
        case 1: launch<true,false,256>(ctx,input,gate,up,output,rows,hidden,k); break;
        case 2: launch<false,true,256>(ctx,input,gate,up,output,rows,hidden,k); break;
        case 3: launch<true,true,256>(ctx,input,gate,up,output,rows,hidden,k); break;
        case 4: launch<true,true,128>(ctx,input,gate,up,output,rows,hidden,k); break;
        default: return -3;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

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

// Existing separate-scale/value storage; one fixed four-value/128-thread test.
__global__ void gate_up_split_f32x4(
    const float* __restrict__ input, const __half* __restrict__ gate_scales,
    const uint32_t* __restrict__ gate_values, const __half* __restrict__ up_scales,
    const uint32_t* __restrict__ up_values, float* __restrict__ output,
    float* __restrict__ gate_output, float* __restrict__ up_output, int hidden, int k) {
    const int column = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int tasks = k / 4;
    const size_t value_base = static_cast<size_t>(column) * tasks;
    const size_t scale_base = static_cast<size_t>(column) * (k / 32);
    const float4* activation = reinterpret_cast<const float4*>(input + static_cast<size_t>(row) * k);
    float gate_acc = 0.0f, up_acc = 0.0f;
    for (int task = tid; task < tasks; task += 128) {
        const float4 a = activation[task];
        const uint32_t gate_q = gate_values[value_base + task];
        const uint32_t up_q = up_values[value_base + task];
        const float gs = __half2float(gate_scales[scale_base + task / 8]);
        const float us = __half2float(up_scales[scale_base + task / 8]);
        const float values[4] = {a.x, a.y, a.z, a.w};
        #pragma unroll
        for (int index = 0; index < 4; ++index) {
            gate_acc += values[index] * (static_cast<float>(static_cast<int8_t>(gate_q >> (index * 8))) * gs);
            up_acc += values[index] * (static_cast<float>(static_cast<int8_t>(up_q >> (index * 8))) * us);
        }
    }
    extern __shared__ float published[];
    published[tid] = gate_acc;
    published[tid + 128] = up_acc;
    __syncthreads();
    if (tid < 32) {
        const float g = reduce_published<128>(published, tid);
        const float u = reduce_published<128>(published + 128, tid);
        if (tid == 0) {
            const size_t index = static_cast<size_t>(row) * hidden + column;
            output[index] = g * (1.0f / (1.0f + expf(-g))) * u;
            if (gate_output) gate_output[index] = g;
            if (up_output) up_output[index] = u;
        }
    }
}

// Keep each column's task assignment and FP32 tree, sharing only activation
// loads and block setup. An odd final column participates in the barrier.
__global__ void gate_up_split_f32x4_two_columns(
    const float* __restrict__ input, const __half* __restrict__ gate_scales,
    const uint32_t* __restrict__ gate_values, const __half* __restrict__ up_scales,
    const uint32_t* __restrict__ up_values, float* __restrict__ output,
    float* __restrict__ gate_output, float* __restrict__ up_output, int hidden, int k) {
    const int first_column = blockIdx.x * 2;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int tasks = k / 4;
    const float4* activation = reinterpret_cast<const float4*>(input + static_cast<size_t>(row) * k);
    float gate_acc[2] = {0.0f, 0.0f}, up_acc[2] = {0.0f, 0.0f};
    for (int task = tid; task < tasks; task += 128) {
        const float4 a = activation[task];
        const float input_values[4] = {a.x, a.y, a.z, a.w};
        #pragma unroll
        for (int local = 0; local < 2; ++local) {
            const int column = first_column + local;
            if (column < hidden) {
                const size_t value_index = static_cast<size_t>(column) * tasks + task;
                const size_t scale_index = static_cast<size_t>(column) * (k / 32) + task / 8;
                const uint32_t gate_q = gate_values[value_index];
                const uint32_t up_q = up_values[value_index];
                const float gs = __half2float(gate_scales[scale_index]);
                const float us = __half2float(up_scales[scale_index]);
                #pragma unroll
                for (int index = 0; index < 4; ++index) {
                    gate_acc[local] += input_values[index] * (static_cast<float>(static_cast<int8_t>(gate_q >> (index * 8))) * gs);
                    up_acc[local] += input_values[index] * (static_cast<float>(static_cast<int8_t>(up_q >> (index * 8))) * us);
                }
            }
        }
    }
    extern __shared__ float published[];
    #pragma unroll
    for (int local = 0; local < 2; ++local) {
        published[local * 256 + tid] = gate_acc[local];
        published[local * 256 + 128 + tid] = up_acc[local];
    }
    __syncthreads();
    if (tid < 32) {
        #pragma unroll
        for (int local = 0; local < 2; ++local) {
            if (first_column + local < hidden) {
                const float g = reduce_published<128>(published + local * 256, tid);
                const float u = reduce_published<128>(published + local * 256 + 128, tid);
                if (tid == 0) {
                    const size_t index = static_cast<size_t>(row) * hidden + first_column + local;
                    output[index] = g * (1.0f / (1.0f + expf(-g))) * u;
                    if (gate_output) gate_output[index] = g;
                    if (up_output) up_output[index] = u;
                }
            }
        }
    }
}

// Same separate-plane math and ownership as the qualified kernel, with
// Qwen's fixed input width exposed to address generation and loop unrolling.
__global__ void gate_up_split_f32x4_k2048(
    const float* __restrict__ input, const __half* __restrict__ gate_scales,
    const uint32_t* __restrict__ gate_values, const __half* __restrict__ up_scales,
    const uint32_t* __restrict__ up_values, float* __restrict__ output,
    float* __restrict__ gate_output, float* __restrict__ up_output, int hidden) {
    constexpr int k = 2048;
    const int column = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    constexpr int tasks = k / 4;
    const size_t value_base = static_cast<size_t>(column) * tasks;
    const size_t scale_base = static_cast<size_t>(column) * (k / 32);
    const float4* activation = reinterpret_cast<const float4*>(input + static_cast<size_t>(row) * k);
    float gate_acc = 0.0f, up_acc = 0.0f;
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int task = tid + step * 128;
        const float4 a = activation[task];
        const uint32_t gate_q = gate_values[value_base + task];
        const uint32_t up_q = up_values[value_base + task];
        const float gs = __half2float(gate_scales[scale_base + task / 8]);
        const float us = __half2float(up_scales[scale_base + task / 8]);
        const float values[4] = {a.x, a.y, a.z, a.w};
        #pragma unroll
        for (int index = 0; index < 4; ++index) {
            gate_acc += values[index] * (static_cast<float>(static_cast<int8_t>(gate_q >> (index * 8))) * gs);
            up_acc += values[index] * (static_cast<float>(static_cast<int8_t>(up_q >> (index * 8))) * us);
        }
    }
    extern __shared__ float published[];
    published[tid] = gate_acc;
    published[tid + 128] = up_acc;
    __syncthreads();
    if (tid < 32) {
        const float g = reduce_published<128>(published, tid);
        const float u = reduce_published<128>(published + 128, tid);
        if (tid == 0) {
            const size_t index = static_cast<size_t>(row) * hidden + column;
            output[index] = g * (1.0f / (1.0f + expf(-g))) * u;
            if (gate_output) gate_output[index] = g;
            if (up_output) up_output[index] = u;
        }
    }
}

// Lossless joint gate/up records: same one-column task assignment and math,
// but one paired payload and one paired scale address per task.
__global__ void gate_up_joint_f32x4(
    const float* __restrict__ input, const uint32_t* __restrict__ scales,
    const uint2* __restrict__ payload, float* __restrict__ output,
    float* __restrict__ gate_output, float* __restrict__ up_output, int hidden, int k) {
    const int column = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int tasks = k / 4;
    const size_t value_base = static_cast<size_t>(column) * tasks;
    const size_t scale_base = static_cast<size_t>(column) * (k / 32);
    const float4* activation = reinterpret_cast<const float4*>(input + static_cast<size_t>(row) * k);
    float gate_acc = 0.0f, up_acc = 0.0f;
    for (int task = tid; task < tasks; task += 128) {
        const float4 a = activation[task];
        const uint2 q = payload[value_base + task];
        const uint32_t s = scales[scale_base + task / 8];
        const float gs = __half2float(__ushort_as_half(static_cast<unsigned short>(s)));
        const float us = __half2float(__ushort_as_half(static_cast<unsigned short>(s >> 16)));
        const float values[4] = {a.x, a.y, a.z, a.w};
        #pragma unroll
        for (int index = 0; index < 4; ++index) {
            gate_acc += values[index] * (static_cast<float>(static_cast<int8_t>(q.x >> (index * 8))) * gs);
            up_acc += values[index] * (static_cast<float>(static_cast<int8_t>(q.y >> (index * 8))) * us);
        }
    }
    extern __shared__ float published[];
    published[tid] = gate_acc;
    published[tid + 128] = up_acc;
    __syncthreads();
    if (tid < 32) {
        const float g = reduce_published<128>(published, tid);
        const float u = reduce_published<128>(published + 128, tid);
        if (tid == 0) {
            const size_t index = static_cast<size_t>(row) * hidden + column;
            output[index] = g * (1.0f / (1.0f + expf(-g))) * u;
            if (gate_output) gate_output[index] = g;
            if (up_output) up_output[index] = u;
        }
    }
}

template<int Variant>
int launch_split_f32(
    M40llmCudaContext* ctx, const void* input, const void* const* scales,
    const void* const* values, void* output, void* gate_output, void* up_output,
    int rows, int hidden, int k) {
    if (!ctx || !input || !scales || !values || !output || input == output) return -1;
    if ((rows != 1 && rows != 4) || hidden <= 0 || k <= 0 || k % 32 ||
        reinterpret_cast<uintptr_t>(input) % 16 || reinterpret_cast<uintptr_t>(output) % 4) return -2;
    if constexpr (Variant == 3) { if (k != 2048) return -2; }
    for (int i = 0; i < 2; ++i) {
        if (!scales[i] || !values[i] || reinterpret_cast<uintptr_t>(scales[i]) % 2 ||
            reinterpret_cast<uintptr_t>(values[i]) % 4) return -2;
    }
    if (gate_output || up_output) {
        if (!gate_output || !up_output || gate_output == up_output || gate_output == output ||
            up_output == output || gate_output == input || up_output == input ||
            reinterpret_cast<uintptr_t>(gate_output) % 4 || reinterpret_cast<uintptr_t>(up_output) % 4) return -3;
    }
    if constexpr (Variant == 1) {
        gate_up_split_f32x4<<<dim3(hidden,rows),128,1024,ctx->decode_stream>>>(
            static_cast<const float*>(input),static_cast<const __half*>(scales[0]),static_cast<const uint32_t*>(values[0]),
            static_cast<const __half*>(scales[1]),static_cast<const uint32_t*>(values[1]),static_cast<float*>(output),
            static_cast<float*>(gate_output),static_cast<float*>(up_output),hidden,k);
    } else if constexpr (Variant == 2) {
        gate_up_split_f32x4_two_columns<<<dim3(hidden/2+hidden%2,rows),128,2048,ctx->decode_stream>>>(
            static_cast<const float*>(input),static_cast<const __half*>(scales[0]),static_cast<const uint32_t*>(values[0]),
            static_cast<const __half*>(scales[1]),static_cast<const uint32_t*>(values[1]),static_cast<float*>(output),
            static_cast<float*>(gate_output),static_cast<float*>(up_output),hidden,k);
    } else {
        gate_up_split_f32x4_k2048<<<dim3(hidden,rows),128,1024,ctx->decode_stream>>>(
            static_cast<const float*>(input),static_cast<const __half*>(scales[0]),static_cast<const uint32_t*>(values[0]),
            static_cast<const __half*>(scales[1]),static_cast<const uint32_t*>(values[1]),static_cast<float*>(output),
            static_cast<float*>(gate_output),static_cast<float*>(up_output),hidden);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -4;
}
}

extern "C" int m40llm_ffn_q8_split_f32_async(
    M40llmCudaContext* ctx, const void* input, const void* const* scales,
    const void* const* values, void* output, void* gate_output, void* up_output,
    int rows, int hidden, int k) {
    return m40llm_q8_ffn::launch_split_f32<1>(ctx,input,scales,values,output,gate_output,up_output,rows,hidden,k);
}

extern "C" int m40llm_ffn_q8_joint_f32_async(
    M40llmCudaContext* ctx, const void* input, const void* scales,
    const void* values, void* output, void* gate_output, void* up_output,
    int rows, int hidden, int k) {
    if (!ctx || !input || !scales || !values || !output || input == output) return -1;
    if ((rows != 1 && rows != 4) || hidden <= 0 || k <= 0 || k % 32 ||
        reinterpret_cast<uintptr_t>(input) % 16 || reinterpret_cast<uintptr_t>(output) % 4 ||
        reinterpret_cast<uintptr_t>(scales) % 4 || reinterpret_cast<uintptr_t>(values) % 8) return -2;
    if (gate_output || up_output) {
        if (!gate_output || !up_output || gate_output == up_output || gate_output == output ||
            up_output == output || gate_output == input || up_output == input ||
            reinterpret_cast<uintptr_t>(gate_output) % 4 || reinterpret_cast<uintptr_t>(up_output) % 4) return -3;
    }
    m40llm_q8_ffn::gate_up_joint_f32x4<<<dim3(hidden,rows),128,1024,ctx->decode_stream>>>(
        static_cast<const float*>(input),static_cast<const uint32_t*>(scales),static_cast<const uint2*>(values),
        static_cast<float*>(output),static_cast<float*>(gate_output),static_cast<float*>(up_output),hidden,k);
    return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

extern "C" int m40llm_ffn_q8_split_f32_two_columns_async(
    M40llmCudaContext* ctx, const void* input, const void* const* scales,
    const void* const* values, void* output, void* gate_output, void* up_output,
    int rows, int hidden, int k) {
    return m40llm_q8_ffn::launch_split_f32<2>(ctx,input,scales,values,output,gate_output,up_output,rows,hidden,k);
}

extern "C" int m40llm_ffn_q8_split_f32_k2048_async(
    M40llmCudaContext* ctx, const void* input, const void* const* scales,
    const void* const* values, void* output, void* gate_output, void* up_output,
    int rows, int hidden, int k) {
    return m40llm_q8_ffn::launch_split_f32<3>(ctx,input,scales,values,output,gate_output,up_output,rows,hidden,k);
}

namespace m40llm_q8_ffn {
template<bool HasPackedLoads, bool HasWarpReduction, int Threads, bool HasFactoredScales = false, bool HasCoalescedPairs = false>
__global__ void gate_up_swiglu(
    const float* __restrict__ input,
    const unsigned char* __restrict__ gate,
    const unsigned char* __restrict__ up,
    float* __restrict__ output, int hidden, int k,
    float* __restrict__ gate_output, float* __restrict__ up_output) {
    const int column = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int blocks = k / 32;
    const float* activation = input + static_cast<size_t>(row) * k;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    static_assert(!(HasFactoredScales && HasCoalescedPairs), "independent experiments");
    constexpr int values_per_task = HasCoalescedPairs ? 2 : 8;
    constexpr int tasks_per_block = 32 / values_per_task;
    // Original variants retain split4 assignment; adjacent pairs change only
    // lane ownership and vector loads, not activation precision or reduction.
    for (int task = tid; task < blocks * tasks_per_block; task += Threads) {
        const int block = task / tasks_per_block;
        const int offset = (task - block * tasks_per_block) * values_per_task;
        const int input_base = block * 32 + offset;
        const size_t weight_base = (static_cast<size_t>(column) * blocks + block) * 34;
        const float gate_scale = __half2float(*reinterpret_cast<const __half*>(gate + weight_base));
        const float up_scale = __half2float(*reinterpret_cast<const __half*>(up + weight_base));
        if constexpr (HasCoalescedPairs) {
            const float2 a = *reinterpret_cast<const float2*>(activation + input_base);
            const uint16_t gate_pair = *reinterpret_cast<const uint16_t*>(gate + weight_base + 2 + offset);
            const uint16_t up_pair = *reinterpret_cast<const uint16_t*>(up + weight_base + 2 + offset);
            gate_acc += a.x * (static_cast<float>(static_cast<int8_t>(gate_pair)) * gate_scale);
            up_acc += a.x * (static_cast<float>(static_cast<int8_t>(up_pair)) * up_scale);
            gate_acc += a.y * (static_cast<float>(static_cast<int8_t>(gate_pair >> 8)) * gate_scale);
            up_acc += a.y * (static_cast<float>(static_cast<int8_t>(up_pair >> 8)) * up_scale);
        } else if constexpr (HasPackedLoads) {
            float gate_dot = 0.0f;
            float up_dot = 0.0f;
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
                    if constexpr (HasFactoredScales) {
                        gate_dot += a * static_cast<float>(gate_q);
                        up_dot += a * static_cast<float>(up_q);
                    } else {
                        gate_acc += a * (static_cast<float>(gate_q) * gate_scale);
                        up_acc += a * (static_cast<float>(up_q) * up_scale);
                    }
                }
            }
            if constexpr (HasFactoredScales) {
                // Keep activations in FP32, but apply the shared scale once
                // per eight-value task. FP32 association differs from native.
                gate_acc += gate_dot * gate_scale;
                up_acc += up_dot * up_scale;
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
            if (tid == 0) {
                const size_t index = static_cast<size_t>(row) * hidden + column;
                output[index] = g * (1.0f / (1.0f + expf(-g))) * u;
                if constexpr (HasFactoredScales || HasCoalescedPairs) {
                    if (gate_output) gate_output[index] = g;
                    if (up_output) up_output[index] = u;
                }
            }
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

template<bool HasPackedLoads, bool HasWarpReduction, int Threads, bool HasFactoredScales = false, bool HasCoalescedPairs = false>
void launch(M40llmCudaContext* ctx, const void* input, const void* gate, const void* up,
            void* output, int rows, int hidden, int k,
            void* gate_output = nullptr, void* up_output = nullptr) {
    gate_up_swiglu<HasPackedLoads,HasWarpReduction,Threads,HasFactoredScales,HasCoalescedPairs><<<dim3(hidden,rows),Threads,2*Threads*sizeof(float),ctx->decode_stream>>>(
        static_cast<const float*>(input),static_cast<const unsigned char*>(gate),
        static_cast<const unsigned char*>(up),static_cast<float*>(output),hidden,k,
        static_cast<float*>(gate_output),static_cast<float*>(up_output));
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
        case 5: launch<true,true,128,true>(ctx,input,gate,up,output,rows,hidden,k); break;
        case 6:
            if (reinterpret_cast<uintptr_t>(input) % 8) return -2;
            launch<true,true,128,false,true>(ctx,input,gate,up,output,rows,hidden,k); break;
        default: return -3;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

extern "C" int m40llm_ffn_q8_f32_factored_capture_async(
    M40llmCudaContext* ctx, const void* input, const void* gate, const void* up,
    void* output, void* gate_output, void* up_output, int rows, int hidden, int k) {
    if (!ctx || !input || !gate || !up || !output || !gate_output || !up_output) return -1;
    if (input == output || input == gate_output || input == up_output ||
        output == gate_output || output == up_output || gate_output == up_output) return -1;
    if ((rows != 1 && rows != 4) || hidden <= 0 || k <= 0 || k % 32 != 0 ||
        reinterpret_cast<uintptr_t>(gate) % 2 || reinterpret_cast<uintptr_t>(up) % 2 ||
        reinterpret_cast<uintptr_t>(input) % 4 || reinterpret_cast<uintptr_t>(output) % 4 ||
        reinterpret_cast<uintptr_t>(gate_output) % 4 || reinterpret_cast<uintptr_t>(up_output) % 4) return -2;
    m40llm_q8_ffn::launch<true,true,128,true>(ctx,input,gate,up,output,rows,hidden,k,gate_output,up_output);
    return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

extern "C" int m40llm_ffn_q8_f32_coalesced_capture_async(
    M40llmCudaContext* ctx, const void* input, const void* gate, const void* up,
    void* output, void* gate_output, void* up_output, int rows, int hidden, int k) {
    if (!ctx || !input || !gate || !up || !output || !gate_output || !up_output) return -1;
    if (input == output || input == gate_output || input == up_output ||
        output == gate_output || output == up_output || gate_output == up_output) return -1;
    if ((rows != 1 && rows != 4) || hidden <= 0 || k <= 0 || k % 32 != 0 ||
        reinterpret_cast<uintptr_t>(gate) % 2 || reinterpret_cast<uintptr_t>(up) % 2 ||
        reinterpret_cast<uintptr_t>(input) % 8 || reinterpret_cast<uintptr_t>(output) % 4 ||
        reinterpret_cast<uintptr_t>(gate_output) % 4 || reinterpret_cast<uintptr_t>(up_output) % 4) return -2;
    m40llm_q8_ffn::launch<true,true,128,false,true>(ctx,input,gate,up,output,rows,hidden,k,gate_output,up_output);
    return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

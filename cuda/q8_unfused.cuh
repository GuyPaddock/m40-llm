// One fixed unfused decomposition of the qualified split-plane FP32 FFN.
#pragma once
namespace m40llm_q8_unfused {

__global__ void project_four_columns(
    const float* __restrict__ input, const __half* __restrict__ scales,
    const uint32_t* __restrict__ values, float* __restrict__ output,
    int hidden, int k) {
    const int first_column = blockIdx.x * 4;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int tasks = k / 4;
    const float4* activation = reinterpret_cast<const float4*>(
        input + static_cast<size_t>(row) * k);
    float acc[4] = {};
    for (int task = tid; task < tasks; task += 128) {
        const float4 a = activation[task];
        const float x[4] = {a.x, a.y, a.z, a.w};
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int column = first_column + c;
            if (column < hidden) {
                const uint32_t q = values[static_cast<size_t>(column) * tasks + task];
                const float s = __half2float(scales[static_cast<size_t>(column) * (k / 32) + task / 8]);
                #pragma unroll
                for (int index = 0; index < 4; ++index) {
                    acc[c] += x[index] * (static_cast<float>(static_cast<int8_t>(q >> (index * 8))) * s);
                }
            }
        }
    }
    extern __shared__ float published[];
    #pragma unroll
    for (int c = 0; c < 4; ++c) published[c * 128 + tid] = acc[c];
    __syncthreads();
    if (tid < 32) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float v = m40llm_q8_ffn::reduce_published<128>(published + c * 128, tid);
            if (tid == 0 && first_column + c < hidden) {
                output[static_cast<size_t>(row) * hidden + first_column + c] = v;
            }
        }
    }
}

__global__ void swiglu_control_order(
    const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ output, size_t count) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float g = gate[i];
        output[i] = g * (1.0f / (1.0f + expf(-g))) * up[i];
    }
}
}

extern "C" int m40llm_ffn_q8_split_unfused_f32_async(
    M40llmCudaContext* ctx, const void* input, const void* const* scales,
    const void* const* values, void* output, void* gate_output, void* up_output,
    int rows, int hidden, int k) {
    if (!ctx || !input || !scales || !values || !output || !gate_output || !up_output) return -1;
    if ((rows != 1 && rows != 4) || hidden <= 0 || k <= 0 || k % 32 ||
        reinterpret_cast<uintptr_t>(input) % 16 || reinterpret_cast<uintptr_t>(output) % 4 ||
        reinterpret_cast<uintptr_t>(gate_output) % 4 || reinterpret_cast<uintptr_t>(up_output) % 4) return -2;
    if (gate_output == up_output || output == gate_output || output == up_output ||
        input == output || input == gate_output || input == up_output) return -2;
    for (int family = 0; family < 2; ++family) {
        if (!scales[family] || !values[family] ||
            reinterpret_cast<uintptr_t>(scales[family]) % 2 ||
            reinterpret_cast<uintptr_t>(values[family]) % 4) return -2;
    }
    const dim3 grid((static_cast<unsigned>(hidden) + 3) / 4, rows);
    void* outputs[2] = {gate_output, up_output};
    for (int family = 0; family < 2; ++family) {
        m40llm_q8_unfused::project_four_columns<<<grid,128,2048,ctx->decode_stream>>>(
            static_cast<const float*>(input),static_cast<const __half*>(scales[family]),
            static_cast<const uint32_t*>(values[family]),static_cast<float*>(outputs[family]),hidden,k);
        if (cudaGetLastError() != cudaSuccess) return -3;
    }
    const size_t count = static_cast<size_t>(rows) * hidden;
    const size_t tiles = (count + 255) / 256;
    const unsigned blocks = static_cast<unsigned>(tiles > 65535 ? 65535 : tiles);
    m40llm_q8_unfused::swiglu_control_order<<<blocks,256,0,ctx->decode_stream>>>(
        static_cast<const float*>(gate_output),static_cast<const float*>(up_output),
        static_cast<float*>(output),count);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

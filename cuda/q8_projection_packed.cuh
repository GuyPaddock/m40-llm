// Fixed load-vectorization probes; qualified scalar-load kernels stay intact.
#pragma once
namespace m40llm_q8_packed_projection {
__device__ __forceinline__ float accumulate_eight(
    float acc, const float* activation, const unsigned char* payload, float scale) {
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const uint16_t packed = *reinterpret_cast<const uint16_t*>(payload + pair * 2);
        #pragma unroll
        for (int lane = 0; lane < 2; ++lane) {
            const int8_t q = static_cast<int8_t>(packed >> (lane * 8));
            acc += activation[pair * 2 + lane] * (static_cast<float>(q) * scale);
        }
    }
    return acc;
}

__global__ void qkv(
    const float* __restrict__ input, const unsigned char* __restrict__ wq,
    const unsigned char* __restrict__ wk, const unsigned char* __restrict__ wv,
    const float* __restrict__ bq, const float* __restrict__ bk, const float* __restrict__ bv,
    float* __restrict__ q, float* __restrict__ kout, float* __restrict__ v,
    int nq, int nk, int nv, int k) {
    const int column = blockIdx.x;
    const int tid = threadIdx.x;
    if (column >= nq + nk + nv) return;
    const unsigned char* weight = wq;
    const float* bias = bq;
    float* output = q;
    int local_column = column;
    if (column >= nq + nk) {
        weight = wv; bias = bv; output = v; local_column -= nq + nk;
    } else if (column >= nq) {
        weight = wk; bias = bk; output = kout; local_column -= nq;
    }
    const int blocks = k / 32;
    float acc = 0.0f;
    for (int task = tid; task < blocks * 4; task += 256) {
        const int block = task / 4;
        const int offset = (task - block * 4) * 8;
        const int input_base = block * 32 + offset;
        const float4 lo = *reinterpret_cast<const float4*>(input + input_base);
        const float4 hi = *reinterpret_cast<const float4*>(input + input_base + 4);
        const float activation[8] = {lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w};
        const size_t base = (static_cast<size_t>(local_column) * blocks + block) * 34;
        const float scale = __half2float(*reinterpret_cast<const __half*>(weight + base));
        acc = accumulate_eight(acc,activation,weight + base + 2 + offset,scale);
    }
    extern __shared__ float partials[];
    partials[tid] = acc;
    __syncthreads();
    if (tid < 32) {
        float value = m40llm_q8_ffn::reduce_published<256>(partials,tid);
        if (tid == 0) {
            if (bias) value += bias[local_column];
            output[local_column] = value;
        }
    }
}

__global__ void output_four_columns(
    const float* __restrict__ input, const unsigned char* __restrict__ weight,
    float* __restrict__ output, int n, int k) {
    const int first_column = blockIdx.x * 4;
    const int tid = threadIdx.x;
    const int blocks = k / 32;
    float acc[4] = {0.0f,0.0f,0.0f,0.0f};
    for (int task = tid; task < blocks * 4; task += 256) {
        const int block = task / 4;
        const int offset = (task - block * 4) * 8;
        const int input_base = block * 32 + offset;
        const float4 lo = *reinterpret_cast<const float4*>(input + input_base);
        const float4 hi = *reinterpret_cast<const float4*>(input + input_base + 4);
        const float activation[8] = {lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w};
        #pragma unroll
        for (int local = 0; local < 4; ++local) {
            const int column = first_column + local;
            if (column < n) {
                const size_t base = (static_cast<size_t>(column) * blocks + block) * 34;
                const float scale = __half2float(*reinterpret_cast<const __half*>(weight + base));
                acc[local] = accumulate_eight(acc[local],activation,weight + base + 2 + offset,scale);
            }
        }
    }
    extern __shared__ float partials[];
    #pragma unroll
    for (int local = 0; local < 4; ++local) partials[local * 256 + tid] = acc[local];
    __syncthreads();
    if (tid < 32) {
        #pragma unroll
        for (int local = 0; local < 4; ++local) {
            const float value = m40llm_q8_ffn::reduce_published<256>(partials + local * 256,tid);
            if (tid == 0 && first_column + local < n) output[first_column + local] = value;
        }
    }
}
}

extern "C" int m40llm_qkv_f32xq8_0_gguf_packed_warp_async(
    M40llmCudaContext* ctx, const void* input, const void* const* weights,
    const void* const* biases, void* const* outputs, const int* widths,
    int k, uint32_t stream_kind) {
    if (!ctx || !input || !weights || !biases || !outputs || !widths) return -1;
    if (k <= 0 || k % 32 || stream_kind > 1 || reinterpret_cast<uintptr_t>(input) % 16) return -2;
    int64_t total = 0;
    for (int i = 0; i < 3; ++i) {
        if (widths[i] <= 0 || !weights[i] || !outputs[i] || outputs[i] == input ||
            reinterpret_cast<uintptr_t>(weights[i]) % 2 || reinterpret_cast<uintptr_t>(outputs[i]) % 4 ||
            (biases[i] && reinterpret_cast<uintptr_t>(biases[i]) % 4)) return -2;
        for (int j = 0; j < i; ++j) if (outputs[i] == outputs[j]) return -2;
        total += widths[i];
    }
    if (total > INT32_MAX) return -2;
    cudaStream_t stream = stream_kind == 0 ? ctx->prefill_stream : ctx->decode_stream;
    m40llm_q8_packed_projection::qkv<<<static_cast<int>(total),256,1024,stream>>>(
        static_cast<const float*>(input),static_cast<const unsigned char*>(weights[0]),
        static_cast<const unsigned char*>(weights[1]),static_cast<const unsigned char*>(weights[2]),
        static_cast<const float*>(biases[0]),static_cast<const float*>(biases[1]),static_cast<const float*>(biases[2]),
        static_cast<float*>(outputs[0]),static_cast<float*>(outputs[1]),static_cast<float*>(outputs[2]),
        widths[0],widths[1],widths[2],k);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

extern "C" int m40llm_gemm_f32xq8_0_gguf_coltile4_packed_warp_async(
    M40llmCudaContext* ctx, const void* input, const void* weight,
    void* output, int n, int k, uint32_t stream_kind) {
    if (!ctx || !input || !weight || !output || input == output || weight == output) return -1;
    if (n <= 0 || k <= 0 || k % 32 || stream_kind > 1 ||
        reinterpret_cast<uintptr_t>(input) % 16 || reinterpret_cast<uintptr_t>(weight) % 2 ||
        reinterpret_cast<uintptr_t>(output) % 4) return -2;
    cudaStream_t stream = stream_kind == 0 ? ctx->prefill_stream : ctx->decode_stream;
    m40llm_q8_packed_projection::output_four_columns<<<n / 4 + (n % 4 != 0),256,4096,stream>>>(
        static_cast<const float*>(input),static_cast<const unsigned char*>(weight),
        static_cast<float*>(output),n,k);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

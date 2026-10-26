// Fixed exact-reduction probes; production consumers remain unchanged.
#pragma once
#include <cmath>

namespace m40llm_decode_reductions {
__global__ void weighted_rms_norm_warp(
    const float* __restrict__ input, const void* __restrict__ weight,
    float* __restrict__ output, size_t rows, size_t dim,
    float eps, uint32_t weight_dtype) {
    const size_t row = blockIdx.x;
    if (row >= rows) return;
    const uint32_t tid = threadIdx.x;
    const size_t base = row * dim;
    float ss = 0.0f;
    for (size_t i = tid; i < dim; i += blockDim.x) {
        const float x = input[base + i];
        ss += x * x;
    }
    extern __shared__ float reduce[];
    reduce[tid] = ss;
    __syncthreads();
    if (tid < 32) {
        const float sum = m40llm_q8_ffn::reduce_published<256>(reduce,tid);
        if (tid == 0) reduce[0] = rsqrtf(sum / static_cast<float>(dim) + eps);
    }
    __syncthreads();
    const float scale = reduce[0];
    const __half* w_f16 = reinterpret_cast<const __half*>(weight);
    const float* w_f32 = reinterpret_cast<const float*>(weight);
    for (size_t i = tid; i < dim; i += blockDim.x) {
        const float w = weight_dtype == 0 ? __half2float(w_f16[i]) : w_f32[i];
        output[base + i] = input[base + i] * scale * w;
    }
}
}

extern "C" int m40llm_rms_norm_f32_weighted_warp_async(
    M40llmCudaContext* ctx, const void* input, const void* weight, void* output,
    uint32_t rows, uint32_t dim, float eps, uint32_t weight_dtype, uint32_t stream_kind) {
    if (!ctx || !input || !weight || !output) return -1;
    if (!rows || rows > INT32_MAX || !dim || weight_dtype > 1 || stream_kind > 1 ||
        !std::isfinite(eps) || eps < 0 ||
        reinterpret_cast<uintptr_t>(input) % 4 || reinterpret_cast<uintptr_t>(output) % 4 ||
        reinterpret_cast<uintptr_t>(weight) % (weight_dtype == 0 ? 2 : 4) ||
        output == input || output == weight ||
        static_cast<uint64_t>(rows) * dim > SIZE_MAX / sizeof(float)) return -2;
    const cudaStream_t stream = stream_kind == 0 ? ctx->prefill_stream : ctx->decode_stream;
    m40llm_decode_reductions::weighted_rms_norm_warp<<<rows,256,1024,stream>>>(
        static_cast<const float*>(input),weight,static_cast<float*>(output),rows,dim,eps,weight_dtype);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

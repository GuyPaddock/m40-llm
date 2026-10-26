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

struct __align__(8) ArgmaxPartial {
    float value;
    uint32_t index;
};
static_assert(sizeof(ArgmaxPartial) == 8, "argmax scratch ABI requires eight-byte partials");

__device__ __forceinline__ void select_better(ArgmaxPartial& best, ArgmaxPartial other) {
    if (other.value > best.value || (other.value == best.value && other.index < best.index)) best = other;
}

__device__ __forceinline__ ArgmaxPartial reduce_argmax(
    ArgmaxPartial best, ArgmaxPartial* published, uint32_t tid) {
    published[tid] = best;
    __syncthreads();
    if (tid < 32) {
        #pragma unroll
        for (uint32_t warp = 1; warp < 8; ++warp) select_better(best,published[tid+warp*32]);
        #pragma unroll
        for (uint32_t stride = 16; stride > 0; stride >>= 1) {
            const ArgmaxPartial other = {
                __shfl_down_sync(0xffffffffu,best.value,stride),
                __shfl_down_sync(0xffffffffu,best.index,stride)};
            if (tid < stride) select_better(best,other);
        }
    }
    return best;
}

__global__ void argmax_partials(
    const float* __restrict__ values, uint32_t len, ArgmaxPartial* __restrict__ partials) {
    const uint32_t tid = threadIdx.x;
    ArgmaxPartial best = {-FLT_MAX,0};
    // Wide indexing prevents a final stride from wrapping near UINT32_MAX.
    for (size_t i = static_cast<size_t>(blockIdx.x)*blockDim.x+tid; i < len;
         i += static_cast<size_t>(gridDim.x)*blockDim.x) {
        select_better(best,{values[i],static_cast<uint32_t>(i)});
    }
    __shared__ ArgmaxPartial published[256];
    best = reduce_argmax(best,published,tid);
    if (tid == 0) partials[blockIdx.x] = best;
}

__global__ void argmax_reduce(
    const ArgmaxPartial* __restrict__ partials, uint32_t count, uint32_t* __restrict__ output) {
    const uint32_t tid = threadIdx.x;
    ArgmaxPartial best = {-FLT_MAX,0};
    if (tid < count) best = partials[tid];
    __shared__ ArgmaxPartial published[256];
    best = reduce_argmax(best,published,tid);
    if (tid == 0) *output = best.index;
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

extern "C" int m40llm_argmax_f32_partitioned_async(
    M40llmCudaContext* ctx, const void* values, uint32_t len,
    void* scratch, size_t scratch_bytes, void* output, uint32_t stream_kind) {
    if (!ctx || !values || !scratch || !output) return -1;
    const uint32_t blocks = std::min(48u,len/256u+(len%256u!=0));
    if (!len || stream_kind > 1 || reinterpret_cast<uintptr_t>(values) % 4 ||
        reinterpret_cast<uintptr_t>(scratch) % 8 || reinterpret_cast<uintptr_t>(output) % 4 ||
        scratch_bytes < blocks*sizeof(m40llm_decode_reductions::ArgmaxPartial) ||
        output == values || output == scratch || scratch == values) return -2;
    if (cudaSetDevice(ctx->device_id) != cudaSuccess) return -3;
    const cudaStream_t stream = stream_kind == 0 ? ctx->prefill_stream : ctx->decode_stream;
    m40llm_decode_reductions::argmax_partials<<<blocks,256,0,stream>>>(
        static_cast<const float*>(values),len,
        static_cast<m40llm_decode_reductions::ArgmaxPartial*>(scratch));
    if (cudaGetLastError() != cudaSuccess) return -4;
    m40llm_decode_reductions::argmax_reduce<<<1,256,0,stream>>>(
        static_cast<const m40llm_decode_reductions::ArgmaxPartial*>(scratch),blocks,
        static_cast<uint32_t*>(output));
    return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

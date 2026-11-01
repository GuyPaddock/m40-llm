// One fixed native four-column output-projection reduction experiment.
#pragma once
namespace m40llm_q8_output {
__global__ void decode_coltile4_warp(
    const float* __restrict__ A, const unsigned char* __restrict__ B,
    float* __restrict__ C, int N, int K) {
    const int col0 = blockIdx.x * 4;
    const int tid = threadIdx.x;
    const int qk = 32;
    const int split = 4;
    const int vals_per_split = qk / split;
    const int block_bytes = 34;
    const int blocks_per_col = K / qk;
    const int tasks = blocks_per_col * split;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int task = tid; task < tasks; task += blockDim.x) {
        const int block_idx = task / split;
        const int split_idx = task - block_idx * split;
        const int q_offset = split_idx * vals_per_split;
        const int k_base = block_idx * qk + q_offset;
        float a_vals[vals_per_split];
        #pragma unroll
        for (int q_idx = 0; q_idx < vals_per_split; ++q_idx) a_vals[q_idx] = A[k_base + q_idx];
        const size_t block0 = (static_cast<size_t>(col0) * blocks_per_col + block_idx) * block_bytes;
        if (col0 < N) {
            const float scale = __half2float(*reinterpret_cast<const __half*>(B + block0));
            #pragma unroll
            for (int q_idx = 0; q_idx < vals_per_split; ++q_idx) {
                const int8_t q = *reinterpret_cast<const int8_t*>(B + block0 + 2 + q_offset + q_idx);
                acc0 += a_vals[q_idx] * (static_cast<float>(q) * scale);
            }
        }
        if (col0 + 1 < N) {
            const size_t block_base = block0 + static_cast<size_t>(blocks_per_col) * block_bytes;
            const float scale = __half2float(*reinterpret_cast<const __half*>(B + block_base));
            #pragma unroll
            for (int q_idx = 0; q_idx < vals_per_split; ++q_idx) {
                const int8_t q = *reinterpret_cast<const int8_t*>(B + block_base + 2 + q_offset + q_idx);
                acc1 += a_vals[q_idx] * (static_cast<float>(q) * scale);
            }
        }
        if (col0 + 2 < N) {
            const size_t block_base = block0 + static_cast<size_t>(2 * blocks_per_col) * block_bytes;
            const float scale = __half2float(*reinterpret_cast<const __half*>(B + block_base));
            #pragma unroll
            for (int q_idx = 0; q_idx < vals_per_split; ++q_idx) {
                const int8_t q = *reinterpret_cast<const int8_t*>(B + block_base + 2 + q_offset + q_idx);
                acc2 += a_vals[q_idx] * (static_cast<float>(q) * scale);
            }
        }
        if (col0 + 3 < N) {
            const size_t block_base = block0 + static_cast<size_t>(3 * blocks_per_col) * block_bytes;
            const float scale = __half2float(*reinterpret_cast<const __half*>(B + block_base));
            #pragma unroll
            for (int q_idx = 0; q_idx < vals_per_split; ++q_idx) {
                const int8_t q = *reinterpret_cast<const int8_t*>(B + block_base + 2 + q_offset + q_idx);
                acc3 += a_vals[q_idx] * (static_cast<float>(q) * scale);
            }
        }
    }
    extern __shared__ float partials[];
    partials[tid] = acc0;
    partials[tid + 256] = acc1;
    partials[tid + 512] = acc2;
    partials[tid + 768] = acc3;
    __syncthreads();
    if (tid < 32) {
        #pragma unroll
        for (int local = 0; local < 4; ++local) {
            const float value = m40llm_q8_ffn::reduce_published<256>(partials + local * 256, tid);
            if (tid == 0 && col0 + local < N) C[col0 + local] = value;
        }
    }
}
}

extern "C" int m40llm_gemm_f32xq8_0_gguf_coltile4_warp_async(
    M40llmCudaContext* ctx, const void* input, const void* weight,
    void* output, int n, int k, uint32_t stream_kind) {
    if (!ctx || !input || !weight || !output || input == output || weight == output) return -1;
    if (n <= 0 || k <= 0 || k % 32 || stream_kind > 1 ||
        reinterpret_cast<uintptr_t>(input) % 4 || reinterpret_cast<uintptr_t>(weight) % 2 ||
        reinterpret_cast<uintptr_t>(output) % 4) return -2;
    cudaStream_t stream = stream_kind == 0 ? ctx->prefill_stream : ctx->decode_stream;
    m40llm_q8_output::decode_coltile4_warp<<<n / 4 + (n % 4 != 0),256,4096,stream>>>(
        static_cast<const float*>(input),static_cast<const unsigned char*>(weight),
        static_cast<float*>(output),n,k);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

// Benchmark-only split storage with the qualified packed projection geometry.
#pragma once
namespace m40llm_q8_split_projection {

__global__ void qkv(
    const float* __restrict__ input,
    const __half* __restrict__ sq, const int8_t* __restrict__ vq,
    const __half* __restrict__ sk, const int8_t* __restrict__ vk,
    const __half* __restrict__ sv, const int8_t* __restrict__ vv,
    const float* __restrict__ bq, const float* __restrict__ bk, const float* __restrict__ bv,
    float* __restrict__ q, float* __restrict__ kout, float* __restrict__ v,
    int nq, int nk, int nv, int k) {
    const int column = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* scales = sq;
    const int8_t* values = vq;
    const float* bias = bq;
    float* output = q;
    int local_column = column;
    int width = nq;
    if (column >= nq + nk) {
        scales = sv; values = vv; bias = bv; output = v; width = nv; local_column -= nq + nk;
    } else if (column >= nq) {
        scales = sk; values = vk; bias = bk; output = kout; width = nk; local_column -= nq;
    }
    const int blocks = k / 32;
    float acc[1] = {0.0f};
    for (int task = tid; task < blocks * 4; task += 256) {
        const int block = task / 4;
        const int offset = (task % 4) * 8;
        const int input_base = block * 32 + offset;
        const float4 lo = *reinterpret_cast<const float4*>(input + input_base);
        const float4 hi = *reinterpret_cast<const float4*>(input + input_base + 4);
        const float activation[8] = {lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w};
        q8_separate_accumulate<1>(scales,values,activation,local_column,width,blocks,block,offset,acc);
    }
    extern __shared__ float partials[];
    partials[tid] = acc[0];
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
    const float* __restrict__ input, const __half* __restrict__ scales,
    const int8_t* __restrict__ values, float* __restrict__ output, int n, int k) {
    const int first_column = blockIdx.x * 4;
    const int tid = threadIdx.x;
    const int blocks = k / 32;
    float acc[4] = {0.0f,0.0f,0.0f,0.0f};
    for (int task = tid; task < blocks * 4; task += 256) {
        const int block = task / 4;
        const int offset = (task % 4) * 8;
        const int input_base = block * 32 + offset;
        const float4 lo = *reinterpret_cast<const float4*>(input + input_base);
        const float4 hi = *reinterpret_cast<const float4*>(input + input_base + 4);
        const float activation[8] = {lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w};
        q8_separate_accumulate<4>(scales,values,activation,first_column,n,blocks,block,offset,acc);
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

int validate_spans(
    const void* input, const void* const* scales, const void* const* values,
    const void* const* biases, void* const* outputs, const int* widths,
    int count, int k, uint32_t stream_kind) {
    if (!input || !scales || !values || !biases || !outputs || !widths) return -1;
    if (k <= 0 || k % 32 || stream_kind > 1 || reinterpret_cast<uintptr_t>(input) % 16) return -2;
    int64_t total = 0;
    for (int i = 0; i < count; ++i) {
        if (widths[i] <= 0 || !scales[i] || !values[i] || !outputs[i] ||
            reinterpret_cast<uintptr_t>(scales[i]) % 2 || reinterpret_cast<uintptr_t>(values[i]) % 8 ||
            reinterpret_cast<uintptr_t>(outputs[i]) % 4 ||
            (biases[i] && reinterpret_cast<uintptr_t>(biases[i]) % 4)) return -2;
        if (outputs[i] == input) return -2;
        for (int j = 0; j < count; ++j) {
            if (outputs[i] == scales[j] || outputs[i] == values[j] || outputs[i] == biases[j] ||
                (i != j && outputs[i] == outputs[j])) return -2;
        }
        total += widths[i];
    }
    return total <= INT32_MAX ? 0 : -2;
}
}

extern "C" int m40llm_qkv_f32xq8_separate_packed_warp_async(
    M40llmCudaContext* ctx, const void* input, const void* const* scales,
    const void* const* values, const void* const* biases, void* const* outputs,
    const int* widths, int k, uint32_t stream_kind) {
    if (!ctx) return -1;
    const int check = m40llm_q8_split_projection::validate_spans(
        input,scales,values,biases,outputs,widths,3,k,stream_kind);
    if (check) return check;
    cudaStream_t stream = stream_kind == 0 ? ctx->prefill_stream : ctx->decode_stream;
    m40llm_q8_split_projection::qkv<<<widths[0]+widths[1]+widths[2],256,1024,stream>>>(
        static_cast<const float*>(input),
        static_cast<const __half*>(scales[0]),static_cast<const int8_t*>(values[0]),
        static_cast<const __half*>(scales[1]),static_cast<const int8_t*>(values[1]),
        static_cast<const __half*>(scales[2]),static_cast<const int8_t*>(values[2]),
        static_cast<const float*>(biases[0]),static_cast<const float*>(biases[1]),static_cast<const float*>(biases[2]),
        static_cast<float*>(outputs[0]),static_cast<float*>(outputs[1]),static_cast<float*>(outputs[2]),
        widths[0],widths[1],widths[2],k);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

extern "C" int m40llm_gemm_f32xq8_separate_coltile4_packed_warp_async(
    M40llmCudaContext* ctx, const void* input, const void* scales,
    const void* values, void* output, int n, int k, uint32_t stream_kind) {
    if (!ctx) return -1;
    const void* bias = nullptr;
    const int check = m40llm_q8_split_projection::validate_spans(
        input,&scales,&values,&bias,&output,&n,1,k,stream_kind);
    if (check) return check;
    cudaStream_t stream = stream_kind == 0 ? ctx->prefill_stream : ctx->decode_stream;
    m40llm_q8_split_projection::output_four_columns<<<n/4+(n%4!=0),256,4096,stream>>>(
        static_cast<const float*>(input),static_cast<const __half*>(scales),
        static_cast<const int8_t*>(values),static_cast<float*>(output),n,k);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}
